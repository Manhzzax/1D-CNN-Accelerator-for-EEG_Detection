import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import json

# Add project root directory to sys.path to enable src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config, get_train_val_test_datasets
from src.eeg_augmentation import mild_eeg_augmentation
from src.group_dro import GroupDROObjective
from src.model import (
    SubjectDiscriminator,
    build_model,
    build_model_from_run,
    gradient_reverse,
    save_model_spec,
)
from src.supervised_contrastive import supervised_contrastive_loss
from src.training_sampling import patient_group_balanced_weights
from src.utils import set_seed, plot_training_history, plot_confusion_matrix, outputs_dir


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false/1/0")


def _training_seed(config):
    value = os.environ.get('CHBMIT_TRAIN_SEED', config['data']['seed'])
    try:
        seed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError('CHBMIT_TRAIN_SEED must be a non-negative integer') from error
    if seed < 0:
        raise ValueError('CHBMIT_TRAIN_SEED must be a non-negative integer')
    return seed


def _window_metrics(targets, probabilities, threshold=0.5):
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    metrics = {
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "sensitivity": float(recall_score(targets, predictions, zero_division=0)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "threshold": float(threshold),
    }
    # AUROC/AP need both classes in targets; keep previous behavior when possible.
    try:
        metrics["auroc"] = float(roc_auc_score(targets, probabilities))
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["average_precision"] = float(average_precision_score(targets, probabilities))
    except ValueError:
        metrics["average_precision"] = float("nan")
    return metrics


def _score_window_loader(model, loader, device, use_amp):
    probabilities = []
    targets = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            targets.extend(batch_targets.numpy())
    return np.asarray(probabilities), np.asarray(targets)


def main():
    config = load_config()
    
    # 1. Set CPU threads limit to avoid shared server overloading
    num_threads = config['training'].get('num_threads', 4)
    torch.set_num_threads(num_threads)
    print(f"PyTorch CPU threads restricted to: {torch.get_num_threads()}")
    
    # 2. Set seed
    training_seed = _training_seed(config)
    set_seed(training_seed)
    print(f"Set training seed to {training_seed} for reproducibility.")
    
    print("=" * 60)
    print("RUNNING MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # 3. Keep an unopened outer test partition out of process memory during
    # validation-only V2 architecture and hyperparameter selection.
    skip_test_evaluation = _env_bool('CHBMIT_SKIP_TEST_EVALUATION', False)
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        include_test=not skip_test_evaluation
    )
    
    batch_size = int(os.environ.get('CHBMIT_TRAIN_BATCH_SIZE', config['training']['batch_size']))
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    class_balanced_batches = _env_bool(
        'CHBMIT_CLASS_BALANCED_BATCHES', config['training'].get('class_balanced_batches', False)
    )
    patient_group_balanced_batches = _env_bool(
        'CHBMIT_PATIENT_GROUP_BALANCED_BATCHES',
        config['training'].get('patient_group_balanced_batches', False),
    )
    train_sampler = None
    sampling_strata = None
    if patient_group_balanced_batches:
        if train_dataset.domain_labels is None:
            raise ValueError('Patient-group-balanced batches require train patient-group labels')
        importance = train_dataset.sampling_weights.to(torch.float64)
        weights, sampling_strata = patient_group_balanced_weights(
            train_dataset.y.numpy(),
            train_dataset.domain_labels.numpy(),
            importance.numpy(),
        )
        sample_weights = torch.from_numpy(weights)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(
            'Patient-group-balanced training batches enabled: equal probability across '
            f'{len(sampling_strata)} observed class/patient-group strata | '
            f'{int(train_dataset.domain_labels.max().item()) + 1} train patient groups'
        )
    elif class_balanced_batches:
        class_counts = torch.bincount(train_dataset.y, minlength=2).to(torch.float64)
        if torch.any(class_counts == 0):
            raise ValueError("Class-balanced batches require both training classes")
        sample_weights = torch.empty(len(train_dataset), dtype=torch.double)
        importance = train_dataset.sampling_weights.to(torch.double)
        for class_index in range(len(class_counts)):
            class_mask = train_dataset.y == class_index
            class_importance = importance[class_mask]
            sample_weights[class_mask] = class_importance / class_importance.sum()
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(
            "Class-balanced training batches enabled; source counts: "
            f"{class_counts.tolist()} | sampling weight range: "
            f"{importance.min().item():.1f}-{importance.max().item():.1f}"
        )
    
    # Enable multiple workers and pinned memory for server optimization
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory)
        if test_dataset is not None
        else None
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 4. Initialize model
    model = build_model().to(device)
    save_model_spec(outputs_dir, model)
    print(
        f"Model: {model.architecture_name} | parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    domain_config = config['training'].get('subject_adversarial', {})
    subject_adversarial = _env_bool(
        'CHBMIT_SUBJECT_ADVERSARIAL', domain_config.get('enabled', False)
    )
    domain_loss_coefficient = float(os.environ.get(
        'CHBMIT_SUBJECT_ADVERSARIAL_COEFFICIENT', domain_config.get('coefficient', 0.05)
    ))
    domain_hidden_features = int(os.environ.get(
        'CHBMIT_SUBJECT_ADVERSARIAL_HIDDEN_FEATURES', domain_config.get('hidden_features', 16)
    ))
    domain_discriminator = None
    train_domain_count = (
        int(train_dataset.domain_labels.max().item()) + 1
        if train_dataset.domain_labels is not None else 0
    )
    if subject_adversarial:
        if not hasattr(model, 'forward_features') or not hasattr(model, 'classifier'):
            raise ValueError(
                'Subject-adversarial training currently requires a model with forward_features and classifier'
            )
        if train_dataset.domain_labels is None:
            raise ValueError('Subject-adversarial training requires train patient-group labels')
        if not 0.0 < domain_loss_coefficient <= 1.0:
            raise ValueError('CHBMIT_SUBJECT_ADVERSARIAL_COEFFICIENT must be in (0, 1]')
        domain_discriminator = SubjectDiscriminator(
            model.classifier.in_features,
            domain_hidden_features,
            train_domain_count,
        ).to(device)
        print(
            'Source-only subject-adversarial training enabled: '
            f'{train_domain_count} train patient groups | GRL coefficient {domain_loss_coefficient:g} | '
            f'training-only head {domain_hidden_features} hidden features'
        )

    group_dro_config = config['training'].get('group_dro', {})
    group_dro_enabled = _env_bool(
        'CHBMIT_GROUP_DRO', group_dro_config.get('enabled', False)
    )
    group_dro_eta = float(os.environ.get(
        'CHBMIT_GROUP_DRO_ETA', group_dro_config.get('eta', 0.1)
    ))
    group_dro_objective = None
    if group_dro_enabled:
        if subject_adversarial:
            raise ValueError('GroupDRO and subject-adversarial training are separate controlled ablations')
        if train_dataset.domain_labels is None:
            raise ValueError('GroupDRO requires train patient-group labels')
        group_dro_objective = GroupDROObjective(train_domain_count, group_dro_eta)
        print(
            'Source-only GroupDRO enabled: '
            f'{train_domain_count} train patient groups | eta {group_dro_eta:g}'
        )

    contrastive_config = config['training'].get('supervised_contrastive', {})
    supervised_contrastive = _env_bool(
        'CHBMIT_SUPERVISED_CONTRASTIVE', contrastive_config.get('enabled', False)
    )
    contrastive_coefficient = float(os.environ.get(
        'CHBMIT_SUPERVISED_CONTRASTIVE_COEFFICIENT',
        contrastive_config.get('coefficient', 0.05),
    ))
    contrastive_temperature = float(os.environ.get(
        'CHBMIT_SUPERVISED_CONTRASTIVE_TEMPERATURE',
        contrastive_config.get('temperature', 0.1),
    ))
    if supervised_contrastive:
        if subject_adversarial or group_dro_enabled:
            raise ValueError(
                'Supervised contrastive, subject-adversarial, and GroupDRO are separate controlled ablations'
            )
        if not hasattr(model, 'forward_features') or not hasattr(model, 'classifier'):
            raise ValueError(
                'Supervised contrastive training requires a model with forward_features and classifier'
            )
        if not 0.0 < contrastive_coefficient <= 1.0:
            raise ValueError('CHBMIT_SUPERVISED_CONTRASTIVE_COEFFICIENT must be in (0, 1]')
        if contrastive_temperature <= 0.0:
            raise ValueError('CHBMIT_SUPERVISED_CONTRASTIVE_TEMPERATURE must be positive')
        print(
            'Training-only supervised contrastive loss enabled: '
            f'coefficient {contrastive_coefficient:g} | temperature {contrastive_temperature:g} | '
            f'inference graph unchanged'
        )

    augmentation_config = config['training'].get('mild_eeg_augmentation', {})
    mild_augmentation = _env_bool(
        'CHBMIT_MILD_EEG_AUGMENTATION', augmentation_config.get('enabled', False)
    )
    augmentation_gain_delta = float(os.environ.get(
        'CHBMIT_MILD_EEG_AUGMENTATION_GAIN_DELTA',
        augmentation_config.get('gain_delta', 0.1),
    ))
    augmentation_noise_std = float(os.environ.get(
        'CHBMIT_MILD_EEG_AUGMENTATION_NOISE_STD',
        augmentation_config.get('noise_std', 0.02),
    ))
    if mild_augmentation:
        if subject_adversarial or group_dro_enabled or supervised_contrastive:
            raise ValueError(
                'Mild EEG augmentation, supervised contrastive, subject-adversarial, and GroupDRO are separate controlled ablations'
            )
        if not 0.0 <= augmentation_gain_delta < 1.0:
            raise ValueError('CHBMIT_MILD_EEG_AUGMENTATION_GAIN_DELTA must be in [0, 1)')
        if augmentation_noise_std < 0.0:
            raise ValueError('CHBMIT_MILD_EEG_AUGMENTATION_NOISE_STD must be non-negative')
        print(
            'Training-only mild EEG augmentation enabled: '
            f'shared gain +/-{augmentation_gain_delta:g} | z-score noise std {augmentation_noise_std:g} | '
            f'inference graph unchanged'
        )
    
    # 5. Set loss, optimizer, and AMP Scaler
    learning_rate = float(os.environ.get('CHBMIT_TRAIN_LEARNING_RATE', config['training']['learning_rate']))
    weight_decay = float(os.environ.get('CHBMIT_TRAIN_WEIGHT_DECAY', config['training']['weight_decay']))
    criterion = nn.CrossEntropyLoss()
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer_parameters = list(model.parameters())
    if domain_discriminator is not None:
        optimizer_parameters.extend(domain_discriminator.parameters())
    optimizer_name = os.environ.get('CHBMIT_TRAIN_OPTIMIZER', 'adam').strip().lower()
    optimizer_class = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError("CHBMIT_TRAIN_OPTIMIZER must be 'adam' or 'adamw'")
    optimizer = optimizer_class(
        optimizer_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler_factor = float(os.environ.get(
        'CHBMIT_LR_SCHEDULER_FACTOR', config['training']['lr_factor']
    ))
    scheduler_patience = int(os.environ.get(
        'CHBMIT_LR_SCHEDULER_PATIENCE', config['training']['lr_patience']
    ))
    if not 0.0 < scheduler_factor < 1.0 or scheduler_patience < 0:
        raise ValueError('Invalid ReduceLROnPlateau factor or patience')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    
    # Check if GPU training and AMP is active
    use_amp = config['training'].get('use_amp', False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    if use_amp:
        print("Automatic Mixed Precision (AMP) training enabled (FP16).")
    else:
        print("Standard single-precision (FP32) training enabled.")

    def compute_training_losses(inputs, targets, domain_targets):
        if domain_discriminator is None and not supervised_contrastive:
            outputs = model(inputs)
            per_sample_classification_losses = per_sample_criterion(outputs, targets)
            classification_loss = per_sample_classification_losses.mean()
            optimization_classification_loss = (
                group_dro_objective(per_sample_classification_losses, domain_targets)
                if group_dro_objective is not None else classification_loss
            )
            return outputs, classification_loss, optimization_classification_loss, None, None
        features = model.forward_features(inputs)
        outputs = model.classifier(features)
        classification_loss = criterion(outputs, targets)
        domain_loss = None
        if domain_discriminator is not None:
            domain_logits = domain_discriminator(
                gradient_reverse(features, domain_loss_coefficient)
            )
            domain_loss = criterion(domain_logits, domain_targets)
        contrastive_loss = (
            supervised_contrastive_loss(features, targets, contrastive_temperature)
            if supervised_contrastive else None
        )
        return outputs, classification_loss, classification_loss, domain_loss, contrastive_loss
        
    # 6. Training loop
    epochs = int(os.environ.get('CHBMIT_TRAIN_EPOCHS', config['training']['epochs']))
    early_stopping = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping.get('enabled', False)
    early_stopping_monitor = early_stopping.get('monitor', 'val_loss')
    min_epochs = int(os.environ.get(
        'CHBMIT_EARLY_STOPPING_MIN_EPOCHS', early_stopping.get('min_epochs', 1)
    ))
    early_stopping_patience = int(os.environ.get(
        'CHBMIT_EARLY_STOPPING_PATIENCE', early_stopping.get('patience', epochs)
    ))
    min_delta = float(os.environ.get(
        'CHBMIT_EARLY_STOPPING_MIN_DELTA', early_stopping.get('min_delta', 0.0)
    ))
    if early_stopping_monitor != 'val_loss':
        raise ValueError("Only val_loss early stopping is supported")
    if min_epochs < 1 or early_stopping_patience < 1 or min_delta < 0:
        raise ValueError("Invalid early_stopping configuration")
    best_val_loss = float('inf')
    best_patience_loss = float('inf')
    best_epoch = 0
    no_improvement_epochs = 0
    stopped_early = False
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    domain_losses = []
    contrastive_losses = []
    group_dro_weights = []
    learning_rates = []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Train epoch
        model.train()
        if domain_discriminator is not None:
            domain_discriminator.train()
        running_loss = 0.0
        running_domain_loss = 0.0
        running_contrastive_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            domain_targets = batch[2] if (subject_adversarial or group_dro_enabled) else None
            inputs, targets = inputs.to(device), targets.to(device)
            if domain_targets is not None:
                domain_targets = domain_targets.to(device)
            if mild_augmentation:
                inputs = mild_eeg_augmentation(
                    inputs, augmentation_gain_delta, augmentation_noise_std
                )
            optimizer.zero_grad()
            
            if use_amp:
                # Forward with mixed precision
                with torch.amp.autocast(device_type="cuda"):
                    outputs, classification_loss, optimization_classification_loss, domain_loss, contrastive_loss = compute_training_losses(
                        inputs, targets, domain_targets
                    )
                    loss = optimization_classification_loss
                    if domain_loss is not None:
                        loss = loss + domain_loss
                    if contrastive_loss is not None:
                        loss = loss + contrastive_coefficient * contrastive_loss
                # Backward and step with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 forward & backward
                outputs, classification_loss, optimization_classification_loss, domain_loss, contrastive_loss = compute_training_losses(
                    inputs, targets, domain_targets
                )
                loss = optimization_classification_loss
                if domain_loss is not None:
                    loss = loss + domain_loss
                if contrastive_loss is not None:
                    loss = loss + contrastive_coefficient * contrastive_loss
                loss.backward()
                optimizer.step()
            
            running_loss += classification_loss.item() * inputs.size(0)
            if domain_loss is not None:
                running_domain_loss += domain_loss.item() * inputs.size(0)
            if contrastive_loss is not None:
                running_contrastive_loss += contrastive_loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * correct_train / total_train
        
        # Validate epoch
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = 100. * correct_val / total_val
        
        scheduler.step(epoch_val_loss)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        learning_rates.append(float(optimizer.param_groups[0]['lr']))
        domain_losses.append(
            running_domain_loss / total_train if domain_discriminator is not None else None
        )
        contrastive_losses.append(
            running_contrastive_loss / total_train if supervised_contrastive else None
        )
        group_dro_weights.append(
            group_dro_objective.weights().cpu().tolist() if group_dro_objective is not None else None
        )
        
        domain_loss_message = (
            f" | Domain Loss: {running_domain_loss / total_train:.4f}"
            if domain_discriminator is not None else ""
        )
        contrastive_loss_message = (
            f" | SupCon Loss: {running_contrastive_loss / total_train:.4f}"
            if supervised_contrastive else ""
        )
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%{domain_loss_message}{contrastive_loss_message}")
        
        # Always preserve the absolute validation-loss minimum. `min_delta`
        # controls only whether patience is reset, not which checkpoint is evaluated.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            model_save_path = os.path.join(outputs_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model to {model_save_path}")

        if epoch_val_loss < best_patience_loss - min_delta:
            best_patience_loss = epoch_val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if (
            early_stopping_enabled
            and epoch + 1 >= min_epochs
            and no_improvement_epochs >= early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch + 1}: validation loss did not improve by "
                f"at least {min_delta} for {no_improvement_epochs} epochs."
            )
            break
            
    print("\nTraining completed successfully.")
    
    # 7. Plot history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    hyperparameters = {
        "training_seed": training_seed,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "class_balanced_batches": class_balanced_batches,
        "patient_group_balanced_batches": patient_group_balanced_batches,
        "patient_group_sampling": {
            "strategy": "equal_observed_class_patient_group_strata" if patient_group_balanced_batches else None,
            "strata": sampling_strata,
        },
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": scheduler_factor,
            "patience": scheduler_patience,
        },
        "subject_adversarial": {
            "enabled": subject_adversarial,
            "gradient_reversal_coefficient": domain_loss_coefficient if subject_adversarial else None,
            "hidden_features": domain_hidden_features if subject_adversarial else None,
            "train_patient_groups": train_domain_count if subject_adversarial else None,
            "training_only_head": subject_adversarial,
        },
        "group_dro": {
            "enabled": group_dro_enabled,
            "eta": group_dro_eta if group_dro_enabled else None,
            "train_patient_groups": train_domain_count if group_dro_enabled else None,
            "training_only_objective": group_dro_enabled,
        },
        "supervised_contrastive": {
            "enabled": supervised_contrastive,
            "coefficient": contrastive_coefficient if supervised_contrastive else None,
            "temperature": contrastive_temperature if supervised_contrastive else None,
            "training_only_objective": supervised_contrastive,
            "inference_parameter_delta": 0,
        },
        "mild_eeg_augmentation": {
            "enabled": mild_augmentation,
            "shared_gain_delta": augmentation_gain_delta if mild_augmentation else None,
            "zscore_noise_std": augmentation_noise_std if mild_augmentation else None,
            "training_only_transform": mild_augmentation,
            "inference_parameter_delta": 0,
        },
    }
    with open(os.path.join(outputs_dir, "hyperparameters.json"), "w") as output_file:
        json.dump(hyperparameters, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    # Persist validation metrics from the selected checkpoint. Validation is 1:1
    # sampled in the locked protocol, so its raw accuracy is interpretable.
    best_model = build_model_from_run(outputs_dir).to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(outputs_dir, "best_model.pth"), map_location=device, weights_only=True)
    )
    validation_probabilities, validation_targets = _score_window_loader(
        best_model, val_loader, device, use_amp
    )
    validation_window_metrics = _window_metrics(validation_targets, validation_probabilities)
    with open(os.path.join(outputs_dir, "validation_window_metrics.json"), "w") as output_file:
        json.dump(validation_window_metrics, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Validation window metrics: {json.dumps(validation_window_metrics, sort_keys=True)}")

    training_summary = {
        "epochs_requested": epochs,
        "epochs_completed": len(train_losses),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_patience_validation_loss": best_patience_loss,
        "stopped_early": stopped_early,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "monitor": early_stopping_monitor,
            "min_epochs": min_epochs,
            "patience": early_stopping_patience,
            "min_delta": min_delta,
        },
        "hyperparameters": hyperparameters,
        "subject_adversarial_domain_loss_per_epoch": domain_losses,
        "supervised_contrastive_loss_per_epoch": contrastive_losses,
        "group_dro_weights_per_epoch": group_dro_weights,
        "history": {
            "train_loss": train_losses,
            "validation_loss": val_losses,
            "train_accuracy": train_accs,
            "validation_accuracy": val_accs,
            "learning_rate": learning_rates,
        },
        "window_validation_metrics": validation_window_metrics,
    }

    # Validation-only search trials must not consume the held-out test metrics.
    if skip_test_evaluation:
        with open(os.path.join(outputs_dir, "training_summary.json"), "w") as output_file:
            json.dump(training_summary, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print("Test evaluation skipped by CHBMIT_SKIP_TEST_EVALUATION.")
        return

    # 8. Evaluate on Test Set
    print("\nEvaluating best model on Test Set...")
    best_model = build_model_from_run(outputs_dir).to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(outputs_dir, "best_model.pth"), map_location=device, weights_only=True)
    )
    best_model.eval()
    
    all_probs, all_targets = _score_window_loader(best_model, test_loader, device, use_amp)
    all_preds = (all_probs >= 0.5).astype(np.int64)
    
    # Metrics calculation
    test_acc = 100. * np.sum(all_preds == all_targets) / len(all_targets)
    print(f"  Test Accuracy: {test_acc:.2f}%")
    
    report = classification_report(all_targets, all_preds, target_names=['Non-Seizure', 'Seizure'], digits=4)
    cm = confusion_matrix(all_targets, all_preds)
    window_metrics = _window_metrics(all_targets, all_probs)
    training_summary["window_test_metrics"] = window_metrics
    
    print("\nClassification Report:")
    print(report)
    
    # Save Report File
    report_path = os.path.join(outputs_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Model Evaluation Report ===\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nWindow-Level Metrics:\n")
        f.write(json.dumps(window_metrics, indent=2, sort_keys=True))
    print(f"Saved classification report to: {report_path}")
    with open(os.path.join(outputs_dir, "window_metrics.json"), "w") as f:
        json.dump(window_metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(outputs_dir, "training_summary.json"), "w") as f:
        json.dump(training_summary, f, indent=2, sort_keys=True)
        f.write("\n")
    np.save(os.path.join(outputs_dir, "test_probabilities.npy"), all_probs)
    np.save(os.path.join(outputs_dir, "test_targets.npy"), all_targets)
    print(f"Window metrics: {json.dumps(window_metrics, sort_keys=True)}")
    
    # Plot and save CM
    plot_confusion_matrix(cm)
    print("=" * 60)

if __name__ == "__main__":
    main()
