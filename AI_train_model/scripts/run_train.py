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
from src.model import build_model, build_model_from_run, save_model_spec
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


def _window_metrics(targets, probabilities):
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "sensitivity": float(recall_score(targets, predictions, zero_division=0)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "average_precision": float(average_precision_score(targets, probabilities)),
        "threshold": 0.5,
    }


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
    set_seed(config['data']['seed'])
    
    print("=" * 60)
    print("RUNNING MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # 3. Get datasets
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets()
    
    batch_size = int(os.environ.get('CHBMIT_TRAIN_BATCH_SIZE', config['training']['batch_size']))
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    class_balanced_batches = _env_bool(
        'CHBMIT_CLASS_BALANCED_BATCHES', config['training'].get('class_balanced_batches', False)
    )
    train_sampler = None
    if class_balanced_batches:
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
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
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
    
    # 5. Set loss, optimizer, and AMP Scaler
    learning_rate = float(os.environ.get('CHBMIT_TRAIN_LEARNING_RATE', config['training']['learning_rate']))
    weight_decay = float(os.environ.get('CHBMIT_TRAIN_WEIGHT_DECAY', config['training']['weight_decay']))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['training']['lr_factor'], 
        patience=config['training']['lr_patience']
    )
    
    # Check if GPU training and AMP is active
    use_amp = config['training'].get('use_amp', False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    if use_amp:
        print("Automatic Mixed Precision (AMP) training enabled (FP16).")
    else:
        print("Standard single-precision (FP32) training enabled.")
        
    # 6. Training loop
    epochs = int(os.environ.get('CHBMIT_TRAIN_EPOCHS', config['training']['epochs']))
    early_stopping = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping.get('enabled', False)
    early_stopping_monitor = early_stopping.get('monitor', 'val_loss')
    min_epochs = int(early_stopping.get('min_epochs', 1))
    early_stopping_patience = int(os.environ.get(
        'CHBMIT_EARLY_STOPPING_PATIENCE', early_stopping.get('patience', epochs)
    ))
    min_delta = float(early_stopping.get('min_delta', 0.0))
    if early_stopping_monitor != 'val_loss':
        raise ValueError("Only val_loss early stopping is supported")
    if min_epochs < 1 or early_stopping_patience < 1 or min_delta < 0:
        raise ValueError("Invalid early_stopping configuration")
    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement_epochs = 0
    stopped_early = False
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Train epoch
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                # Forward with mixed precision
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                # Backward and step with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 forward & backward
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
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
        
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Save best model
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            no_improvement_epochs = 0
            model_save_path = os.path.join(outputs_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model to {model_save_path}")
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
        "batch_size": batch_size,
        "class_balanced_batches": class_balanced_batches,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
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
        "stopped_early": stopped_early,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "monitor": early_stopping_monitor,
            "min_epochs": min_epochs,
            "patience": early_stopping_patience,
            "min_delta": min_delta,
        },
        "hyperparameters": hyperparameters,
        "window_validation_metrics": validation_window_metrics,
    }

    # Validation-only search trials must not consume the held-out test metrics.
    if _env_bool('CHBMIT_SKIP_TEST_EVALUATION', False):
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
