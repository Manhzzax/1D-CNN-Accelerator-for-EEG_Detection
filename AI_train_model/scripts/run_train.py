import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from src.model import EEG1DCNN
from src.utils import set_seed, plot_training_history, plot_confusion_matrix, outputs_dir

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
    
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    
    # Enable multiple workers and pinned memory for server optimization
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
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
    model = EEG1DCNN().to(device)
    
    # 5. Set loss, optimizer, and AMP Scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
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
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    
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
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_save_path = os.path.join(outputs_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model to {model_save_path}")
            
    print("\nTraining completed successfully.")
    
    # 7. Plot history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 8. Evaluate on Test Set
    print("\nEvaluating best model on Test Set...")
    best_model = EEG1DCNN().to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(outputs_dir, "best_model.pth"), map_location=device, weights_only=True)
    )
    best_model.eval()
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = best_model(inputs)
            else:
                outputs = best_model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(np.int64)
    all_targets = np.array(all_targets)
    
    # Metrics calculation
    test_acc = 100. * np.sum(all_preds == all_targets) / len(all_targets)
    print(f"  Test Accuracy: {test_acc:.2f}%")
    
    report = classification_report(all_targets, all_preds, target_names=['Non-Seizure', 'Seizure'], digits=4)
    cm = confusion_matrix(all_targets, all_preds)
    window_metrics = {
        "accuracy": float(test_acc / 100.0),
        "balanced_accuracy": float(balanced_accuracy_score(all_targets, all_preds)),
        "sensitivity": float(recall_score(all_targets, all_preds, zero_division=0)),
        "precision": float(precision_score(all_targets, all_preds, zero_division=0)),
        "f1": float(f1_score(all_targets, all_preds, zero_division=0)),
        "auroc": float(roc_auc_score(all_targets, all_probs)),
        "average_precision": float(average_precision_score(all_targets, all_probs)),
        "threshold": 0.5,
    }
    
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
    np.save(os.path.join(outputs_dir, "test_probabilities.npy"), all_probs)
    np.save(os.path.join(outputs_dir, "test_targets.npy"), all_targets)
    print(f"Window metrics: {json.dumps(window_metrics, sort_keys=True)}")
    
    # Plot and save CM
    plot_confusion_matrix(cm)
    print("=" * 60)

if __name__ == "__main__":
    main()
