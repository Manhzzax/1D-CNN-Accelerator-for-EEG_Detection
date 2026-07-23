import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Helper path logic
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
outputs_dir = os.path.join(project_dir, "outputs")

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed} for reproducibility.")

def fold_batchnorm(conv, bn):
    """
    Folds BatchNorm1d parameters into Conv1d weights and biases.
    W_folded = W * gamma / sqrt(var + eps)
    B_folded = (B - mean) * gamma / sqrt(var + eps) + beta
    """
    # Get parameters
    w_conv = conv.weight.data
    if conv.bias is not None:
        b_conv = conv.bias.data
    else:
        b_conv = torch.zeros(conv.out_channels)
        
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps
    
    # Calculate scale factor
    scale = gamma / torch.sqrt(var + eps)
    
    # Fold weights and biases
    w_folded = w_conv * scale.view(-1, 1, 1)
    b_folded = (b_conv - mean) * scale + beta
    
    return w_folded, b_folded

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Plots training and validation loss and accuracy curves.
    Saves to outputs/loss_accuracy_curves.png.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    filepath = os.path.join(outputs_dir, "loss_accuracy_curves.png")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs, label='Val Acc', color='red')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved training curve plots to {filepath}")

def plot_confusion_matrix(cm, labels=['Non-Seizure', 'Seizure']):
    """
    Plots the final test set confusion matrix as a Seaborn heatmap.
    Saves to outputs/confusion_matrix.png.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    filepath = os.path.join(outputs_dir, "confusion_matrix.png")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to {filepath}")
