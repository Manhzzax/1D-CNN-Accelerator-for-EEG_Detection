import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

# Helper path logic
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
config_path = os.path.join(project_dir, "config", "config.yaml")

def load_config():
    """
    Loads project configurations from config.yaml.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class EEGDataset(Dataset):
    """
    Custom PyTorch Dataset for EEG signals.
    Shape input to (batch_size, channels, length) for Conv1d.
    For CHB-MIT: channels = 23, length = 256.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) # shape: (N, 23, 256)
        self.y = torch.tensor(y, dtype=torch.long)     # shape: (N,)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_train_val_test_datasets():
    """
    Loads preprocessed CHB-MIT segments (.npz), splits into Train/Val/Test,
    applies channel-wise standardization using fitted parameters on the training set,
    and returns PyTorch Datasets.
    """
    config = load_config()
    data_dir = os.path.join(project_dir, "data")
    preprocessed_path = os.path.join(data_dir, config['data']['preprocessed_filename'])
    
    if not os.path.exists(preprocessed_path):
        print(f"\nERROR: Preprocessed dataset not found at: {preprocessed_path}")
        print("Please run MNE preprocessing first using:")
        print("  python main.py --mode preprocess\n")
        raise FileNotFoundError(f"Missing preprocessed dataset file: {preprocessed_path}")
        
    print(f"Loading preprocessed dataset from {preprocessed_path}...")
    data = np.load(preprocessed_path)
    X = data['X'] # shape: (N, 23, 256)
    y = data['y'] # shape: (N,)
    
    print(f"Loaded dataset shape: X={X.shape} | y={y.shape}")
    
    # Split: Train (80%), Val (10%), Test (10%)
    test_ratio = config['data']['split_ratios']['test']
    val_ratio = config['data']['split_ratios']['val']
    train_ratio = config['data']['split_ratios']['train']
    val_adjusted_ratio = val_ratio / (train_ratio + val_ratio)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=config['data']['seed'], stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adjusted_ratio, random_state=config['data']['seed'], stratify=y_train_val
    )
    
    # Channel-wise Z-score standardization:
    # 1. Reshape inputs from (N, 23, 256) to (N * 256, 23) to scale channels independently
    N_tr, C, L = X_train.shape
    N_va = X_val.shape[0]
    N_te = X_test.shape[0]
    
    X_train_flat = X_train.transpose(0, 2, 1).reshape(N_tr * L, C)
    X_val_flat = X_val.transpose(0, 2, 1).reshape(N_va * L, C)
    X_test_flat = X_test.transpose(0, 2, 1).reshape(N_te * L, C)
    
    # Fit scaler on training set flat matrix only to avoid leakage
    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler.transform(X_val_flat)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    
    # 2. Reshape scaled matrices back to (N, 23, 256)
    X_train = X_train_scaled_flat.reshape(N_tr, L, C).transpose(0, 2, 1)
    X_val = X_val_scaled_flat.reshape(N_va, L, C).transpose(0, 2, 1)
    X_test = X_test_scaled_flat.reshape(N_te, L, C).transpose(0, 2, 1)
    
    # Save the scaler mean and scale for verification or hardware deployment scaling
    outputs_dir = os.path.join(project_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    np.save(os.path.join(outputs_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(outputs_dir, "scaler_scale.npy"), scaler.scale_)
    
    print(f"Dataset Split Sizes:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    config = load_config()
    print("Loaded configuration successfully:")
    print(config)
