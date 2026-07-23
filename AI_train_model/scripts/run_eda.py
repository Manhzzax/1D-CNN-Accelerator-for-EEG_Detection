import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root directory to sys.path to enable src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config

def main():
    config = load_config()
    outputs_dir = os.path.join(project_dir, "outputs")
    data_dir = os.path.join(project_dir, "data")
    preprocessed_path = os.path.join(data_dir, config['data']['preprocessed_filename'])
    
    print("=" * 60)
    print("RUNNING CHB-MIT PREPROCESSED DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    if not os.path.exists(preprocessed_path):
        print(f"Error: Preprocessed dataset not found at {preprocessed_path}.")
        print("Please run preprocessing first using:")
        print("  python main.py --mode preprocess")
        return
        
    print(f"Loading preprocessed dataset: {preprocessed_path}")
    data = np.load(preprocessed_path)
    X = data['X'] # shape: (N, 23, 256)
    y = data['y'] # shape: (N,)
    channels = data['channels']
    
    print(f"Dataset Shape: X={X.shape} | y={y.shape}")
    print(f"Number of channels: {len(channels)}")
    
    # 1. Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    binary_counts = dict(zip(unique, counts))
    print("\nClass Distribution:")
    print(f"  Non-Seizure (0): {binary_counts.get(0.0, 0)} samples ({binary_counts.get(0.0, 0)/len(y)*100:.2f}%)")
    print(f"  Seizure (1):     {binary_counts.get(1.0, 0)} samples ({binary_counts.get(1.0, 0)/len(y)*100:.2f}%)")
    
    # 2. Plot Class Distribution
    os.makedirs(outputs_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.countplot(x=y, palette='coolwarm')
    plt.xticks([0, 1], ['Non-Seizure (0)', 'Seizure (1)'])
    plt.title('CHB-MIT Balanced Class Distribution')
    plt.xlabel('Class Group')
    plt.ylabel('Number of Samples')
    
    dist_path = os.path.join(outputs_dir, "class_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to: {dist_path}")
    
    # 3. Plot EEG Waveforms for Seizure vs Non-Seizure (Binary comparison)
    # Pick a few representative channels to show in the plot (e.g. 4 channels)
    ch_indices = [0, 5, 10, 15] # FP1-F7, F3-C3, C4-P4, P8-O2
    ch_names = [channels[i] for i in ch_indices]
    
    seizure_idx = np.where(y == 1)[0][0]
    normal_idx = np.where(y == 0)[0][0]
    
    seizure_sample = X[seizure_idx] # (23, 256)
    normal_sample = X[normal_idx]   # (23, 256)
    
    plt.figure(figsize=(14, 10))
    
    # Plot Seizure Waveforms
    for i, ch_idx in enumerate(ch_indices):
        plt.subplot(8, 1, i + 1)
        plt.plot(seizure_sample[ch_idx], color='red', alpha=0.8)
        plt.ylabel('Amp')
        plt.title(f"Seizure (Ictal) EEG - Channel {ch_names[i]}" if i == 0 else f"Channel {ch_names[i]}", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        
    # Plot Normal Waveforms
    for i, ch_idx in enumerate(ch_indices):
        plt.subplot(8, 1, i + 5)
        plt.plot(normal_sample[ch_idx], color='blue', alpha=0.8)
        plt.ylabel('Amp')
        plt.title(f"Non-Seizure (Interictal) EEG - Channel {ch_names[i]}" if i == 0 else f"Channel {ch_names[i]}", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        if i == len(ch_indices) - 1:
            plt.xlabel('Time Sample (256 Hz)')
            
    plt.tight_layout()
    binary_wave_path = os.path.join(outputs_dir, "eeg_sample_signals_binary.png")
    plt.savefig(binary_wave_path, dpi=300)
    plt.close()
    print(f"Saved binary sample EEG waveforms to: {binary_wave_path}")
    
    # 4. Signal Statistics
    print("\nEEG Signal Value Statistics:")
    print(f"  Global Mean: {X.mean():.4f}")
    print(f"  Global Std:  {X.std():.4f}")
    print(f"  Global Min:  {X.min():.4f}")
    print(f"  Global Max:  {X.max():.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
