import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root directory to sys.path to enable src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config, get_train_val_test_datasets
from src.model import EEG1DCNN
from src.quantization import verify_and_export_quantized_model
from src.utils import outputs_dir

def main():
    config = load_config()
    
    print("=" * 60)
    print("RUNNING BATCHNORM FOLDING & Q15 QUANTIZATION EXPORT")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load best model weights
    model_path = os.path.join(outputs_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model not found at {model_path}. Please run run_train.py first.")
        sys.exit(1)
        
    model = EEG1DCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded trained model weights from {model_path}")
    
    # 2. Get test set loader
    _, _, test_dataset = get_train_val_test_datasets()
    
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)
    pin_memory = config['training'].get('pin_memory', True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # 3. Fold BN, quantize, and export weight files
    verify_and_export_quantized_model(model, test_loader, device)
    print("=" * 60)

if __name__ == "__main__":
    main()
