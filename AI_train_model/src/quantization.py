import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils import fold_batchnorm, outputs_dir, project_dir
from src.data_loader import load_config

class FoldedEEG1DCNN(torch.nn.Module):
    """
    A helper model class containing Conv1d layers with BN folded.
    Used to verify the accuracy of folded and quantized weights.
    De-quantizes parameters back to float for execution in PyTorch.
    Adapts dynamically to the channel and length settings in config.yaml.
    """
    def __init__(self, w1_q, b1_q, w2_q, b2_q, w_fc1_q, b_fc1_q, w_fc2_q, b_fc2_q, w_fc3_q, b_fc3_q, 
                 scale_c1, scale_c2, scale_f1, scale_f2, scale_f3):
        super(FoldedEEG1DCNN, self).__init__()
        
        config = load_config()
        in_channels = config['model']['input_channels']
        input_length = config['model']['input_length']
        num_classes = config['model']['num_classes']
        
        # Conv layers
        self.conv1 = torch.nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        
        # FC layers
        flat_features = 32 * (input_length // 4)
        self.fc1 = torch.nn.Linear(flat_features, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, num_classes)
        
        # Load the quantized parameters scaled back to float using per-layer scales
        self.conv1.weight.data = w1_q.float() / scale_c1
        self.conv1.bias.data = b1_q.float() / scale_c1
        self.conv2.weight.data = w2_q.float() / scale_c2
        self.conv2.bias.data = b2_q.float() / scale_c2
        self.fc1.weight.data = w_fc1_q.float() / scale_f1
        self.fc1.bias.data = b_fc1_q.float() / scale_f1
        self.fc2.weight.data = w_fc2_q.float() / scale_f2
        self.fc2.bias.data = b_fc2_q.float() / scale_f2
        self.fc3.weight.data = w_fc3_q.float() / scale_f3
        self.fc3.bias.data = b_fc3_q.float() / scale_f3
        
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_optimal_q_bits(tensor):
    """
    Finds the optimal number of fractional bits Q for a 16-bit signed integer
    such that no clipping occurs. Max value must fit in 2^(15 - Q).
    Q = 15 - ceil(log2(max(|tensor|)))
    """
    max_val = torch.max(torch.abs(tensor)).item()
    if max_val == 0:
        return 15 # Default to Q15 if all zero
    
    # Calculate ceil(log2(max_val))
    k = int(np.ceil(np.log2(max_val)))
    integer_bits = max(1, k + 1) # 1 bit for sign, k bits for magnitude
    q_bits = 16 - integer_bits
    
    # Restrict q_bits to standard fixed-point ranges [0, 15]
    q_bits = max(0, min(15, q_bits))
    return q_bits

def quantize_tensor(tensor, q_bits, clamp_min, clamp_max):
    """
    Quantizes a floating point tensor to fixed-point format and clamps to signed int bounds.
    """
    scale = 2 ** q_bits
    quantized = torch.round(tensor * scale)
    quantized = torch.clamp(quantized, clamp_min, clamp_max)
    return quantized.short() # 16-bit signed integer

def export_weight_file(tensor_q, filename):
    """
    Exports a quantized tensor as a text document containing integers, one per line.
    """
    flat_arr = tensor_q.cpu().numpy().flatten()
    filepath = os.path.join(outputs_dir, filename)
    with open(filepath, "w") as f:
        for val in flat_arr:
            f.write(f"{val}\n")
    print(f"Exported {filename} ({len(flat_arr)} values, file size: {os.path.getsize(filepath)/1024:.2f} KB)")

def verify_and_export_quantized_model(model, test_loader, device):
    """
    Folds BN layers, quantizes parameters, measures accuracy, and exports text files.
    """
    config = load_config()
    clamp_min = config['quantization']['clamp_min']
    clamp_max = config['quantization']['clamp_max']
    
    print("\nFolding Batch Normalization layers...")
    w1_folded, b1_folded = fold_batchnorm(model.conv1, model.bn1)
    w2_folded, b2_folded = fold_batchnorm(model.conv2, model.bn2)
    
    # FC weights
    w_fc1 = model.fc1.weight.data
    b_fc1 = model.fc1.bias.data
    w_fc2 = model.fc2.weight.data
    b_fc2 = model.fc2.bias.data
    w_fc3 = model.fc3.weight.data
    b_fc3 = model.fc3.bias.data
    
    # Print ranges of folded parameters
    print("\nParameter Float Ranges:")
    for name, tensor in [
        ("conv1.weight", w1_folded), ("conv1.bias", b1_folded),
        ("conv2.weight", w2_folded), ("conv2.bias", b2_folded),
        ("fc1.weight", w_fc1), ("fc1.bias", b_fc1),
        ("fc2.weight", w_fc2), ("fc2.bias", b_fc2),
        ("fc3.weight", w_fc3), ("fc3.bias", b_fc3)
    ]:
        print(f"  {name:15s} | Min: {tensor.min().item():.6f} | Max: {tensor.max().item():.6f} | Mean: {tensor.mean().item():.6f}")
        
    # Calculate optimal Q-bits per layer
    q_c1 = get_optimal_q_bits(w1_folded)
    q_c2 = get_optimal_q_bits(w2_folded)
    q_f1 = get_optimal_q_bits(w_fc1)
    q_f2 = get_optimal_q_bits(w_fc2)
    q_f3 = get_optimal_q_bits(w_fc3)
    
    print("\nOptimal Quantization Scales (Fractional Bits):")
    print(f"  conv1 layer: Q{q_c1} (Scale: {2**q_c1})")
    print(f"  conv2 layer: Q{q_c2} (Scale: {2**q_c2})")
    print(f"  fc1 layer:   Q{q_f1} (Scale: {2**q_f1})")
    print(f"  fc2 layer:   Q{q_f2} (Scale: {2**q_f2})")
    print(f"  fc3 layer:   Q{q_f3} (Scale: {2**q_f3})")
    
    # Quantize parameters dynamically
    w1_q = quantize_tensor(w1_folded, q_c1, clamp_min, clamp_max)
    b1_q = quantize_tensor(b1_folded, q_c1, clamp_min, clamp_max)
    w2_q = quantize_tensor(w2_folded, q_c2, clamp_min, clamp_max)
    b2_q = quantize_tensor(b2_folded, q_c2, clamp_min, clamp_max)
    
    w_fc1_q = quantize_tensor(w_fc1, q_f1, clamp_min, clamp_max)
    b_fc1_q = quantize_tensor(b_fc1, q_f1, clamp_min, clamp_max)
    w_fc2_q = quantize_tensor(w_fc2, q_f2, clamp_min, clamp_max)
    b_fc2_q = quantize_tensor(b_fc2, q_f2, clamp_min, clamp_max)
    w_fc3_q = quantize_tensor(w_fc3, q_f3, clamp_min, clamp_max)
    b_fc3_q = quantize_tensor(b_fc3, q_f3, clamp_min, clamp_max)
    
    # Verify accuracy of float model
    model.eval()
    correct_float = 0
    total_float = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_float += targets.size(0)
            correct_float += predicted.eq(targets).sum().item()
    float_acc = 100. * correct_float / total_float
    print(f"\nOriginal float model accuracy (with BN): {float_acc:.2f}%")
    
    # Evaluate folded, quantized model using per-layer scales
    quant_model = FoldedEEG1DCNN(
        w1_q, b1_q, w2_q, b2_q, w_fc1_q, b_fc1_q, w_fc2_q, b_fc2_q, w_fc3_q, b_fc3_q,
        2**q_c1, 2**q_c2, 2**q_f1, 2**q_f2, 2**q_f3
    ).to(device)
    
    quant_model.eval()
    correct_quant = 0
    total_quant = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = quant_model(inputs)
            _, predicted = outputs.max(1)
            total_quant += targets.size(0)
            correct_quant += predicted.eq(targets).sum().item()
    quant_acc = 100. * correct_quant / total_quant
    print(f"Quantized dynamic model accuracy (folded, integer): {quant_acc:.2f}%")
    print(f"Accuracy difference: {quant_acc - float_acc:.2f}%")
    
    # Log quantization report
    report_path = os.path.join(outputs_dir, "quantization_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Quantization Report ===\n")
        f.write(f"Original Float Model Accuracy: {float_acc:.2f}%\n")
        f.write(f"Quantized Q-Dynamic Accuracy:  {quant_acc:.2f}%\n")
        f.write(f"Accuracy Loss:                 {float_acc - quant_acc:.2f}%\n\n")
        f.write("Layer Scaling Configuration:\n")
        f.write(f"  conv1 layer: Q{q_c1} (Scale: {2**q_c1})\n")
        f.write(f"  conv2 layer: Q{q_c2} (Scale: {2**q_c2})\n")
        f.write(f"  fc1 layer:   Q{q_f1} (Scale: {2**q_f1})\n")
        f.write(f"  fc2 layer:   Q{q_f2} (Scale: {2**q_f2})\n")
        f.write(f"  fc3 layer:   Q{q_f3} (Scale: {2**q_f3})\n")
    print(f"Saved quantization report to {report_path}")
    
    # Export layers
    print("\nExporting quantized layers to text files...")
    export_weight_file(w1_q, "conv1_weight_q16")
    export_weight_file(b1_q, "conv1_bias_q16")
    export_weight_file(w2_q, "conv2_weight_q16")
    export_weight_file(b2_q, "conv2_bias_q16")
    export_weight_file(w_fc1_q, "fc1_weight_q16")
    export_weight_file(b_fc1_q, "fc1_bias_q16")
    export_weight_file(w_fc2_q, "fc2_weight_q16")
    export_weight_file(b_fc2_q, "fc2_bias_q16")
    export_weight_file(w_fc3_q, "fc3_weight_q16")
    export_weight_file(b_fc3_q, "fc3_bias_q16")
