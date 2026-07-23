import os
import yaml
import torch
import torch.nn as nn

# Helper path logic
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
config_path = os.path.join(project_dir, "config", "config.yaml")

def load_config():
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class EEG1DCNN(nn.Module):
    def __init__(self, in_channels=None, input_length=None, num_classes=None):
        """
        EEG 1D-CNN Model for Seizure Detection (adapted for CHB-MIT multi-channel data)
        Input shape: (batch_size, in_channels, input_length) -> e.g. (batch, 23, 256)
        Total parameters: ~70K (well under the 100K limit)
        """
        super(EEG1DCNN, self).__init__()
        
        # Load from config if not specified
        config = load_config()
        if in_channels is None:
            in_channels = config['model']['input_channels']
        if input_length is None:
            input_length = config['model']['input_length']
        if num_classes is None:
            num_classes = config['model']['num_classes']
            
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # input_length -> input_length // 2
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # input_length // 2 -> input_length // 4
        
        # Calculate dynamic flattened size
        self.flat_features = 32 * (input_length // 4)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_features, 32)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC 1
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        # FC 2
        x = self.fc2(x)
        x = self.relu4(x)
        
        # FC 3
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Test the model parameters and shape for CHB-MIT settings
    model = EEG1DCNN(in_channels=23, input_length=256, num_classes=2)
    print(model)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test input
    test_input = torch.randn(2, 23, 256)
    test_output = model(test_input)
    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape: {test_output.shape}")
