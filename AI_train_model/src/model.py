import json
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
        Input shape: (batch_size, in_channels, input_length) -> e.g. (batch, 17, 256)
        Total parameters: ~70K (well under the 100K limit)
        """
        super(EEG1DCNN, self).__init__()
        self.architecture_name = "baseline_1dcnn"
        
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


class SeparableEEG1DCNN(nn.Module):
    """EEGNet-inspired raw-signal CNN with explicit temporal then spatial mixing."""

    def __init__(
        self,
        in_channels,
        num_classes,
        temporal_filters_per_channel=2,
        spatial_filters=32,
        temporal_kernel=31,
        refinement_kernel=15,
        dropout=0.25,
    ):
        super().__init__()
        if temporal_kernel % 2 == 0 or refinement_kernel % 2 == 0:
            raise ValueError("Separable temporal kernels must be odd for length-preserving padding")
        self.architecture_name = "separable_1dcnn"
        temporal_channels = in_channels * temporal_filters_per_channel
        self.temporal_depthwise = nn.Conv1d(
            in_channels,
            temporal_channels,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            groups=in_channels,
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm1d(temporal_channels)
        self.spatial_pointwise = nn.Conv1d(temporal_channels, spatial_filters, kernel_size=1, bias=False)
        self.spatial_bn = nn.BatchNorm1d(spatial_filters)
        self.refine_depthwise = nn.Conv1d(
            spatial_filters,
            spatial_filters,
            kernel_size=refinement_kernel,
            padding=refinement_kernel // 2,
            groups=spatial_filters,
            bias=False,
        )
        self.refine_pointwise = nn.Conv1d(spatial_filters, spatial_filters, kernel_size=1, bias=False)
        self.refine_bn = nn.BatchNorm1d(spatial_filters)
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(spatial_filters, num_classes)

    def forward(self, x):
        x = self.activation(self.temporal_bn(self.temporal_depthwise(x)))
        x = self.pool(self.activation(self.spatial_bn(self.spatial_pointwise(x))))
        x = self.activation(self.refine_bn(self.refine_pointwise(self.refine_depthwise(x))))
        x = self.pool(x)
        x = self.dropout(self.global_pool(x).squeeze(-1))
        return self.classifier(x)


def build_model(architecture=None, model_config=None):
    """Create a configured model; environment override keeps ablations isolated."""
    if model_config is None:
        model_config = load_config()["model"]
    architecture = architecture or os.environ.get(
        "CHBMIT_MODEL_ARCHITECTURE", model_config.get("architecture", "baseline_1dcnn")
    )
    if architecture == "baseline_1dcnn":
        return EEG1DCNN(
            in_channels=model_config["input_channels"],
            input_length=model_config["input_length"],
            num_classes=model_config["num_classes"],
        )
    if architecture == "separable_1dcnn":
        options = model_config["separable_1dcnn"]
        return SeparableEEG1DCNN(
            in_channels=model_config["input_channels"],
            num_classes=model_config["num_classes"],
            temporal_filters_per_channel=int(options["temporal_filters_per_channel"]),
            spatial_filters=int(options["spatial_filters"]),
            temporal_kernel=int(options["temporal_kernel"]),
            refinement_kernel=int(options["refinement_kernel"]),
            dropout=float(options["dropout"]),
        )
    raise ValueError(f"Unknown CHB-MIT model architecture: {architecture}")


def save_model_spec(output_dir, model):
    """Persist the architecture contract alongside a checkpoint."""
    model_config = load_config()["model"].copy()
    model_config["architecture"] = model.architecture_name
    spec = {
        "architecture": model.architecture_name,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_config": model_config,
    }
    with open(os.path.join(output_dir, "model_spec.json"), "w", encoding="utf-8") as output_file:
        json.dump(spec, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def build_model_from_run(output_dir):
    """Load the architecture recorded with a run; legacy checkpoints are baseline CNNs."""
    spec_path = os.path.join(output_dir, "model_spec.json")
    architecture = "baseline_1dcnn"
    if os.path.isfile(spec_path):
        with open(spec_path, "r", encoding="utf-8") as input_file:
            spec = json.load(input_file)
        architecture = spec["architecture"]
        return build_model(architecture, spec.get("model_config"))
    return build_model(architecture)

if __name__ == "__main__":
    # Test the model parameters and shape for CHB-MIT settings
    model = EEG1DCNN(in_channels=17, input_length=256, num_classes=2)
    print(model)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test input
    test_input = torch.randn(2, 17, 256)
    test_output = model(test_input)
    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape: {test_output.shape}")
