import json
import os
from copy import deepcopy
import yaml
import torch
import torch.nn as nn
from torch.autograd import Function

from .runtime_config import apply_runtime_overrides

# Helper path logic
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
config_path = os.path.join(project_dir, "config", "config.yaml")

def load_config():
    with open(config_path, "r") as f:
        config = apply_runtime_overrides(yaml.safe_load(f))
    return config


def _apply_separable_environment_overrides(model_config):
    """Apply explicit trial settings before creating a new separable model."""
    overrides = {
        "CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL": (
            "temporal_filters_per_channel", int,
        ),
        "CHBMIT_SEPARABLE_SPATIAL_FILTERS": ("spatial_filters", int),
        "CHBMIT_SEPARABLE_TEMPORAL_KERNEL": ("temporal_kernel", int),
        "CHBMIT_SEPARABLE_REFINEMENT_KERNEL": ("refinement_kernel", int),
        "CHBMIT_SEPARABLE_DROPOUT": ("dropout", float),
    }
    options = model_config["separable_1dcnn"]
    for environment_name, (option_name, parser) in overrides.items():
        value = os.environ.get(environment_name)
        if value is not None:
            options[option_name] = parser(value)
    if options["temporal_filters_per_channel"] < 1 or options["spatial_filters"] < 1:
        raise ValueError("Separable filter counts must be positive")
    if not 0.0 <= options["dropout"] < 1.0:
        raise ValueError("CHBMIT_SEPARABLE_DROPOUT must be in [0, 1)")
    return model_config


def _apply_multiscale_separable_environment_overrides(model_config):
    """Apply explicit trial settings before creating a multiscale separable model."""
    overrides = {
        "CHBMIT_MULTISCALE_TEMPORAL_FILTERS_PER_BRANCH": (
            "temporal_filters_per_branch", int,
        ),
        "CHBMIT_MULTISCALE_SPATIAL_FILTERS": ("spatial_filters", int),
        "CHBMIT_MULTISCALE_SHORT_KERNEL": ("short_kernel", int),
        "CHBMIT_MULTISCALE_LONG_KERNEL": ("long_kernel", int),
        "CHBMIT_MULTISCALE_REFINEMENT_KERNEL": ("refinement_kernel", int),
        "CHBMIT_MULTISCALE_DROPOUT": ("dropout", float),
    }
    options = model_config["multiscale_separable_1dcnn"]
    for environment_name, (option_name, parser) in overrides.items():
        value = os.environ.get(environment_name)
        if value is not None:
            options[option_name] = parser(value)
    if options["temporal_filters_per_branch"] < 1 or options["spatial_filters"] < 1:
        raise ValueError("Multiscale separable filter counts must be positive")
    if any(options[name] < 1 or options[name] % 2 == 0 for name in (
        "short_kernel", "long_kernel", "refinement_kernel",
    )):
        raise ValueError("Multiscale separable kernels must be positive odd integers")
    if not 0.0 <= options["dropout"] < 1.0:
        raise ValueError("CHBMIT_MULTISCALE_DROPOUT must be in [0, 1)")
    return model_config


def _apply_hierarchical_separable_environment_overrides(model_config):
    """Apply explicit 31/7/3-style trial settings before model construction."""
    overrides = {
        "CHBMIT_HIERARCHICAL_TEMPORAL_FILTERS_PER_CHANNEL": (
            "temporal_filters_per_channel", int,
        ),
        "CHBMIT_HIERARCHICAL_SPATIAL_FILTERS": ("spatial_filters", int),
        "CHBMIT_HIERARCHICAL_TEMPORAL_KERNEL": ("temporal_kernel", int),
        "CHBMIT_HIERARCHICAL_SECOND_KERNEL": ("second_kernel", int),
        "CHBMIT_HIERARCHICAL_THIRD_KERNEL": ("third_kernel", int),
        "CHBMIT_HIERARCHICAL_DROPOUT": ("dropout", float),
    }
    options = model_config["hierarchical_separable_1dcnn"]
    for environment_name, (option_name, parser) in overrides.items():
        value = os.environ.get(environment_name)
        if value is not None:
            options[option_name] = parser(value)
    pointwise_value = os.environ.get("CHBMIT_HIERARCHICAL_THIRD_POINTWISE")
    if pointwise_value is not None:
        normalized = pointwise_value.strip().lower()
        if normalized not in {"true", "false", "1", "0"}:
            raise ValueError("CHBMIT_HIERARCHICAL_THIRD_POINTWISE must be true/false/1/0")
        options["third_pointwise"] = normalized in {"true", "1"}
    if options["temporal_filters_per_channel"] < 1 or options["spatial_filters"] < 1:
        raise ValueError("Hierarchical separable filter counts must be positive")
    if any(options[name] < 1 or options[name] % 2 == 0 for name in (
        "temporal_kernel", "second_kernel", "third_kernel",
    )):
        raise ValueError("Hierarchical separable kernels must be positive odd integers")
    if not 0.0 <= options["dropout"] < 1.0:
        raise ValueError("CHBMIT_HIERARCHICAL_DROPOUT must be in [0, 1)")
    return model_config


def _apply_residual_hierarchical_environment_overrides(model_config):
    """Apply explicit residual 31/7/3 trial settings before construction."""
    overrides = {
        "CHBMIT_RESIDUAL_HIERARCHICAL_TEMPORAL_FILTERS_PER_CHANNEL": (
            "temporal_filters_per_channel", int,
        ),
        "CHBMIT_RESIDUAL_HIERARCHICAL_SPATIAL_FILTERS": ("spatial_filters", int),
        "CHBMIT_RESIDUAL_HIERARCHICAL_TEMPORAL_KERNEL": ("temporal_kernel", int),
        "CHBMIT_RESIDUAL_HIERARCHICAL_SECOND_KERNEL": ("second_kernel", int),
        "CHBMIT_RESIDUAL_HIERARCHICAL_THIRD_KERNEL": ("third_kernel", int),
        "CHBMIT_RESIDUAL_HIERARCHICAL_DROPOUT": ("dropout", float),
    }
    options = model_config["residual_hierarchical_separable_1dcnn"]
    for environment_name, (option_name, parser) in overrides.items():
        value = os.environ.get(environment_name)
        if value is not None:
            options[option_name] = parser(value)
    if options["temporal_filters_per_channel"] < 1 or options["spatial_filters"] < 1:
        raise ValueError("Residual hierarchical filter counts must be positive")
    if any(options[name] < 1 or options[name] % 2 == 0 for name in (
        "temporal_kernel", "second_kernel", "third_kernel",
    )):
        raise ValueError("Residual hierarchical kernels must be positive odd integers")
    if not 0.0 <= options["dropout"] < 1.0:
        raise ValueError("CHBMIT_RESIDUAL_HIERARCHICAL_DROPOUT must be in [0, 1)")
    return model_config


def effective_model_config(model_config=None):
    """Return the exact model contract for a newly created model or saved run."""
    if model_config is not None:
        return deepcopy(model_config)
    config = deepcopy(load_config()["model"])
    config = _apply_separable_environment_overrides(config)
    config = _apply_multiscale_separable_environment_overrides(config)
    config = _apply_hierarchical_separable_environment_overrides(config)
    return _apply_residual_hierarchical_environment_overrides(config)

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

    def forward_features(self, x):
        x = self.activation(self.temporal_bn(self.temporal_depthwise(x)))
        x = self.pool(self.activation(self.spatial_bn(self.spatial_pointwise(x))))
        x = self.activation(self.refine_bn(self.refine_pointwise(self.refine_depthwise(x))))
        x = self.pool(x)
        return self.dropout(self.global_pool(x).squeeze(-1))

    def forward(self, x):
        return self.classifier(self.forward_features(x))


class HierarchicalSeparableEEG1DCNN(nn.Module):
    """Three-stage separable EEG CNN with progressively shorter temporal kernels."""

    def __init__(
        self,
        in_channels,
        num_classes,
        temporal_filters_per_channel=3,
        spatial_filters=32,
        temporal_kernel=31,
        second_kernel=7,
        third_kernel=3,
        third_pointwise=False,
        dropout=0.25,
    ):
        super().__init__()
        if any(kernel % 2 == 0 for kernel in (temporal_kernel, second_kernel, third_kernel)):
            raise ValueError("Hierarchical separable temporal kernels must be odd")
        self.architecture_name = "hierarchical_separable_1dcnn"
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
        self.second_depthwise = nn.Conv1d(
            spatial_filters,
            spatial_filters,
            kernel_size=second_kernel,
            padding=second_kernel // 2,
            groups=spatial_filters,
            bias=False,
        )
        self.second_pointwise = nn.Conv1d(spatial_filters, spatial_filters, kernel_size=1, bias=False)
        self.second_bn = nn.BatchNorm1d(spatial_filters)
        self.third_depthwise = nn.Conv1d(
            spatial_filters,
            spatial_filters,
            kernel_size=third_kernel,
            padding=third_kernel // 2,
            groups=spatial_filters,
            bias=False,
        )
        self.third_pointwise = (
            nn.Conv1d(spatial_filters, spatial_filters, kernel_size=1, bias=False)
            if third_pointwise else nn.Identity()
        )
        self.third_bn = nn.BatchNorm1d(spatial_filters)
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(spatial_filters, num_classes)

    def forward_features(self, x):
        x = self.activation(self.temporal_bn(self.temporal_depthwise(x)))
        x = self.pool(self.activation(self.spatial_bn(self.spatial_pointwise(x))))
        x = self.activation(self.second_bn(self.second_pointwise(self.second_depthwise(x))))
        x = self.pool(x)
        x = self.activation(self.third_bn(self.third_pointwise(self.third_depthwise(x))))
        return self.dropout(self.global_pool(x).squeeze(-1))

    def forward(self, x):
        return self.classifier(self.forward_features(x))


class ResidualHierarchicalSeparableEEG1DCNN(nn.Module):
    """31/7/3 hierarchy with identity residuals and no extra convolution weights."""

    def __init__(
        self,
        in_channels,
        num_classes,
        temporal_filters_per_channel=3,
        spatial_filters=32,
        temporal_kernel=31,
        second_kernel=7,
        third_kernel=3,
        dropout=0.25,
    ):
        super().__init__()
        if any(kernel % 2 == 0 for kernel in (temporal_kernel, second_kernel, third_kernel)):
            raise ValueError("Residual hierarchical temporal kernels must be odd")
        self.architecture_name = "residual_hierarchical_separable_1dcnn"
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
        self.second_depthwise = nn.Conv1d(
            spatial_filters,
            spatial_filters,
            kernel_size=second_kernel,
            padding=second_kernel // 2,
            groups=spatial_filters,
            bias=False,
        )
        self.second_pointwise = nn.Conv1d(spatial_filters, spatial_filters, kernel_size=1, bias=False)
        self.second_bn = nn.BatchNorm1d(spatial_filters)
        self.third_depthwise = nn.Conv1d(
            spatial_filters,
            spatial_filters,
            kernel_size=third_kernel,
            padding=third_kernel // 2,
            groups=spatial_filters,
            bias=False,
        )
        self.third_bn = nn.BatchNorm1d(spatial_filters)
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(spatial_filters, num_classes)

    def forward_features(self, x):
        x = self.activation(self.temporal_bn(self.temporal_depthwise(x)))
        x = self.pool(self.activation(self.spatial_bn(self.spatial_pointwise(x))))
        second_skip = x
        x = self.activation(self.second_bn(self.second_pointwise(self.second_depthwise(x))) + second_skip)
        x = self.pool(x)
        third_skip = x
        x = self.activation(self.third_bn(self.third_depthwise(x)) + third_skip)
        return self.dropout(self.global_pool(x).squeeze(-1))

    def forward(self, x):
        return self.classifier(self.forward_features(x))


class _GradientReversal(Function):
    """Identity forward pass with a scaled reversed gradient for source-domain invariance."""

    @staticmethod
    def forward(context, inputs, coefficient):
        context.coefficient = float(coefficient)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(context, gradients):
        return -context.coefficient * gradients, None


def gradient_reverse(inputs, coefficient):
    return _GradientReversal.apply(inputs, coefficient)


class SubjectDiscriminator(nn.Module):
    """Training-only patient-domain head; it is never exported for inference."""

    def __init__(self, input_features, hidden_features, domain_count):
        super().__init__()
        if hidden_features < 1 or domain_count < 2:
            raise ValueError("Subject discriminator requires positive hidden width and at least two domains")
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, domain_count),
        )

    def forward(self, inputs):
        return self.network(inputs)


class MultiScaleSeparableEEG1DCNN(nn.Module):
    """Compact EEGNet-style CNN with short and long depthwise temporal paths."""

    def __init__(
        self,
        in_channels,
        num_classes,
        temporal_filters_per_branch=1,
        spatial_filters=32,
        short_kernel=15,
        long_kernel=63,
        refinement_kernel=15,
        dropout=0.25,
    ):
        super().__init__()
        if any(kernel % 2 == 0 for kernel in (short_kernel, long_kernel, refinement_kernel)):
            raise ValueError("Multiscale separable temporal kernels must be odd")
        self.architecture_name = "multiscale_separable_1dcnn"
        branch_channels = in_channels * temporal_filters_per_branch
        self.short_depthwise = nn.Conv1d(
            in_channels,
            branch_channels,
            kernel_size=short_kernel,
            padding=short_kernel // 2,
            groups=in_channels,
            bias=False,
        )
        self.long_depthwise = nn.Conv1d(
            in_channels,
            branch_channels,
            kernel_size=long_kernel,
            padding=long_kernel // 2,
            groups=in_channels,
            bias=False,
        )
        temporal_channels = branch_channels * 2
        self.temporal_bn = nn.BatchNorm1d(temporal_channels)
        self.spatial_pointwise = nn.Conv1d(
            temporal_channels, spatial_filters, kernel_size=1, bias=False
        )
        self.spatial_bn = nn.BatchNorm1d(spatial_filters)
        self.refine_depthwise = nn.Conv1d(
            spatial_filters,
            spatial_filters,
            kernel_size=refinement_kernel,
            padding=refinement_kernel // 2,
            groups=spatial_filters,
            bias=False,
        )
        self.refine_pointwise = nn.Conv1d(
            spatial_filters, spatial_filters, kernel_size=1, bias=False
        )
        self.refine_bn = nn.BatchNorm1d(spatial_filters)
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(spatial_filters, num_classes)

    def forward(self, x):
        short_features = self.short_depthwise(x)
        long_features = self.long_depthwise(x)
        x = torch.cat((short_features, long_features), dim=1)
        x = self.activation(self.temporal_bn(x))
        x = self.pool(self.activation(self.spatial_bn(self.spatial_pointwise(x))))
        x = self.pool(self.activation(self.refine_bn(self.refine_pointwise(self.refine_depthwise(x)))))
        x = self.dropout(self.global_pool(x).squeeze(-1))
        return self.classifier(x)


class ParallelMultiKernelEEG1DCNN(nn.Module):
    """Raw EEG CNN with short and long temporal receptive fields in parallel."""

    def __init__(
        self,
        in_channels,
        num_classes,
        branch_filters=16,
        short_kernel=15,
        long_kernel=31,
        refinement_filters=32,
        refinement_kernel=5,
        dropout=0.25,
    ):
        super().__init__()
        if any(kernel % 2 == 0 for kernel in (short_kernel, long_kernel, refinement_kernel)):
            raise ValueError("Parallel multi-kernel convolutions must use odd kernels")
        self.architecture_name = "parallel_multikernel_1dcnn"
        self.short_branch = nn.Conv1d(
            in_channels, branch_filters, kernel_size=short_kernel, padding=short_kernel // 2, bias=False
        )
        self.long_branch = nn.Conv1d(
            in_channels, branch_filters, kernel_size=long_kernel, padding=long_kernel // 2, bias=False
        )
        merged_channels = branch_filters * 2
        self.merge_bn = nn.BatchNorm1d(merged_channels)
        self.refine = nn.Conv1d(
            merged_channels,
            refinement_filters,
            kernel_size=refinement_kernel,
            padding=refinement_kernel // 2,
            bias=False,
        )
        self.refine_bn = nn.BatchNorm1d(refinement_filters)
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(refinement_filters, num_classes)

    def forward(self, x):
        short_features = self.short_branch(x)
        long_features = self.long_branch(x)
        x = self.pool(self.activation(self.merge_bn(torch.cat((short_features, long_features), dim=1))))
        x = self.pool(self.activation(self.refine_bn(self.refine(x))))
        x = self.dropout(self.global_pool(x).squeeze(-1))
        return self.classifier(x)


def build_model(architecture=None, model_config=None):
    """Create a configured model; environment override keeps ablations isolated."""
    model_config = effective_model_config(model_config)
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
    if architecture == "hierarchical_separable_1dcnn":
        options = model_config["hierarchical_separable_1dcnn"]
        return HierarchicalSeparableEEG1DCNN(
            in_channels=model_config["input_channels"],
            num_classes=model_config["num_classes"],
            temporal_filters_per_channel=int(options["temporal_filters_per_channel"]),
            spatial_filters=int(options["spatial_filters"]),
            temporal_kernel=int(options["temporal_kernel"]),
            second_kernel=int(options["second_kernel"]),
            third_kernel=int(options["third_kernel"]),
            third_pointwise=bool(options["third_pointwise"]),
            dropout=float(options["dropout"]),
        )
    if architecture == "residual_hierarchical_separable_1dcnn":
        options = model_config["residual_hierarchical_separable_1dcnn"]
        return ResidualHierarchicalSeparableEEG1DCNN(
            in_channels=model_config["input_channels"],
            num_classes=model_config["num_classes"],
            temporal_filters_per_channel=int(options["temporal_filters_per_channel"]),
            spatial_filters=int(options["spatial_filters"]),
            temporal_kernel=int(options["temporal_kernel"]),
            second_kernel=int(options["second_kernel"]),
            third_kernel=int(options["third_kernel"]),
            dropout=float(options["dropout"]),
        )
    if architecture == "multiscale_separable_1dcnn":
        options = model_config["multiscale_separable_1dcnn"]
        return MultiScaleSeparableEEG1DCNN(
            in_channels=model_config["input_channels"],
            num_classes=model_config["num_classes"],
            temporal_filters_per_branch=int(options["temporal_filters_per_branch"]),
            spatial_filters=int(options["spatial_filters"]),
            short_kernel=int(options["short_kernel"]),
            long_kernel=int(options["long_kernel"]),
            refinement_kernel=int(options["refinement_kernel"]),
            dropout=float(options["dropout"]),
        )
    if architecture == "parallel_multikernel_1dcnn":
        options = model_config["parallel_multikernel_1dcnn"]
        return ParallelMultiKernelEEG1DCNN(
            in_channels=model_config["input_channels"],
            num_classes=model_config["num_classes"],
            branch_filters=int(options["branch_filters"]),
            short_kernel=int(options["short_kernel"]),
            long_kernel=int(options["long_kernel"]),
            refinement_filters=int(options["refinement_filters"]),
            refinement_kernel=int(options["refinement_kernel"]),
            dropout=float(options["dropout"]),
        )
    raise ValueError(f"Unknown CHB-MIT model architecture: {architecture}")


def save_model_spec(output_dir, model):
    """Persist the architecture contract alongside a checkpoint."""
    model_config = effective_model_config()
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
