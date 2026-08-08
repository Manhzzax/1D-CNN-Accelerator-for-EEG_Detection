"""FP32 hardware-first 1-D CNN reference from the Q1 specification."""

from __future__ import annotations


def build_reference_model():
    try:
        import torch.nn as nn
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("torch is required; install project dependencies") from error

    class DepthwiseSeparable(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1, dilation: int = 1):
            super().__init__()
            padding = dilation * (kernel - 1) // 2
            self.depthwise = nn.Conv1d(in_channels, in_channels, kernel, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
            self.norm = nn.BatchNorm1d(in_channels)
            self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
            self.out_norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, value):
            return self.relu(self.out_norm(self.pointwise(self.relu(self.norm(self.depthwise(value))))))

    class ResidualDepthwise(nn.Module):
        def __init__(self, channels: int, kernel: int, dilation: int):
            super().__init__()
            self.body = DepthwiseSeparable(channels, channels, kernel, dilation=dilation)

        def forward(self, value):
            return value + self.body(value)

    class Reference1DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(19, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True))
            self.block1 = DepthwiseSeparable(32, 64, 5, stride=2)
            self.block2 = ResidualDepthwise(64, 3, dilation=2)
            self.block3 = DepthwiseSeparable(64, 96, 5, stride=2, dilation=2)
            self.block4 = ResidualDepthwise(96, 3, dilation=4)
            self.block5 = DepthwiseSeparable(96, 128, 3, stride=2, dilation=4)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 2))

        def forward(self, value):
            return self.head(self.block5(self.block4(self.block3(self.block2(self.block1(self.stem(value)))))))

    return Reference1DCNN()

