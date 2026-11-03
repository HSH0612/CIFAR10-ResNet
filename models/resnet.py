"""A CIFAR-sized ResNet trained entirely from random initialization."""

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class CifarResNet(nn.Module):
    def __init__(self, width: int = 40, num_classes: int = 10) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        channels = width
        stages = []
        for stage_index, target in enumerate((width, 2 * width, 4 * width, 8 * width)):
            stride = 1 if stage_index == 0 else 2
            stages.append(nn.Sequential(BasicBlock(channels, target, stride), BasicBlock(target, target)))
            channels = target
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return self.classifier(self.pool(x).flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
