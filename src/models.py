"""
ResNet-18 and MobileNet-V2 adapted for CIFAR-10 (32x32 input).

ResNet-18:  3x3 initial convolution, no initial maxpool.
MobileNet-V2: stride-1 initial convolution.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 CIFAR-10 images."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Replace 7x7 conv with 3x3, stride 1, padding 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool
    model.maxpool = nn.Identity()
    return model


def mobilenetv2_cifar10(num_classes: int = 10) -> nn.Module:
    """MobileNet-V2 adapted for 32x32 CIFAR-10 images."""
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    # Change first conv from stride-2 to stride-1 for 32x32 input
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        first_conv.in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=1,
        padding=first_conv.padding,
        bias=False,
    )
    return model


def get_model(name: str) -> nn.Module:
    """Factory function. name: 'resnet18' or 'mobilenetv2'."""
    if name == "resnet18":
        return resnet18_cifar10()
    elif name == "mobilenetv2":
        return mobilenetv2_cifar10()
    else:
        raise ValueError(f"Unknown model: {name}")
