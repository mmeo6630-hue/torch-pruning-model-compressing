"""ResNet-18 adapted to 32x32 CIFAR-10 inputs.

torchvision's ImageNet stem (7x7 stride-2 conv + 3x3 stride-2 max-pool) throws
away 3/4 of a 32x32 image before the first residual block. The standard CIFAR
fix is a 3x3 stride-1 stem with no pooling, which is what `resnet18_cifar`
builds. Everything after the stem is untouched torchvision code, so the
residual/downsample topology that makes structured pruning hard is preserved --
that topology is the whole point of the demo.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18

from .config import NUM_CLASSES


def resnet18_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 with a CIFAR stem."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_baseline(
    ckpt_path,
    device: Optional[torch.device] = None,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Rebuild the baseline architecture and load trained weights into it."""
    model = resnet18_cifar(num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model.to(device) if device is not None else model


def save_pruned(model: nn.Module, path) -> None:
    """Persist a pruned model.

    Structured pruning changes layer shapes, so a bare `state_dict` no longer
    matches `resnet18_cifar()`. We pickle the whole module instead.
    """
    torch.save(model, path)


def load_pruned(path, device: Optional[torch.device] = None) -> nn.Module:
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model.to(device) if device is not None else model
