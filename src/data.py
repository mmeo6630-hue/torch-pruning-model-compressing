"""CIFAR-10 data pipeline.

The test loader can be restricted to a small *stratified* subset. During the
live demo we evaluate 8 model variants, and a 2,000-image subset cuts each
evaluation from ~12s to ~2.5s on CPU while keeping the accuracy estimate within
roughly +/-1% of the full 10,000-image test set.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import DATA_DIR

# Channel statistics of the CIFAR-10 training split.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

TEST_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def stratified_indices(targets, per_class: int, seed: int = 42) -> list:
    """Pick `per_class` samples from every class, deterministically.

    A class-balanced subset keeps the accuracy estimate unbiased, which a plain
    `range(n)` slice would not guarantee.
    """
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(targets):
        c_idx = np.where(targets == c)[0]
        idx.extend(rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False))
    return sorted(int(i) for i in idx)


def get_dataloaders(
    batch_size: int = 128,
    test_batch_size: int = 256,
    eval_subset: Optional[int] = None,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader).

    Args:
        eval_subset: total number of test images to keep (split evenly across
            the 10 classes). None means the full 10,000-image test set.
    """
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=download, transform=TRAIN_TF)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=download, transform=TEST_TF)

    if eval_subset is not None and eval_subset < len(test_ds):
        per_class = eval_subset // 10
        test_ds = Subset(test_ds, stratified_indices(test_ds.targets, per_class))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, test_loader


def example_inputs(device: Optional[torch.device] = None) -> torch.Tensor:
    """A single dummy CIFAR-10 batch, used to trace the dependency graph."""
    x = torch.randn(1, 3, 32, 32)
    return x.to(device) if device is not None else x
