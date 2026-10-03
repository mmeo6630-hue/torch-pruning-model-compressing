"""Shared configuration: paths, device selection, reproducibility."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

# --- Paths -------------------------------------------------------------------
# Resolve relative to the repository root so scripts and notebooks agree.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BASELINE_CKPT = CKPT_DIR / "resnet18_cifar10.pt"
BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Task constants ----------------------------------------------------------
NUM_CLASSES = 10
INPUT_SIZE = (1, 3, 32, 32)  # CIFAR-10, batch of 1
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Pruning ratios reported in the benchmark sweep.
PRUNING_RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
# The headline ratio used for the before/after comparison table.
MAIN_RATIO = 0.5


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device: CUDA (Colab) > MPS (Apple) > CPU."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible enough to compare pruning ratios fairly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
