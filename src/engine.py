"""Training / evaluation loops shared by the notebook and the benchmark script."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy in percent."""
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    max_steps: Optional[int] = None,
    scheduler=None,
    log_every: int = 0,
) -> float:
    """One pass over `loader` (or `max_steps` batches). Returns mean loss."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.train().to(device)
    running, seen = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.item() * y.numel()
        seen += y.numel()
        if log_every and step % log_every == 0:
            print(f"    step {step:4d}  loss {loss.item():.4f}")
    return running / max(seen, 1)


def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    max_steps_per_epoch: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Short recovery training after pruning.

    Uses SGD with a cosine schedule at a low LR: pruning removes capacity but
    keeps the surviving weights, so the model only needs to re-settle, not
    retrain from scratch.
    """
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
    )
    steps = max_steps_per_epoch or len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs * steps, 1))

    history: Dict[str, List[float]] = {"epoch": [], "loss": [], "acc": [], "seconds": []}
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(
            model, train_loader, optimizer, device,
            max_steps=max_steps_per_epoch, scheduler=scheduler,
        )
        acc = evaluate(model, test_loader, device)
        dt = time.perf_counter() - t0
        history["epoch"].append(ep)
        history["loss"].append(loss)
        history["acc"].append(acc)
        history["seconds"].append(dt)
        if verbose:
            print(f"  [finetune] epoch {ep}/{epochs}  loss {loss:.4f}  acc {acc:.2f}%  ({dt:.1f}s)")
    return history
