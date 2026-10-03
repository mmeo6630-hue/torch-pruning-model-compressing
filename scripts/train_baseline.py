#!/usr/bin/env python3
"""Train the ResNet-18 / CIFAR-10 baseline that the demo prunes.

Run once, offline. The notebook only loads the resulting checkpoint, so the
live demo never spends its 10 minutes training.

    python scripts/train_baseline.py --epochs 30

Writes:
    checkpoints/resnet18_cifar10.pt   best-accuracy weights + metadata
    results/baseline_training_log.csv per-epoch loss / accuracy / time
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASELINE_CKPT, RESULTS_DIR, get_device, set_seed
from src.data import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import resnet18_cifar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(BASELINE_CKPT))
    args = ap.parse_args()

    set_seed(42)
    device = get_device(args.device)
    print(f"device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, eval_subset=None, num_workers=args.workers
    )
    model = resnet18_cifar().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    log_path = RESULTS_DIR / "baseline_training_log.csv"
    best_acc, t_start = 0.0, time.perf_counter()

    with open(log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "lr", "train_loss", "test_acc", "seconds"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            lr_now = optimizer.param_groups[0]["lr"]
            loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
            scheduler.step()
            acc = evaluate(model, test_loader, device)
            dt = time.perf_counter() - t0

            print(f"epoch {epoch:3d}/{args.epochs}  lr {lr_now:.4f}  "
                  f"loss {loss:.4f}  test_acc {acc:.2f}%  ({dt:.1f}s)", flush=True)
            writer.writerow([epoch, f"{lr_now:.6f}", f"{loss:.4f}", f"{acc:.2f}", f"{dt:.1f}"])
            fh.flush()

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "accuracy": acc,
                        "epoch": epoch,
                        "arch": "resnet18_cifar",
                        "args": vars(args),
                    },
                    args.out,
                )

    total = time.perf_counter() - t_start
    print(f"\nbest test accuracy: {best_acc:.2f}%  |  total {total/60:.1f} min")
    print(f"checkpoint -> {args.out}")
    (RESULTS_DIR / "baseline_summary.json").write_text(
        json.dumps({"best_accuracy": best_acc, "epochs": args.epochs,
                    "minutes": total / 60, "device": str(device)}, indent=2)
    )


if __name__ == "__main__":
    main()
