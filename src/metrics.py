"""Quantitative metrics for the benchmark report.

Required by the assignment: Parameters, MACs, Accuracy.
Added because the topic is resource-constrained devices:
  * CPU batch-1 latency  -- what an edge device actually experiences
  * on-disk model size   -- Flash footprint
  * peak activation memory -- the RAM the forward pass needs on top of weights
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp

from .engine import evaluate

BYTES_PER_MB = 1024 ** 2


def count_params(model: nn.Module) -> int:
    """Total number of stored parameters (zeros included)."""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_params(model: nn.Module) -> int:
    """Parameters that are not exactly zero.

    This is where unstructured pruning shows up: it drives this number down
    while `count_params` stays flat, because the zeros are still stored.
    """
    return sum(int(torch.count_nonzero(p)) for p in model.parameters())


def weight_sparsity(model: nn.Module) -> float:
    """Fraction of zeros among Conv2d/Linear weights, in percent."""
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += int((m.weight == 0).sum())
            total += m.weight.numel()
    return 100.0 * zeros / max(total, 1)


def macs_and_params(model: nn.Module, example_inputs: torch.Tensor) -> Tuple[int, int]:
    """(MACs, params) for one forward pass, via torch_pruning's op counter.

    Counted on CPU so the number never depends on which accelerator is present.
    """
    model_cpu = model.to("cpu").eval()
    macs, params = tp.utils.count_ops_and_params(model_cpu, example_inputs.to("cpu"))
    return int(macs), int(params)


def model_size_mb(model: nn.Module) -> float:
    """Size of the serialized state_dict in MiB (a proxy for Flash usage)."""
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / BYTES_PER_MB
    finally:
        os.remove(path)


@torch.inference_mode()
def activation_memory_mb(model: nn.Module, example_inputs: torch.Tensor) -> float:
    """Total float32 bytes produced by all leaf modules during one forward pass.

    An upper bound on the activation RAM an inference engine must juggle. It
    scales with channel count, so structured pruning reduces it and
    unstructured pruning does not.
    """
    model = model.to("cpu").eval()
    total = 0
    handles = []

    def hook(_module, _inp, out):
        nonlocal total
        if isinstance(out, torch.Tensor):
            total += out.numel() * out.element_size()

    for m in model.modules():
        if not list(m.children()):  # leaf modules only, avoids double counting
            handles.append(m.register_forward_hook(hook))
    try:
        model(example_inputs.to("cpu"))
    finally:
        for h in handles:
            h.remove()
    return total / BYTES_PER_MB


@torch.inference_mode()
def latency_ms(
    model: nn.Module,
    example_inputs: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    runs: int = 50,
    warmup: int = 10,
    repeats: int = 1,
) -> Dict[str, float]:
    """Wall-clock latency of one forward pass, in milliseconds.

    Measures `repeats` independent blocks of `runs` timings and reports the block
    with the smallest median. Timing noise on a shared machine is one-sided -
    another process can only ever make a forward pass look slower, never faster -
    so the fastest block is the closest estimate of what the model actually
    costs. With `repeats=1` this is a plain median, which is fine when nothing
    else is running.
    """
    model = model.to(device).eval()
    x = example_inputs.to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        model(x)
    sync()

    best = None
    for _ in range(max(repeats, 1)):
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            sync()
            samples.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(samples)
        block = {"median": float(np.median(arr)), "p90": float(np.percentile(arr, 90))}
        if best is None or block["median"] < best["median"]:
            best = block
    return best


def benchmark(
    model: nn.Module,
    name: str,
    example_inputs: torch.Tensor,
    test_loader=None,
    device: Optional[torch.device] = None,
    latency_runs: int = 50,
    tag: str = "",
    ratio: Optional[float] = None,
) -> Dict[str, object]:
    """Collect every metric for one model variant into a single flat row.

    The metric helpers move the model between CPU and the accelerator as they
    go. We put it back where we found it so callers can keep using it.
    """
    origin = next(model.parameters()).device
    macs, params = macs_and_params(model, example_inputs)
    row: Dict[str, object] = {
        "model": name,
        "method": tag,
        "ratio": ratio,
        "params_M": params / 1e6,
        "nonzero_params_M": count_nonzero_params(model) / 1e6,
        "sparsity_pct": weight_sparsity(model),
        "macs_M": macs / 1e6,
        "size_MB": model_size_mb(model),
        "act_mem_MB": activation_memory_mb(model, example_inputs),
    }

    cpu_lat = latency_ms(model, example_inputs, torch.device("cpu"), runs=latency_runs)
    row["cpu_latency_ms"] = cpu_lat["median"]
    row["cpu_latency_p90_ms"] = cpu_lat["p90"]

    if test_loader is not None and device is not None:
        row["accuracy"] = evaluate(model, test_loader, device)
    else:
        row["accuracy"] = float("nan")

    model.to(origin)
    return row


def comparison_table(rows, baseline_key: str = "model") -> "object":
    """Build a pandas DataFrame with retention/reduction columns vs. row 0."""
    import pandas as pd

    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["params_vs_base_pct"] = 100.0 * df["params_M"] / base["params_M"]
    df["macs_vs_base_pct"] = 100.0 * df["macs_M"] / base["macs_M"]
    df["speedup_x"] = base["cpu_latency_ms"] / df["cpu_latency_ms"]
    df["acc_drop_pp"] = df["accuracy"] - base["accuracy"]
    return df
