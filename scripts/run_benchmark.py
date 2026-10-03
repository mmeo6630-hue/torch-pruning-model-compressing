#!/usr/bin/env python3
"""Full benchmark sweep: unstructured vs. structured pruning, 20% -> 70%.

Run offline. The notebook reads the CSV this produces so the live demo can show
fully fine-tuned numbers without spending 40 minutes on them.

    python scripts/run_benchmark.py --finetune-epochs 3

For every ratio it records four variants:
    unstructured        weights zeroed, no recovery training
    unstructured+ft     same, then fine-tuned with the masks still attached
    structured          channels physically removed via DepGraph, no recovery
    structured+ft       same, then fine-tuned

Writes results/benchmark.csv and results/benchmark.md.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (BASELINE_CKPT, BENCHMARK_CSV, MAIN_RATIO, PRUNING_RATIOS,
                        RESULTS_DIR, get_device, set_seed)
from src.data import example_inputs, get_dataloaders
from src.engine import finetune
from src.metrics import benchmark
from src.model import load_baseline, save_pruned
from src.pruning import (global_unstructured_prune, make_masks_permanent,
                         structured_prune)

COLUMNS = [
    "model", "method", "ratio", "accuracy", "params_M", "macs_M",
    "nonzero_params_M", "sparsity_pct", "size_MB", "act_mem_MB",
    "cpu_latency_ms", "cpu_latency_p90_ms",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(BASELINE_CKPT))
    ap.add_argument("--ratios", type=float, nargs="+", default=list(PRUNING_RATIOS))
    ap.add_argument("--finetune-epochs", type=int, default=3)
    ap.add_argument("--finetune-lr", type=float, default=0.01)
    ap.add_argument("--eval-subset", type=int, default=None,
                    help="restrict the test set (default: all 10,000 images)")
    ap.add_argument("--latency-runs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--global-pruning", dest="global_pruning", action="store_true", default=True)
    ap.add_argument("--uniform-pruning", dest="global_pruning", action="store_false")
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--latency-only", action="store_true",
                    help="re-measure only the latency columns of an existing CSV")
    ap.add_argument("--latency-repeats", type=int, default=1,
                    help="timing blocks per model; the fastest block is reported")
    ap.add_argument("--out", default=str(BENCHMARK_CSV))
    args = ap.parse_args()

    set_seed(42)
    device = get_device(args.device)
    print(f"device: {device}  |  global_pruning={args.global_pruning}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, eval_subset=args.eval_subset, num_workers=args.workers
    )
    print(f"test images: {len(test_loader.dataset)}")

    ex = example_inputs()
    baseline = load_baseline(args.ckpt)

    if args.latency_only:
        _remeasure_latency(baseline, ex, args)
        return

    rows = [benchmark(baseline, "ResNet18 baseline", ex, test_loader, device,
                      latency_runs=args.latency_runs, tag="none", ratio=0.0)]
    print(f"baseline: acc {rows[0]['accuracy']:.2f}%  "
          f"params {rows[0]['params_M']:.2f}M  MACs {rows[0]['macs_M']:.0f}M")

    t_start = time.perf_counter()
    for ratio in args.ratios:
        print(f"\n{'='*66}\nratio {ratio:.0%}\n{'='*66}")

        # --- unstructured ---------------------------------------------------
        uns = global_unstructured_prune(baseline, ratio, permanent=True)
        rows.append(benchmark(uns, f"unstructured {ratio:.0%}", ex, test_loader, device,
                              latency_runs=args.latency_runs, tag="unstructured", ratio=ratio))
        print(f"  unstructured      acc {rows[-1]['accuracy']:.2f}%  "
              f"params {rows[-1]['params_M']:.2f}M  MACs {rows[-1]['macs_M']:.0f}M")

        if not args.skip_finetune:
            uns_ft = global_unstructured_prune(baseline, ratio, permanent=False)
            finetune(uns_ft, train_loader, test_loader, device,
                     epochs=args.finetune_epochs, lr=args.finetune_lr)
            make_masks_permanent(uns_ft)
            rows.append(benchmark(uns_ft, f"unstructured {ratio:.0%} +ft", ex, test_loader, device,
                                  latency_runs=args.latency_runs,
                                  tag="unstructured+ft", ratio=ratio))
            print(f"  unstructured+ft   acc {rows[-1]['accuracy']:.2f}%")

        # --- structured -----------------------------------------------------
        st = structured_prune(baseline, ex, ratio=ratio, global_pruning=args.global_pruning)
        rows.append(benchmark(st, f"structured {ratio:.0%}", ex, test_loader, device,
                              latency_runs=args.latency_runs, tag="structured", ratio=ratio))
        print(f"  structured        acc {rows[-1]['accuracy']:.2f}%  "
              f"params {rows[-1]['params_M']:.2f}M  MACs {rows[-1]['macs_M']:.0f}M")

        if not args.skip_finetune:
            st_ft = copy.deepcopy(st)
            finetune(st_ft, train_loader, test_loader, device,
                     epochs=args.finetune_epochs, lr=args.finetune_lr)
            rows.append(benchmark(st_ft, f"structured {ratio:.0%} +ft", ex, test_loader, device,
                                  latency_runs=args.latency_runs,
                                  tag="structured+ft", ratio=ratio))
            print(f"  structured+ft     acc {rows[-1]['accuracy']:.2f}%")
            if abs(ratio - MAIN_RATIO) < 1e-9:
                out = Path(args.ckpt).parent / f"resnet18_cifar10_pruned{int(ratio*100)}.pt"
                save_pruned(st_ft, out)
                print(f"  saved headline pruned model -> {out}")

        pd.DataFrame(rows)[COLUMNS].to_csv(args.out, index=False)  # checkpoint as we go

    df = pd.DataFrame(rows)[COLUMNS]
    df.to_csv(args.out, index=False)
    (RESULTS_DIR / "benchmark.md").write_text(_markdown_report(df, args))
    _update_readme(df)
    print(f"\ntotal {(time.perf_counter()-t_start)/60:.1f} min")
    print(f"csv -> {args.out}\nmd  -> {RESULTS_DIR/'benchmark.md'}")


def _remeasure_latency(baseline: nn.Module, ex: torch.Tensor, args) -> None:
    """Re-time every variant in an existing CSV, leaving the other columns alone.

    Latency depends only on the shape of the network, never on the values in it,
    and every architecture in the sweep is reproducible from the baseline plus a
    ratio. So the timings can be redone on an idle machine without repeating the
    hour of fine-tuning that produced the accuracies.
    """
    from src.metrics import latency_ms

    df = pd.read_csv(args.out)

    def arch_key(row) -> str:
        """Rows sharing a key run the exact same computation.

        Zeroing weights changes no tensor shape, so the baseline and every
        unstructured variant are one architecture. Fine-tuning changes values,
        not shapes, so `structured 50%` and `structured 50% +ft` are one too.
        Timing each architecture once and sharing the result keeps the report
        free of differences that are really just machine drift.
        """
        method = str(row["method"])
        return f"structured@{float(row['ratio']):.2f}" if method.startswith("structured") else "dense"

    architectures: Dict[str, nn.Module] = {"dense": baseline}
    for _, row in df.iterrows():
        key = arch_key(row)
        if key not in architectures:
            architectures[key] = structured_prune(
                baseline, ex, ratio=float(row["ratio"]), global_pruning=args.global_pruning
            )

    print(f"{len(df)} variants share {len(architectures)} distinct architectures")
    print(f"timing each with {args.latency_repeats} x {args.latency_runs} runs\n")

    # Warm every architecture before timing any of them, so the first one measured
    # does not absorb one-time allocator and kernel-selection costs.
    with torch.inference_mode():
        for model in architectures.values():
            model.to("cpu").eval()
            for _ in range(20):
                model(ex)

    timings = {}
    for key, model in architectures.items():
        lat = latency_ms(model, ex, torch.device("cpu"),
                         runs=args.latency_runs, warmup=20, repeats=args.latency_repeats)
        timings[key] = lat
        print(f"  {key:20} {lat['median']:6.2f} ms")

    for i, row in df.iterrows():
        lat = timings[arch_key(row)]
        df.at[i, "cpu_latency_ms"] = lat["median"]
        df.at[i, "cpu_latency_p90_ms"] = lat["p90"]

    df.to_csv(args.out, index=False)
    (RESULTS_DIR / "benchmark.md").write_text(_markdown_report(df, args))
    _update_readme(df)
    print(f"\ncsv -> {args.out}")


def _signed(value: float) -> str:
    """Format with a typographic minus, matching the other rows of the table."""
    return f"{'−' if value < 0 else '+'}{abs(value):.2f}"


def _update_readme(df: pd.DataFrame) -> None:
    """Fill the HEADLINE and RESULTS blocks in README.md from the measured data.

    Keeps the numbers in the README identical to the ones in the CSV instead of
    relying on someone to retype them after a re-run.
    """
    readme = RESULTS_DIR.parent / "README.md"
    if not readme.exists():
        return

    base = df.iloc[0]
    main = df[(df["ratio"] == MAIN_RATIO) & (df["method"] == "structured+ft")]
    if main.empty:
        main = df[(df["ratio"] == MAIN_RATIO) & (df["method"] == "structured")]
    if main.empty:
        return
    m = main.iloc[0]
    label = (f"Pruned {MAIN_RATIO:.0%} + fine-tune" if m["method"] == "structured+ft"
             else f"Pruned {MAIN_RATIO:.0%}, no fine-tune")

    headline = [
        f"| | Baseline | {label} | Change |",
        "|---|---|---|---|",
        f"| **Parameters** | {base['params_M']:.2f} M | {m['params_M']:.2f} M | "
        f"**−{100 * (1 - m['params_M'] / base['params_M']):.1f}%** |",
        f"| **MACs** | {base['macs_M']:.0f} M | {m['macs_M']:.0f} M | "
        f"**−{100 * (1 - m['macs_M'] / base['macs_M']):.1f}%** |",
        f"| **Top-1 accuracy** | {base['accuracy']:.2f}% | {m['accuracy']:.2f}% | "
        f"**{_signed(m['accuracy'] - base['accuracy'])} pts** |",
        f"| Model size | {base['size_MB']:.1f} MB | {m['size_MB']:.1f} MB | "
        f"−{100 * (1 - m['size_MB'] / base['size_MB']):.1f}% |",
        f"| CPU latency, batch 1 | {base['cpu_latency_ms']:.2f} ms | "
        f"{m['cpu_latency_ms']:.2f} ms | "
        f"**{base['cpu_latency_ms'] / m['cpu_latency_ms']:.2f}× faster** |",
        f"| Activation memory | {base['act_mem_MB']:.2f} MB | {m['act_mem_MB']:.2f} MB | "
        f"−{100 * (1 - m['act_mem_MB'] / base['act_mem_MB']):.1f}% |",
    ]

    out = df.copy()
    view = pd.DataFrame({
        "Model": out["model"],
        "Accuracy (%)": out["accuracy"].round(2),
        "vs base (pts)": (out["accuracy"] - base["accuracy"]).round(2),
        "Params (M)": out["params_M"].round(2),
        "MACs (M)": out["macs_M"].round(0),
        "Size (MB)": out["size_MB"].round(1),
        "CPU lat (ms)": out["cpu_latency_ms"].round(2),
        "Speed-up": (base["cpu_latency_ms"] / out["cpu_latency_ms"]).round(2),
    })
    results = [
        view.to_markdown(index=False),
        "",
        "![accuracy](results/figures/fig1_accuracy_vs_ratio.png)",
        "",
        "![pareto](results/figures/fig3_accuracy_vs_macs.png)",
    ]

    text = readme.read_text()
    for tag, body in (("HEADLINE", headline), ("RESULTS", results)):
        start, end = f"<!-- {tag}:BEGIN -->", f"<!-- {tag}:END -->"
        i, j = text.find(start), text.find(end)
        if i == -1 or j == -1:
            continue
        text = text[: i + len(start)] + "\n" + "\n".join(body) + "\n" + text[j:]
    readme.write_text(text)
    print("README.md updated")


def _markdown_report(df: pd.DataFrame, args) -> str:
    base = df.iloc[0]
    out = df.copy()
    out["params_%"] = (100 * out["params_M"] / base["params_M"]).round(1)
    out["macs_%"] = (100 * out["macs_M"] / base["macs_M"]).round(1)
    out["speedup"] = (base["cpu_latency_ms"] / out["cpu_latency_ms"]).round(2)
    out["acc_drop_pp"] = (out["accuracy"] - base["accuracy"]).round(2)

    show = out[["model", "accuracy", "acc_drop_pp", "params_M", "params_%",
                "macs_M", "macs_%", "size_MB", "cpu_latency_ms", "speedup"]].round(3)

    main = out[(out["ratio"] == MAIN_RATIO) & (out["method"].isin(["structured", "structured+ft"]))]
    lines = [
        "# Benchmark: Torch-Pruning on ResNet-18 / CIFAR-10",
        "",
        f"- device (accuracy + fine-tuning): `{get_device(args.device)}`",
        f"- latency: CPU, batch size 1, input 1x3x32x32, {args.latency_repeats} "
        f"block(s) of {args.latency_runs} runs; the fastest block's median is reported",
        "- rows sharing an architecture share one latency measurement "
        "(unstructured pruning changes no tensor shape)",
        "- MACs / parameters: `torch_pruning.utils.count_ops_and_params`, measured on CPU",
        ("- fine-tuning: skipped (`--skip-finetune`)" if args.skip_finetune else
         f"- fine-tuning: {args.finetune_epochs} epochs, SGD lr={args.finetune_lr}, "
         "cosine schedule"),
        f"- channel selection: group L2 magnitude, "
        f"{'global' if args.global_pruning else 'uniform per-layer'} ranking",
        "",
        "## Required comparison (before vs. after structured pruning)",
        "",
    ]
    if not main.empty:
        req = pd.concat([out.iloc[[0]], main])[
            ["model", "params_M", "macs_M", "accuracy"]
        ].round(3)
        req.columns = ["Model", "Parameters (M)", "MACs (M)", "Accuracy (%)"]
        lines += [req.to_markdown(index=False), ""]
    lines += ["## Full sweep", "", show.to_markdown(index=False), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
