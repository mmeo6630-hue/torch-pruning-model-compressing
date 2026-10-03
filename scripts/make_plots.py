#!/usr/bin/env python3
"""Render the benchmark figures used in the slides and the README.

    python scripts/make_plots.py

Reads results/benchmark.csv (written by run_benchmark.py) and, when the baseline
checkpoint is present, also draws the two model-structure figures.

Encoding used throughout: hue = pruning method (blue = structured,
orange = unstructured); line style = fine-tuning (solid = fine-tuned,
dashed = no recovery training). Two orthogonal facts, two orthogonal channels,
so no figure needs more than two hues.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASELINE_CKPT, BENCHMARK_CSV, FIGURES_DIR, MAIN_RATIO

# --- Design tokens (validated categorical palette, light surface) ------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
AXIS = "#d6d5d0"
STRUCTURED = "#2a78d6"   # categorical slot 1
UNSTRUCTURED = "#eb6834"  # categorical slot 2
BACKDROP = "#e3e2de"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "figure.dpi": 150,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",          # solid hairline; dashed grids read as thresholds
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_SOFT,
})

SERIES = [
    # (method tag, label, color, linestyle, marker, filled?)
    ("structured+ft", "Structured + fine-tune", STRUCTURED, "-", "o", True),
    ("structured", "Structured, no fine-tune", STRUCTURED, "--", "o", False),
    ("unstructured+ft", "Unstructured + fine-tune", UNSTRUCTURED, "-", "s", True),
    ("unstructured", "Unstructured, no fine-tune", UNSTRUCTURED, "--", "s", False),
]


def _plot_series(ax, df, ycol, tag, label, color, ls, marker, filled):
    sub = df[df["method"] == tag].sort_values("ratio")
    if sub.empty:
        return None
    ax.plot(
        sub["ratio"] * 100, sub[ycol],
        color=color, linestyle=ls, marker=marker, label=label,
        markerfacecolor=color if filled else SURFACE,
        markeredgecolor=color, markeredgewidth=1.6,
        clip_on=False, zorder=3,
    )
    return sub


def _baseline_rule(ax, value, text, xtext=None, below=False):
    """Recessive reference line for the unpruned model."""
    ax.axhline(value, color=INK_MUTED, linewidth=1.0, linestyle=(0, (1, 2)), zorder=1)
    x = xtext if xtext is not None else ax.get_xlim()[0]
    dy, va = (-6, "top") if below else (5, "bottom")
    ax.annotate(text, xy=(x, value), xytext=(0, dy), textcoords="offset points",
                color=INK_MUTED, fontsize=8.5, va=va, ha="left")


# ---------------------------------------------------------------------------
def fig_accuracy(df: pd.DataFrame) -> None:
    base = df.iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for tag, label, color, ls, marker, filled in SERIES:
        sub = _plot_series(ax, df, "accuracy", tag, label, color, ls, marker, filled)
        # Direct-label the endpoint only; the axis carries the rest.
        if sub is not None and tag in ("structured+ft", "structured"):
            last = sub.iloc[-1]
            ax.annotate(f"{last['accuracy']:.1f}%",
                        xy=(last["ratio"] * 100, last["accuracy"]),
                        xytext=(8, -2), textcoords="offset points",
                        color=INK_SOFT, fontsize=9, va="center")

    ax.margins(y=0.12)
    _baseline_rule(ax, base["accuracy"], f"unpruned baseline  {base['accuracy']:.2f}%")
    ax.set_xlabel("Channels / weights removed (%)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Top-1 accuracy vs. pruning ratio")
    ax.legend(loc="lower left", ncols=1)
    fig.savefig(FIGURES_DIR / "fig1_accuracy_vs_ratio.png")
    plt.close(fig)


def fig_cost(df: pd.DataFrame) -> None:
    """Parameters, MACs and latency in three panels.

    Deliberately three panels rather than one chart with several y-scales: a
    dual-axis plot would let the reader compare slopes that share no unit.
    Cost does not depend on fine-tuning, so each panel needs only two lines.
    """
    base = df.iloc[0]
    panels = [
        ("params_M", "Parameters (M)", "Parameters"),
        ("macs_M", "MACs (M)", "MACs per image"),
        ("cpu_latency_ms", "Latency (ms)", "CPU latency, batch 1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))

    for ax, (col, ylab, title) in zip(axes, panels):
        for tag, color, marker, label in (
            ("structured", STRUCTURED, "o", "Structured"),
            ("unstructured", UNSTRUCTURED, "s", "Unstructured"),
        ):
            sub = df[df["method"] == tag].sort_values("ratio")
            if sub.empty:
                continue
            ax.plot(sub["ratio"] * 100, sub[col], color=color, marker=marker,
                    label=label, markerfacecolor=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, clip_on=False, zorder=3)
        # The unstructured line sits exactly on the baseline value, so a separate
        # baseline rule would just draw the same level twice. Say it in words.
        ax.annotate("same as baseline", xy=(0.99, base[col]),
                    xycoords=("axes fraction", "data"), xytext=(0, 8),
                    textcoords="offset points", color=INK_MUTED, fontsize=8.5,
                    ha="right", va="bottom")
        ax.set_xlabel("Channels / weights removed (%)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(bottom=0, top=base[col] * 1.18)

    axes[0].legend(loc="center right")
    fig.suptitle("Unstructured pruning changes none of these; structured pruning changes all three",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES_DIR / "fig2_cost_vs_ratio.png")
    plt.close(fig)


def fig_pareto(df: pd.DataFrame) -> None:
    base = df.iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for tag, label, color, ls, marker, filled in SERIES:
        if not tag.endswith("+ft"):
            continue
        sub = df[df["method"] == tag].sort_values("macs_M")
        if sub.empty:
            continue
        ax.plot(sub["macs_M"], sub["accuracy"], color=color, linestyle=ls,
                marker=marker, label=label, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.4, clip_on=False, zorder=3)
        if tag == "structured+ft":
            for _, r in sub.iterrows():
                ax.annotate(f"{r['ratio']:.0%}", xy=(r["macs_M"], r["accuracy"]),
                            xytext=(0, 9), textcoords="offset points",
                            color=INK_MUTED, fontsize=8, ha="center")
        else:
            # Every unstructured point has the same MACs, so per-point ratio
            # labels would pile up on one vertical line. One note says it better.
            mid = sub.iloc[len(sub) // 2]
            ax.annotate("20-70% pruned:\nMACs never move",
                        xy=(mid["macs_M"], mid["accuracy"]),
                        xytext=(-8, -42), textcoords="offset points",
                        color=INK_MUTED, fontsize=8.5, ha="right", va="top")

    ax.plot([base["macs_M"]], [base["accuracy"]], marker="*", markersize=13,
            color=INK_SOFT, linestyle="none", label="Unpruned baseline",
            clip_on=False, zorder=4)
    ax.set_xlabel("MACs per image (M)  -  lower is cheaper")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy per unit of compute (labels = pruning ratio)")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "fig3_accuracy_vs_macs.png")
    plt.close(fig)


def fig_channels(ratio: float = MAIN_RATIO) -> None:
    """How global ranking distributes the cuts across the network."""
    import torch
    from src.data import example_inputs
    from src.model import load_baseline
    from src.pruning import channel_counts, structured_prune

    if not Path(BASELINE_CKPT).exists():
        print("  (skip channel figure: no baseline checkpoint yet)")
        return

    model = load_baseline(BASELINE_CKPT)
    ex = example_inputs()
    pruned = structured_prune(model, ex, ratio=ratio, global_pruning=True)
    before, after = channel_counts(model), channel_counts(pruned)

    names = list(before)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.bar(x, [before[n] for n in names], width=0.78, color=BACKDROP,
           label="Before pruning", zorder=2)
    ax.bar(x, [after[n] for n in names], width=0.78, color=STRUCTURED,
           label="After pruning", zorder=3)

    for xi, n in zip(x, names):
        kept = 100 * after[n] / before[n]
        ax.annotate(f"{kept:.0f}%", xy=(xi, after[n]), xytext=(0, 3),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK_SOFT)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Output channels")
    kept_pct = [100 * after[n] / before[n] for n in names]
    ax.set_title(f"Channels kept per layer at {ratio:.0%} global pruning: "
                 f"{min(kept_pct):.0f}%-{max(kept_pct):.0f}% (label = kept)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGURES_DIR / "fig4_channels_per_layer.png")
    plt.close(fig)


def fig_redundancy(layers=("layer1.0.conv2", "layer3.0.conv2", "layer4.1.conv2")) -> None:
    """Per-channel filter norms: the redundancy that makes pruning possible."""
    from src.model import load_baseline
    from src.pruning import channel_l2_norms

    if not Path(BASELINE_CKPT).exists():
        print("  (skip redundancy figure: no baseline checkpoint yet)")
        return

    norms = channel_l2_norms(load_baseline(BASELINE_CKPT))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    weak = []

    for ax, name in zip(axes, layers):
        v = norms[name].numpy()
        v = v / v.max()
        # Fixed bin edges so the bars span the same 0-1 range in every panel.
        ax.hist(v, bins=np.linspace(0, 1, 26), color=STRUCTURED,
                edgecolor=SURFACE, linewidth=0.8, zorder=3)
        med = float(np.median(v))
        weak.append(100.0 * float((v < 0.5).mean()))
        ax.axvline(med, color=INK_MUTED, linewidth=1.0, linestyle=(0, (1, 2)), zorder=4)
        # Top-right: these distributions all skew left, so that corner stays clear.
        ax.text(0.97, 0.95, f"median {med:.2f}", transform=ax.transAxes,
                color=INK_MUTED, fontsize=8.5, va="top", ha="right")
        # Anchor at 0 so "how close to nothing does the weakest filter get?" is
        # readable off the axis instead of hidden by an auto-zoomed range.
        ax.set_xlim(0, 1.02)
        ax.set_title(f"{name}  ({len(v)} channels)", fontsize=10)
        ax.set_xlabel("Filter L2 norm (normalised)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Number of channels")
    fig.suptitle("Per-channel filter norms: "
                 f"{min(weak):.0f}-{max(weak):.0f}% of channels sit below half the strongest",
                 x=0.005, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES_DIR / "fig5_channel_norm_hist.png")
    plt.close(fig)


def main() -> None:
    made = []
    if Path(BENCHMARK_CSV).exists():
        df = pd.read_csv(BENCHMARK_CSV)
        fig_accuracy(df); made.append("fig1_accuracy_vs_ratio.png")
        fig_cost(df); made.append("fig2_cost_vs_ratio.png")
        fig_pareto(df); made.append("fig3_accuracy_vs_macs.png")
    else:
        print(f"  (skip sweep figures: {BENCHMARK_CSV} not found -- run run_benchmark.py first)")
    fig_channels(); made.append("fig4_channels_per_layer.png")
    fig_redundancy(); made.append("fig5_channel_norm_hist.png")
    print("figures ->", FIGURES_DIR)
    for m in made:
        if (FIGURES_DIR / m).exists():
            print("  ", m)


if __name__ == "__main__":
    main()
