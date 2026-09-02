#!/usr/bin/env python3
"""Plot paired five-seed primary effects for GLoMo* versus Ours.

Positive effects consistently favour Ours: for MAE the sign is reversed
(baseline - ours), and for all other metrics it is ours - baseline. The
figure visualises paired seed variability and 95% t intervals; all effects
are labelled as Holm-corrected non-significant.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


SEEDS = (5576, 42, 2026, 2027, 3407)
METRICS = (
    ("mae", "MAE", "lower"),
    ("corr", "Corr", "higher"),
    ("acc2", "Acc-2", "higher"),
    ("f1", "F1", "higher"),
    ("acc2_non_zero", "Acc-2 (nz)", "higher"),
    ("f1_non_zero", "F1 (nz)", "higher"),
    ("acc7", "ACC-7", "higher"),
)
JITTER = np.linspace(-0.15, 0.15, len(SEEDS))


def load_metric(root: Path, dataset: str, variant: str, seed: int, key: str) -> float:
    path = root / f"{dataset}_{variant}_valmae_seed{seed}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing primary-run metric file: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("save_best_by") != "valid_mae":
        raise ValueError(f"Expected validation-MAE selection: {path}")
    return float(payload[key])


def effects(root: Path, dataset: str, key: str, direction: str) -> np.ndarray:
    baseline = np.asarray([load_metric(root, dataset, "baseline", seed, key) for seed in SEEDS])
    ours = np.asarray([load_metric(root, dataset, "ours", seed, key) for seed in SEEDS])
    return baseline - ours if direction == "lower" else ours - baseline


def draw_panel(ax, root: Path, dataset: str, title: str):
    labels = [label for _, label, _ in METRICS]
    positions = np.arange(len(METRICS))[::-1]
    all_effects, summaries = [], []
    for (key, _, direction), pos in zip(METRICS, positions):
        delta = effects(root, dataset, key, direction)
        mean = delta.mean()
        half_ci = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
        all_effects.extend(delta.tolist())
        summaries.append((pos, delta, mean, half_ci))

    limit = max(0.012, max(abs(min(all_effects)), abs(max(all_effects))) * 1.55)
    ax.axvline(0, color="#333333", linewidth=1.0, zorder=0)
    for pos, delta, mean, half_ci in summaries:
        ax.scatter(delta, np.full_like(delta, pos, dtype=float) + JITTER,
                   color="#4C78A8", s=28, alpha=0.82, zorder=3)
        ax.errorbar(mean, pos, xerr=half_ci, fmt="D", color="#E45756",
                    markeredgecolor="black", markersize=5.2, capsize=3,
                    linewidth=1.3, zorder=4)
        ax.text(limit * 0.96, pos, "Holm ns", va="center", ha="right",
                fontsize=7.4, color="#555555")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(-limit, limit)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.set_title(title, fontsize=10.5, pad=7)
    ax.set_xlabel("Paired effect size (positive favors Ours)")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--out-dir", default="revision/overleaf_source/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    draw_panel(axes[0], Path(args.experiments_root), "mosi", "CMU-MOSI")
    draw_panel(axes[1], Path(args.experiments_root), "mosei", "CMU-MOSEI")
    axes[1].tick_params(axis="y", labelleft=True)
    fig.text(0.5, 0.01,
             "Dots: five paired seeds; diamonds: mean; bars: 95% t interval. "
             "All Holm-adjusted paired t-tests are non-significant (p ≥ 0.05).",
             ha="center", fontsize=8.2)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.4)
    fig.savefig(out_dir / "fig_five_seed_effects.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_five_seed_effects.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
