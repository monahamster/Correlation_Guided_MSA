#!/usr/bin/env python3
"""Generate the three-seed controlled-corruption robustness curves.

The script reads metrics.json files saved by the reproducible test-time
corruption runs. It plots mean Pearson correlation with one-standard-deviation
bands for GLoMo* and the proposed model at every tested corruption level.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (5576, 42, 2026)
DATASETS = ("mosi", "mosei")
MODELS = (("baseline", r"GLoMo$^{*}$", "#4C78A8"), ("ours", "Ours", "#E45756"))
CONDITIONS = {
    "audio": (("clean", 0.0), ("audio_n010", 0.1), ("audio_n020", 0.2),
              ("audio_n030", 0.3), ("audio_n050", 0.5), ("audio_n100", 1.0)),
    "visual": (("clean", 0.0), ("visual_m010", 0.1), ("visual_m030", 0.3),
               ("visual_m050", 0.5), ("visual_m070", 0.7), ("visual_m100", 1.0)),
}


def load_metric(experiments_root: Path, dataset: str, model: str, seed: int, condition: str) -> float:
    path = experiments_root / f"{dataset}_{model}_valmae_seed{seed}_corrupt_{condition}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment metric file: {path}")
    with path.open() as handle:
        return float(json.load(handle)["corr"])


def series(experiments_root: Path, dataset: str, model: str, corruption: str):
    levels, means, stds = [], [], []
    for condition, level in CONDITIONS[corruption]:
        values = [load_metric(experiments_root, dataset, model, seed, condition) for seed in SEEDS]
        levels.append(level)
        means.append(np.mean(values))
        stds.append(np.std(values, ddof=1))
    return np.asarray(levels), np.asarray(means), np.asarray(stds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--out-dir", default="revision/overleaf_source/figures")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.5), sharex="col")
    for row, dataset in enumerate(DATASETS):
        for col, corruption in enumerate(("audio", "visual")):
            ax = axes[row, col]
            for model, label, color in MODELS:
                x, mean, std = series(experiments_root, dataset, model, corruption)
                ax.plot(x, mean, marker="o", linewidth=2.0, markersize=4.5, label=label, color=color)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
            ax.grid(axis="y", alpha=0.25, linewidth=0.7)
            ax.set_title(f"{'CMU-MOSI' if dataset == 'mosi' else 'CMU-MOSEI'} / "
                         f"{'audio noise' if corruption == 'audio' else 'visual masking'}", fontsize=10)
            ax.set_xticks(x)
            ax.set_ylim(bottom=min(ax.get_ylim()[0], 0.70), top=max(ax.get_ylim()[1], 0.79))
            if col == 0:
                ax.set_ylabel("Pearson correlation")
            if row == 1:
                ax.set_xlabel(r"Noise level $\sigma$" if corruption == "audio" else r"Mask ratio $p$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "fig_robustness_curves.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_robustness_curves.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
