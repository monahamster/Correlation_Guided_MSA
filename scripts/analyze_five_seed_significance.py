#!/usr/bin/env python3
"""Compute paired five-seed significance tests for the primary comparison.

The same random seeds are used for GLoMo* and Ours, so each seed yields one
paired observation. The script reports two-sided paired t-tests and exact
Wilcoxon signed-rank tests. Holm-adjusted p-values control family-wise error
across every reported dataset--metric comparison.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


METRICS = (
    ("mae", "MAE", "lower"),
    ("corr", "Corr", "higher"),
    ("acc2", "Acc-2", "higher"),
    ("f1", "F1", "higher"),
    ("acc2_non_zero", "Acc-2 (nz)", "higher"),
    ("f1_non_zero", "F1 (nz)", "higher"),
    ("acc7", "ACC-7", "higher"),
)


def parse_seeds(value: str):
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) < 5:
        raise argparse.ArgumentTypeError("At least five paired seeds are required.")
    return seeds


def load_metrics(root: Path, dataset: str, variant: str, seed: int):
    path = root / f"{dataset}_{variant}_valmae_seed{seed}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing primary-run metric file: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("save_best_by") != "valid_mae":
        raise ValueError(f"{path} was not selected by validation MAE.")
    return payload


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in the input order."""
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--seeds", type=parse_seeds, default=(5576, 42, 2026, 2027, 3407))
    parser.add_argument("--out-prefix", default="experiments/five_seed_significance")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    rows = []
    for dataset in ("mosi", "mosei"):
        baseline = [load_metrics(root, dataset, "baseline", seed) for seed in args.seeds]
        ours = [load_metrics(root, dataset, "ours", seed) for seed in args.seeds]
        for key, label, direction in METRICS:
            base_values = np.asarray([record[key] for record in baseline], dtype=float)
            ours_values = np.asarray([record[key] for record in ours], dtype=float)
            # Positive delta consistently means a directional improvement for Ours.
            delta = base_values - ours_values if direction == "lower" else ours_values - base_values
            t_stat, t_p = ttest_rel(ours_values, base_values, alternative="two-sided")
            try:
                _, w_p = wilcoxon(ours_values, base_values, alternative="two-sided", method="exact")
            except ValueError:
                w_p = float("nan")
            rows.append({
                "dataset": dataset.upper(),
                "metric": label,
                "direction": direction,
                "seeds": ",".join(map(str, args.seeds)),
                "baseline_mean": float(base_values.mean()),
                "baseline_std": float(base_values.std(ddof=1)),
                "ours_mean": float(ours_values.mean()),
                "ours_std": float(ours_values.std(ddof=1)),
                "improvement_delta": float(delta.mean()),
                "paired_t_statistic": float(t_stat),
                "paired_t_p": float(t_p),
                "wilcoxon_p": float(w_p),
            })

    adjusted = holm_adjust(np.asarray([row["paired_t_p"] for row in rows], dtype=float))
    for row, p_holm in zip(rows, adjusted):
        row["paired_t_p_holm"] = float(p_holm)
        row["significant_holm_0_05"] = bool(p_holm < 0.05)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['dataset']:5s} {row['metric']:11s} "
            f"delta={row['improvement_delta']:+.4f} "
            f"t_p={row['paired_t_p']:.4f} holm={row['paired_t_p_holm']:.4f} "
            f"wilcoxon={row['wilcoxon_p']:.4f}"
        )


if __name__ == "__main__":
    main()
