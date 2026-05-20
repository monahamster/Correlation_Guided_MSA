#!/usr/bin/env python3
"""Generate fig_confusion_mosi_7class.pdf from saved MOSI prediction files."""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_predictions(data_root: Path, exp_name: str) -> pd.DataFrame:
    csv_path = data_root / exp_name / "analysis" / "predictions.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    zip_path = data_root / f"{exp_name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith("analysis/predictions.csv")]
            if not candidates:
                raise FileNotFoundError(f"No analysis/predictions.csv found in {zip_path}")
            return pd.read_csv(zf.open(candidates[0]))
    raise FileNotFoundError(f"Cannot find {csv_path} or {zip_path}")


def confusion_counts(y_true, y_pred, labels=range(7)):
    mat = np.zeros((7, 7), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) <= 6 and 0 <= int(p) <= 6:
            mat[int(t), int(p)] += 1
    return mat


def draw_matrix(ax, mat, title):
    im = ax.imshow(mat, interpolation="nearest")
    display_labels = ["-3", "-2", "-1", "0", "1", "2", "3"]
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels(display_labels, fontsize=8)
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    row_sum = mat.sum(axis=1, keepdims=True)
    pct = np.divide(mat, np.maximum(row_sum, 1)) * 100
    threshold = mat.max() / 2.0 if mat.max() > 0 else 0
    for i in range(7):
        for j in range(7):
            color = "white" if mat[i, j] > threshold else "black"
            ax.text(j, i, f"{mat[i, j]}\n{pct[i, j]:.1f}%", ha="center", va="center", fontsize=5.5, color=color)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("mosi_baseline_seed5576", "CMU-MOSI / GLoMo$^{*}$"),
        ("mosi_ours_seed5576", "CMU-MOSI / Ours"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    for ax, (exp_name, title) in zip(axes, specs):
        df = read_predictions(data_root, exp_name)
        mat = confusion_counts(df["label_7"].to_numpy(), df["pred_7"].to_numpy())
        draw_matrix(ax, mat, title)
    fig.suptitle("Seven-class Confusion Matrices on CMU-MOSI", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "fig_confusion_mosi_7class.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_confusion_mosi_7class.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
