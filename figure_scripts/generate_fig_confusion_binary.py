#!/usr/bin/env python3
"""Generate fig_confusion_binary.pdf from saved MOSI/MOSEI prediction files.

Expected input structure under --data-root:
  mosi_baseline_seed5576.zip  or  mosi_baseline_seed5576/analysis/predictions.csv
  mosi_ours_seed5576.zip      or  mosi_ours_seed5576/analysis/predictions.csv
  mosei_baseline_seed5576.zip or  mosei_baseline_seed5576/analysis/predictions.csv
  mosei_ours_seed5576.zip     or  mosei_ours_seed5576/analysis/predictions.csv
"""
import argparse
import os
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


def confusion_counts(y_true, y_pred, labels=(0, 1)):
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            mat[label_to_idx[t], label_to_idx[p]] += 1
    return mat


def draw_matrix(ax, mat, title):
    im = ax.imshow(mat, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg.", "Pos."], fontsize=9)
    ax.set_yticklabels(["Neg.", "Pos."], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    row_sum = mat.sum(axis=1, keepdims=True)
    pct = np.divide(mat, np.maximum(row_sum, 1)) * 100
    threshold = mat.max() / 2.0 if mat.max() > 0 else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > threshold else "black"
            ax.text(j, i, f"{mat[i, j]}\n{pct[i, j]:.1f}%", ha="center", va="center", fontsize=8, color=color)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".", help="Directory containing experiment folders or zip files.")
    parser.add_argument("--out-dir", default="figures", help="Output figure directory.")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("mosi_baseline_seed5576", "CMU-MOSI / GLoMo$^{*}$"),
        ("mosi_ours_seed5576", "CMU-MOSI / Ours"),
        ("mosei_baseline_seed5576", "CMU-MOSEI / GLoMo$^{*}$"),
        ("mosei_ours_seed5576", "CMU-MOSEI / Ours"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for ax, (exp_name, title) in zip(axes.ravel(), specs):
        df = read_predictions(data_root, exp_name)
        mat = confusion_counts(df["label_2"].to_numpy(), df["pred_2"].to_numpy(), labels=(0, 1))
        draw_matrix(ax, mat, title)
    fig.suptitle("Binary Confusion Matrices", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig_confusion_binary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_confusion_binary.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
