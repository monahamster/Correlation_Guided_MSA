#!/usr/bin/env python3
"""Generate fig_silhouette_selected.pdf from saved final representation files."""
import argparse
import zipfile
from pathlib import Path
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def read_npz(data_root: Path, exp_name: str):
    npz_path = data_root / exp_name / "analysis" / "repr.npz"
    if npz_path.exists():
        return np.load(npz_path)
    zip_path = data_root / f"{exp_name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith("analysis/repr.npz")]
            if not candidates:
                raise FileNotFoundError(f"No analysis/repr.npz found in {zip_path}")
            return np.load(BytesIO(zf.read(candidates[0])))
    raise FileNotFoundError(f"Cannot find {npz_path} or {zip_path}")


def safe_silhouette(x, y):
    # Silhouette is undefined if there is only one class or every point is its own class.
    y = np.asarray(y)
    valid_classes = np.unique(y)
    if len(valid_classes) < 2 or len(valid_classes) >= len(y):
        return np.nan
    return float(silhouette_score(x, y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mosi_base = read_npz(data_root, "mosi_baseline_seed5576")
    mosi_ours = read_npz(data_root, "mosi_ours_seed5576")
    mosei_base = read_npz(data_root, "mosei_baseline_seed5576")
    mosei_ours = read_npz(data_root, "mosei_ours_seed5576")

    scores = {
        "MOSI / 7-class": [
            safe_silhouette(mosi_base["repr"], mosi_base["label_7"]),
            safe_silhouette(mosi_ours["repr"], mosi_ours["label_7"]),
        ],
        "MOSEI / binary": [
            safe_silhouette(mosei_base["repr"], mosei_base["label_2"]),
            safe_silhouette(mosei_ours["repr"], mosei_ours["label_2"]),
        ],
    }

    labels = list(scores.keys())
    glomo_scores = [scores[k][0] for k in labels]
    ours_scores = [scores[k][1] for k in labels]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    b1 = ax.bar(x - width / 2, glomo_scores, width, label="GLoMo$^{*}$")
    b2 = ax.bar(x + width / 2, ours_scores, width, label="Ours")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.axhline(0, linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 0.01 if h >= 0 else -0.01
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:.3f}", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_silhouette_selected.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_silhouette_selected.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
