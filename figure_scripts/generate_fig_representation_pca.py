#!/usr/bin/env python3
"""Generate fig_representation_pca.pdf from saved final representation files."""
import argparse
import zipfile
from pathlib import Path
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def pca_pair(x1, x2):
    # Fit PCA jointly for each dataset so baseline and ours are comparable in the same projection space.
    x = np.vstack([x1, x2])
    x = StandardScaler().fit_transform(x)
    z = PCA(n_components=2, random_state=42).fit_transform(x)
    return z[: len(x1)], z[len(x1) :]


def scatter(ax, z, labels, title, class_names=None):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    for lab in unique:
        mask = labels == lab
        name = str(lab) if class_names is None else class_names.get(int(lab), str(lab))
        ax.scatter(z[mask, 0], z[mask, 1], s=6, alpha=0.7, label=name, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


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

    z_mosi_base, z_mosi_ours = pca_pair(mosi_base["repr"], mosi_ours["repr"])
    z_mosei_base, z_mosei_ours = pca_pair(mosei_base["repr"], mosei_ours["repr"])

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    mosi_names = {0: "-3", 1: "-2", 2: "-1", 3: "0", 4: "1", 5: "2", 6: "3"}
    mosei_names = {0: "Neg.", 1: "Pos."}

    scatter(axes[0, 0], z_mosi_base, mosi_base["label_7"], "CMU-MOSI / GLoMo$^{*}$", mosi_names)
    scatter(axes[0, 1], z_mosi_ours, mosi_ours["label_7"], "CMU-MOSI / Ours", mosi_names)
    scatter(axes[1, 0], z_mosei_base, mosei_base["label_2"], "CMU-MOSEI / GLoMo$^{*}$", mosei_names)
    scatter(axes[1, 1], z_mosei_ours, mosei_ours["label_2"], "CMU-MOSEI / Ours", mosei_names)

    # One compact legend for each row.
    handles0, labels0 = axes[0, 0].get_legend_handles_labels()
    handles1, labels1 = axes[1, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles0, labels0, title="MOSI label", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    axes[1, 1].legend(handles1, labels1, title="MOSEI label", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

    fig.suptitle("PCA Visualization of Final Fusion Representations", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out_dir / "fig_representation_pca.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_representation_pca.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
