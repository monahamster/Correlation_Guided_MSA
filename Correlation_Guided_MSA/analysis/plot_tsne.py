import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from umap import UMAP
except Exception:  # pragma: no cover
    UMAP = None


def load_repr(repr_path, label_key):
    data = np.load(repr_path, allow_pickle=True)
    return data["repr"], data[label_key]


def reduce_repr(reprs, method, perplexity, random_state):
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(reprs)
    if method == "umap":
        if UMAP is None:
            raise ImportError("umap-learn is not installed. Please install it or use --method tsne/pca.")
        return UMAP(n_components=2, random_state=random_state).fit_transform(reprs)
    n_samples = reprs.shape[0]
    perplexity = min(perplexity, max(5.0, float(n_samples - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(reprs)


def plot_panel(ax, coords, labels, title, label_key):
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="Spectral", s=14, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    return scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr", type=str, required=True, help="Current model repr.npz")
    parser.add_argument("--baseline-repr", type=str, default="", help="Optional baseline repr.npz")
    parser.add_argument(
        "--label-key",
        type=str,
        choices=["label_reg", "label_2", "label_7", "pred_reg", "pred_2", "pred_7"],
        default="label_7",
    )
    parser.add_argument("--method", type=str, choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    reprs, labels = load_repr(args.repr, args.label_key)
    coords = reduce_repr(reprs, args.method, args.perplexity, args.random_state)

    if args.baseline_repr:
        baseline_repr, baseline_labels = load_repr(args.baseline_repr, args.label_key)
        baseline_coords = reduce_repr(baseline_repr, args.method, args.perplexity, args.random_state)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        scatter = plot_panel(axes[0], baseline_coords, baseline_labels, "Baseline", args.label_key)
        plot_panel(axes[1], coords, labels, "Ours", args.label_key)
        fig.suptitle(f"{args.method.upper()} of sentence representations ({args.label_key})")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = plot_panel(ax, coords, labels, f"{args.method.upper()} of sentence representations", args.label_key)

    cbar = fig.colorbar(scatter, ax=fig.axes, shrink=0.9)
    cbar.set_label(args.label_key)
    fig.tight_layout()

    default_name = f"{args.method}_compare_{args.label_key}.png" if args.baseline_repr else f"{args.method}_{args.label_key}.png"
    output = Path(args.output) if args.output else Path(args.repr).with_name(default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"saved representation figure to {output}")


if __name__ == "__main__":
    main()
