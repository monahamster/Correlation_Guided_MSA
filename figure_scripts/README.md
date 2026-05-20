# Figure generation scripts

This folder follows the requirement that each figure has a corresponding generation script.

## Scripts and outputs

| Script | Output figure |
|---|---|
| `generate_fig_confusion_binary.py` | `figures/fig_confusion_binary.pdf` |
| `generate_fig_confusion_mosi_7class.py` | `figures/fig_confusion_mosi_7class.pdf` |
| `generate_fig_silhouette_selected.py` | `figures/fig_silhouette_selected.pdf` |
| `generate_fig_representation_pca.py` | `figures/fig_representation_pca.pdf` |

Each script also saves a `.png` preview with the same base name.

## Required input files

Place either the extracted experiment folders or the zip files in the data root:

```text
mosi_baseline_seed5576.zip
mosi_ours_seed5576.zip
mosei_baseline_seed5576.zip
mosei_ours_seed5576.zip
```

or extracted folders with the same names. The scripts read:

```text
analysis/predictions.csv
analysis/repr.npz
```

## Example usage

Run from the project root:

```bash
python scripts/generate_fig_confusion_binary.py --data-root . --out-dir figures
python scripts/generate_fig_confusion_mosi_7class.py --data-root . --out-dir figures
python scripts/generate_fig_silhouette_selected.py --data-root . --out-dir figures
python scripts/generate_fig_representation_pca.py --data-root . --out-dir figures
```

If your zip files are stored elsewhere, replace `--data-root .` with that directory.
