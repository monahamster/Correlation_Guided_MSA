# October 2026 revision: code and experimental records

This directory connects each reported run to the exact code, configuration, validation history and test predictions used for it. `registry.json` is the machine-readable index; `docs/EXPERIMENT_INDEX.csv` provides a compact view. Shared runs are listed once and can belong to multiple tables.

## Recompute results without training

From the repository root, in a Python 3.8 environment:

```bash
pip install -r revision_2026_10/requirements-results.txt
python revision_2026_10/tools/verify_release.py
python revision_2026_10/tools/reproduce_results.py --output /tmp/cgmsa-results --plots
python revision_2026_10/tools/plot_projections.py --output /tmp/cgmsa-projections
```

Use new output directories. The first result command verifies all 78 prediction files, recomputes eight metrics and 224 table entries (mean and sample standard deviation), checks validation-MAE checkpoint selection, and reproduces the four paired MAE comparisons. It also draws confusion matrices and reruns the original robustness and efficiency plotting scripts on copies of the archived summaries. Projection plots use the stored PCA/t-SNE coordinates; fitting new projections requires the fusion features described in `docs/ARTIFACTS.md`.

These commands use the included numeric records. They do not download datasets, load model checkpoints or start training. Robustness curves are redrawn from archived evaluation summaries; this command does not re-infer every perturbation.

## Prepare a downstream training run

Use Linux, Python 3.8.10 and the package versions in `requirements-training.txt`. Exact archived environment records are under each snapshot's `protocol/environment.json`. CUDA hardware and drivers must be compatible with PyTorch 2.0.1.

1. Supply the dataset features, BERT files, fixed word-feature caches and selected correlation checkpoints listed in `docs/ARTIFACTS.md`.
2. Copy `assets.example.json` to an external path and fill in those locations.
3. Choose a run ID from `docs/EXPERIMENT_INDEX.csv` and prepare a new workspace:

```bash
python revision_2026_10/tools/prepare_run.py   --id mosi_dynamic_seed5576   --assets /path/to/assets.json   --workspace /path/to/new/mosi_dynamic_seed5576   --gpu 0 --check-inputs
```

The preparation command does not train. It verifies the archived source, copies it, relocates filesystem paths, checks required input hashes, and writes `launch.json`, `run.sh` and `relocation.json`. Non-path experiment arguments remain unchanged. It requires only the assets consumed by that run and records which unrelated input checks were omitted. The full original manifest remains in `source_snapshots/`.

To launch the prepared experiment explicitly:

```bash
bash /path/to/new/mosi_dynamic_seed5576/run.sh
```

The original training entry point checks the protocol and environment again. The output directory must be new. It trains for the fixed epoch budget and selects the earliest minimum validation MAE; it does not use test performance to choose a checkpoint. The portable evaluator uses the same inference and metric rules as the archived evaluator:

```bash
python revision_2026_10/tools/evaluate_run.py --workspace /path/to/new/mosi_dynamic_seed5576 --run-dir /path/to/completed/run --split dev --output /tmp/cgmsa-dev
python revision_2026_10/tools/evaluate_run.py --workspace /path/to/new/mosi_dynamic_seed5576 --run-dir /path/to/completed/run --split test --output /tmp/cgmsa-test
```

A completed run is the directory containing `run_config.json`, `history.jsonl`, `dev_predictions.npz` and `best_model.pt`. The evaluator uses that selected checkpoint. For replay of an original archived checkpoint, omit `--run-dir` and supply `--checkpoint /path/to/original/best_model.pt`; its hash must match the registry. Run dev verification before test inference. `--limit 2 --gpu -1` performs a small CPU dev preflight; limits are prohibited for test evaluation. Existing outputs are not overwritten. The original evaluator remains in `analysis/clean_original/`.

## Prepare the selected pretraining recipe

```bash
python revision_2026_10/tools/prepare_pretraining.py   --dataset mosi --cache-root /path/to/repair_cache   --workspace /path/to/new/mosi_pretraining --gpu 0
bash /path/to/new/mosi_pretraining/run.sh
```

Use `--dataset mosei` and another new workspace for MOSEI. Preparation changes only cache/manifest paths in a copy of the original recipe. It preserves the selected objective, masks, donor sampling, 20-epoch budget and checkpoint rule. Cache SHA-256 checks remain active. Both recipes use pretraining seed 5576; downstream seed variation does not retrain this branch. See `docs/PROTOCOL.md` for the two selected objectives.

## Contents

| Path | Contents |
|---|---|
| `source_snapshots/` | Ten immutable source snapshots; exact code used by the archived runs |
| `runs/` | 78 configurations, histories, dev/test predictions and metrics |
| `results/clean/` | Test evaluation manifest, comparison tables and validation records |
| `pretraining/` | Selected pretraining code, histories, donor maps and checkpoint hashes |
| `development/` | Objective, positional-encoding, regularization and readout screening records |
| `analysis/` | Robustness, efficiency, non-text diagnostics, cases and representation records |
| `tools/` | Portable preparation, result checking and plotting commands |
| `provenance.json` | Original location and SHA-256 for copied evidence |

Original paths are retained in archived records to connect them to the experiment logs. Use the preparation tools to work at new paths. Checkpoints, raw feature datasets, BERT weights, media, manuscripts and reviewer correspondence are not part of this Git source release. No new training was performed while packaging it.
