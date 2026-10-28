# Assets and provenance

The Git release contains source code and numeric experimental evidence. The following assets are external:

| Asset | Required for | Record |
|---|---|---|
| `mosi.pkl`, `mosei.pkl` | Downstream training | Snapshot `protocol/input_manifest.json` |
| BERT encoder/tokenizer files | Downstream training and fixed feature extraction | Same input manifest |
| `repair_cache/{mosi,mosei}/{train,dev,test}.fixed_contextual_word_v1.pt` | Guided runs and correlation pretraining | `pretraining/input_manifest.json` |
| Selected correlation weights | Guided downstream training | `pretraining/{mosi,mosei}/checkpoint.json` |
| 78 selected downstream checkpoints | Fresh inference, perturbations and efficiency measurement | `registry.json` |
| Final fusion feature arrays | Refitting PCA/t-SNE and feature statistics | Original extraction and representation scripts |

These assets remain in the experiment storage; this package does not provide public download links. `assets.example.json` shows the required path configuration. The two small correlation checkpoints are also prepared in a separate local artifact bundle, outside Git, for review before distribution. Dataset and BERT assets are not redistributed here.

The original fixed-word cache builder is retained as `Correlation_Pretraining/modality_correlation/fixed_word_features.py` in the source snapshots. Exact input-hash verification requires the archived cache bytes. Rebuilding a cache can change serialization or metadata and must be validated as a new input artifact; the preparation tools do not silently accept a mismatched hash. Availability of those archived assets is therefore required for hash-verified training reproduction.

Four seed-5576 PCA/t-SNE coordinate files are included in `analysis/representations/projections/` for redrawing the figures. These coordinates are fitted independently by model. Labels are used for coloring. To refit projections, the separate local representation-feature bundle contains the extracted arrays and original extraction-completion record. The original fitting settings and scripts are retained in `analysis/manuscript_original/p6_analysis_visuals_20261026/`.

`provenance.json` records copied-file hashes; `source_snapshots/*/protocol/frozen_source.json` preserves the training-time source checks. Source snapshots are not edited for portability. `prepare_run.py` creates a separate copy and updates path-bearing metadata plus its derived source manifest; `relocation.json` records the changes. Original histories, metrics and predictions remain unchanged.
