# Packaging validation, 2026-10-28

- Recomputed all 78 clean runs, 224 metric summaries and four paired MAE tests from released predictions.
- Materialized all 78 configurations in separate workspaces; the original argument parser and protocol guard accepted every configuration. Non-path arguments and source hashes were checked.
- Imported all ten snapshot families on CPU and passed their original source/environment guards.
- Prepared both selected pretraining recipes and verified all six fixed-cache input hashes. No pretraining was started.
- Replayed two MOSI Dynamic dev examples on CPU with the archived checkpoint and checked agreement against the saved predictions. This is a small inference preflight, not a new full training or full-test run.
- Redrew confusion, robustness, efficiency and stored projection plots.
- Checked copied-file SHA-256 values, Python syntax and excluded-asset/credential patterns.

The detailed JSON records in this directory describe these checks. Training was not repeated during packaging. External datasets, encoder files and weights remain required for fresh training or inference.

All 1,062 copied evidence files were also verified against their staged Git blob bytes. Newly authored tools and documentation passed whitespace checks; historical source whitespace is preserved for byte-exact provenance.
