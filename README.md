# Correlation_Guided_MSA

This repository contains the codebase for the current project built around correlation-guided multimodal sentiment analysis and related extensions.

## Overview

The repository is organized around three layers:

- source code for the main models and training entry points
- archived experiment records used to support reproducibility
- script-based figure generation linked to specific experiment outputs

The current paper-facing codebase focuses on CMU-MOSI and CMU-MOSEI.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

If your environment already provides PyTorch separately, adjust the installation order as needed for your platform.

## Repository Structure

- `Correlation_Guided_MSA/`
  - Main codebase for CMU-MOSI and CMU-MOSEI experiments.
- `Correlation_Pretraining/`
  - Correlation pretraining utilities and related components.
- `figure_scripts/`
  - Figure generation scripts used in the paper.
- `experiments/`
  - Experiment records kept for reproducibility and submission archiving.
- `datasets/`
  - Local dataset files used by the code.

## Main Entry Points

### Sentiment Analysis

- `Correlation_Guided_MSA/main_cgmsa.py`

### Correlation Pretraining

- `Correlation_Pretraining/modality_correlation/main_correlation_glomo.py`

## Minimal Usage

Typical workflows in this repository follow the pattern below:

1. prepare datasets under `datasets/`
2. optionally pretrain correlation modules in `Correlation_Pretraining/`
3. run sentiment experiments from `Correlation_Guided_MSA/`
4. generate paper figures with `figure_scripts/`

Examples:

```bash
cd Correlation_Guided_MSA
python main_cgmsa.py ...
```

```bash
cd Correlation_Pretraining/modality_correlation
python main_correlation_glomo.py ...
```

```bash
cd figure_scripts
bash generate_all_figures.sh
```

## Experiment Records

The `experiments/` directory is kept as an archival record for submission and reproducibility:

- one result corresponds to one experiment record
- one table corresponds to one reproducible experiment group
- one figure corresponds to one generation script

Historical experiment directory names and recorded command files are preserved as-is and are not renamed retroactively.

Each archived experiment may contain:

- `command.sh`
- `train.log`
- `metrics.json`
- selected analysis outputs used by figure-generation scripts

These records are preserved as experiment evidence rather than rewritten to follow later repository renaming.

## Reproducibility Notes

- Each reported result is intended to correspond to a concrete archived experiment record.
- Each paper figure is intended to correspond to a dedicated generation script in `figure_scripts/`.
- Some archived metadata files contain local absolute paths from the original training environment. These paths are preserved for traceability and should be adapted when reproducing runs in a different environment.
- Historical experiment commands may reference earlier script names from before the repository renaming. They are preserved intentionally as part of the original experiment record.

## Data and Checkpoints

- Datasets are expected to be prepared locally under `datasets/`.
- Large checkpoints and intermediate artifacts are not all tracked in the repository.
- Only selected analysis files required for paper figures are included in version control.

## Availability

The manuscript PDF is managed separately from the tracked code submission. The repository is structured to support code release, experiment traceability, and figure reproducibility.

## License

See `LICENSE` for the repository license and review third-party component licenses before redistributing derived assets or checkpoints.

## Notes

- The current manuscript PDF is managed separately and is not part of the tracked code submission by default.
- Historical experiment metadata may still contain older absolute paths or script names. These records are intentionally preserved for traceability.
