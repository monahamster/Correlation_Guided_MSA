# Correlation_Guided_MSA

This repository contains the codebase for the current project built around correlation-guided multimodal sentiment analysis and related extensions.

## Repository Structure

- `Correlation_Guided_MSA/`
  - Main codebase for CMU-MOSI and CMU-MOSEI experiments.
- `Correlation_Guided_Humor/`
  - Auxiliary codebase for humor/sarcasm task experiments.
- `Correlation_Guided_MER/`
  - Auxiliary codebase for multimodal emotion recognition experiments.
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

### Humor / Sarcasm

- `Correlation_Guided_Humor/main_cg_humor.py`

### Emotion Recognition

- `Correlation_Guided_MER/main_cg_mer.py`

### Correlation Pretraining

- `Correlation_Pretraining/modality_correlation/main_correlation_glomo.py`

## Experiment Records

The `experiments/` directory is kept as an archival record for submission and reproducibility:

- one result corresponds to one experiment record
- one table corresponds to one reproducible experiment group
- one figure corresponds to one generation script

Historical experiment directory names and recorded command files are preserved as-is and are not renamed retroactively.

## Notes

- The current manuscript PDF is managed separately and is not part of the tracked code submission by default.
- Historical experiment metadata may still contain older absolute paths or script names. These records are intentionally preserved for traceability.
