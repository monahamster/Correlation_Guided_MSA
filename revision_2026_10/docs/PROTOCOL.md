# Frozen protocol

## Data and inputs

GLoMo input features are used consistently by the backbone and correlation pretraining branch. Text/audio/visual dimensions are 768/74/47 for MOSI and 768/74/35 for MOSEI. Dataset content hashes, splits and sample IDs are preserved in the input manifests and run records. The supplied data are precomputed features; this release does not include original video/audio.

## Pretraining and guidance

The correlation encoder is pretrained separately and frozen during downstream training. Both datasets use aligned-word cosine readout. MOSI uses the joint word/sentence triplet objective with beta 0.5 and no added positional encoding; MOSEI uses the sentence triplet objective (beta 0) with the selected positional encoding. The selected checkpoints are epochs 8 and 15 respectively (one-based), selected using validation sentence-triplet loss. Both recipes use seed 5576, 20 epochs, batch size 32, Adam learning rate 1e-4, triplet margin 0.2, and five validation donor maps. The original code and history define all remaining details.

Baseline disables the correlation guidance paths while retaining the GLoMo fusion/MoE architecture and prediction losses. Dynamic applies sample-dependent guidance to both routing and fusion. Constant uses the mean dynamic scales over the complete training split as fixed buffers. Module ablations enable fusion only, MoE only or both. Modality ablations are indexed separately and retain their original masking protocol.

## Downstream training

| Setting | MOSI | MOSEI |
|---|---:|---:|
| Epoch budget | 70 | 100 |
| Batch size | 240 | 64 |
| Maximum sequence length | 60 | 80 |
| Hidden dimension | 48 | 192 |
| Transformer layers | 4 | 3 |
| Learning rate | 4e-5 | 1e-5 |
| MoE guidance coefficient | 0.05 | 0.1 |
| Fusion guidance coefficient | 0.2 | 0.2 |

Main comparisons use downstream seeds 5576, 42, 2026, 2027 and 3407. Module and modality ablations use 5576, 42 and 2026. Means and sample standard deviations use their stated seed groups. The frozen pretrained branch is shared across downstream seeds.

One optimizer instance is maintained through each training run. Gradient accumulation is 1. The checkpoint is the earliest minimum sample-weighted validation MAE over the full epoch budget, with no early stopping and no test-based selection. Histories use zero-based epoch indices. Regression, auxiliary seven-class supervision and MoE balancing losses are preserved from the archived implementation.

## Evaluation

All reported sentiment metrics use the scalar regression output, including seven-level accuracy. Acc-2 uses nonnegative versus negative labels/predictions; Acc-2 nonzero excludes neutral ground-truth labels and uses positive versus nonpositive. F1 is weighted F1. Acc-7 clips scores to [-3,3] and rounds to integer levels using NumPy's ties-to-even rule. MAE is the primary metric. Four paired five-seed MAE tests compare Dynamic with Baseline/Constant on the two datasets, using two-sided t tests and Holm adjustment. Report secondary metrics alongside MAE.

Random seeds, data ordering and initialization are controlled. The archived environment uses `warn_only=True` for deterministic algorithms because the CUDA adaptive-max-pool backward kernel is nondeterministic in PyTorch 2.0.1. A fresh training run may therefore differ numerically even with matching inputs; recomputation from the released predictions is independently checked.
