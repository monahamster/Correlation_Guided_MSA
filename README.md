# Correlation-Guided Multimodal Sentiment Analysis

## October 2026 revision

The code, frozen protocols and experimental results used in the October revision are organized in **[revision_2026_10](revision_2026_10/README.md)**.

Start there to recompute the reported results or prepare a training run. The release contains 78 evaluated runs, their exact source snapshots, configurations, epoch histories and test predictions. Main comparisons use five downstream seeds; module and modality ablations use three.

- [Reproduction instructions](revision_2026_10/README.md)
- [Protocol](revision_2026_10/docs/PROTOCOL.md)
- [Results](revision_2026_10/docs/RESULTS.md)
- [Experiment index](revision_2026_10/docs/EXPERIMENT_INDEX.csv)
- [Required external assets](revision_2026_10/docs/ARTIFACTS.md)

The top-level `Correlation_Guided_MSA/`, `Correlation_Pretraining/`, `experiments/` and `figure_scripts/` directories preserve the earlier implementation and records. Their earlier commands and results are historical; the October instructions above define the current experiments. The previous README is retained in [the archive](revision_2026_10/docs/README_before_october.md).

## Code origin and license

The method uses the GLoMo backbone and a CorMulT-based correlation pretraining branch. Original source notices and the repository [LICENSE](LICENSE) are retained. Dataset, pretrained encoder and third-party component licenses continue to apply to those assets.
