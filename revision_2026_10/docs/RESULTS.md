# Results

Main results use five downstream seeds. Values are mean ± sample standard deviation; accuracies and F1 are fractions.

| Dataset | Variant | MAE | Corr | Acc-2 | F1 | Acc-2 nonzero | F1 nonzero | Acc-7 |
|---|---|---|---|---|---|---|---|---|
| MOSI | baseline | 0.7572 ± 0.0179 | 0.7626 ± 0.0138 | 0.8210 ± 0.0084 | 0.8203 ± 0.0081 | 0.8394 ± 0.0078 | 0.8393 ± 0.0074 | 0.4488 ± 0.0070 |
| MOSI | constant | 0.7528 ± 0.0119 | 0.7650 ± 0.0086 | 0.8187 ± 0.0133 | 0.8181 ± 0.0130 | 0.8382 ± 0.0128 | 0.8382 ± 0.0124 | 0.4543 ± 0.0105 |
| MOSI | dynamic | 0.7547 ± 0.0114 | 0.7656 ± 0.0089 | 0.8216 ± 0.0166 | 0.8208 ± 0.0163 | 0.8391 ± 0.0176 | 0.8388 ± 0.0173 | 0.4450 ± 0.0095 |
| MOSEI | baseline | 0.5418 ± 0.0037 | 0.7556 ± 0.0051 | 0.8276 ± 0.0158 | 0.8296 ± 0.0128 | 0.8495 ± 0.0037 | 0.8476 ± 0.0041 | 0.5353 ± 0.0057 |
| MOSEI | constant | 0.5426 ± 0.0044 | 0.7546 ± 0.0066 | 0.8316 ± 0.0069 | 0.8336 ± 0.0050 | 0.8529 ± 0.0034 | 0.8512 ± 0.0048 | 0.5350 ± 0.0033 |
| MOSEI | dynamic | 0.5409 ± 0.0031 | 0.7558 ± 0.0070 | 0.8316 ± 0.0118 | 0.8337 ± 0.0095 | 0.8548 ± 0.0048 | 0.8531 ± 0.0060 | 0.5353 ± 0.0034 |

Dynamic has slightly lower mean MAE than Baseline on both datasets. MOSI secondary metrics are mixed; MOSEI binary metrics improve. None of the four paired MAE comparisons reaches significance after Holm adjustment. The release therefore reports the full metric set and seed variability.

The complete module and modality ablations are in `results/clean/test_comparison.json`. They use three seeds and must not be compared as if their means used the five-seed main sample. Additional analysis summaries and exact scripts are under `analysis/`.

Run `tools/reproduce_results.py` to regenerate the complete 224-row summary CSV, confidence intervals, paired tests and plots from the numeric records.
