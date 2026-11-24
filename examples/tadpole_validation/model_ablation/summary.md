Synthetic-data validation; these are not real-data reproduction results.

Every metric shows finite-value folds n/expected. Missing, NaN and infinite
values are excluded. SD uses ddof=0 with at least two finite folds;
NA means that fold-to-fold spread cannot be reported.

| Config | Folds | mAUC | F1 (weighted) | Acc | BCA | Stable Acc | Stable BCA | Conv. Acc | Conv. BCA | Conv. Sens | Conv. AUROC (T4) | Sens@Spec90 (T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 default (encoder, d128, ce) | 2/2 | 0.437 ± 0.068 (n=2/2) | 0.409 ± 0.099 (n=2/2) | 0.454 ± 0.101 (n=2/2) | 0.577 ± 0.014 (n=2/2) | 0.583 ± 0.083 (n=2/2) | 0.659 ± 0.034 (n=2/2) | 0.167 ± 0.167 (n=2/2) | 0.369 ± 0.203 (n=2/2) | 0.200 ± 0.200 (n=2/2) | 0.325 ± 0.025 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| M2a weighted cross-entropy | 2/2 | 0.447 ± 0.046 (n=2/2) | 0.392 ± 0.074 (n=2/2) | 0.426 ± 0.074 (n=2/2) | 0.562 ± 0.001 (n=2/2) | 0.542 ± 0.042 (n=2/2) | 0.608 ± 0.017 (n=2/2) | 0.167 ± 0.167 (n=2/2) | 0.372 ± 0.206 (n=2/2) | 0.200 ± 0.200 (n=2/2) | 0.300 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| M2b focal loss | 2/2 | 0.434 ± 0.071 (n=2/2) | 0.386 ± 0.075 (n=2/2) | 0.426 ± 0.074 (n=2/2) | 0.559 ± 0.004 (n=2/2) | 0.542 ± 0.042 (n=2/2) | 0.639 ± 0.014 (n=2/2) | 0.167 ± 0.167 (n=2/2) | 0.369 ± 0.203 (n=2/2) | 0.200 ± 0.200 (n=2/2) | 0.317 ± 0.033 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| M3a paper encoder-decoder (d256, h4, 4+8 layers) | 2/2 | 0.693 ± 0.028 (n=2/2) | 0.255 ± 0.055 (n=2/2) | 0.369 ± 0.075 (n=2/2) | 0.551 ± 0.020 (n=2/2) | 0.333 ± 0.083 (n=2/2) | 0.567 ± 0.005 (n=2/2) | 0.417 ± 0.417 (n=2/2) | 0.333 ± 0.167 (n=2/2) | 0.167 ± 0.167 (n=2/2) | 0.279 ± 0.154 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| M3b embedding 64 | 2/2 | 0.596 ± 0.122 (n=2/2) | 0.292 ± 0.074 (n=2/2) | 0.459 ± 0.070 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.250 ± 0.083 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.917 ± 0.083 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.333 ± 0.000 (n=2/2) | 0.392 ± 0.192 (n=2/2) | 0.167 ± 0.167 (n=2/2) |
| M3c embedding 256 | 2/2 | 0.558 ± 0.044 (n=2/2) | 0.345 ± 0.039 (n=2/2) | 0.371 ± 0.018 (n=2/2) | 0.544 ± 0.020 (n=2/2) | 0.292 ± 0.125 (n=2/2) | 0.486 ± 0.077 (n=2/2) | 0.517 ± 0.317 (n=2/2) | 0.367 ± 0.133 (n=2/2) | 0.200 ± 0.133 (n=2/2) | 0.292 ± 0.142 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| M3d 2 layers | 2/2 | 0.542 ± 0.043 (n=2/2) | 0.204 ± 0.067 (n=2/2) | 0.258 ± 0.036 (n=2/2) | 0.482 ± 0.029 (n=2/2) | 0.208 ± 0.042 (n=2/2) | 0.425 ± 0.050 (n=2/2) | 0.383 ± 0.217 (n=2/2) | 0.381 ± 0.014 (n=2/2) | 0.133 ± 0.067 (n=2/2) | 0.404 ± 0.029 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| M3e dropout 0.1 | 2/2 | 0.461 ± 0.075 (n=2/2) | 0.440 ± 0.057 (n=2/2) | 0.484 ± 0.072 (n=2/2) | 0.596 ± 0.002 (n=2/2) | 0.500 ± 0.083 (n=2/2) | 0.629 ± 0.058 (n=2/2) | 0.417 ± 0.417 (n=2/2) | 0.467 ± 0.300 (n=2/2) | 0.300 ± 0.300 (n=2/2) | 0.333 ± 0.017 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| M3f learning rate 1e-3 | 2/2 | 0.797 ± 0.041 (n=2/2) | 0.427 ± 0.069 (n=2/2) | 0.515 ± 0.015 (n=2/2) | 0.651 ± 0.006 (n=2/2) | 0.500 ± 0.167 (n=2/2) | 0.697 ± 0.053 (n=2/2) | 0.517 ± 0.317 (n=2/2) | 0.436 ± 0.203 (n=2/2) | 0.200 ± 0.133 (n=2/2) | 0.683 ± 0.183 (n=2/2) | 0.417 ± 0.417 (n=2/2) |

Incomplete metric coverage (including metrics available only in CSV):

Undefined completed folds include absent metrics and non-finite values.

| Config | Metric | Valid/Expected | Undefined completed folds | Missing run folds |
| --- | --- | --- | --- | --- |
| M0 default (encoder, d128, ce) | converter.mauc | 1/2 | [0] | [] |
| M2a weighted cross-entropy | converter.mauc | 1/2 | [0] | [] |
| M2b focal loss | converter.mauc | 1/2 | [0] | [] |
| M3a paper encoder-decoder (d256, h4, 4+8 layers) | converter.mauc | 1/2 | [0] | [] |
| M3b embedding 64 | converter.mauc | 1/2 | [0] | [] |
| M3c embedding 256 | converter.mauc | 1/2 | [0] | [] |
| M3d 2 layers | converter.mauc | 1/2 | [0] | [] |
| M3e dropout 0.1 | converter.mauc | 1/2 | [0] | [] |
| M3f learning rate 1e-3 | converter.mauc | 1/2 | [0] | [] |

Fold coverage (partial summaries are not full CV results):

- M0 default (encoder, d128, ce): 2/2 folds; missing []
- M2a weighted cross-entropy: 2/2 folds; missing []
- M2b focal loss: 2/2 folds; missing []
- M3a paper encoder-decoder (d256, h4, 4+8 layers): 2/2 folds; missing []
- M3b embedding 64: 2/2 folds; missing []
- M3c embedding 256: 2/2 folds; missing []
- M3d 2 layers: 2/2 folds; missing []
- M3e dropout 0.1: 2/2 folds; missing []
- M3f learning rate 1e-3: 2/2 folds; missing []
