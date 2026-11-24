Synthetic-data validation; these are not real-data reproduction results.

Every metric shows finite-value folds n/expected. Missing, NaN and infinite
values are excluded. SD uses ddof=0 with at least two finite folds;
NA means that fold-to-fold spread cannot be reported.

| Config | Folds | mAUC | F1 (weighted) | Acc | BCA | Stable Acc | Stable BCA | Conv. Acc | Conv. BCA | Conv. Sens | Conv. AUROC (T4) | Sens@Spec90 (T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw/visit_transformer | 2/2 | 0.437 ± 0.068 (n=2/2) | 0.409 ± 0.099 (n=2/2) | 0.454 ± 0.101 (n=2/2) | 0.577 ± 0.014 (n=2/2) | 0.583 ± 0.083 (n=2/2) | 0.659 ± 0.034 (n=2/2) | 0.167 ± 0.167 (n=2/2) | 0.369 ± 0.203 (n=2/2) | 0.200 ± 0.200 (n=2/2) | 0.325 ± 0.025 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| raw/lstm | 2/2 | 0.593 ± 0.037 (n=2/2) | 0.118 ± 0.070 (n=2/2) | 0.230 ± 0.064 (n=2/2) | 0.527 ± 0.027 (n=2/2) | 0.333 ± 0.083 (n=2/2) | 0.531 ± 0.031 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.250 ± 0.083 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.396 ± 0.071 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| balanced/visit_transformer | 2/2 | 0.599 ± 0.112 (n=2/2) | 0.548 ± 0.008 (n=2/2) | 0.556 ± 0.000 (n=2/2) | 0.642 ± 0.013 (n=2/2) | 0.775 ± 0.025 (n=2/2) | 0.764 ± 0.014 (n=2/2) | 0.325 ± 0.075 (n=2/2) | 0.421 ± 0.171 (n=2/2) | 0.250 ± 0.167 (n=2/2) | 0.492 ± 0.092 (n=2/2) | 0.100 ± 0.100 (n=2/2) |
| raw/minrnn_monthly | 2/2 | 0.654 ± 0.010 (n=2/2) | 0.272 ± 0.087 (n=2/2) | 0.260 ± 0.093 (n=2/2) | 0.473 ± 0.067 (n=2/2) | 0.333 ± 0.083 (n=2/2) | 0.514 ± 0.048 (n=2/2) | 0.100 ± 0.100 (n=2/2) | 0.275 ± 0.042 (n=2/2) | 0.033 ± 0.033 (n=2/2) | 0.529 ± 0.154 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| raw/tst | 2/2 | 0.724 ± 0.057 (n=2/2) | 0.412 ± 0.044 (n=2/2) | 0.430 ± 0.041 (n=2/2) | 0.634 ± 0.007 (n=2/2) | 0.500 ± 0.083 (n=2/2) | 0.667 ± 0.021 (n=2/2) | 0.267 ± 0.067 (n=2/2) | 0.342 ± 0.108 (n=2/2) | 0.100 ± 0.033 (n=2/2) | 0.450 ± 0.250 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| raw/copy_last | 2/2 | 0.826 ± 0.035 (n=2/2) | 0.666 ± 0.040 (n=2/2) | 0.686 ± 0.020 (n=2/2) | 0.804 ± 0.039 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.236 ± 0.069 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| balanced/copy_last | 2/2 | 0.775 ± 0.058 (n=2/2) | 0.481 ± 0.074 (n=2/2) | 0.500 ± 0.056 (n=2/2) | 0.736 ± 0.057 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.233 ± 0.067 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) |

Incomplete metric coverage (including metrics available only in CSV):

Undefined completed folds include absent metrics and non-finite values.

| Config | Metric | Valid/Expected | Undefined completed folds | Missing run folds |
| --- | --- | --- | --- | --- |
| raw/visit_transformer | converter.mauc | 1/2 | [0] | [] |
| raw/lstm | converter.mauc | 1/2 | [0] | [] |
| balanced/visit_transformer | converter.mauc | 1/2 | [0] | [] |
| raw/minrnn_monthly | converter.mauc | 1/2 | [0] | [] |
| raw/tst | converter.mauc | 1/2 | [0] | [] |
| raw/copy_last | converter.mauc | 1/2 | [0] | [] |
| balanced/copy_last | converter.mauc | 1/2 | [0] | [] |

Welch's t-test across folds:

| Data | Comparison | Metric | t | p | Ref. valid folds | Other valid folds |
| --- | --- | --- | --- | --- | --- | --- |
| raw | visit_transformer vs lstm | mAUC | -2.00 | 0.2203 | 2/2 | 2/2 |
| raw | visit_transformer vs lstm | BCA | 1.62 | 0.2860 | 2/2 | 2/2 |
| raw | visit_transformer vs lstm | Conv. Acc | 1.00 | 0.5000 | 2/2 | 2/2 |
| raw | visit_transformer vs lstm | Conv. BCA | 0.54 | 0.6631 | 2/2 | 2/2 |

Fold coverage (partial summaries are not full CV results):

- raw/visit_transformer: 2/2 folds; missing []
- raw/lstm: 2/2 folds; missing []
- balanced/visit_transformer: 2/2 folds; missing []
- raw/minrnn_monthly: 2/2 folds; missing []
- raw/tst: 2/2 folds; missing []
- raw/copy_last: 2/2 folds; missing []
- balanced/copy_last: 2/2 folds; missing []
