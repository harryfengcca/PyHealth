Synthetic-data validation; these are not real-data reproduction results.

Every metric shows finite-value folds n/expected. Missing, NaN and infinite
values are excluded. SD uses ddof=0 with at least two finite folds;
NA means that fold-to-fold spread cannot be reported.

| Config | Folds | mAUC | F1 (weighted) | Acc | BCA | Stable Acc | Stable BCA | Conv. Acc | Conv. BCA | Conv. Sens | Conv. AUROC (T4) | Sens@Spec90 (T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw/paper_transformer | 2/2 | 0.658 ± 0.012 (n=2/2) | 0.255 ± 0.055 (n=2/2) | 0.369 ± 0.075 (n=2/2) | 0.551 ± 0.020 (n=2/2) | 0.333 ± 0.083 (n=2/2) | 0.567 ± 0.005 (n=2/2) | 0.417 ± 0.417 (n=2/2) | 0.333 ± 0.167 (n=2/2) | 0.167 ± 0.167 (n=2/2) | 0.258 ± 0.108 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| raw/tst | 2/2 | 0.724 ± 0.057 (n=2/2) | 0.412 ± 0.044 (n=2/2) | 0.430 ± 0.041 (n=2/2) | 0.634 ± 0.007 (n=2/2) | 0.500 ± 0.083 (n=2/2) | 0.667 ± 0.021 (n=2/2) | 0.267 ± 0.067 (n=2/2) | 0.342 ± 0.108 (n=2/2) | 0.100 ± 0.033 (n=2/2) | 0.450 ± 0.250 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| raw/minrnn_monthly | 2/2 | 0.654 ± 0.010 (n=2/2) | 0.272 ± 0.087 (n=2/2) | 0.260 ± 0.093 (n=2/2) | 0.473 ± 0.067 (n=2/2) | 0.333 ± 0.083 (n=2/2) | 0.514 ± 0.048 (n=2/2) | 0.100 ± 0.100 (n=2/2) | 0.275 ± 0.042 (n=2/2) | 0.033 ± 0.033 (n=2/2) | 0.529 ± 0.154 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| raw/lstm_monthly | 2/2 | 0.741 ± 0.024 (n=2/2) | 0.194 ± 0.057 (n=2/2) | 0.288 ± 0.065 (n=2/2) | 0.565 ± 0.044 (n=2/2) | 0.417 ± 0.083 (n=2/2) | 0.576 ± 0.049 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.247 ± 0.081 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.442 ± 0.042 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| raw/gru_monthly | 2/2 | 0.775 ± 0.016 (n=2/2) | 0.302 ± 0.103 (n=2/2) | 0.345 ± 0.067 (n=2/2) | 0.582 ± 0.035 (n=2/2) | 0.500 ± 0.083 (n=2/2) | 0.621 ± 0.066 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.247 ± 0.081 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.617 ± 0.117 (n=2/2) | 0.083 ± 0.083 (n=2/2) |
| balanced/paper_transformer | 2/2 | 0.575 ± 0.047 (n=2/2) | 0.465 ± 0.068 (n=2/2) | 0.611 ± 0.056 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.325 ± 0.075 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.900 ± 0.100 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.333 ± 0.000 (n=2/2) | 0.308 ± 0.108 (n=2/2) | 0.100 ± 0.100 (n=2/2) |
| balanced/tst | 2/2 | 0.767 ± 0.044 (n=2/2) | 0.025 ± 0.003 (n=2/2) | 0.111 ± 0.000 (n=2/2) | 0.493 ± 0.007 (n=2/2) | 0.225 ± 0.025 (n=2/2) | 0.486 ± 0.014 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.246 ± 0.079 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.617 ± 0.117 (n=2/2) | 0.350 ± 0.150 (n=2/2) |
| balanced/minrnn_monthly | 2/2 | 0.497 ± 0.081 (n=2/2) | 0.356 ± 0.088 (n=2/2) | 0.278 ± 0.056 (n=2/2) | 0.434 ± 0.019 (n=2/2) | 0.450 ± 0.050 (n=2/2) | 0.569 ± 0.042 (n=2/2) | 0.125 ± 0.125 (n=2/2) | 0.279 ± 0.029 (n=2/2) | 0.042 ± 0.042 (n=2/2) | 0.500 ± 0.167 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| balanced/lstm_monthly | 2/2 | 0.676 ± 0.088 (n=2/2) | 0.082 ± 0.050 (n=2/2) | 0.167 ± 0.056 (n=2/2) | 0.523 ± 0.032 (n=2/2) | 0.350 ± 0.150 (n=2/2) | 0.549 ± 0.063 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.242 ± 0.075 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.533 ± 0.133 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| balanced/gru_monthly | 2/2 | 0.792 ± 0.181 (n=2/2) | 0.104 ± 0.067 (n=2/2) | 0.222 ± 0.111 (n=2/2) | 0.586 ± 0.114 (n=2/2) | 0.425 ± 0.175 (n=2/2) | 0.597 ± 0.125 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.237 ± 0.071 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.667 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| raw/copy_last | 2/2 | 0.826 ± 0.035 (n=2/2) | 0.666 ± 0.040 (n=2/2) | 0.686 ± 0.020 (n=2/2) | 0.804 ± 0.039 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.236 ± 0.069 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) |
| balanced/copy_last | 2/2 | 0.775 ± 0.058 (n=2/2) | 0.481 ± 0.074 (n=2/2) | 0.500 ± 0.056 (n=2/2) | 0.736 ± 0.057 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 1.000 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.233 ± 0.067 (n=2/2) | 0.000 ± 0.000 (n=2/2) | 0.500 ± 0.000 (n=2/2) | 0.000 ± 0.000 (n=2/2) |

Incomplete metric coverage (including metrics available only in CSV):

Undefined completed folds include absent metrics and non-finite values.

| Config | Metric | Valid/Expected | Undefined completed folds | Missing run folds |
| --- | --- | --- | --- | --- |
| raw/paper_transformer | converter.mauc | 1/2 | [0] | [] |
| raw/tst | converter.mauc | 1/2 | [0] | [] |
| raw/minrnn_monthly | converter.mauc | 1/2 | [0] | [] |
| raw/lstm_monthly | converter.mauc | 1/2 | [0] | [] |
| raw/gru_monthly | converter.mauc | 1/2 | [0] | [] |
| balanced/paper_transformer | converter.mauc | 1/2 | [0] | [] |
| balanced/tst | converter.mauc | 1/2 | [0] | [] |
| balanced/minrnn_monthly | converter.mauc | 1/2 | [0] | [] |
| balanced/lstm_monthly | converter.mauc | 1/2 | [0] | [] |
| balanced/gru_monthly | converter.mauc | 1/2 | [0] | [] |
| raw/copy_last | converter.mauc | 1/2 | [0] | [] |
| balanced/copy_last | converter.mauc | 1/2 | [0] | [] |

Welch's t-test across folds:

| Data | Comparison | Metric | t | p | Ref. valid folds | Other valid folds |
| --- | --- | --- | --- | --- | --- | --- |
| raw | paper_transformer vs tst | mAUC | -1.13 | 0.4486 | 2/2 | 2/2 |
| raw | paper_transformer vs tst | BCA | -3.98 | 0.1201 | 2/2 | 2/2 |
| raw | paper_transformer vs tst | Conv. Acc | 0.36 | 0.7803 | 2/2 | 2/2 |
| raw | paper_transformer vs tst | Conv. BCA | -0.04 | 0.9709 | 2/2 | 2/2 |
| raw | paper_transformer vs minrnn_monthly | mAUC | 0.27 | 0.8102 | 2/2 | 2/2 |
| raw | paper_transformer vs minrnn_monthly | BCA | 1.12 | 0.4406 | 2/2 | 2/2 |
| raw | paper_transformer vs minrnn_monthly | Conv. Acc | 0.74 | 0.5842 | 2/2 | 2/2 |
| raw | paper_transformer vs minrnn_monthly | Conv. BCA | 0.34 | 0.7866 | 2/2 | 2/2 |
| raw | paper_transformer vs lstm_monthly | mAUC | -3.12 | 0.1296 | 2/2 | 2/2 |
| raw | paper_transformer vs lstm_monthly | BCA | -0.29 | 0.8108 | 2/2 | 2/2 |
| raw | paper_transformer vs lstm_monthly | Conv. Acc | 1.00 | 0.5000 | 2/2 | 2/2 |
| raw | paper_transformer vs lstm_monthly | Conv. BCA | 0.47 | 0.7023 | 2/2 | 2/2 |
| raw | paper_transformer vs gru_monthly | mAUC | -5.85 | 0.0339 | 2/2 | 2/2 |
| raw | paper_transformer vs gru_monthly | BCA | -0.78 | 0.5368 | 2/2 | 2/2 |
| raw | paper_transformer vs gru_monthly | Conv. Acc | 1.00 | 0.5000 | 2/2 | 2/2 |
| raw | paper_transformer vs gru_monthly | Conv. BCA | 0.47 | 0.7023 | 2/2 | 2/2 |
| balanced | paper_transformer vs tst | mAUC | -2.96 | 0.0984 | 2/2 | 2/2 |
| balanced | paper_transformer vs tst | BCA | 1.00 | 0.5000 | 2/2 | 2/2 |
| balanced | paper_transformer vs tst | Conv. Acc | 9.00 | 0.0704 | 2/2 | 2/2 |
| balanced | paper_transformer vs tst | Conv. BCA | 3.21 | 0.1922 | 2/2 | 2/2 |
| balanced | paper_transformer vs minrnn_monthly | mAUC | 0.83 | 0.5099 | 2/2 | 2/2 |
| balanced | paper_transformer vs minrnn_monthly | BCA | 3.57 | 0.1740 | 2/2 | 2/2 |
| balanced | paper_transformer vs minrnn_monthly | Conv. Acc | 4.84 | 0.0440 | 2/2 | 2/2 |
| balanced | paper_transformer vs minrnn_monthly | Conv. BCA | 7.57 | 0.0836 | 2/2 | 2/2 |
| balanced | paper_transformer vs lstm_monthly | mAUC | -1.02 | 0.4415 | 2/2 | 2/2 |
| balanced | paper_transformer vs lstm_monthly | BCA | -0.72 | 0.6014 | 2/2 | 2/2 |
| balanced | paper_transformer vs lstm_monthly | Conv. Acc | 9.00 | 0.0704 | 2/2 | 2/2 |
| balanced | paper_transformer vs lstm_monthly | Conv. BCA | 3.44 | 0.1799 | 2/2 | 2/2 |
| balanced | paper_transformer vs gru_monthly | mAUC | -1.16 | 0.4342 | 2/2 | 2/2 |
| balanced | paper_transformer vs gru_monthly | BCA | -0.76 | 0.5881 | 2/2 | 2/2 |
| balanced | paper_transformer vs gru_monthly | Conv. Acc | 9.00 | 0.0704 | 2/2 | 2/2 |
| balanced | paper_transformer vs gru_monthly | Conv. BCA | 3.71 | 0.1678 | 2/2 | 2/2 |

Fold coverage (partial summaries are not full CV results):

- raw/paper_transformer: 2/2 folds; missing []
- raw/tst: 2/2 folds; missing []
- raw/minrnn_monthly: 2/2 folds; missing []
- raw/lstm_monthly: 2/2 folds; missing []
- raw/gru_monthly: 2/2 folds; missing []
- balanced/paper_transformer: 2/2 folds; missing []
- balanced/tst: 2/2 folds; missing []
- balanced/minrnn_monthly: 2/2 folds; missing []
- balanced/lstm_monthly: 2/2 folds; missing []
- balanced/gru_monthly: 2/2 folds; missing []
- raw/copy_last: 2/2 folds; missing []
- balanced/copy_last: 2/2 folds; missing []
