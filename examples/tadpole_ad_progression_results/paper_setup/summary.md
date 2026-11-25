Every metric shows finite-value folds n/expected. Missing, NaN and infinite
values are excluded. SD uses ddof=0 with at least two finite folds;
NA means that fold-to-fold spread cannot be reported.

| Config | Folds | mAUC | F1 (weighted) | Acc | BCA | Stable Acc | Stable BCA | Conv. Acc | Conv. BCA | Conv. Sens | Conv. AUROC (T4) | Sens@Spec90 (T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw/paper_transformer | 10/10 | 0.956 ± 0.013 (n=10/10) | 0.830 ± 0.037 (n=10/10) | 0.834 ± 0.035 (n=10/10) | 0.880 ± 0.023 (n=10/10) | 0.910 ± 0.030 (n=10/10) | 0.941 ± 0.019 (n=10/10) | 0.604 ± 0.121 (n=10/10) | 0.564 ± 0.057 (n=10/10) | 0.279 ± 0.069 (n=10/10) | 0.861 ± 0.043 (n=10/10) | 0.566 ± 0.127 (n=10/10) |
| raw/tst | 10/10 | 0.954 ± 0.014 (n=10/10) | 0.839 ± 0.036 (n=10/10) | 0.844 ± 0.034 (n=10/10) | 0.888 ± 0.022 (n=10/10) | 0.922 ± 0.025 (n=10/10) | 0.949 ± 0.014 (n=10/10) | 0.607 ± 0.108 (n=10/10) | 0.555 ± 0.038 (n=10/10) | 0.256 ± 0.035 (n=10/10) | 0.856 ± 0.047 (n=10/10) | 0.587 ± 0.118 (n=10/10) |
| raw/minrnn_monthly | 10/10 | 0.939 ± 0.010 (n=10/10) | 0.763 ± 0.015 (n=10/10) | 0.767 ± 0.012 (n=10/10) | 0.841 ± 0.015 (n=10/10) | 0.991 ± 0.008 (n=10/10) | 0.994 ± 0.005 (n=10/10) | 0.091 ± 0.049 (n=10/10) | 0.336 ± 0.028 (n=10/10) | 0.037 ± 0.018 (n=10/10) | 0.822 ± 0.033 (n=10/10) | 0.589 ± 0.048 (n=10/10) |
| raw/lstm_monthly | 10/10 | 0.910 ± 0.019 (n=10/10) | 0.784 ± 0.026 (n=10/10) | 0.787 ± 0.023 (n=10/10) | 0.853 ± 0.020 (n=10/10) | 0.975 ± 0.017 (n=10/10) | 0.983 ± 0.011 (n=10/10) | 0.219 ± 0.079 (n=10/10) | 0.387 ± 0.036 (n=10/10) | 0.091 ± 0.027 (n=10/10) | 0.757 ± 0.062 (n=10/10) | 0.472 ± 0.098 (n=10/10) |
| raw/gru_monthly | 10/10 | 0.904 ± 0.018 (n=10/10) | 0.763 ± 0.028 (n=10/10) | 0.768 ± 0.025 (n=10/10) | 0.841 ± 0.020 (n=10/10) | 0.986 ± 0.010 (n=10/10) | 0.991 ± 0.006 (n=10/10) | 0.109 ± 0.091 (n=10/10) | 0.342 ± 0.037 (n=10/10) | 0.045 ± 0.037 (n=10/10) | 0.742 ± 0.037 (n=10/10) | 0.396 ± 0.091 (n=10/10) |
| balanced/paper_transformer | 10/10 | 0.911 ± 0.022 (n=10/10) | 0.792 ± 0.049 (n=10/10) | 0.800 ± 0.048 (n=10/10) | 0.833 ± 0.039 (n=10/10) | 0.817 ± 0.065 (n=10/10) | 0.873 ± 0.048 (n=10/10) | 0.783 ± 0.072 (n=10/10) | 0.665 ± 0.040 (n=10/10) | 0.410 ± 0.058 (n=10/10) | 0.844 ± 0.045 (n=10/10) | 0.597 ± 0.150 (n=10/10) |
| balanced/tst | 10/10 | 0.914 ± 0.026 (n=10/10) | 0.792 ± 0.058 (n=10/10) | 0.800 ± 0.058 (n=10/10) | 0.837 ± 0.038 (n=10/10) | 0.827 ± 0.049 (n=10/10) | 0.881 ± 0.035 (n=10/10) | 0.772 ± 0.096 (n=10/10) | 0.663 ± 0.048 (n=10/10) | 0.407 ± 0.064 (n=10/10) | 0.853 ± 0.053 (n=10/10) | 0.585 ± 0.177 (n=10/10) |
| balanced/minrnn_monthly | 10/10 | 0.884 ± 0.018 (n=10/10) | 0.594 ± 0.058 (n=10/10) | 0.592 ± 0.047 (n=10/10) | 0.750 ± 0.029 (n=10/10) | 0.958 ± 0.031 (n=10/10) | 0.968 ± 0.025 (n=10/10) | 0.225 ± 0.089 (n=10/10) | 0.394 ± 0.038 (n=10/10) | 0.096 ± 0.037 (n=10/10) | 0.840 ± 0.036 (n=10/10) | 0.571 ± 0.120 (n=10/10) |
| balanced/lstm_monthly | 10/10 | 0.840 ± 0.034 (n=10/10) | 0.660 ± 0.048 (n=10/10) | 0.655 ± 0.044 (n=10/10) | 0.779 ± 0.032 (n=10/10) | 0.922 ± 0.067 (n=10/10) | 0.948 ± 0.041 (n=10/10) | 0.387 ± 0.081 (n=10/10) | 0.463 ± 0.038 (n=10/10) | 0.168 ± 0.040 (n=10/10) | 0.768 ± 0.068 (n=10/10) | 0.454 ± 0.179 (n=10/10) |
| balanced/gru_monthly | 10/10 | 0.844 ± 0.024 (n=10/10) | 0.606 ± 0.051 (n=10/10) | 0.603 ± 0.042 (n=10/10) | 0.756 ± 0.026 (n=10/10) | 0.974 ± 0.023 (n=10/10) | 0.981 ± 0.018 (n=10/10) | 0.233 ± 0.094 (n=10/10) | 0.396 ± 0.040 (n=10/10) | 0.098 ± 0.038 (n=10/10) | 0.826 ± 0.035 (n=10/10) | 0.548 ± 0.126 (n=10/10) |
| raw/copy_last | 10/10 | 0.838 ± 0.011 (n=10/10) | 0.743 ± 0.004 (n=10/10) | 0.751 ± 0.003 (n=10/10) | 0.831 ± 0.010 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.299 ± 0.013 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.500 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) |
| balanced/copy_last | 10/10 | 0.727 ± 0.016 (n=10/10) | 0.477 ± 0.017 (n=10/10) | 0.500 ± 0.004 (n=10/10) | 0.703 ± 0.014 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.298 ± 0.010 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.500 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) |

Welch's t-test across folds:

| Data | Comparison | Metric | t | p | Ref. valid folds | Other valid folds |
| --- | --- | --- | --- | --- | --- | --- |
| raw | paper_transformer vs tst | mAUC | 0.35 | 0.7324 | 10/10 | 10/10 |
| raw | paper_transformer vs tst | BCA | -0.76 | 0.4571 | 10/10 | 10/10 |
| raw | paper_transformer vs tst | Conv. Acc | -0.06 | 0.9548 | 10/10 | 10/10 |
| raw | paper_transformer vs tst | Conv. BCA | 0.39 | 0.6986 | 10/10 | 10/10 |
| raw | paper_transformer vs minrnn_monthly | mAUC | 3.10 | 0.0065 | 10/10 | 10/10 |
| raw | paper_transformer vs minrnn_monthly | BCA | 4.29 | 0.0006 | 10/10 | 10/10 |
| raw | paper_transformer vs minrnn_monthly | Conv. Acc | 11.83 | 0.0000 | 10/10 | 10/10 |
| raw | paper_transformer vs minrnn_monthly | Conv. BCA | 10.68 | 0.0000 | 10/10 | 10/10 |
| raw | paper_transformer vs lstm_monthly | mAUC | 6.02 | 0.0000 | 10/10 | 10/10 |
| raw | paper_transformer vs lstm_monthly | BCA | 2.73 | 0.0139 | 10/10 | 10/10 |
| raw | paper_transformer vs lstm_monthly | Conv. Acc | 8.03 | 0.0000 | 10/10 | 10/10 |
| raw | paper_transformer vs lstm_monthly | Conv. BCA | 7.80 | 0.0000 | 10/10 | 10/10 |
| raw | paper_transformer vs gru_monthly | mAUC | 7.02 | 0.0000 | 10/10 | 10/10 |
| raw | paper_transformer vs gru_monthly | BCA | 3.83 | 0.0013 | 10/10 | 10/10 |
| raw | paper_transformer vs gru_monthly | Conv. Acc | 9.81 | 0.0000 | 10/10 | 10/10 |
| raw | paper_transformer vs gru_monthly | Conv. BCA | 9.71 | 0.0000 | 10/10 | 10/10 |
| balanced | paper_transformer vs tst | mAUC | -0.29 | 0.7721 | 10/10 | 10/10 |
| balanced | paper_transformer vs tst | BCA | -0.21 | 0.8392 | 10/10 | 10/10 |
| balanced | paper_transformer vs tst | Conv. Acc | 0.27 | 0.7910 | 10/10 | 10/10 |
| balanced | paper_transformer vs tst | Conv. BCA | 0.12 | 0.9094 | 10/10 | 10/10 |
| balanced | paper_transformer vs minrnn_monthly | mAUC | 2.90 | 0.0097 | 10/10 | 10/10 |
| balanced | paper_transformer vs minrnn_monthly | BCA | 5.16 | 0.0001 | 10/10 | 10/10 |
| balanced | paper_transformer vs minrnn_monthly | Conv. Acc | 14.61 | 0.0000 | 10/10 | 10/10 |
| balanced | paper_transformer vs minrnn_monthly | Conv. BCA | 14.66 | 0.0000 | 10/10 | 10/10 |
| balanced | paper_transformer vs lstm_monthly | mAUC | 5.29 | 0.0001 | 10/10 | 10/10 |
| balanced | paper_transformer vs lstm_monthly | BCA | 3.22 | 0.0049 | 10/10 | 10/10 |
| balanced | paper_transformer vs lstm_monthly | Conv. Acc | 10.98 | 0.0000 | 10/10 | 10/10 |
| balanced | paper_transformer vs lstm_monthly | Conv. BCA | 11.03 | 0.0000 | 10/10 | 10/10 |
| balanced | paper_transformer vs gru_monthly | mAUC | 6.20 | 0.0000 | 10/10 | 10/10 |
| balanced | paper_transformer vs gru_monthly | BCA | 4.95 | 0.0002 | 10/10 | 10/10 |
| balanced | paper_transformer vs gru_monthly | Conv. Acc | 13.93 | 0.0000 | 10/10 | 10/10 |
| balanced | paper_transformer vs gru_monthly | Conv. BCA | 14.24 | 0.0000 | 10/10 | 10/10 |

Fold coverage (partial summaries are not full CV results):

- raw/paper_transformer: 10/10 folds; missing []
- raw/tst: 10/10 folds; missing []
- raw/minrnn_monthly: 10/10 folds; missing []
- raw/lstm_monthly: 10/10 folds; missing []
- raw/gru_monthly: 10/10 folds; missing []
- balanced/paper_transformer: 10/10 folds; missing []
- balanced/tst: 10/10 folds; missing []
- balanced/minrnn_monthly: 10/10 folds; missing []
- balanced/lstm_monthly: 10/10 folds; missing []
- balanced/gru_monthly: 10/10 folds; missing []
- raw/copy_last: 10/10 folds; missing []
- balanced/copy_last: 10/10 folds; missing []
