Every metric shows finite-value folds n/expected. Missing, NaN and infinite
values are excluded. SD uses ddof=0 with at least two finite folds;
NA means that fold-to-fold spread cannot be reported.

| Config | Folds | mAUC | F1 (weighted) | Acc | BCA | Stable Acc | Stable BCA | Conv. Acc | Conv. BCA | Conv. Sens | Conv. AUROC (T4) | Sens@Spec90 (T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw/visit_transformer | 10/10 | 0.956 ± 0.013 (n=10/10) | 0.835 ± 0.039 (n=10/10) | 0.839 ± 0.035 (n=10/10) | 0.884 ± 0.024 (n=10/10) | 0.914 ± 0.035 (n=10/10) | 0.943 ± 0.022 (n=10/10) | 0.615 ± 0.075 (n=10/10) | 0.566 ± 0.031 (n=10/10) | 0.277 ± 0.039 (n=10/10) | 0.856 ± 0.044 (n=10/10) | 0.584 ± 0.116 (n=10/10) |
| raw/lstm | 10/10 | 0.952 ± 0.016 (n=10/10) | 0.837 ± 0.040 (n=10/10) | 0.842 ± 0.036 (n=10/10) | 0.885 ± 0.023 (n=10/10) | 0.902 ± 0.039 (n=10/10) | 0.935 ± 0.024 (n=10/10) | 0.662 ± 0.099 (n=10/10) | 0.593 ± 0.050 (n=10/10) | 0.312 ± 0.067 (n=10/10) | 0.855 ± 0.044 (n=10/10) | 0.566 ± 0.120 (n=10/10) |
| raw/gru | 10/10 | 0.955 ± 0.014 (n=10/10) | 0.835 ± 0.030 (n=10/10) | 0.839 ± 0.028 (n=10/10) | 0.883 ± 0.019 (n=10/10) | 0.907 ± 0.029 (n=10/10) | 0.938 ± 0.018 (n=10/10) | 0.636 ± 0.076 (n=10/10) | 0.577 ± 0.032 (n=10/10) | 0.292 ± 0.045 (n=10/10) | 0.862 ± 0.037 (n=10/10) | 0.579 ± 0.103 (n=10/10) |
| raw/copy_last | 10/10 | 0.838 ± 0.011 (n=10/10) | 0.743 ± 0.004 (n=10/10) | 0.751 ± 0.003 (n=10/10) | 0.831 ± 0.010 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.299 ± 0.013 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.500 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) |
| balanced/visit_transformer | 10/10 | 0.910 ± 0.027 (n=10/10) | 0.783 ± 0.057 (n=10/10) | 0.800 ± 0.051 (n=10/10) | 0.837 ± 0.033 (n=10/10) | 0.814 ± 0.061 (n=10/10) | 0.876 ± 0.036 (n=10/10) | 0.785 ± 0.079 (n=10/10) | 0.654 ± 0.038 (n=10/10) | 0.384 ± 0.049 (n=10/10) | 0.847 ± 0.054 (n=10/10) | 0.582 ± 0.142 (n=10/10) |
| balanced/lstm | 10/10 | 0.904 ± 0.027 (n=10/10) | 0.791 ± 0.080 (n=10/10) | 0.805 ± 0.071 (n=10/10) | 0.839 ± 0.047 (n=10/10) | 0.809 ± 0.060 (n=10/10) | 0.869 ± 0.039 (n=10/10) | 0.801 ± 0.106 (n=10/10) | 0.679 ± 0.062 (n=10/10) | 0.430 ± 0.089 (n=10/10) | 0.844 ± 0.051 (n=10/10) | 0.530 ± 0.141 (n=10/10) |
| balanced/gru | 10/10 | 0.909 ± 0.030 (n=10/10) | 0.809 ± 0.064 (n=10/10) | 0.819 ± 0.059 (n=10/10) | 0.851 ± 0.042 (n=10/10) | 0.829 ± 0.068 (n=10/10) | 0.884 ± 0.043 (n=10/10) | 0.809 ± 0.076 (n=10/10) | 0.680 ± 0.040 (n=10/10) | 0.429 ± 0.058 (n=10/10) | 0.859 ± 0.051 (n=10/10) | 0.564 ± 0.171 (n=10/10) |
| balanced/copy_last | 10/10 | 0.727 ± 0.016 (n=10/10) | 0.477 ± 0.017 (n=10/10) | 0.500 ± 0.004 (n=10/10) | 0.703 ± 0.014 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 1.000 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.298 ± 0.010 (n=10/10) | 0.000 ± 0.000 (n=10/10) | 0.500 ± 0.000 (n=10/10) | 0.000 ± 0.000 (n=10/10) |

Welch's t-test across folds:

| Data | Comparison | Metric | t | p | Ref. valid folds | Other valid folds |
| --- | --- | --- | --- | --- | --- | --- |
| raw | visit_transformer vs lstm | mAUC | 0.48 | 0.6361 | 10/10 | 10/10 |
| raw | visit_transformer vs lstm | BCA | -0.09 | 0.9289 | 10/10 | 10/10 |
| raw | visit_transformer vs lstm | Conv. Acc | -1.13 | 0.2733 | 10/10 | 10/10 |
| raw | visit_transformer vs lstm | Conv. BCA | -1.38 | 0.1880 | 10/10 | 10/10 |
| raw | visit_transformer vs gru | mAUC | 0.17 | 0.8687 | 10/10 | 10/10 |
| raw | visit_transformer vs gru | BCA | 0.08 | 0.9410 | 10/10 | 10/10 |
| raw | visit_transformer vs gru | Conv. Acc | -0.58 | 0.5682 | 10/10 | 10/10 |
| raw | visit_transformer vs gru | Conv. BCA | -0.77 | 0.4524 | 10/10 | 10/10 |
| balanced | visit_transformer vs lstm | mAUC | 0.51 | 0.6172 | 10/10 | 10/10 |
| balanced | visit_transformer vs lstm | BCA | -0.08 | 0.9337 | 10/10 | 10/10 |
| balanced | visit_transformer vs lstm | Conv. Acc | -0.36 | 0.7252 | 10/10 | 10/10 |
| balanced | visit_transformer vs lstm | Conv. BCA | -1.04 | 0.3168 | 10/10 | 10/10 |
| balanced | visit_transformer vs gru | mAUC | 0.13 | 0.9003 | 10/10 | 10/10 |
| balanced | visit_transformer vs gru | BCA | -0.76 | 0.4583 | 10/10 | 10/10 |
| balanced | visit_transformer vs gru | Conv. Acc | -0.65 | 0.5241 | 10/10 | 10/10 |
| balanced | visit_transformer vs gru | Conv. BCA | -1.41 | 0.1743 | 10/10 | 10/10 |

Fold coverage (partial summaries are not full CV results):

- raw/visit_transformer: 10/10 folds; missing []
- raw/lstm: 10/10 folds; missing []
- raw/gru: 10/10 folds; missing []
- raw/copy_last: 10/10 folds; missing []
- balanced/visit_transformer: 10/10 folds; missing []
- balanced/lstm: 10/10 folds; missing []
- balanced/gru: 10/10 folds; missing []
- balanced/copy_last: 10/10 folds; missing []
