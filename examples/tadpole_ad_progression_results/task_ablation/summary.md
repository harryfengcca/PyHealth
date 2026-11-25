Every metric shows finite-value folds n/expected. Missing, NaN and infinite
values are excluded. SD uses ddof=0 with at least two finite folds;
NA means that fold-to-fold spread cannot be reported.

| Config | Folds | mAUC | F1 (weighted) | Acc | BCA | Stable Acc | Stable BCA | Conv. Acc | Conv. BCA | Conv. Sens | Conv. AUROC (T4) | Sens@Spec90 (T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 default (ffill, to_target, drop reverters) [LSTM] | 5/5 | 0.954 ± 0.008 (n=5/5) | 0.830 ± 0.012 (n=5/5) | 0.834 ± 0.011 (n=5/5) | 0.878 ± 0.006 (n=5/5) | 0.903 ± 0.019 (n=5/5) | 0.935 ± 0.011 (n=5/5) | 0.625 ± 0.043 (n=5/5) | 0.573 ± 0.024 (n=5/5) | 0.287 ± 0.033 (n=5/5) | 0.860 ± 0.019 (n=5/5) | 0.544 ± 0.048 (n=5/5) |
| T1a no forward fill [LSTM] | 5/5 | 0.950 ± 0.007 (n=5/5) | 0.831 ± 0.014 (n=5/5) | 0.836 ± 0.013 (n=5/5) | 0.881 ± 0.006 (n=5/5) | 0.913 ± 0.014 (n=5/5) | 0.942 ± 0.007 (n=5/5) | 0.604 ± 0.031 (n=5/5) | 0.556 ± 0.016 (n=5/5) | 0.261 ± 0.021 (n=5/5) | 0.845 ± 0.018 (n=5/5) | 0.563 ± 0.063 (n=5/5) |
| T1b forward fill + missing indicators [LSTM] | 5/5 | 0.952 ± 0.009 (n=5/5) | 0.836 ± 0.015 (n=5/5) | 0.841 ± 0.013 (n=5/5) | 0.884 ± 0.008 (n=5/5) | 0.905 ± 0.006 (n=5/5) | 0.937 ± 0.003 (n=5/5) | 0.648 ± 0.039 (n=5/5) | 0.583 ± 0.023 (n=5/5) | 0.298 ± 0.036 (n=5/5) | 0.859 ± 0.022 (n=5/5) | 0.575 ± 0.059 (n=5/5) |
| T2a no time feature [LSTM] | 5/5 | 0.950 ± 0.011 (n=5/5) | 0.836 ± 0.013 (n=5/5) | 0.841 ± 0.012 (n=5/5) | 0.883 ± 0.006 (n=5/5) | 0.908 ± 0.006 (n=5/5) | 0.938 ± 0.003 (n=5/5) | 0.638 ± 0.044 (n=5/5) | 0.578 ± 0.021 (n=5/5) | 0.290 ± 0.026 (n=5/5) | 0.853 ± 0.023 (n=5/5) | 0.563 ± 0.061 (n=5/5) |
| T2b months since previous visit [LSTM] | 5/5 | 0.951 ± 0.008 (n=5/5) | 0.825 ± 0.005 (n=5/5) | 0.830 ± 0.005 (n=5/5) | 0.875 ± 0.006 (n=5/5) | 0.906 ± 0.009 (n=5/5) | 0.937 ± 0.008 (n=5/5) | 0.599 ± 0.019 (n=5/5) | 0.555 ± 0.011 (n=5/5) | 0.264 ± 0.019 (n=5/5) | 0.853 ± 0.018 (n=5/5) | 0.550 ± 0.072 (n=5/5) |
| T3 keep reverters [LSTM] | 5/5 | 0.926 ± 0.015 (n=5/5) | 0.790 ± 0.023 (n=5/5) | 0.795 ± 0.022 (n=5/5) | 0.845 ± 0.016 (n=5/5) | 0.881 ± 0.034 (n=5/5) | 0.919 ± 0.023 (n=5/5) | 0.638 ± 0.044 (n=5/5) | 0.586 ± 0.021 (n=5/5) | 0.309 ± 0.027 (n=5/5) | 0.851 ± 0.039 (n=5/5) | 0.516 ± 0.126 (n=5/5) |
| T0 default (ffill, to_target, drop reverters) [Transformer] | 5/5 | 0.953 ± 0.005 (n=5/5) | 0.833 ± 0.015 (n=5/5) | 0.839 ± 0.014 (n=5/5) | 0.882 ± 0.009 (n=5/5) | 0.909 ± 0.019 (n=5/5) | 0.939 ± 0.011 (n=5/5) | 0.625 ± 0.039 (n=5/5) | 0.567 ± 0.011 (n=5/5) | 0.276 ± 0.012 (n=5/5) | 0.851 ± 0.016 (n=5/5) | 0.565 ± 0.047 (n=5/5) |
| T1a no forward fill [Transformer] | 5/5 | 0.952 ± 0.006 (n=5/5) | 0.836 ± 0.017 (n=5/5) | 0.840 ± 0.016 (n=5/5) | 0.884 ± 0.010 (n=5/5) | 0.927 ± 0.020 (n=5/5) | 0.951 ± 0.012 (n=5/5) | 0.578 ± 0.039 (n=5/5) | 0.543 ± 0.016 (n=5/5) | 0.244 ± 0.016 (n=5/5) | 0.846 ± 0.011 (n=5/5) | 0.591 ± 0.063 (n=5/5) |
| T1b forward fill + missing indicators [Transformer] | 5/5 | 0.954 ± 0.003 (n=5/5) | 0.840 ± 0.011 (n=5/5) | 0.846 ± 0.010 (n=5/5) | 0.887 ± 0.006 (n=5/5) | 0.904 ± 0.015 (n=5/5) | 0.936 ± 0.007 (n=5/5) | 0.672 ± 0.026 (n=5/5) | 0.586 ± 0.010 (n=5/5) | 0.293 ± 0.013 (n=5/5) | 0.852 ± 0.009 (n=5/5) | 0.575 ± 0.047 (n=5/5) |
| T2a no time feature [Transformer] | 5/5 | 0.954 ± 0.003 (n=5/5) | 0.834 ± 0.011 (n=5/5) | 0.839 ± 0.011 (n=5/5) | 0.883 ± 0.005 (n=5/5) | 0.911 ± 0.015 (n=5/5) | 0.940 ± 0.007 (n=5/5) | 0.620 ± 0.029 (n=5/5) | 0.566 ± 0.018 (n=5/5) | 0.275 ± 0.026 (n=5/5) | 0.857 ± 0.012 (n=5/5) | 0.549 ± 0.066 (n=5/5) |
| T2b months since previous visit [Transformer] | 5/5 | 0.951 ± 0.004 (n=5/5) | 0.836 ± 0.015 (n=5/5) | 0.841 ± 0.013 (n=5/5) | 0.885 ± 0.009 (n=5/5) | 0.917 ± 0.026 (n=5/5) | 0.945 ± 0.015 (n=5/5) | 0.612 ± 0.044 (n=5/5) | 0.561 ± 0.017 (n=5/5) | 0.268 ± 0.018 (n=5/5) | 0.845 ± 0.010 (n=5/5) | 0.583 ± 0.063 (n=5/5) |
| T3 keep reverters [Transformer] | 5/5 | 0.930 ± 0.008 (n=5/5) | 0.789 ± 0.009 (n=5/5) | 0.796 ± 0.010 (n=5/5) | 0.846 ± 0.009 (n=5/5) | 0.874 ± 0.011 (n=5/5) | 0.916 ± 0.008 (n=5/5) | 0.664 ± 0.051 (n=5/5) | 0.600 ± 0.023 (n=5/5) | 0.326 ± 0.030 (n=5/5) | 0.853 ± 0.025 (n=5/5) | 0.571 ± 0.071 (n=5/5) |

Incomplete metric coverage (including metrics available only in CSV):

Undefined completed folds include absent metrics and non-finite values.

| Config | Metric | Valid/Expected | Undefined completed folds | Missing run folds |
| --- | --- | --- | --- | --- |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [LSTM] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [LSTM] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [LSTM] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [LSTM] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [LSTM] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T3 keep reverters [LSTM] | reverter.mauc | 4/5 | [4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T0 default (ffill, to_target, drop reverters) [Transformer] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1a no forward fill [Transformer] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T1b forward fill + missing indicators [Transformer] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2a no time feature [Transformer] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.accuracy | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.bca | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.f1_macro | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.f1_weighted | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.mauc | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.n | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.sens | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.sens_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.sens_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.sens_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.spec | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.spec_AD | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.spec_CN | 0/5 | [0, 1, 2, 3, 4] | [] |
| T2b months since previous visit [Transformer] | reverter.spec_MCI | 0/5 | [0, 1, 2, 3, 4] | [] |
| T3 keep reverters [Transformer] | reverter.mauc | 4/5 | [4] | [] |

Fold coverage (partial summaries are not full CV results):

- T0 default (ffill, to_target, drop reverters) [LSTM]: 5/5 folds; missing []
- T1a no forward fill [LSTM]: 5/5 folds; missing []
- T1b forward fill + missing indicators [LSTM]: 5/5 folds; missing []
- T2a no time feature [LSTM]: 5/5 folds; missing []
- T2b months since previous visit [LSTM]: 5/5 folds; missing []
- T3 keep reverters [LSTM]: 5/5 folds; missing []
- T0 default (ffill, to_target, drop reverters) [Transformer]: 5/5 folds; missing []
- T1a no forward fill [Transformer]: 5/5 folds; missing []
- T1b forward fill + missing indicators [Transformer]: 5/5 folds; missing []
- T2a no time feature [Transformer]: 5/5 folds; missing []
- T2b months since previous visit [Transformer]: 5/5 folds; missing []
- T3 keep reverters [Transformer]: 5/5 folds; missing []
