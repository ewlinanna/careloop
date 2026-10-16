# CareLoop evaluation (patient-grouped test split)
Test encounters: 19918, positive rate 0.115

| model | ROC-AUC | PR-AUC | Brier | P@top10% | Lift@top10% |
|---|---|---|---|---|---|
| logistic_regression | 0.670 | 0.225 | 0.2245 | 0.264 | 2.29 |
| lightgbm | 0.674 | 0.241 | 0.1144 | 0.285 | 2.48 |
| catboost | 0.678 | 0.239 | 0.1148 | 0.282 | 2.45 |
| ensemble_uncalibrated | 0.679 | 0.240 | 0.1393 | 0.280 | 2.43 |
| ensemble_calibrated | 0.679 | 0.230 | 0.0964 | 0.283 | 2.46 |

ROC-AUC around 0.65-0.70 is realistic for this dataset. Accuracy is not reported (imbalanced target).