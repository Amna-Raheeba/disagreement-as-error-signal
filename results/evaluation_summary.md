# Evaluation summary

Seed 7. Train 800 rows; explained 250 validation and 250 test rows per model (3 models x 4 explainers).

## Pooled AUC for predicting model error (higher is better)

| Signal | AUC |
|---|---|
| Explainer disagreement | 0.535 |
| Model confidence (1 - margin) | 0.682 |
| Ensemble variance | 0.640 |
| Disagreement + confidence (rank average) | 0.651 |
| Random | 0.508 |

## Per model

| Model | Accuracy | AUC disagreement | AUC confidence | AUC ensemble var | AUC combined | Mean disagreement (errors) | Mean disagreement (correct) |
|---|---|---|---|---|---|---|---|
| knn | 0.708 | 0.554 | 0.677 | 0.695 | 0.681 | 0.296 | 0.277 |
| logistic | 0.796 | 0.458 | 0.674 | 0.585 | 0.579 | 0.058 | 0.061 |
| tree | 0.704 | 0.595 | 0.697 | 0.640 | 0.692 | 0.233 | 0.205 |

## By planted region (pooled over models)

| Region | Model-rows | Error rate | AUC disagreement | AUC confidence |
|---|---|---|---|---|
| interaction | 93 | 0.387 | 0.485 | 0.497 |
| normal | 567 | 0.247 | 0.605 | 0.724 |
| shift | 90 | 0.244 | 0.565 | 0.601 |

## Distrust-threshold triage (calibrated on validation, review target 20%)

| Model | Threshold | Review rate | Error capture | Error rate in queue | Error rate outside |
|---|---|---|---|---|---|
| knn | 0.386 | 0.192 | 0.219 | 0.333 | 0.282 |
| logistic | 0.086 | 0.256 | 0.255 | 0.203 | 0.204 |
| tree | 0.307 | 0.172 | 0.203 | 0.349 | 0.285 |

Mean error capture at a ~20% review rate: 0.226 of all errors are routed to review by the distrust threshold.
With the threshold calibrated on the combined disagreement + confidence score instead, mean error capture is 0.269.
