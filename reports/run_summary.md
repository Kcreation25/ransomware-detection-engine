# Training Run Summary

**Run date:** 2026-11-01  
**Run type:** Quick prototype / smoke benchmark — not the full benchmark configuration.

## Data

- Labeled BitcoinHeist source file: 2,916,697 rows.
- Stratified preview sample: 200,000 rows.
- Preview split: 130,000 train / 30,000 validation / 40,000 held-out test.
- Supplied unlabeled file: 1,048,575 rows. All addresses overlapped the labeled source, so no independent unlabeled records remained after the overlap check.
- To avoid cross-split contamination, the Isolation Forest for this preview was fitted on the training split only.

## Model and configuration

- Supervised model: Random Forest.
- Imbalance strategy: class weighting.
- Hybrid component: Isolation Forest.
- Only one model/strategy combination was run for this preview; the full configured model/resampling search was not run.

## Held-out preview metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.980225 |
| Balanced accuracy | 0.696721 |
| Precision | 0.336750 |
| Recall | 0.404930 |
| F1 | 0.367706 |
| MCC | 0.359317 |
| ROC-AUC | 0.869489 |
| PR-AUC | 0.326232 |

## Confusion matrix

| | Predicted non-malicious | Predicted malicious |
|---|---:|---:|
| Actual non-malicious | 38,979 | 453 |
| Actual malicious | 338 | 230 |

## Prediction smoke test

The saved model scored 1,000 rows from the supplied unlabeled CSV. The prediction output is a demonstration only; labels are model predictions and are not ground-truth annotations. A small sample is available in predictions_sample.csv.

## Limitations and responsible interpretation

- These are preview metrics from a 200,000-row sample, not full-dataset benchmark results.
- Accuracy is affected by class imbalance; use precision, recall, F1, PR-AUC, MCC and the confusion matrix when interpreting performance.
- The separate unlabeled file did not provide independent addresses. The Isolation Forest therefore used only the training partition for this run.
- A malicious prediction is a risk signal for further investigation, not proof of criminal activity or attribution.
- Re-run the complete benchmark on a properly prepared, independent dataset before presenting final project results.
