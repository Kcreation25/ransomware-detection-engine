# Training Run Summary

**Run type:** Quick prototype benchmark; not the full benchmark configuration.

## Data and split
- Original labeled BitcoinHeist dataset: 2,916,697 rows.
- Stratified preview sample: 200,000 rows.
- Split: 130,000 train / 30,000 validation / 40,000 held-out test.
- The supplied `no label.csv` contains 1,048,575 addresses, all of which are present in the full labeled dataset. To avoid leakage, this run fitted Isolation Forest on the training split only.

## Benchmark configuration
- Model: Random Forest
- Imbalance strategy: class weighting
- One model/strategy combination was run for this quick preview.
- This is not the complete model/resampling benchmark.

## Held-out test metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9802 |
| Balanced accuracy | 0.6967 |
| Precision | 0.3367 |
| Recall | 0.4049 |
| F1 | 0.3677 |
| MCC | 0.3593 |
| ROC-AUC | 0.8695 |
| PR-AUC | 0.3262 |

## Confusion matrix

| | Predicted non-malicious | Predicted malicious |
|---|---:|---:|
| Actual non-malicious | 38,979 | 453 |
| Actual malicious | 338 | 230 |

## Prediction smoke test
The saved model scored 1,000 rows from the supplied unlabeled CSV and wrote `reports/predictions.csv` locally. These predicted labels are not ground truth.

## Limitations
- Do not present these preview metrics as full-dataset benchmark results.
- The complete benchmark can take considerably longer.
- A malicious prediction is a risk flag for further investigation, not proof of criminal activity.
