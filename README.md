# Synthetic Churn Prediction Demo

This is an independent, reproducible demonstration of a subscription churn classification workflow using **fictional synthetic data only**. It is separate from the historical case study in the repository root.

## What it demonstrates

- Reproducible synthetic subscriber data generation
- A clearly documented synthetic target: churn within the next 30 days
- Stratified train/test split
- Preprocessing in a scikit-learn pipeline
- Logistic Regression baseline with class weighting
- Evaluation with ROC-AUC, average precision, precision, recall, F1 and a confusion matrix
- Precision–recall curve and coefficient chart
- An illustrative SQL feature-aggregation query

## Run it locally

Python 3.10+ is recommended.

```bash
cd synthetic_demo
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python churn_demo.py
```

The script generates files under `outputs/` and `figures/`. The dataset is generated in memory; no real subscriber records are used.

## Example results (synthetic data only)

A reference run on 5,000 generated rows produced:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.811 |
| Average precision | 0.508 |
| Precision at threshold 0.50 | 0.343 |
| Recall at threshold 0.50 | 0.697 |
| F1 at threshold 0.50 | 0.460 |

The test set contained 1,250 generated rows. Confusion matrix in `[[TN, FP], [FN, TP]]` order: `[[811, 251], [57, 131]]`.

**These values are illustrative results from a synthetic data-generating process—not performance metrics from the historical project, a real customer dataset, or evidence of business impact.** Values can vary with package versions.

## Visual results

The charts below are generated from the synthetic test set and are included so the workflow can be reviewed directly on GitHub.

### Precision–recall curve

![Precision–recall curve for the synthetic test set](figures/precision_recall_curve.png)

This curve illustrates the trade-off between precision and recall as the classification threshold changes. Because the synthetic positive class is relatively uncommon, this view complements ROC-AUC.

### Confusion matrix

![Confusion matrix for the synthetic test set](figures/confusion_matrix.png)

At the illustrative threshold of 0.50, the reference run produced 811 true negatives, 251 false positives, 57 false negatives, and 131 true positives. These counts apply only to this generated test sample.

### Logistic Regression coefficients

![Largest Logistic Regression coefficients for the synthetic demo](figures/top_coefficients.png)

The chart displays the largest absolute coefficients from the fitted model. Numeric features were standardised and categorical features one-hot encoded. These coefficients describe associations in the artificial data-generating setup; they are not causal effects.

**Reminder:** all charts and metrics here are based on synthetic data only. They are not results from the historical project or a real customer population.

## How the target is generated

The code creates an illustrative probability from recent inactivity, reading frequency, tenure, payment history, plan, discount use, acquisition channel, and monthly-equivalent price. A random draw from that probability becomes `churn_next_30d`. Since the label is artificially constructed from the same features, the resulting scores only validate that the demo workflow runs; they do not estimate real-world churn performance.

## Evaluation and limitations

- ROC-AUC and the precision–recall curve use predicted probabilities, not hard class labels.
- The 0.50 classification threshold is a demonstration choice. A real use case should select a threshold on validation data based on the cost of missed churners and false positives.
- The demo uses a random stratified split. Real longitudinal subscription data should consider a time-based split, prediction-time feature availability, and repeated-customer leakage.
- Model coefficients are not causal effects.
- This is a baseline demonstration, not a production deployment.

## Files

- `churn_demo.py` — data generator, model pipeline, evaluation and chart output
- `requirements.txt` — Python dependencies
- `example_feature_aggregation.sql` — illustrative SQL pattern with fictional table names
- `tests/test_churn_demo.py` — small automated tests
- `figures/` — generated charts
- `outputs/metrics.json` — generated example metrics

## Relationship to the historical case study

The repository root describes the historical workflow from prior scripts. This demo is independently written and uses synthetic data only. It does not reproduce the original training pipeline or make claims about the original model's performance.
