# Subscription Churn Prediction: From Behavioural Data to Risk Scoring

**Project type:** Historical data analytics and machine learning project  
**Tools:** Python, pandas, SQL/Hive, scikit-learn, imbalanced-learn, XGBoost

## Overview

This project explored how subscription lifecycle information and customer engagement behaviour could be used to identify subscribers at risk of churn.

The workflow combined behavioural feature engineering, classification model comparison, recall-focused hyperparameter tuning, and churn-risk scoring. Separate scoring scripts applied a trained model to generate churn probabilities for downstream analysis.

This case study documents the historical workflow based on the project scripts. Confidential source code, internal infrastructure details, and original customer data are not included.

## Business question

**Can subscription history and engagement behaviour help identify customers who may be at risk of cancelling?**

The modelling workflow focused on identifying churn cases, making recall an important consideration during model tuning.

## Data and feature engineering

The training workflow combined subscription records with behavioural aggregates, including comment and voting activity calculated over 21-day windows.

Features covered several categories:

- **Subscription lifecycle:** date-derived subscription features, offer type, subscription term, and discount usage.
- **Payments:** historical payment counts and payment-related indicators.
- **Engagement:** page views, reading behaviour, completed-reading share, comments, and voting activity.
- **Behavioural patterns:** acquisition source, device usage, time-of-day patterns, and weekend activity.

The workflow also included missing-value handling, categorical encoding, and variance inflation factor (VIF) analysis as part of feature review.

## Modelling approach

The training script compared Logistic Regression and Gradient Boosting classification approaches, including selected experiments with class-imbalance handling.

| Method | Role in the workflow |
|---|---|
| Logistic Regression | Baseline classification model |
| Gradient Boosting | Non-linear classification model |
| SMOTE | Class-imbalance handling in selected experiments |
| StandardScaler | Feature scaling in a Logistic Regression experiment |
| GridSearchCV | Hyperparameter tuning |
| Recall scoring | Objective used in the Gradient Boosting parameter search |

The parameter search evaluated different tree depths and learning rates using cross-validation, with recall as the optimisation metric.

## Evaluation

The scripts included classification reports, confusion matrices, ROC-related analysis, and feature-importance inspection.

These methods help assess the model from different perspectives: detecting churners, understanding false positives and missed cases, evaluating discrimination, and investigating influential features.

A review of the historical code also identified an evaluation issue: some ROC-AUC calculations and ROC-curve inputs used hard class predictions rather than predicted probabilities. ROC analysis should use continuous prediction scores.

The original metric outputs were not available in the material reviewed, so this case study does not report historical performance figures or claim a measured business impact.

## Scoring and monitoring

Separate scoring scripts loaded a trained XGBoost model and generated churn-risk probabilities for downstream analysis.

A further export calculated daily average feature values by observation date, providing a basis for examining changes in input features over time.

The scripts demonstrate probability scoring and feature-summary exports. They do not, by themselves, establish a fully automated production monitoring system or a quantified improvement in retention.

## Key learnings

- Churn modelling depends heavily on feature engineering and the quality of the underlying data.
- Optimising recall can help detect more churn cases, but may increase false positives; precision and business costs should also be considered.
- ROC-AUC and ROC curves should use predicted probabilities or other continuous scores.
- For longitudinal customer data, time-based validation and careful prevention of target leakage are important.
- Scoring and monitoring considerations extend beyond the initial model-training process.

## Scope and confidentiality

This is a retrospective case study of a historical project workflow. The original scripts did not provide the metric outputs needed to verify model performance, and no retention uplift or financial impact is claimed.

Original employer code, customer data, internal endpoints, private infrastructure details, and credentials are intentionally excluded.

**Portfolio note:** Any public implementation accompanying this case study should use independently written code and synthetic or appropriately licensed public data. Synthetic demonstration results must not be represented as results from the historical project.
