"""Synthetic subscription churn demo. Contains no real customer data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50


def make_synthetic_data(n_rows: int = 5000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return fictional subscriber-level rows and an artificially generated churn label."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100 for a meaningful demo split.")

    rng = np.random.default_rng(random_state)
    tenure_days = rng.integers(1, 1500, n_rows)
    days_since_last_activity = np.clip(
        rng.gamma(shape=2.0, scale=8.0, size=n_rows), 0, 90
    ).round()
    reading_days_30d = rng.binomial(
        30, np.clip(0.65 - days_since_last_activity / 80, 0.03, 0.8)
    )
    payments_12m = rng.poisson(lam=np.clip(tenure_days / 365 * 8, 0.1, 30))
    discount_used = rng.binomial(1, 0.35, n_rows)
    plan = rng.choice(["basic", "standard", "premium"], size=n_rows, p=[0.45, 0.4, 0.15])
    acquisition_channel = rng.choice(
        ["organic", "search", "social", "referral"], size=n_rows, p=[0.35, 0.3, 0.2, 0.15]
    )
    monthly_equivalent_price = np.select(
        [plan == "basic", plan == "standard", plan == "premium"], [8.0, 12.0, 18.0]
    ) + rng.normal(0, 0.75, n_rows)

    # This formula intentionally creates an artificial relationship for teaching only.
    logit = (
        -1.25
        + 0.055 * days_since_last_activity
        - 0.075 * reading_days_30d
        - 0.035 * np.minimum(payments_12m, 20)
        - 0.0007 * tenure_days
        + 0.45 * discount_used
        + 0.20 * (plan == "basic")
        + 0.18 * (acquisition_channel == "social")
        + 0.035 * (monthly_equivalent_price - 12)
    )
    probability = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
    churn_next_30d = rng.binomial(1, probability)

    return pd.DataFrame({
        "tenure_days": tenure_days,
        "days_since_last_activity": days_since_last_activity.astype(int),
        "reading_days_30d": reading_days_30d,
        "payments_12m": payments_12m,
        "discount_used": discount_used,
        "plan": plan,
        "acquisition_channel": acquisition_channel,
        "monthly_equivalent_price": monthly_equivalent_price.round(2),
        "churn_next_30d": churn_next_30d,
    })


def build_model() -> Pipeline:
    """Create a leakage-conscious preprocessing and Logistic Regression pipeline."""
    numeric_features = [
        "tenure_days", "days_since_last_activity", "reading_days_30d",
        "payments_12m", "discount_used", "monthly_equivalent_price",
    ]
    categorical_features = ["plan", "acquisition_channel"]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])


def run_demo(n_rows: int = 5000, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Train, evaluate and export a baseline model on generated data."""
    OUTPUTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    data = make_synthetic_data(n_rows=n_rows)
    features = data.drop(columns="churn_next_30d")
    target = data["churn_next_30d"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=RANDOM_STATE, stratify=target
    )

    model = build_model()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    metrics = {
        "data_note": "Synthetic fictional data; these metrics are not historical business results.",
        "rows": int(len(data)),
        "synthetic_positive_rate": float(target.mean()),
        "test_rows": int(len(y_test)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prediction_export = X_test.reset_index(drop=True).copy()
    prediction_export["actual_churn_next_30d"] = y_test.reset_index(drop=True)
    prediction_export["predicted_churn_probability"] = probabilities
    prediction_export["predicted_churn_at_threshold"] = predictions
    prediction_export.to_csv(OUTPUTS / "synthetic_test_predictions.csv", index=False)

    # Precision–recall curve must use probabilities, not hard class labels.
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve (Synthetic Test Set)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    img = ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], labels=["Actual stay", "Actual churn"])
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(img, ax=ax, label="Subscribers")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    coef_frame = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_frame["absolute_coefficient"] = coef_frame["coefficient"].abs()
    coef_frame = coef_frame.sort_values("absolute_coefficient", ascending=False)
    coef_frame.to_csv(OUTPUTS / "model_coefficients.csv", index=False)
    top = coef_frame.head(12).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set(xlabel="Logistic regression coefficient", title="Largest Coefficients (Synthetic Demo)")
    fig.tight_layout()
    fig.savefig(FIGURES / "top_coefficients.png", dpi=160)
    plt.close(fig)

    print("SYNTHETIC DEMO ONLY — not historical or production performance.")
    print(f"Rows: {len(data)} | synthetic positive rate: {target.mean():.3f} | test rows: {len(y_test)}")
    for key in ("roc_auc", "average_precision", "precision", "recall", "f1"):
        print(f"{key}: {metrics[key]:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {[int(tn), int(fp), int(fn), int(tp)]}")
    print(f"Outputs saved under: {ROOT}")
    return metrics


if __name__ == "__main__":
    run_demo()
