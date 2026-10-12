from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churn_demo import build_model, make_synthetic_data


def test_data_generator_is_reproducible():
    pd.testing.assert_frame_equal(
        make_synthetic_data(250, 7),
        make_synthetic_data(250, 7),
    )


def test_target_is_binary_and_schema_has_expected_columns():
    data = make_synthetic_data(250, 8)
    assert "churn_next_30d" in data.columns
    assert set(data["churn_next_30d"].unique()).issubset({0, 1})
    assert len(data) == 250


def test_pipeline_fits_and_returns_probabilities():
    data = make_synthetic_data(500, 9)
    X = data.drop(columns="churn_next_30d")
    y = data["churn_next_30d"]
    model = build_model().fit(X, y)
    probabilities = model.predict_proba(X.iloc[:20])[:, 1]
    assert len(probabilities) == 20
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
