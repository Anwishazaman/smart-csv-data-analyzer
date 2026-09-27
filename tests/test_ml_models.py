import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import ml_models


def test_regression_comparison_reports_cross_validation_and_test_scores(monkeypatch):
    frame = pd.DataFrame({
        "experience": np.arange(60, dtype=float),
        "salary": np.arange(60, dtype=float) * 2500 + 40000,
    })
    monkeypatch.setattr(
        ml_models,
        "_models",
        lambda problem_type: {"Linear Regression": LinearRegression()},
    )

    result = ml_models.train_and_compare(frame, "salary")
    comparison = result["comparison"]

    assert result["cv_folds"] == 5
    assert result["best_name"] == "Linear Regression"
    assert {"CV R2 score", "CV RMSE", "R2 score", "RMSE"}.issubset(comparison.columns)
    assert comparison.loc[0, "CV R2 score"] > 0.99


def test_classification_comparison_reports_cross_validated_f1(monkeypatch):
    frame = pd.DataFrame({
        "measurement": np.arange(80, dtype=float),
        "label": ["low" if value < 40 else "high" for value in range(80)],
    })
    monkeypatch.setattr(
        ml_models,
        "_models",
        lambda problem_type: {"Logistic Regression": LogisticRegression(max_iter=1000)},
    )

    result = ml_models.train_and_compare(frame, "label")
    comparison = result["comparison"]

    assert result["cv_folds"] == 5
    assert result["best_name"] == "Logistic Regression"
    assert {"CV Accuracy", "CV F1 score", "Accuracy", "F1 score"}.issubset(comparison.columns)


def test_global_shap_importance_aggregates_across_sample(monkeypatch):
    frame = pd.DataFrame({
        "first_feature": np.arange(40, dtype=float),
        "second_feature": np.arange(40, dtype=float) ** 2,
        "target": np.arange(40, dtype=float) * 3,
    })
    monkeypatch.setattr(
        ml_models,
        "_models",
        lambda problem_type: {"Linear Regression": LinearRegression()},
    )
    result = ml_models.train_and_compare(frame, "target")

    class FakeShap:
        @staticmethod
        def Explainer(model, background):
            def explain(rows):
                contribution_values = np.tile(
                    np.arange(1, rows.shape[1] + 1, dtype=float),
                    (len(rows), 1),
                )
                return SimpleNamespace(values=contribution_values)

            return explain

    monkeypatch.setattr(ml_models, "shap", FakeShap)
    importance = ml_models.explain_global_importance(
        result["best_model"], result["training_features"], max_rows=10
    )

    assert list(importance.columns) == ["Feature", "Mean absolute SHAP"]
    assert len(importance) == 2
    assert importance.iloc[0]["Mean absolute SHAP"] >= importance.iloc[1]["Mean absolute SHAP"]
