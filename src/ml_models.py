"""Train and compare simple models for a selected target column."""

import pandas as pd
try:
    import shap
except ImportError:
    shap = None

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None


def detect_problem_type(target):
    """Choose classification for labels and regression for continuous numbers."""
    if (
        target.dtype == "object"
        or str(target.dtype) == "category"
        or target.dtype == "bool"
        or target.nunique() <= 10
    ):
        return "classification"
    return "regression"


def _make_preprocessor(features):
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numbers",
                Pipeline([
                    ("fill", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categories",
                Pipeline([
                    ("fill", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers)


def _models(problem_type, class_count=None):
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, random_state=42
            ),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
            )
        return models

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            objective="reg:squarederror",
        )
    return models


def _feature_importance(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_).mean(axis=0)
    else:
        return None

    importance = pd.DataFrame({"feature": names, "importance": values})
    return importance.sort_values("importance", ascending=False).head(15)


def train_and_compare(dataframe, target_column):
    """Preprocess, split, train several models, and return a comparison."""
    data = dataframe.dropna(subset=[target_column]).copy()
    target = data.pop(target_column)

    if target.nunique() < 2:
        raise ValueError("The selected target needs at least two different values.")

    problem_type = detect_problem_type(target)
    features = data
    target_encoder = None

    if problem_type == "classification":
        target_encoder = LabelEncoder()
        target = pd.Series(
            target_encoder.fit_transform(target.astype(str)),
            index=target.index,
            name=target.name,
        )

    if problem_type == "classification" and target.value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for a train/test split.")

    stratify = target if problem_type == "classification" else None
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    comparison = []
    fitted_models = {}
    for model_name, model in _models(problem_type).items():
        pipeline = Pipeline([
            ("preprocess", _make_preprocessor(x_train)),
            ("model", model),
        ])
        try:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            if problem_type == "classification":
                primary_metric = accuracy_score(y_test, predictions)
                row = {
                    "Model": model_name,
                    "Accuracy": round(primary_metric, 4),
                    "F1 score": round(f1_score(y_test, predictions, average="weighted"), 4),
                }
            else:
                primary_metric = r2_score(y_test, predictions)
                row = {
                    "Model": model_name,
                    "R2 score": round(primary_metric, 4),
                    "RMSE": round(mean_squared_error(y_test, predictions) ** 0.5, 2),
                }
            comparison.append(row)
            fitted_models[model_name] = pipeline
        except Exception as error:
            comparison.append({"Model": model_name, "Error": str(error)})

    successful = [row for row in comparison if "Error" not in row]
    if not successful:
        raise ValueError("None of the models could be trained on this data.")

    metric_name = "Accuracy" if problem_type == "classification" else "R2 score"
    best_row = max(successful, key=lambda row: row[metric_name])
    best_name = best_row["Model"]

    return {
        "problem_type": problem_type,
        "comparison": pd.DataFrame(comparison),
        "best_name": best_name,
        "best_model": fitted_models[best_name],
        "feature_importance": _feature_importance(fitted_models[best_name]),
        "x_test_rows": len(x_test),
        "test_features": x_test,
        "test_target": y_test,
        "target_encoder": target_encoder,
    }


def explain_prediction(pipeline, features):
    """Return SHAP contributions for the first row in a test dataframe."""
    if shap is None:
        raise ValueError("SHAP is not installed. Run: pip install shap")
    if features.empty:
        raise ValueError("There is no test row to explain.")

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed_background = preprocessor.transform(features)
    transformed = transformed_background[:1]
    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()
        transformed = transformed_background[:1]
    feature_names = preprocessor.get_feature_names_out()

    try:
        if model.__class__.__name__.startswith("XGB"):
            explainer = shap.TreeExplainer(
                model, feature_perturbation="tree_path_dependent"
            )
        else:
            explainer = shap.Explainer(model, transformed_background)
        explanation = explainer(transformed)
    except Exception as error:
        raise ValueError(f"SHAP could not explain this model: {error}") from error

    values = explanation.values
    base_values = explanation.base_values

    if values.ndim == 3:
        prediction = int(model.predict(transformed)[0])
        values = values[0, :, prediction]
        base_value = base_values[0, prediction]
    else:
        values = values[0]
        base_value = base_values[0]

    contribution_table = pd.DataFrame({
        "Feature": feature_names,
        "SHAP contribution": values,
    })
    contribution_table["Effect"] = contribution_table["SHAP contribution"].apply(
        lambda value: "increased prediction" if value > 0 else "decreased prediction"
    )
    contribution_table["Absolute contribution"] = contribution_table[
        "SHAP contribution"
    ].abs()
    contribution_table = contribution_table.sort_values(
        "Absolute contribution", ascending=False
    ).head(15)

    return {
        "base_value": float(base_value),
        "prediction": model.predict(transformed)[0],
        "contributions": contribution_table.drop(columns=["Absolute contribution"]),
    }
