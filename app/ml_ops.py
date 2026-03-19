from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_ops import NA_TEXT_VALUES, split_feature_types
from .models import get_model, is_incremental_model


@dataclass
class IncrementalModelBundle:
    """Store fitted preprocessing and incremental model state."""

    preprocessor: ColumnTransformer
    model: object
    features: List[str]
    target: str
    task_type: str
    model_name: str


class TrainingResult:
    """Store outputs from one training run."""

    def __init__(self, pipeline, metrics: Dict[str, float], importance_df: pd.DataFrame, model_bundle=None):
        self.pipeline = pipeline
        self.metrics = metrics
        self.importance_df = importance_df
        self.model_bundle = model_bundle


def build_preprocessor(df: pd.DataFrame, features: List[str]) -> ColumnTransformer:
    """Build the shared preprocessing graph."""
    numeric_features, categorical_features = split_feature_types(df, features)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame | None,
    features: List[str],
    target: str,
    model_name: str,
    task_type: str,
    continue_bundle: IncrementalModelBundle | None = None,
    model_params: Dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Train a full or incremental model and evaluate it."""
    if continue_bundle is not None:
        return _continue_incremental_model(continue_bundle, train_df, eval_df, features, target)

    if is_incremental_model(model_name):
        return _fit_incremental_model(train_df, eval_df, features, target, model_name, task_type, model_params or {}, test_size, random_state)
    return _fit_full_model(train_df, eval_df, features, target, model_name, task_type, model_params or {}, test_size, random_state)


def predict_dataframe(model_or_bundle, df: pd.DataFrame, features: List[str]) -> pd.Series:
    """Run predictions on a dataframe using the stored training features."""
    feature_frame = _normalize_feature_frame(df[features].copy())
    if hasattr(model_or_bundle, "named_steps"):
        return pd.Series(model_or_bundle.predict(feature_frame), index=df.index)
    if hasattr(model_or_bundle, "preprocessor") and hasattr(model_or_bundle, "model"):
        transformed = model_or_bundle.preprocessor.transform(feature_frame)
        return pd.Series(model_or_bundle.model.predict(transformed), index=df.index)
    raise ValueError("Loaded model format is not supported for prediction.")


def extract_importance(model_or_pipeline, df: pd.DataFrame, features: List[str], preprocessor: ColumnTransformer | None = None) -> pd.DataFrame:
    """Extract feature importance when the estimator exposes it."""
    numeric_features, categorical_features = split_feature_types(df, features)
    model = model_or_pipeline.named_steps["model"] if hasattr(model_or_pipeline, "named_steps") else model_or_pipeline
    active_preprocessor = model_or_pipeline.named_steps["preprocessor"] if hasattr(model_or_pipeline, "named_steps") else preprocessor
    feature_names = compute_feature_names(active_preprocessor, numeric_features, categorical_features)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = np.abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False).reset_index(drop=True)


def compute_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Expand encoded feature names from the fitted preprocessor."""
    feature_names = list(numeric_features)
    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(ohe.get_feature_names_out(categorical_features).tolist())
    return feature_names


def _fit_full_model(train_df, eval_df, features, target, model_name, task_type, model_params, test_size, random_state) -> TrainingResult:
    dataset = _prepare_training_frame(train_df, features, target)
    if dataset.empty:
        raise ValueError("Training dataset is empty after preparing target and feature values.")

    if eval_df is None or eval_df.empty:
        X = dataset[features]
        y = dataset[target]
        stratify = _build_stratify_target(y, task_type, test_size)
        X_train, X_eval, y_train, y_eval = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_frame = pd.concat([X_train, y_train], axis=1)
        eval_frame = pd.concat([X_eval, y_eval], axis=1)
    else:
        train_frame = dataset
        eval_frame = _prepare_training_frame(eval_df, features, target)
        if eval_frame.empty:
            raise ValueError("Evaluation dataset is empty after filtering by time range.")

    preprocessor = build_preprocessor(train_frame, features)
    model = get_model(model_name, task_type, model_params=model_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train_frame[features], train_frame[target])
    preds = pipeline.predict(eval_frame[features])

    metrics = _compute_metrics(eval_frame[target], preds, task_type)
    importance_df = extract_importance(pipeline, train_frame, features)
    return TrainingResult(pipeline, metrics, importance_df)


def _fit_incremental_model(train_df, eval_df, features, target, model_name, task_type, model_params, test_size, random_state) -> TrainingResult:
    dataset = _prepare_training_frame(train_df, features, target)
    if dataset.empty:
        raise ValueError("Training dataset is empty after preparing target and feature values.")

    if eval_df is None or eval_df.empty:
        X = dataset[features]
        y = dataset[target]
        stratify = _build_stratify_target(y, task_type, test_size)
        X_train, X_eval, y_train, y_eval = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_frame = pd.concat([X_train, y_train], axis=1)
        eval_frame = pd.concat([X_eval, y_eval], axis=1)
    else:
        train_frame = dataset
        eval_frame = _prepare_training_frame(eval_df, features, target)
        if eval_frame.empty:
            raise ValueError("Evaluation dataset is empty after filtering by time range.")

    preprocessor = build_preprocessor(train_frame, features)
    X_train = preprocessor.fit_transform(train_frame[features])
    X_eval = preprocessor.transform(eval_frame[features])
    model = get_model(model_name, task_type, model_params=model_params)

    if task_type == "classification":
        model.partial_fit(X_train, train_frame[target], classes=np.unique(train_frame[target]))
    else:
        model.partial_fit(X_train, train_frame[target])

    preds = model.predict(X_eval)
    metrics = _compute_metrics(eval_frame[target], preds, task_type)
    importance_df = extract_importance(model, train_frame, features, preprocessor=preprocessor)
    bundle = IncrementalModelBundle(preprocessor, model, list(features), target, task_type, model_name)
    return TrainingResult(None, metrics, importance_df, model_bundle=bundle)


def _continue_incremental_model(bundle: IncrementalModelBundle, train_df: pd.DataFrame, eval_df: pd.DataFrame | None, features, target) -> TrainingResult:
    train_frame = _prepare_training_frame(train_df, features, target)
    if train_frame.empty:
        raise ValueError("Training dataset is empty after preparing target and feature values.")

    if eval_df is None or eval_df.empty:
        eval_frame = train_frame
    else:
        eval_frame = _prepare_training_frame(eval_df, features, target)
        if eval_frame.empty:
            raise ValueError("Evaluation dataset is empty after filtering by time range.")

    X_train = bundle.preprocessor.transform(train_frame[features])
    X_eval = bundle.preprocessor.transform(eval_frame[features])

    if bundle.task_type == "classification":
        bundle.model.partial_fit(X_train, train_frame[target])
    else:
        bundle.model.partial_fit(X_train, train_frame[target])

    preds = bundle.model.predict(X_eval)
    metrics = _compute_metrics(eval_frame[target], preds, bundle.task_type)
    importance_df = extract_importance(bundle.model, train_frame, features, preprocessor=bundle.preprocessor)
    return TrainingResult(None, metrics, importance_df, model_bundle=bundle)


def _compute_metrics(y_true, y_pred, task_type: str) -> Dict[str, float]:
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _prepare_training_frame(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    frame = df[features + [target]].copy()
    frame[features] = _normalize_feature_frame(frame[features])
    frame[target] = _normalize_target_series(frame[target])
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    return frame


def _normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            stripped = series.astype("string").str.strip()
            normalized[column] = series.mask(stripped.fillna("").eq("") | stripped.str.lower().isin(NA_TEXT_VALUES), pd.NA)
    return normalized


def _normalize_target_series(series: pd.Series) -> pd.Series:
    normalized = series.copy()
    if pd.api.types.is_object_dtype(normalized) or pd.api.types.is_string_dtype(normalized):
        stripped = normalized.astype("string").str.strip()
        normalized = normalized.mask(stripped.fillna("").eq("") | stripped.str.lower().isin(NA_TEXT_VALUES), pd.NA)
    return normalized.fillna(0)


def _build_stratify_target(y: pd.Series, task_type: str, test_size: float):
    if task_type != "classification":
        return None

    value_counts = y.value_counts(dropna=False)
    class_count = len(value_counts)
    if class_count < 2 or value_counts.min() < 2:
        return None

    test_rows = max(int(round(len(y) * test_size)), 1)
    train_rows = len(y) - test_rows
    if test_rows < class_count or train_rows < class_count:
        return None
    return y
