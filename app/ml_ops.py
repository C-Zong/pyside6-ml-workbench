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

from .data_ops import split_feature_types
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
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Train a full or incremental model and evaluate it."""
    if continue_bundle is not None:
        return _continue_incremental_model(continue_bundle, train_df, eval_df, features, target)

    if is_incremental_model(model_name):
        return _fit_incremental_model(train_df, eval_df, features, target, model_name, task_type, test_size, random_state)
    return _fit_full_model(train_df, eval_df, features, target, model_name, task_type, test_size, random_state)


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


def _fit_full_model(train_df, eval_df, features, target, model_name, task_type, test_size, random_state) -> TrainingResult:
    dataset = train_df[features + [target]].dropna().copy()
    if dataset.empty:
        raise ValueError("Training dataset is empty after dropping missing target rows.")

    if eval_df is None or eval_df.empty:
        X = dataset[features]
        y = dataset[target]
        stratify = y if task_type == "classification" and y.nunique(dropna=True) > 1 else None
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
        eval_frame = eval_df[features + [target]].dropna().copy()
        if eval_frame.empty:
            raise ValueError("Evaluation dataset is empty after filtering by time range.")

    preprocessor = build_preprocessor(train_frame, features)
    model = get_model(model_name, task_type)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train_frame[features], train_frame[target])
    preds = pipeline.predict(eval_frame[features])

    metrics = _compute_metrics(eval_frame[target], preds, task_type)
    importance_df = extract_importance(pipeline, train_frame, features)
    return TrainingResult(pipeline, metrics, importance_df)


def _fit_incremental_model(train_df, eval_df, features, target, model_name, task_type, test_size, random_state) -> TrainingResult:
    dataset = train_df[features + [target]].dropna().copy()
    if dataset.empty:
        raise ValueError("Training dataset is empty after dropping missing target rows.")

    if eval_df is None or eval_df.empty:
        X = dataset[features]
        y = dataset[target]
        stratify = y if task_type == "classification" and y.nunique(dropna=True) > 1 else None
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
        eval_frame = eval_df[features + [target]].dropna().copy()
        if eval_frame.empty:
            raise ValueError("Evaluation dataset is empty after filtering by time range.")

    preprocessor = build_preprocessor(train_frame, features)
    X_train = preprocessor.fit_transform(train_frame[features])
    X_eval = preprocessor.transform(eval_frame[features])
    model = get_model(model_name, task_type)

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
    train_frame = train_df[features + [target]].dropna().copy()
    if train_frame.empty:
        raise ValueError("Training dataset is empty after dropping missing target rows.")

    if eval_df is None or eval_df.empty:
        eval_frame = train_frame
    else:
        eval_frame = eval_df[features + [target]].dropna().copy()
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
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }
