from __future__ import annotations

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .data_ops import split_feature_types
from .models import get_model


class TrainingResult:
    def __init__(self, pipeline, metrics: Dict[str, float], importance_df: pd.DataFrame):
        self.pipeline = pipeline
        self.metrics = metrics
        self.importance_df = importance_df


def build_pipeline(df: pd.DataFrame, features: List[str], target: str, model_name: str, task_type: str) -> Pipeline:
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = get_model(model_name, task_type)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_feature_names(pipeline: Pipeline, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []
    feature_names.extend(numeric_features)

    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_names)

    return feature_names


def extract_importance(pipeline: Pipeline, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    numeric_features, categorical_features = split_feature_types(df, features)
    model = pipeline.named_steps["model"]
    feature_names = compute_feature_names(pipeline, numeric_features, categorical_features)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = np.abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({"feature": feature_names, "importance": values})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def train_model(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model_name: str,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    X = df[features]
    y = df[target]

    stratify = y if task_type == "classification" and y.nunique(dropna=True) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = build_pipeline(df, features, target, model_name, task_type)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        }
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(mean_squared_error(y_test, preds, squared=False)),
            "r2": float(r2_score(y_test, preds)),
        }

    importance_df = extract_importance(pipeline, df, features)
    return TrainingResult(pipeline, metrics, importance_df)
