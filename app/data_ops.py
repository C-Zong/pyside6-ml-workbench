from __future__ import annotations

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np


SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv")


def load_table(file_path: str) -> pd.DataFrame:
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(file_path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def merge_tables(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = "inner",
) -> pd.DataFrame:
    return pd.merge(left_df, right_df, left_on=left_key, right_on=right_key, how=how)


def column_profile(df: pd.DataFrame, col: str) -> Dict:
    series = df[col]
    vc = series.astype("object").fillna("<NULL>").value_counts(dropna=False).head(20)
    profile = {
        "column": col,
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "null_count": int(series.isna().sum()),
        "null_ratio": float(series.isna().mean()) if len(series) else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "top_values": vc.to_dict(),
    }
    if pd.api.types.is_numeric_dtype(series):
        profile.update(
            {
                "min": None if series.dropna().empty else float(series.min()),
                "max": None if series.dropna().empty else float(series.max()),
                "mean": None if series.dropna().empty else float(series.mean()),
            }
        )
    return profile


def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    s = df[target_col]
    if pd.api.types.is_numeric_dtype(s):
        unique_n = s.nunique(dropna=True)
        if unique_n <= 20:
            return "classification"
        return "regression"
    return "classification"


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    numeric_fill: str = "median",
    categorical_fill: str = "mode",
    trim_strings: bool = True,
) -> pd.DataFrame:
    cleaned = df.copy()

    if trim_strings:
        obj_cols = cleaned.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            cleaned[col] = cleaned[col].astype("string").str.strip()
            cleaned[col] = cleaned[col].replace({"": pd.NA, "null": pd.NA, "NULL": pd.NA, "None": pd.NA})

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        if numeric_fill == "median":
            val = cleaned[col].median()
        else:
            val = cleaned[col].mean()
        cleaned[col] = cleaned[col].fillna(val)

    for col in cat_cols:
        if categorical_fill == "mode":
            mode = cleaned[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else "Unknown"
        else:
            fill_val = "Unknown"
        cleaned[col] = cleaned[col].fillna(fill_val)

    return cleaned


def split_feature_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    numeric_features = []
    categorical_features = []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    return numeric_features, categorical_features
