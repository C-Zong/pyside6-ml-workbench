from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NA_TEXT_VALUES = {"na", "n/a", "null", "none"}


def load_table(file_path: str) -> pd.DataFrame:
    """Load a table from CSV or Excel into a dataframe."""
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(file_path)
    if lower.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def save_table(df: pd.DataFrame, file_path: str) -> None:
    """Save a dataframe to CSV or Excel based on the target extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        df.to_csv(file_path, index=False)
        return
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df.to_excel(file_path, index=False)
        return
    raise ValueError(f"Unsupported export type: {file_path}")


def build_column_profile(df: pd.DataFrame, column: str) -> Dict:
    """Compute quality metrics and value distribution for a column."""
    series = df[column]
    stripped = series.astype("string").str.strip()
    rows = len(series)
    zero_mask = _build_zero_mask(series, stripped)
    empty_mask = stripped.fillna("").eq("")
    na_mask = series.isna() | stripped.str.lower().isin(NA_TEXT_VALUES)

    distribution = stripped.fillna("<NA>").replace("", "<EMPTY>").value_counts(dropna=False).head(20)
    return {
        "column": column,
        "dtype": str(series.dtype),
        "rows": int(rows),
        "zero_ratio": float(zero_mask.mean()) if rows else 0.0,
        "empty_ratio": float(empty_mask.mean()) if rows else 0.0,
        "na_ratio": float(na_mask.mean()) if rows else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "top_values": distribution.to_dict(),
    }


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    numeric_fill: str = "median",
    categorical_fill: str = "mode",
    trim_strings: bool = True,
) -> pd.DataFrame:
    """Apply the default cleaning routine to a dataframe."""
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
        fill_value = cleaned[col].median() if numeric_fill == "median" else cleaned[col].mean()
        cleaned[col] = cleaned[col].fillna(fill_value)

    for col in cat_cols:
        if categorical_fill == "mode":
            mode = cleaned[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        else:
            fill_value = "Unknown"
        cleaned[col] = cleaned[col].fillna(fill_value)

    return cleaned


def replace_column_with_zero(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Set an entire column to zero."""
    updated = df.copy()
    updated[column] = 0
    return updated


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove a single column from a dataframe."""
    return df.drop(columns=[column]).copy()


def delete_rows_by_spec(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    """Delete rows using a comma-separated list of indexes and ranges."""
    indexes = _parse_index_spec(spec)
    missing = [index for index in indexes if index < 0 or index >= len(df)]
    if missing:
        raise ValueError(f"Row index out of range: {missing[0]}")
    return df.drop(df.index[indexes]).reset_index(drop=True)


def drop_bad_rows(df: pd.DataFrame, columns: List[str], threshold: float) -> pd.DataFrame:
    """Drop rows whose bad-value ratio across selected columns meets the threshold."""
    if not columns:
        raise ValueError("Please choose at least one column for bad-row filtering.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    subset = df[columns]
    bad_mask = pd.DataFrame({column: _build_bad_mask(subset[column]) for column in columns})
    ratio = bad_mask.mean(axis=1)
    filtered = df.loc[ratio < threshold].reset_index(drop=True)
    return filtered


def create_formatted_column(
    df: pd.DataFrame,
    new_column: str,
    left_column: str,
    right_column: str,
    separator: str = "-",
    left_width: int = 0,
    right_width: int = 0,
) -> pd.DataFrame:
    """Create a new text column from two source columns with optional zero padding."""
    if not new_column.strip():
        raise ValueError("Please provide a new column name.")

    updated = df.copy()
    left_values = updated[left_column].astype("string").fillna("")
    right_values = updated[right_column].astype("string").fillna("")

    if left_width > 0:
        left_values = left_values.str.zfill(left_width)
    if right_width > 0:
        right_values = right_values.str.zfill(right_width)

    updated[new_column] = left_values + separator + right_values
    return updated


def split_column_by_delimiter_occurrence(
    df: pd.DataFrame,
    source_column: str,
    left_new_column: str,
    right_new_column: str,
    delimiter: str,
    occurrence: int,
) -> pd.DataFrame:
    """Split one column into two new columns at the chosen delimiter occurrence."""
    if not left_new_column.strip() or not right_new_column.strip():
        raise ValueError("Please provide both output column names.")
    if not delimiter:
        raise ValueError("Please provide a delimiter.")
    if occurrence < 1:
        raise ValueError("Occurrence must be at least 1.")

    updated = df.copy()
    series = updated[source_column].astype("string").fillna("")
    split_parts = series.apply(lambda value: _split_text_by_occurrence(value, delimiter, occurrence))
    updated[left_new_column] = split_parts.str[0]
    updated[right_new_column] = split_parts.str[1]
    return updated


def create_time_range_dataset(
    df: pd.DataFrame,
    time_column: str,
    start_text: str,
    end_text: str,
) -> pd.DataFrame:
    """Filter rows by a parsed time column and return the range as a new dataset."""
    if not time_column:
        raise ValueError("Please choose a Time Column.")

    series = pd.to_datetime(df[time_column], errors="coerce")
    if series.isna().all():
        raise ValueError("Selected Time Column could not be parsed as dates.")

    start = pd.to_datetime(start_text) if start_text else None
    end = pd.to_datetime(end_text) if end_text else None

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= series >= start
    if end is not None:
        mask &= series <= end

    filtered = df.loc[mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("The selected time range returned no rows.")
    return filtered


def merge_tables(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge two tables or append them when the schemas match."""
    if how == "append":
        if list(left_df.columns) != list(right_df.columns):
            raise ValueError("Append requires the two datasets to have the same columns in the same order.")
        return pd.concat([left_df, right_df], ignore_index=True)
    return pd.merge(left_df, right_df, left_on=left_key, right_on=right_key, how=how)


def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Infer whether a target column is classification or regression."""
    series = df[target_col]
    if pd.api.types.is_numeric_dtype(series):
        return "classification" if series.nunique(dropna=True) <= 20 else "regression"
    return "classification"


def split_feature_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    """Split feature names into numeric and categorical groups."""
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    return numeric_features, categorical_features


def _build_zero_mask(series: pd.Series, stripped: pd.Series) -> pd.Series:
    numeric_mask = pd.Series(False, index=series.index)
    if pd.api.types.is_numeric_dtype(series):
        numeric_mask = series.fillna(1).eq(0)
    return numeric_mask | stripped.eq("0")


def _build_bad_mask(series: pd.Series) -> pd.Series:
    stripped = series.astype("string").str.strip()
    return _build_zero_mask(series, stripped) | stripped.fillna("").eq("") | series.isna() | stripped.str.lower().isin(NA_TEXT_VALUES)


def _parse_index_spec(spec: str) -> List[int]:
    indexes: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if end < start:
                raise ValueError(f"Invalid range: {part}")
            indexes.update(range(start, end + 1))
        else:
            indexes.add(int(part))

    if not indexes:
        raise ValueError("Please provide at least one row index.")
    return sorted(indexes)


def _split_text_by_occurrence(value: str, delimiter: str, occurrence: int) -> tuple[str, str]:
    parts = value.split(delimiter)
    if len(parts) <= occurrence:
        return value, ""
    left = delimiter.join(parts[:occurrence])
    right = delimiter.join(parts[occurrence:])
    return left, right
