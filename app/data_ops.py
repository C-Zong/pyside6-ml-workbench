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
        return pd.read_csv(file_path, low_memory=False)
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


def replace_bad_values_with_zero(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Replace null-like values in selected columns with zero."""
    if not columns:
        raise ValueError("Please choose at least one column to zero-fill.")

    updated = df.copy()
    for column in columns:
        series = updated[column]
        stripped = series.astype("string").str.strip()
        bad_mask = series.isna() | stripped.fillna("").eq("") | stripped.str.lower().isin(NA_TEXT_VALUES)
        updated.loc[bad_mask, column] = 0
    return updated


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove one or more columns from a dataframe."""
    if not columns:
        raise ValueError("Please choose at least one column to delete.")
    return df.drop(columns=columns).copy()


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


def coalesce_merged_columns(
    df: pd.DataFrame,
    strategy: str = "first_non_empty",
    left_key: str | None = None,
    right_key: str | None = None,
) -> pd.DataFrame:
    """Collapse merged columns and optionally merge the join keys into one column."""
    updated = df.copy()
    suffix_pairs = _find_merge_suffix_pairs(updated.columns)
    for base_name, left_name, right_name in suffix_pairs:
        updated[base_name] = _coalesce_pair(updated[left_name], updated[right_name], strategy)
        updated = updated.drop(columns=[left_name, right_name])
    if left_key and right_key and left_key != right_key and left_key in updated.columns and right_key in updated.columns:
        updated[left_key] = _coalesce_pair(updated[left_key], updated[right_key], strategy)
        updated = updated.drop(columns=[right_key])
    return updated


def aggregate_dataset(
    df: pd.DataFrame,
    group_key: str,
    specs: List[Dict[str, str]],
) -> pd.DataFrame:
    """Aggregate one or more value columns by key for merge-safe summaries."""
    if not group_key:
        raise ValueError("Please choose a Group Key.")
    if not specs:
        raise ValueError("Please add at least one aggregate rule.")

    agg_map: Dict[str, tuple[str, str]] = {}
    for spec in specs:
        value_column = spec.get("value_column", "")
        agg_function = spec.get("agg_function", "")
        output_column = spec.get("output_column", "")
        if not value_column or not agg_function:
            raise ValueError("Each aggregate rule needs a Value Column and Function.")
        if not output_column.strip():
            raise ValueError("Each aggregate rule needs an Output Column Name.")
        agg_map[output_column] = (value_column, agg_function)

    return df.groupby(group_key, dropna=False).agg(**agg_map).reset_index()


def convert_aggregate_column_to_binary(
    df: pd.DataFrame,
    source_column: str,
    threshold: float = 0,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Convert one aggregate result column into a 0/1 column."""
    if not source_column:
        raise ValueError("Please choose a Binary Source Column.")

    result_column = output_column.strip() if output_column else source_column
    updated = df.copy()
    numeric_values = pd.to_numeric(updated[source_column], errors="coerce").fillna(0)
    updated[result_column] = (numeric_values > threshold).astype(int)
    return updated


def summarize_merge_risk(left_df: pd.DataFrame, right_df: pd.DataFrame, left_key: str, right_key: str) -> Dict[str, object]:
    """Estimate whether a merge is likely to duplicate rows heavily."""
    if not left_key or not right_key:
        raise ValueError("Please choose both Left Key and Right Key.")

    left_rows = len(left_df)
    right_rows = len(right_df)
    left_unique = int(left_df[left_key].nunique(dropna=False))
    right_unique = int(right_df[right_key].nunique(dropna=False))
    left_duplicates = max(left_rows - left_unique, 0)
    right_duplicates = max(right_rows - right_unique, 0)
    left_is_unique = left_duplicates == 0
    right_is_unique = right_duplicates == 0

    if left_is_unique and right_is_unique:
        risk_level = "Low"
    elif left_is_unique or right_is_unique:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "left_rows": left_rows,
        "right_rows": right_rows,
        "left_unique": left_unique,
        "right_unique": right_unique,
        "left_duplicates": left_duplicates,
        "right_duplicates": right_duplicates,
        "left_is_unique": left_is_unique,
        "right_is_unique": right_is_unique,
        "risk_level": risk_level,
    }


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


def _find_merge_suffix_pairs(columns) -> List[tuple[str, str, str]]:
    pairs: List[tuple[str, str, str]] = []
    column_set = set(columns)
    for column in columns:
        if not str(column).endswith("_x"):
            continue
        base_name = str(column)[:-2]
        right_name = f"{base_name}_y"
        if right_name in column_set:
            pairs.append((base_name, str(column), right_name))
    return pairs


def _coalesce_pair(left: pd.Series, right: pd.Series, strategy: str) -> pd.Series:
    left_missing = _is_missing_series(left)
    right_missing = _is_missing_series(right)

    if strategy == "prefer_right":
        return right.where(~right_missing, left)
    if strategy == "prefer_left":
        return left.where(~left_missing, right)
    if strategy != "first_non_empty":
        raise ValueError(f"Unsupported coalesce strategy: {strategy}")
    return left.where(~left_missing, right.where(~right_missing, left))


def _is_missing_series(series: pd.Series) -> pd.Series:
    stripped = series.astype("string").str.strip()
    return series.isna() | stripped.fillna("").eq("") | stripped.str.lower().isin(NA_TEXT_VALUES)
