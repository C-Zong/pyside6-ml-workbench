from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure


def build_distribution_figure(df: pd.DataFrame, column: str) -> Figure:
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series.dropna(), bins=20)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
    else:
        vc = series.astype("object").fillna("<NULL>").value_counts(dropna=False).head(15)
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(f"Top values of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


def build_importance_figure(importance_df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    top_df = importance_df.head(15).sort_values("importance")
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig
