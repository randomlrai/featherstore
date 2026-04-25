"""Compare two saved feature groups across versions or snapshots."""

from __future__ import annotations

import pandas as pd
from typing import Optional

from featherstore.diff import compute_diff


def compare_groups(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "a",
    label_b: str = "b",
) -> dict:
    """Compare two DataFrames and return a structured comparison report.

    Args:
        df_a: First DataFrame (baseline).
        df_b: Second DataFrame (comparison target).
        label_a: Human-readable label for df_a.
        label_b: Human-readable label for df_b.

    Returns:
        A dict with keys: labels, shape, columns, diff.
    """
    diff = compute_diff(df_a, df_b)

    added_cols = [c for c in df_b.columns if c not in df_a.columns]
    removed_cols = [c for c in df_a.columns if c not in df_b.columns]
    common_cols = [c for c in df_a.columns if c in df_b.columns]

    return {
        "labels": {"a": label_a, "b": label_b},
        "shape": {
            label_a: df_a.shape,
            label_b: df_b.shape,
            "row_delta": df_b.shape[0] - df_a.shape[0],
            "col_delta": df_b.shape[1] - df_a.shape[1],
        },
        "columns": {
            "added": added_cols,
            "removed": removed_cols,
            "common": common_cols,
        },
        "diff": diff,
    }


def compare_numeric_stats(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return a side-by-side numeric summary for shared numeric columns.

    Args:
        df_a: Baseline DataFrame.
        df_b: Comparison DataFrame.
        columns: Optional list of columns to restrict comparison to.

    Returns:
        A DataFrame with describe() stats for each shared numeric column.
    """
    num_a = df_a.select_dtypes(include="number")
    num_b = df_b.select_dtypes(include="number")
    shared = [c for c in num_a.columns if c in num_b.columns]
    if columns:
        shared = [c for c in shared if c in columns]
    if not shared:
        return pd.DataFrame()

    desc_a = num_a[shared].describe().add_suffix("_a")
    desc_b = num_b[shared].describe().add_suffix("_b")
    return pd.concat([desc_a, desc_b], axis=1).sort_index(axis=1)
