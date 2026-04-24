"""Utilities for computing diffs between feature group versions."""

from __future__ import annotations

import pandas as pd
from typing import Optional


def compute_diff(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    key_column: Optional[str] = None,
) -> dict:
    """Compute a structural and statistical diff between two DataFrames.

    Args:
        df_old: The previous version of the feature group.
        df_new: The new version of the feature group.
        key_column: Optional column to use as a row key for row-level diff.

    Returns:
        A dict summarising added/removed columns, row count changes,
        and per-column value change rates.
    """
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)

    added_columns = sorted(new_cols - old_cols)
    removed_columns = sorted(old_cols - new_cols)
    common_columns = sorted(old_cols & new_cols)

    row_delta = len(df_new) - len(df_old)

    column_stats: dict[str, dict] = {}
    for col in common_columns:
        try:
            if pd.api.types.is_numeric_dtype(df_old[col]) and pd.api.types.is_numeric_dtype(df_new[col]):
                old_mean = float(df_old[col].mean())
                new_mean = float(df_new[col].mean())
                column_stats[col] = {
                    "old_mean": old_mean,
                    "new_mean": new_mean,
                    "mean_delta": new_mean - old_mean,
                }
            else:
                column_stats[col] = {}
        except Exception:
            column_stats[col] = {}

    rows_changed: Optional[int] = None
    if key_column and key_column in df_old.columns and key_column in df_new.columns:
        merged = df_old.set_index(key_column).join(
            df_new.set_index(key_column),
            how="inner",
            lsuffix="_old",
            rsuffix="_new",
        )
        changed = 0
        for col in common_columns:
            if col == key_column:
                continue
            old_c = f"{col}_old"
            new_c = f"{col}_new"
            if old_c in merged.columns and new_c in merged.columns:
                changed += int((merged[old_c] != merged[new_c]).sum())
        rows_changed = changed

    return {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "common_columns": common_columns,
        "row_count_old": len(df_old),
        "row_count_new": len(df_new),
        "row_delta": row_delta,
        "rows_changed": rows_changed,
        "column_stats": column_stats,
    }
