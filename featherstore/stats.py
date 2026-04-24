"""Feature statistics and profiling for stored feature groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _stats_path(store_path: str) -> Path:
    return Path(store_path) / "stats.json"


def load_stats(store_path: str) -> dict[str, Any]:
    """Load all stored stats. Returns empty dict if file missing."""
    path = _stats_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_stats(store_path: str, stats: dict[str, Any]) -> None:
    """Persist stats to disk."""
    path = _stats_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for a DataFrame."""
    summary: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": {},
    }
    for col in df.columns:
        series = df[col]
        col_stats: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(series):
            col_stats["min"] = series.min()
            col_stats["max"] = series.max()
            col_stats["mean"] = round(float(series.mean()), 6)
            col_stats["std"] = round(float(series.std()), 6)
        else:
            col_stats["unique_count"] = int(series.nunique())
        summary["columns"][col] = col_stats
    return summary


def record_stats(store_path: str, group: str, df: pd.DataFrame) -> dict[str, Any]:
    """Compute and store stats for a feature group."""
    all_stats = load_stats(store_path)
    stats = compute_stats(df)
    all_stats[group] = stats
    save_stats(store_path, all_stats)
    return stats


def get_stats(store_path: str, group: str) -> dict[str, Any] | None:
    """Retrieve stored stats for a group. Returns None if not found."""
    all_stats = load_stats(store_path)
    return all_stats.get(group)
