"""Data profiling utilities for FeatherStore groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _profile_path(store_path: str, group: str) -> Path:
    return Path(store_path) / group / "profile.json"


def load_profile(store_path: str, group: str) -> dict[str, Any]:
    path = _profile_path(store_path, group)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_profile(store_path: str, group: str, profile: dict[str, Any]) -> None:
    path = _profile_path(store_path, group)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)


def compute_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a rich profile of a DataFrame including dtypes, nulls, and cardinality."""
    profile: dict[str, Any] = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_info["min"] = float(series.min()) if not series.isna().all() else None
            col_info["max"] = float(series.max()) if not series.isna().all() else None
            col_info["mean"] = float(series.mean()) if not series.isna().all() else None
            col_info["std"] = float(series.std()) if not series.isna().all() else None
        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            non_null = series.dropna()
            col_info["min_length"] = int(non_null.str.len().min()) if len(non_null) else None
            col_info["max_length"] = int(non_null.str.len().max()) if len(non_null) else None

        profile["columns"][col] = col_info

    return profile


def record_profile(store_path: str, group: str, df: pd.DataFrame) -> dict[str, Any]:
    """Compute and persist a profile for the given group."""
    profile = compute_profile(df)
    save_profile(store_path, group, profile)
    return profile


def diff_profiles(
    old_profile: dict[str, Any], new_profile: dict[str, Any]
) -> dict[str, Any]:
    """Compare two profiles and return a summary of changes.

    Returns a dict with keys 'added_columns', 'removed_columns', and
    'changed_columns', where changed columns list fields whose values differ
    between the old and new profile.
    """
    old_cols = set(old_profile.get("columns", {}))
    new_cols = set(new_profile.get("columns", {}))

    added = sorted(new_cols - old_cols)
    removed = sorted(old_cols - new_cols)

    changed: dict[str, list[str]] = {}
    for col in old_cols & new_cols:
        old_info = old_profile["columns"][col]
        new_info = new_profile["columns"][col]
        differing_fields = [
            field
            for field in old_info.keys() | new_info.keys()
            if old_info.get(field) != new_info.get(field)
        ]
        if differing_fields:
            changed[col] = differing_fields

    return {
        "row_count_old": old_profile.get("row_count"),
        "row_count_new": new_profile.get("row_count"),
        "added_columns": added,
        "removed_columns": removed,
        "changed_columns": changed,
    }
