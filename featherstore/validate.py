"""Schema validation utilities for FeatherStore groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _schema_path(store_path: str, group: str) -> Path:
    return Path(store_path) / group / "schema.json"


def load_schema(store_path: str, group: str) -> dict[str, Any]:
    """Load the saved schema for a group, or return empty dict if missing."""
    path = _schema_path(store_path, group)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_schema(store_path: str, group: str, schema: dict[str, Any]) -> None:
    """Persist the schema for a group to disk."""
    path = _schema_path(store_path, group)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def extract_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Extract column names and dtypes from a DataFrame as a serialisable schema."""
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def validate_schema(
    df: pd.DataFrame, expected: dict[str, Any]
) -> list[str]:
    """Compare a DataFrame against an expected schema.

    Returns a list of human-readable violation strings (empty if valid).
    """
    if not expected:
        return []

    violations: list[str] = []
    expected_cols: list[str] = expected.get("columns", [])
    expected_dtypes: dict[str, str] = expected.get("dtypes", {})

    actual_cols = list(df.columns)
    missing = [c for c in expected_cols if c not in actual_cols]
    extra = [c for c in actual_cols if c not in expected_cols]

    if missing:
        violations.append(f"Missing columns: {missing}")
    if extra:
        violations.append(f"Unexpected columns: {extra}")

    for col, exp_dtype in expected_dtypes.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if actual_dtype != exp_dtype:
                violations.append(
                    f"Column '{col}' dtype mismatch: expected {exp_dtype}, got {actual_dtype}"
                )

    return violations


def record_schema(store_path: str, group: str, df: pd.DataFrame) -> dict[str, Any]:
    """Extract and save a schema for the given group. Returns the schema."""
    schema = extract_schema(df)
    save_schema(store_path, group, schema)
    return schema
