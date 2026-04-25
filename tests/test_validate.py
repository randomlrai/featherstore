"""Tests for featherstore.validate and ValidateMixin integration."""

from __future__ import annotations

import pandas as pd
import pytest

from featherstore.validate import (
    extract_schema,
    load_schema,
    record_schema,
    save_schema,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})


# ---------------------------------------------------------------------------
# Unit tests — pure functions
# ---------------------------------------------------------------------------

def test_load_schema_missing_returns_empty(store_path):
    assert load_schema(store_path, "nonexistent") == {}


def test_save_and_load_schema_roundtrip(store_path, sample_df):
    schema = extract_schema(sample_df)
    save_schema(store_path, "grp", schema)
    loaded = load_schema(store_path, "grp")
    assert loaded["columns"] == schema["columns"]
    assert loaded["dtypes"] == schema["dtypes"]


def test_extract_schema_columns(sample_df):
    schema = extract_schema(sample_df)
    assert schema["columns"] == ["a", "b", "c"]


def test_extract_schema_dtypes(sample_df):
    schema = extract_schema(sample_df)
    assert "a" in schema["dtypes"]
    assert schema["dtypes"]["b"] == str(sample_df["b"].dtype)


def test_validate_schema_no_violations(sample_df):
    schema = extract_schema(sample_df)
    assert validate_schema(sample_df, schema) == []


def test_validate_schema_missing_column(sample_df):
    schema = extract_schema(sample_df)
    df_missing = sample_df.drop(columns=["c"])
    violations = validate_schema(df_missing, schema)
    assert any("Missing" in v for v in violations)


def test_validate_schema_extra_column(sample_df):
    schema = extract_schema(sample_df)
    df_extra = sample_df.copy()
    df_extra["d"] = 0
    violations = validate_schema(df_extra, schema)
    assert any("Unexpected" in v for v in violations)


def test_validate_schema_dtype_mismatch(sample_df):
    schema = extract_schema(sample_df)
    df_cast = sample_df.copy()
    df_cast["a"] = df_cast["a"].astype(float)
    violations = validate_schema(df_cast, schema)
    assert any("dtype mismatch" in v for v in violations)


def test_validate_schema_empty_expected(sample_df):
    """Empty schema should never produce violations."""
    assert validate_schema(sample_df, {}) == []


def test_record_schema_persists(store_path, sample_df):
    record_schema(store_path, "grp", sample_df)
    loaded = load_schema(store_path, "grp")
    assert loaded["columns"] == list(sample_df.columns)
