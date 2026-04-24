"""Tests for featherstore.stats module."""

import pytest
import pandas as pd
import numpy as np

from featherstore.stats import (
    load_stats,
    save_stats,
    compute_stats,
    record_stats,
    get_stats,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, np.nan],
            "score": [0.1, 0.5, 0.9, 0.7],
            "category": ["a", "b", "a", "c"],
        }
    )


def test_load_stats_missing_returns_empty(store_path):
    result = load_stats(store_path)
    assert result == {}


def test_save_and_load_stats_roundtrip(store_path):
    data = {"group_a": {"row_count": 10, "column_count": 3}}
    save_stats(store_path, data)
    loaded = load_stats(store_path)
    assert loaded["group_a"]["row_count"] == 10


def test_compute_stats_row_and_col_count(sample_df):
    stats = compute_stats(sample_df)
    assert stats["row_count"] == 4
    assert stats["column_count"] == 3


def test_compute_stats_numeric_fields(sample_df):
    stats = compute_stats(sample_df)
    age_stats = stats["columns"]["age"]
    assert "min" in age_stats
    assert "max" in age_stats
    assert "mean" in age_stats
    assert "std" in age_stats


def test_compute_stats_null_count(sample_df):
    stats = compute_stats(sample_df)
    assert stats["columns"]["age"]["null_count"] == 1
    assert stats["columns"]["score"]["null_count"] == 0


def test_compute_stats_categorical_unique_count(sample_df):
    stats = compute_stats(sample_df)
    cat_stats = stats["columns"]["category"]
    assert cat_stats["unique_count"] == 3
    assert "min" not in cat_stats


def test_record_stats_persists(store_path, sample_df):
    record_stats(store_path, "features", sample_df)
    all_stats = load_stats(store_path)
    assert "features" in all_stats
    assert all_stats["features"]["row_count"] == 4


def test_get_stats_returns_none_for_missing(store_path):
    result = get_stats(store_path, "nonexistent")
    assert result is None


def test_get_stats_returns_stored(store_path, sample_df):
    record_stats(store_path, "my_group", sample_df)
    result = get_stats(store_path, "my_group")
    assert result is not None
    assert result["column_count"] == 3


def test_record_stats_overwrites_existing(store_path):
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    record_stats(store_path, "grp", df1)
    record_stats(store_path, "grp", df2)
    result = get_stats(store_path, "grp")
    assert result["row_count"] == 5
