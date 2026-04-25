"""Unit tests for featherstore.compare module."""

import pandas as pd
import pytest

from featherstore.compare import compare_groups, compare_numeric_stats


@pytest.fixture
def base_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})


@pytest.fixture
def modified_df():
    return pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [4.0, 5.0, 6.0, 7.0], "d": [10, 20, 30, 40]}
    )


def test_compare_groups_labels(base_df, modified_df):
    report = compare_groups(base_df, modified_df, label_a="v1", label_b="v2")
    assert report["labels"] == {"a": "v1", "b": "v2"}


def test_compare_groups_shape(base_df, modified_df):
    report = compare_groups(base_df, modified_df)
    assert report["shape"]["row_delta"] == 1
    assert report["shape"]["col_delta"] == 0  # both have 3 cols


def test_compare_groups_added_columns(base_df, modified_df):
    report = compare_groups(base_df, modified_df)
    assert "d" in report["columns"]["added"]


def test_compare_groups_removed_columns(base_df, modified_df):
    report = compare_groups(base_df, modified_df)
    assert "c" in report["columns"]["removed"]


def test_compare_groups_common_columns(base_df, modified_df):
    report = compare_groups(base_df, modified_df)
    assert set(report["columns"]["common"]) == {"a", "b"}


def test_compare_groups_diff_present(base_df, modified_df):
    report = compare_groups(base_df, modified_df)
    assert "diff" in report


def test_compare_identical_no_col_changes(base_df):
    report = compare_groups(base_df, base_df.copy())
    assert report["columns"]["added"] == []
    assert report["columns"]["removed"] == []
    assert report["shape"]["row_delta"] == 0


def test_compare_numeric_stats_returns_dataframe(base_df, modified_df):
    result = compare_numeric_stats(base_df, modified_df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_compare_numeric_stats_suffixes(base_df, modified_df):
    result = compare_numeric_stats(base_df, modified_df)
    assert any(col.endswith("_a") for col in result.columns)
    assert any(col.endswith("_b") for col in result.columns)


def test_compare_numeric_stats_column_filter(base_df, modified_df):
    result = compare_numeric_stats(base_df, modified_df, columns=["a"])
    assert all("a" in col for col in result.columns)


def test_compare_numeric_stats_no_shared_numeric():
    df_a = pd.DataFrame({"x": ["p", "q"]})
    df_b = pd.DataFrame({"y": ["r", "s"]})
    result = compare_numeric_stats(df_a, df_b)
    assert result.empty
