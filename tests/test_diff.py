"""Tests for featherstore.diff module."""

import pandas as pd
import pytest

from featherstore.diff import compute_diff


@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
        }
    )


def test_no_changes_returns_empty_diffs(base_df):
    result = compute_diff(base_df, base_df.copy())
    assert result["added_columns"] == []
    assert result["removed_columns"] == []
    assert result["row_delta"] == 0


def test_added_column_detected(base_df):
    new_df = base_df.copy()
    new_df["extra"] = 99
    result = compute_diff(base_df, new_df)
    assert "extra" in result["added_columns"]
    assert result["removed_columns"] == []


def test_removed_column_detected(base_df):
    new_df = base_df.drop(columns=["label"])
    result = compute_diff(base_df, new_df)
    assert "label" in result["removed_columns"]
    assert result["added_columns"] == []


def test_row_delta_positive(base_df):
    extra = pd.DataFrame({"id": [4], "value": [40.0], "label": ["d"]})
    new_df = pd.concat([base_df, extra], ignore_index=True)
    result = compute_diff(base_df, new_df)
    assert result["row_delta"] == 1
    assert result["row_count_new"] == 4


def test_row_delta_negative(base_df):
    new_df = base_df.iloc[:2].copy()
    result = compute_diff(base_df, new_df)
    assert result["row_delta"] == -1


def test_numeric_column_stats_mean_delta(base_df):
    new_df = base_df.copy()
    new_df["value"] = [20.0, 30.0, 40.0]
    result = compute_diff(base_df, new_df)
    stats = result["column_stats"]["value"]
    assert stats["old_mean"] == pytest.approx(20.0)
    assert stats["new_mean"] == pytest.approx(30.0)
    assert stats["mean_delta"] == pytest.approx(10.0)


def test_non_numeric_column_has_empty_stats(base_df):
    new_df = base_df.copy()
    new_df["label"] = ["x", "y", "z"]
    result = compute_diff(base_df, new_df)
    assert result["column_stats"]["label"] == {}


def test_rows_changed_with_key_column(base_df):
    new_df = base_df.copy()
    new_df.loc[0, "value"] = 999.0  # change one row
    result = compute_diff(base_df, new_df, key_column="id")
    assert result["rows_changed"] is not None
    assert result["rows_changed"] >= 1


def test_rows_changed_none_without_key(base_df):
    new_df = base_df.copy()
    result = compute_diff(base_df, new_df)
    assert result["rows_changed"] is None


def test_common_columns_sorted(base_df):
    result = compute_diff(base_df, base_df.copy())
    assert result["common_columns"] == sorted(base_df.columns.tolist())
