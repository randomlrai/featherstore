"""Tests for featherstore.dedupe."""

import pandas as pd
import pytest

from featherstore.dedupe import dedupe_report, drop_duplicates, find_duplicates


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 2, 3, 3, 3],
            "score": [10, 20, 20, 30, 30, 30],
            "label": ["a", "b", "b", "c", "c", "c"],
        }
    )


# ---------------------------------------------------------------------------
# find_duplicates
# ---------------------------------------------------------------------------

def test_find_duplicates_default_keep_first(base_df):
    dupes = find_duplicates(base_df)
    # rows at index 2, 4, 5 are duplicates when keep='first'
    assert len(dupes) == 3
    assert list(dupes.index) == [2, 4, 5]


def test_find_duplicates_keep_false(base_df):
    dupes = find_duplicates(base_df, keep=False)
    # all rows that have any duplicate are returned
    assert len(dupes) == 5  # row 0 is unique; rows 1-5 all have duplicates


def test_find_duplicates_subset(base_df):
    dupes = find_duplicates(base_df, subset=["user_id"])
    # user_id 2 appears twice (1 dup), user_id 3 appears three times (2 dups)
    assert len(dupes) == 3


def test_find_duplicates_no_dupes():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert find_duplicates(df).empty


# ---------------------------------------------------------------------------
# drop_duplicates
# ---------------------------------------------------------------------------

def test_drop_duplicates_reduces_rows(base_df):
    clean = drop_duplicates(base_df)
    assert len(clean) == 3  # rows 0, 1, 3 remain


def test_drop_duplicates_keep_last(base_df):
    clean = drop_duplicates(base_df, keep="last")
    assert len(clean) == 3
    # last occurrence of user_id 3 is index 5
    assert 5 in clean.index


def test_drop_duplicates_subset(base_df):
    clean = drop_duplicates(base_df, subset=["user_id"])
    assert len(clean) == 3
    assert list(clean["user_id"]) == [1, 2, 3]


def test_drop_duplicates_returns_copy(base_df):
    clean = drop_duplicates(base_df)
    clean["score"] = 0
    assert base_df["score"].iloc[0] == 10  # original unchanged


# ---------------------------------------------------------------------------
# dedupe_report
# ---------------------------------------------------------------------------

def test_dedupe_report_counts(base_df):
    report = dedupe_report(base_df)
    assert report["total_rows"] == 6
    assert report["duplicate_rows"] == 3
    assert report["unique_rows"] == 3


def test_dedupe_report_rate(base_df):
    report = dedupe_report(base_df)
    assert abs(report["duplicate_rate"] - 0.5) < 1e-6


def test_dedupe_report_subset_key(base_df):
    report = dedupe_report(base_df, subset=["user_id"])
    assert report["subset"] == ["user_id"]


def test_dedupe_report_no_subset_is_none(base_df):
    report = dedupe_report(base_df)
    assert report["subset"] is None


def test_dedupe_report_empty_df():
    df = pd.DataFrame({"x": pd.Series([], dtype=int)})
    report = dedupe_report(df)
    assert report["total_rows"] == 0
    assert report["duplicate_rate"] == 0.0
