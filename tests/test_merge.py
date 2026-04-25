"""Unit tests for featherstore.merge (pure function layer)."""

import pandas as pd
import pytest

from featherstore.merge import merge_groups, merge_on_index


@pytest.fixture
def left():
    return pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})


@pytest.fixture
def right():
    return pd.DataFrame({"id": [1, 2, 4], "b": [100, 200, 400]})


@pytest.fixture
def extra():
    return pd.DataFrame({"id": [1, 2, 3], "c": ["x", "y", "z"]})


def test_merge_groups_inner(left, right):
    result = merge_groups([left, right], on=["id"], how="inner")
    assert list(result["id"]) == [1, 2]
    assert "a" in result.columns and "b" in result.columns


def test_merge_groups_outer(left, right):
    result = merge_groups([left, right], on=["id"], how="outer")
    assert len(result) == 4  # ids 1, 2, 3, 4


def test_merge_groups_auto_detects_shared_key(left, extra):
    # 'id' is the shared column; 'a' and 'c' are unique
    result = merge_groups([left, extra])  # on=None → auto-detect
    assert "a" in result.columns
    assert "c" in result.columns
    assert len(result) == 3


def test_merge_groups_no_common_columns_raises():
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"b": [2]})
    with pytest.raises(ValueError, match="No common columns"):
        merge_groups([df1, df2])


def test_merge_groups_requires_two_frames(left):
    with pytest.raises(ValueError, match="at least two"):
        merge_groups([left])


def test_merge_groups_invalid_how_raises(left, right):
    with pytest.raises(ValueError, match="Unsupported join strategy"):
        merge_groups([left, right], on=["id"], how="cross")


def test_merge_three_frames(left, right, extra):
    result = merge_groups([left, right, extra], on=["id"], how="inner")
    assert set(result.columns) == {"id", "a", "b", "c"}
    assert len(result) == 2  # ids 1 and 2 are common to all three


def test_merge_on_index():
    df1 = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"b": [10, 20, 30]}, index=[0, 1, 2])
    result = merge_on_index([df1, df2])
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 3


def test_merge_on_index_requires_two_frames():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="at least two"):
        merge_on_index([df])


def test_no_duplicate_columns_after_merge(left, right):
    result = merge_groups([left, right], on=["id"], how="outer")
    assert not any(c.endswith("_dup") for c in result.columns)
