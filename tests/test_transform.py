"""Tests for featherstore.transform pipeline utilities."""

import pytest
import pandas as pd
import numpy as np

from featherstore.transform import (
    TransformPipeline,
    drop_nulls,
    rename_columns,
    cast_columns,
    select_columns,
)


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": ["x", "y", "z", "w"],
        "c": [1.1, 2.2, 3.3, 4.4],
    })


# --- TransformPipeline ---

def test_empty_pipeline_returns_copy(base_df):
    pipe = TransformPipeline("empty")
    result = pipe.run(base_df)
    pd.testing.assert_frame_equal(result, base_df)
    assert result is not base_df


def test_add_step_increases_length(base_df):
    pipe = TransformPipeline("p")
    assert len(pipe) == 0
    pipe.add_step("drop", drop_nulls)
    assert len(pipe) == 1


def test_step_names_ordered(base_df):
    pipe = TransformPipeline("p")
    pipe.add_step("first", lambda df: df)
    pipe.add_step("second", lambda df: df)
    assert pipe.step_names == ["first", "second"]


def test_add_step_non_callable_raises():
    pipe = TransformPipeline("p")
    with pytest.raises(TypeError, match="must be callable"):
        pipe.add_step("bad", "not_a_function")


def test_run_none_raises():
    pipe = TransformPipeline("p")
    with pytest.raises(ValueError, match="must not be None"):
        pipe.run(None)


def test_run_step_exception_wrapped(base_df):
    def bad_step(df):
        raise ValueError("oops")

    pipe = TransformPipeline("p")
    pipe.add_step("bad", bad_step)
    with pytest.raises(RuntimeError, match="bad"):
        pipe.run(base_df)


def test_pipeline_chaining(base_df):
    pipe = (
        TransformPipeline("chain")
        .add_step("drop_nulls", drop_nulls)
        .add_step("rename", rename_columns({"a": "alpha"}))
    )
    result = pipe.run(base_df)
    assert "alpha" in result.columns
    assert result["alpha"].isna().sum() == 0


# --- Helper functions ---

def test_drop_nulls_removes_null_rows(base_df):
    result = drop_nulls(base_df)
    assert result.shape[0] == 3
    assert result["a"].isna().sum() == 0


def test_drop_nulls_subset(base_df):
    result = drop_nulls(base_df, subset=["b"])
    assert result.shape[0] == 4  # no nulls in "b"


def test_rename_columns(base_df):
    fn = rename_columns({"a": "alpha", "b": "beta"})
    result = fn(base_df)
    assert "alpha" in result.columns
    assert "beta" in result.columns
    assert "a" not in result.columns


def test_cast_columns(base_df):
    fn = cast_columns({"a": "float32"})
    result = fn(base_df.dropna())
    assert result["a"].dtype == np.float32


def test_cast_columns_ignores_missing():
    df = pd.DataFrame({"x": [1, 2]})
    fn = cast_columns({"x": "float64", "nonexistent": "int32"})
    result = fn(df)  # should not raise
    assert result["x"].dtype == np.float64


def test_select_columns(base_df):
    fn = select_columns(["a", "c"])
    result = fn(base_df)
    assert list(result.columns) == ["a", "c"]


def test_select_columns_missing_raises(base_df):
    fn = select_columns(["a", "missing"])
    with pytest.raises(KeyError, match="missing"):
        fn(base_df)
