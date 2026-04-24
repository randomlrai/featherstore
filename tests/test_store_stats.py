"""Integration tests: FeatherStore exposes stats after save."""

import pytest
import pandas as pd
import numpy as np

from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
            "label": ["x", "y", "x", "y"],
        }
    )


def test_stats_none_for_unsaved_group(store):
    result = store.get_stats("nonexistent")
    assert result is None


def test_stats_available_after_save(store, sample_df):
    store.save("my_features", sample_df)
    stats = store.get_stats("my_features")
    assert stats is not None
    assert stats["row_count"] == 4
    assert stats["column_count"] == 3


def test_stats_numeric_summary(store, sample_df):
    store.save("my_features", sample_df)
    stats = store.get_stats("my_features")
    col = stats["columns"]["feature_a"]
    assert col["min"] == 1.0
    assert col["max"] == 4.0


def test_stats_updated_on_overwrite(store):
    df1 = pd.DataFrame({"v": [1, 2]})
    df2 = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
    store.save("grp", df1)
    store.save("grp", df2)
    stats = store.get_stats("grp")
    assert stats["row_count"] == 5


def test_stats_null_count_tracked(store):
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    store.save("nulls", df)
    stats = store.get_stats("nulls")
    assert stats["columns"]["a"]["null_count"] == 1
