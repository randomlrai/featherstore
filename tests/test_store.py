"""Tests for featherstore.store.FeatherStore."""

import pytest
import pandas as pd

from featherstore.store import FeatherStore


@pytest.fixture()
def store(tmp_path):
    """Provide a fresh FeatherStore backed by a temp directory."""
    fs = FeatherStore(store_path=str(tmp_path / "test_store"))
    yield fs
    fs.close()


@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "age": [25, 32, 41],
            "spend_30d": [120.5, 340.0, 89.99],
        }
    )


def test_save_and_load_roundtrip(store, sample_df):
    store.save("user_features", sample_df, description="Basic user features")
    loaded = store.load("user_features")
    pd.testing.assert_frame_equal(sample_df, loaded)


def test_load_with_column_selection(store, sample_df):
    store.save("user_features", sample_df)
    loaded = store.load("user_features", columns=["user_id", "age"])
    assert list(loaded.columns) == ["user_id", "age"]
    assert len(loaded) == 3


def test_load_unknown_group_raises(store):
    with pytest.raises(KeyError, match="not found in store"):
        store.load("nonexistent")


def test_list_groups_empty(store):
    result = store.list_groups()
    assert len(result) == 0
    assert "name" in result.columns


def test_list_groups_after_save(store, sample_df):
    store.save("user_features", sample_df, description="test desc")
    store.save("item_features", sample_df)
    groups = store.list_groups()
    assert set(groups["name"]) == {"user_features", "item_features"}


def test_save_overwrites_existing(store, sample_df):
    store.save("user_features", sample_df)
    updated_df = sample_df.copy()
    updated_df["age"] = [99, 99, 99]
    store.save("user_features", updated_df)
    loaded = store.load("user_features")
    assert list(loaded["age"]) == [99, 99, 99]
    # Catalog should still have only one entry
    groups = store.list_groups()
    assert len(groups[groups["name"] == "user_features"]) == 1


def test_delete_removes_group(store, sample_df, tmp_path):
    store.save("user_features", sample_df)
    store.delete("user_features")
    assert len(store.list_groups()) == 0
    with pytest.raises(KeyError):
        store.load("user_features")


def test_delete_unknown_group_raises(store):
    with pytest.raises(KeyError, match="not found in store"):
        store.delete("ghost_group")


def test_repr(store):
    assert "FeatherStore" in repr(store)
