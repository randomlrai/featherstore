"""Integration tests for VisibilityMixin via FeatherStore."""

import pytest
import pandas as pd
from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})


def test_set_and_get_visibility(store, sample_df):
    store.save("features", sample_df)
    store.set_visibility("features", "public", note="shared dataset")
    entry = store.get_visibility("features")
    assert entry is not None
    assert entry["level"] == "public"
    assert entry["note"] == "shared dataset"


def test_get_visibility_none_for_unknown(store):
    assert store.get_visibility("nonexistent") is None


def test_remove_visibility(store, sample_df):
    store.save("features", sample_df)
    store.set_visibility("features", "private")
    store.remove_visibility("features")
    assert store.get_visibility("features") is None


def test_is_public_true(store, sample_df):
    store.save("features", sample_df)
    store.set_visibility("features", "public")
    assert store.is_public("features") is True


def test_is_public_false_for_private(store, sample_df):
    store.save("features", sample_df)
    store.set_visibility("features", "private")
    assert store.is_public("features") is False


def test_is_private_true(store, sample_df):
    store.save("features", sample_df)
    store.set_visibility("features", "private")
    assert store.is_private("features") is True


def test_list_by_visibility_after_multiple_saves(store, sample_df):
    store.save("a", sample_df)
    store.save("b", sample_df)
    store.save("c", sample_df)
    store.set_visibility("a", "internal")
    store.set_visibility("b", "public")
    store.set_visibility("c", "internal")
    internal = store.list_by_visibility("internal")
    assert set(internal) == {"a", "c"}


def test_set_visibility_invalid_level_raises(store, sample_df):
    store.save("features", sample_df)
    with pytest.raises(ValueError):
        store.set_visibility("features", "classified")
