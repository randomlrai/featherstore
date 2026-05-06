"""Integration tests for OwnershipMixin via FeatherStore."""

import pandas as pd
import pytest
from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})


def test_set_and_get_owner(store, sample_df):
    store.save("features", sample_df)
    store.set_owner("features", "alice", team="ml", email="alice@example.com")
    meta = store.get_owner("features")
    assert meta["owner"] == "alice"
    assert meta["team"] == "ml"
    assert meta["email"] == "alice@example.com"


def test_get_owner_none_for_unknown(store):
    assert store.get_owner("nonexistent") is None


def test_remove_owner(store, sample_df):
    store.save("features", sample_df)
    store.set_owner("features", "alice")
    assert store.remove_owner("features") is True
    assert store.get_owner("features") is None


def test_list_by_owner(store, sample_df):
    store.save("features", sample_df)
    store.save("labels", sample_df)
    store.save("scores", sample_df)
    store.set_owner("features", "alice")
    store.set_owner("labels", "alice")
    store.set_owner("scores", "bob")
    result = store.list_by_owner("alice")
    assert set(result) == {"features", "labels"}


def test_list_by_team(store, sample_df):
    store.save("features", sample_df)
    store.save("labels", sample_df)
    store.set_owner("features", "alice", team="ml")
    store.set_owner("labels", "bob", team="data-eng")
    result = store.list_by_team("ml")
    assert result == ["features"]


def test_remove_owner_false_when_not_set(store, sample_df):
    store.save("features", sample_df)
    assert store.remove_owner("features") is False
