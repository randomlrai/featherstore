"""Integration tests for FreshnessMixin via FeatherStore."""

import time
import pytest
import pandas as pd
from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})


def test_touch_records_freshness(store):
    entry = store.touch("grp")
    assert "last_updated" in entry


def test_get_freshness_after_touch(store):
    store.touch("grp")
    info = store.get_freshness("grp")
    assert info is not None
    assert "last_updated" in info


def test_get_freshness_unknown_returns_none(store):
    assert store.get_freshness("nonexistent") is None


def test_is_stale_fresh_group(store):
    store.touch("grp")
    assert store.is_stale("grp", max_age_seconds=3600) is False


def test_is_stale_expired_group(store):
    store.touch("grp")
    time.sleep(0.05)
    assert store.is_stale("grp", max_age_seconds=0) is True


def test_is_stale_unknown_group(store):
    assert store.is_stale("ghost", max_age_seconds=60) is True


def test_remove_freshness(store):
    store.touch("grp")
    removed = store.remove_freshness("grp")
    assert removed is True
    assert store.get_freshness("grp") is None


def test_list_freshness_empty(store):
    assert store.list_freshness() == {}


def test_list_freshness_after_saves(store, sample_df):
    store.save(sample_df, "alpha")
    store.touch("alpha")
    store.touch("beta")
    listing = store.list_freshness()
    assert "alpha" in listing
    assert "beta" in listing
