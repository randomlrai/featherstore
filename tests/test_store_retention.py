"""Integration tests for RetentionMixin via FeatherStore."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from featherstore.store import FeatherStore
from featherstore.retention import save_retention


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})


def test_set_and_get_retention(store, sample_df):
    store.save(sample_df, "features")
    policy = store.set_retention("features", 7)
    assert policy["days"] == 7
    fetched = store.get_retention("features")
    assert fetched is not None
    assert fetched["days"] == 7


def test_get_retention_none_for_unknown(store):
    assert store.get_retention("nonexistent") is None


def test_remove_retention(store, sample_df):
    store.save(sample_df, "features")
    store.set_retention("features", 10)
    assert store.remove_retention("features") is True
    assert store.get_retention("features") is None


def test_is_expired_false_for_fresh_policy(store, sample_df):
    store.save(sample_df, "features")
    store.set_retention("features", 30)
    assert store.is_expired("features") is False


def test_is_expired_true_for_past_expiry(store, sample_df):
    store.save(sample_df, "features")
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    data = {"features": {"days": 1, "set_at": past, "expires_at": past}}
    save_retention(store.store_path, data)
    assert store.is_expired("features") is True


def test_list_expired_returns_expired_groups(store, sample_df):
    store.save(sample_df, "features")
    store.save(sample_df, "labels")
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    future = (datetime.utcnow() + timedelta(days=30)).isoformat()
    data = {
        "features": {"days": 1, "set_at": past, "expires_at": past},
        "labels": {"days": 30, "set_at": past, "expires_at": future},
    }
    save_retention(store.store_path, data)
    expired = store.list_expired()
    assert "features" in expired
    assert "labels" not in expired
