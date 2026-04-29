"""Integration tests for QuotaMixin via FeatherStore."""

import pytest
import pandas as pd
from pathlib import Path

from featherstore.store import FeatherStore
from featherstore.quota import QuotaExceededError


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def test_set_and_get_quota(store):
    store.set_quota("features", 1024 * 1024)
    q = store.get_quota("features")
    assert q is not None
    assert q["max_bytes"] == 1024 * 1024


def test_get_quota_none_for_unknown(store):
    assert store.get_quota("unknown_group") is None


def test_remove_quota(store):
    store.set_quota("features", 512)
    assert store.remove_quota("features") is True
    assert store.get_quota("features") is None


def test_remove_quota_missing_returns_false(store):
    assert store.remove_quota("ghost") is False


def test_check_quota_after_save(store, sample_df):
    store.save(sample_df, "features")
    store.set_quota("features", 1024 * 1024)  # 1 MB — generous
    report = store.check_quota("features")
    assert report["group"] == "features"
    assert report["used_bytes"] > 0
    assert report["exceeded"] is False


def test_enforce_quota_passes_under_limit(store, sample_df):
    store.save(sample_df, "features")
    store.set_quota("features", 1024 * 1024)
    store.enforce_quota("features")  # should not raise


def test_enforce_quota_raises_over_limit(store, sample_df):
    store.save(sample_df, "features")
    store.set_quota("features", 1)  # 1 byte — impossible
    with pytest.raises(QuotaExceededError):
        store.enforce_quota("features")


def test_list_quotas_empty(store):
    assert store.list_quotas() == {}


def test_list_quotas_multiple(store):
    store.set_quota("features", 100)
    store.set_quota("labels", 200)
    quotas = store.list_quotas()
    assert "features" in quotas
    assert "labels" in quotas
