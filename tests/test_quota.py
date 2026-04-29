"""Unit tests for featherstore/quota.py."""

import pytest
from pathlib import Path

from featherstore.quota import (
    load_quotas,
    save_quotas,
    set_quota,
    remove_quota,
    get_quota,
    get_group_size,
    check_quota,
    enforce_quota,
    QuotaExceededError,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def _make_group_file(store_path, group, filename, size_bytes):
    group_dir = Path(store_path) / group
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / filename).write_bytes(b"x" * size_bytes)


def test_load_quotas_missing_returns_empty(store_path):
    assert load_quotas(store_path) == {}


def test_save_and_load_quotas_roundtrip(store_path):
    data = {"features": {"max_bytes": 1024}}
    save_quotas(store_path, data)
    loaded = load_quotas(store_path)
    assert loaded == data


def test_set_quota_creates_entry(store_path):
    result = set_quota(store_path, "features", 2048)
    assert result["max_bytes"] == 2048


def test_set_quota_persists(store_path):
    set_quota(store_path, "features", 4096)
    quotas = load_quotas(store_path)
    assert quotas["features"]["max_bytes"] == 4096


def test_set_quota_invalid_raises(store_path):
    with pytest.raises(ValueError, match="positive"):
        set_quota(store_path, "features", 0)


def test_set_quota_negative_raises(store_path):
    with pytest.raises(ValueError):
        set_quota(store_path, "features", -100)


def test_remove_quota_returns_true(store_path):
    set_quota(store_path, "features", 1024)
    assert remove_quota(store_path, "features") is True


def test_remove_quota_missing_returns_false(store_path):
    assert remove_quota(store_path, "nonexistent") is False


def test_remove_quota_deletes_entry(store_path):
    set_quota(store_path, "features", 1024)
    remove_quota(store_path, "features")
    assert get_quota(store_path, "features") is None


def test_get_quota_none_for_missing(store_path):
    assert get_quota(store_path, "features") is None


def test_get_group_size_missing_group(store_path):
    assert get_group_size(store_path, "ghost") == 0


def test_get_group_size_counts_bytes(store_path):
    _make_group_file(store_path, "features", "data.parquet", 500)
    _make_group_file(store_path, "features", "extra.parquet", 300)
    assert get_group_size(store_path, "features") == 800


def test_check_quota_no_quota_set(store_path):
    _make_group_file(store_path, "features", "data.parquet", 100)
    report = check_quota(store_path, "features")
    assert report["quota_bytes"] is None
    assert report["exceeded"] is False


def test_check_quota_under_limit(store_path):
    _make_group_file(store_path, "features", "data.parquet", 100)
    set_quota(store_path, "features", 1000)
    report = check_quota(store_path, "features")
    assert report["exceeded"] is False
    assert report["used_bytes"] == 100


def test_check_quota_over_limit(store_path):
    _make_group_file(store_path, "features", "data.parquet", 2000)
    set_quota(store_path, "features", 1000)
    report = check_quota(store_path, "features")
    assert report["exceeded"] is True


def test_enforce_quota_no_error_under_limit(store_path):
    _make_group_file(store_path, "features", "data.parquet", 50)
    set_quota(store_path, "features", 1000)
    enforce_quota(store_path, "features")  # should not raise


def test_enforce_quota_raises_when_exceeded(store_path):
    _make_group_file(store_path, "features", "data.parquet", 5000)
    set_quota(store_path, "features", 100)
    with pytest.raises(QuotaExceededError, match="features"):
        enforce_quota(store_path, "features")
