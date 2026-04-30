"""Unit tests for featherstore/retention.py."""

import pytest
from datetime import datetime, timedelta
from featherstore.retention import (
    load_retention,
    save_retention,
    set_retention,
    remove_retention,
    get_retention,
    is_expired,
    list_expired,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_retention_missing_returns_empty(store_path):
    assert load_retention(store_path) == {}


def test_save_and_load_retention_roundtrip(store_path):
    data = {"my_group": {"days": 7, "set_at": "2024-01-01", "expires_at": "2024-01-08"}}
    save_retention(store_path, data)
    assert load_retention(store_path) == data


def test_set_retention_creates_entry(store_path):
    policy = set_retention(store_path, "features", 30)
    assert policy["days"] == 30
    assert "set_at" in policy
    assert "expires_at" in policy


def test_set_retention_persists(store_path):
    set_retention(store_path, "features", 10)
    data = load_retention(store_path)
    assert "features" in data
    assert data["features"]["days"] == 10


def test_set_retention_invalid_days_raises(store_path):
    with pytest.raises(ValueError):
        set_retention(store_path, "features", 0)
    with pytest.raises(ValueError):
        set_retention(store_path, "features", -5)


def test_set_retention_overwrites_existing(store_path):
    set_retention(store_path, "features", 7)
    set_retention(store_path, "features", 14)
    assert load_retention(store_path)["features"]["days"] == 14


def test_get_retention_returns_policy(store_path):
    set_retention(store_path, "features", 5)
    policy = get_retention(store_path, "features")
    assert policy is not None
    assert policy["days"] == 5


def test_get_retention_unknown_returns_none(store_path):
    assert get_retention(store_path, "nonexistent") is None


def test_remove_retention_returns_true_when_exists(store_path):
    set_retention(store_path, "features", 7)
    assert remove_retention(store_path, "features") is True
    assert get_retention(store_path, "features") is None


def test_remove_retention_returns_false_when_missing(store_path):
    assert remove_retention(store_path, "ghost") is False


def test_is_expired_false_for_future_policy(store_path):
    set_retention(store_path, "features", 30)
    assert is_expired(store_path, "features") is False


def test_is_expired_true_for_past_policy(store_path):
    # Manually insert an already-expired policy
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    data = {"old_group": {"days": 1, "set_at": past, "expires_at": past}}
    save_retention(store_path, data)
    assert is_expired(store_path, "old_group") is True


def test_is_expired_false_for_no_policy(store_path):
    assert is_expired(store_path, "no_policy_group") is False


def test_list_expired_returns_correct_groups(store_path):
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    data = {
        "expired": {"days": 1, "set_at": past, "expires_at": past},
        "active": {"days": 30, "set_at": past,
                   "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat()},
    }
    save_retention(store_path, data)
    expired = list_expired(store_path)
    assert "expired" in expired
    assert "active" not in expired
