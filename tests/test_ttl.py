"""Tests for featherstore/ttl.py"""

import time
import pytest
from pathlib import Path

from featherstore.ttl import (
    load_ttl,
    save_ttl,
    set_ttl,
    remove_ttl,
    is_expired,
    list_expired,
    get_ttl,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_ttl_missing_returns_empty(store_path):
    result = load_ttl(store_path)
    assert result == {}


def test_save_and_load_ttl_roundtrip(store_path):
    data = {"features": {"group": "features", "expires_at": "2099-01-01T00:00:00+00:00"}}
    save_ttl(store_path, data)
    loaded = load_ttl(store_path)
    assert loaded == data


def test_set_ttl_creates_entry(store_path):
    entry = set_ttl(store_path, "my_group", expires_in_seconds=3600)
    assert entry["group"] == "my_group"
    assert entry["expires_in_seconds"] == 3600
    assert "expires_at" in entry
    assert "set_at" in entry


def test_set_ttl_persists(store_path):
    set_ttl(store_path, "my_group", expires_in_seconds=3600)
    ttl_data = load_ttl(store_path)
    assert "my_group" in ttl_data


def test_set_ttl_invalid_seconds_raises(store_path):
    with pytest.raises(ValueError, match="positive"):
        set_ttl(store_path, "my_group", expires_in_seconds=0)

    with pytest.raises(ValueError, match="positive"):
        set_ttl(store_path, "my_group", expires_in_seconds=-10)


def test_set_ttl_overwrites_existing(store_path):
    set_ttl(store_path, "my_group", expires_in_seconds=3600)
    set_ttl(store_path, "my_group", expires_in_seconds=7200)
    entry = get_ttl(store_path, "my_group")
    assert entry["expires_in_seconds"] == 7200


def test_remove_ttl_returns_true_if_existed(store_path):
    set_ttl(store_path, "my_group", expires_in_seconds=3600)
    result = remove_ttl(store_path, "my_group")
    assert result is True
    assert get_ttl(store_path, "my_group") is None


def test_remove_ttl_returns_false_if_missing(store_path):
    result = remove_ttl(store_path, "nonexistent")
    assert result is False


def test_is_expired_returns_none_if_no_ttl(store_path):
    assert is_expired(store_path, "no_ttl_group") is None


def test_is_expired_returns_false_for_future(store_path):
    set_ttl(store_path, "future_group", expires_in_seconds=9999)
    assert is_expired(store_path, "future_group") is False


def test_is_expired_returns_true_after_expiry(store_path):
    set_ttl(store_path, "past_group", expires_in_seconds=1)
    time.sleep(1.1)
    assert is_expired(store_path, "past_group") is True


def test_list_expired_returns_expired_groups(store_path):
    set_ttl(store_path, "alive", expires_in_seconds=9999)
    set_ttl(store_path, "dead", expires_in_seconds=1)
    time.sleep(1.1)
    expired = list_expired(store_path)
    assert "dead" in expired
    assert "alive" not in expired


def test_list_expired_empty_when_none_expired(store_path):
    set_ttl(store_path, "alive", expires_in_seconds=9999)
    assert list_expired(store_path) == []


def test_get_ttl_returns_none_for_missing(store_path):
    assert get_ttl(store_path, "ghost") is None
