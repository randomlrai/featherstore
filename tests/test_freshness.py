"""Unit tests for featherstore/freshness.py."""

import time
import pytest
from pathlib import Path
from featherstore.freshness import (
    load_freshness,
    save_freshness,
    record_freshness,
    get_freshness,
    is_stale,
    remove_freshness,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_freshness_missing_returns_empty(store_path):
    assert load_freshness(store_path) == {}


def test_save_and_load_freshness_roundtrip(store_path):
    data = {"features": {"last_updated": "2024-01-01T00:00:00+00:00"}}
    save_freshness(store_path, data)
    assert load_freshness(store_path) == data


def test_record_freshness_creates_entry(store_path):
    entry = record_freshness(store_path, "my_group")
    assert "last_updated" in entry
    data = load_freshness(store_path)
    assert "my_group" in data


def test_record_freshness_overwrites_previous(store_path):
    record_freshness(store_path, "grp")
    time.sleep(0.05)
    entry2 = record_freshness(store_path, "grp")
    data = load_freshness(store_path)
    assert data["grp"]["last_updated"] == entry2["last_updated"]


def test_get_freshness_returns_entry(store_path):
    record_freshness(store_path, "grp")
    entry = get_freshness(store_path, "grp")
    assert entry is not None
    assert "last_updated" in entry


def test_get_freshness_unknown_returns_none(store_path):
    assert get_freshness(store_path, "ghost") is None


def test_is_stale_unknown_group_returns_true(store_path):
    assert is_stale(store_path, "missing", max_age_seconds=60) is True


def test_is_stale_fresh_group_returns_false(store_path):
    record_freshness(store_path, "grp")
    assert is_stale(store_path, "grp", max_age_seconds=3600) is False


def test_is_stale_expired_group_returns_true(store_path):
    record_freshness(store_path, "grp")
    # max_age of 0 seconds means immediately stale
    time.sleep(0.05)
    assert is_stale(store_path, "grp", max_age_seconds=0) is True


def test_remove_freshness_existing(store_path):
    record_freshness(store_path, "grp")
    result = remove_freshness(store_path, "grp")
    assert result is True
    assert get_freshness(store_path, "grp") is None


def test_remove_freshness_nonexistent_returns_false(store_path):
    assert remove_freshness(store_path, "ghost") is False
