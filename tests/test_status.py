"""Tests for featherstore/status.py"""

import pytest
from featherstore.status import (
    load_statuses,
    save_statuses,
    set_status,
    remove_status,
    get_status,
    list_by_status,
    VALID_STATUSES,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_statuses_missing_returns_empty(store_path):
    result = load_statuses(store_path)
    assert result == {}


def test_save_and_load_statuses_roundtrip(store_path):
    data = {"features": {"status": "active"}, "labels": {"status": "deprecated"}}
    save_statuses(store_path, data)
    result = load_statuses(store_path)
    assert result == data


def test_set_status_creates_entry(store_path):
    entry = set_status(store_path, "features", "active")
    assert entry["status"] == "active"
    statuses = load_statuses(store_path)
    assert "features" in statuses
    assert statuses["features"]["status"] == "active"


def test_set_status_with_reason(store_path):
    entry = set_status(store_path, "old_features", "deprecated", reason="replaced by v2")
    assert entry["reason"] == "replaced by v2"
    stored = load_statuses(store_path)["old_features"]
    assert stored["reason"] == "replaced by v2"


def test_set_status_no_reason_omits_key(store_path):
    entry = set_status(store_path, "features", "draft")
    assert "reason" not in entry


def test_set_status_invalid_raises(store_path):
    with pytest.raises(ValueError, match="Invalid status"):
        set_status(store_path, "features", "unknown_status")


def test_set_status_overwrites_existing(store_path):
    set_status(store_path, "features", "draft")
    set_status(store_path, "features", "active")
    result = get_status(store_path, "features")
    assert result["status"] == "active"


def test_remove_status_returns_true(store_path):
    set_status(store_path, "features", "active")
    removed = remove_status(store_path, "features")
    assert removed is True
    assert get_status(store_path, "features") is None


def test_remove_status_missing_returns_false(store_path):
    removed = remove_status(store_path, "nonexistent")
    assert removed is False


def test_get_status_unknown_returns_none(store_path):
    result = get_status(store_path, "ghost_group")
    assert result is None


def test_list_by_status_filters_correctly(store_path):
    set_status(store_path, "a", "active")
    set_status(store_path, "b", "deprecated")
    set_status(store_path, "c", "active")
    active = list_by_status(store_path, "active")
    assert set(active) == {"a", "c"}
    deprecated = list_by_status(store_path, "deprecated")
    assert deprecated == ["b"]


def test_list_by_status_empty_when_none_match(store_path):
    set_status(store_path, "a", "draft")
    result = list_by_status(store_path, "archived")
    assert result == []


def test_all_valid_statuses_accepted(store_path):
    for i, status in enumerate(VALID_STATUSES):
        set_status(store_path, f"group_{i}", status)
    statuses = load_statuses(store_path)
    assert len(statuses) == len(VALID_STATUSES)
