"""Tests for featherstore/priority.py"""

import pytest
from featherstore.priority import (
    load_priorities,
    save_priorities,
    set_priority,
    remove_priority,
    get_priority,
    list_by_priority,
    get_priority_order,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_priorities_missing_returns_empty(store_path):
    result = load_priorities(store_path)
    assert result == {}


def test_save_and_load_priorities_roundtrip(store_path):
    data = {"features": "high", "labels": "critical"}
    save_priorities(store_path, data)
    result = load_priorities(store_path)
    assert result == data


def test_set_priority_creates_entry(store_path):
    result = set_priority(store_path, "features", "high")
    assert result["features"] == "high"


def test_set_priority_persists(store_path):
    set_priority(store_path, "embeddings", "critical")
    assert get_priority(store_path, "embeddings") == "critical"


def test_set_priority_case_insensitive(store_path):
    set_priority(store_path, "raw", "HIGH")
    assert get_priority(store_path, "raw") == "high"


def test_set_priority_invalid_raises(store_path):
    with pytest.raises(ValueError, match="Invalid priority"):
        set_priority(store_path, "features", "urgent")


def test_set_priority_overwrites_existing(store_path):
    set_priority(store_path, "features", "low")
    set_priority(store_path, "features", "critical")
    assert get_priority(store_path, "features") == "critical"


def test_remove_priority_deletes_entry(store_path):
    set_priority(store_path, "features", "medium")
    remove_priority(store_path, "features")
    assert get_priority(store_path, "features") is None


def test_remove_priority_nonexistent_no_error(store_path):
    result = remove_priority(store_path, "nonexistent")
    assert isinstance(result, dict)


def test_get_priority_unknown_returns_none(store_path):
    assert get_priority(store_path, "ghost") is None


def test_list_by_priority_returns_matching_groups(store_path):
    set_priority(store_path, "features", "high")
    set_priority(store_path, "labels", "high")
    set_priority(store_path, "raw", "low")
    result = list_by_priority(store_path, "high")
    assert set(result) == {"features", "labels"}


def test_list_by_priority_empty_when_none_match(store_path):
    set_priority(store_path, "features", "low")
    result = list_by_priority(store_path, "critical")
    assert result == []


def test_get_priority_order_sorted_correctly(store_path):
    set_priority(store_path, "raw", "low")
    set_priority(store_path, "features", "high")
    set_priority(store_path, "targets", "critical")
    set_priority(store_path, "meta", "medium")
    order = get_priority_order(store_path)
    levels = [entry["priority"] for entry in order]
    assert levels == sorted(levels, key=["critical", "high", "medium", "low"].index)


def test_get_priority_order_returns_all_groups(store_path):
    set_priority(store_path, "a", "high")
    set_priority(store_path, "b", "low")
    order = get_priority_order(store_path)
    groups = {entry["group"] for entry in order}
    assert groups == {"a", "b"}
