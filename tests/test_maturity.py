"""Tests for featherstore/maturity.py"""

import pytest
from pathlib import Path
from featherstore.maturity import (
    load_maturity,
    save_maturity,
    set_maturity,
    remove_maturity,
    get_maturity,
    list_by_level,
    VALID_LEVELS,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_maturity_missing_returns_empty(store_path):
    result = load_maturity(store_path)
    assert result == {}


def test_save_and_load_maturity_roundtrip(store_path):
    data = {"features_v1": {"level": "stable", "note": "", "updated_at": "2024-01-01T00:00:00+00:00"}}
    save_maturity(store_path, data)
    loaded = load_maturity(store_path)
    assert loaded == data


def test_set_maturity_creates_entry(store_path):
    meta = set_maturity(store_path, "features_v1", "stable")
    assert meta["level"] == "stable"
    assert "updated_at" in meta


def test_set_maturity_with_note(store_path):
    meta = set_maturity(store_path, "features_v1", "beta", note="needs more testing")
    assert meta["note"] == "needs more testing"


def test_set_maturity_persists(store_path):
    set_maturity(store_path, "features_v1", "experimental")
    loaded = load_maturity(store_path)
    assert "features_v1" in loaded
    assert loaded["features_v1"]["level"] == "experimental"


def test_set_maturity_overwrites_existing(store_path):
    set_maturity(store_path, "features_v1", "experimental")
    set_maturity(store_path, "features_v1", "stable", note="promoted")
    meta = get_maturity(store_path, "features_v1")
    assert meta["level"] == "stable"
    assert meta["note"] == "promoted"


def test_set_maturity_invalid_level_raises(store_path):
    with pytest.raises(ValueError, match="Invalid maturity level"):
        set_maturity(store_path, "features_v1", "unknown")


def test_remove_maturity_returns_true(store_path):
    set_maturity(store_path, "features_v1", "stable")
    result = remove_maturity(store_path, "features_v1")
    assert result is True


def test_remove_maturity_deletes_entry(store_path):
    set_maturity(store_path, "features_v1", "stable")
    remove_maturity(store_path, "features_v1")
    assert get_maturity(store_path, "features_v1") is None


def test_remove_maturity_missing_returns_false(store_path):
    result = remove_maturity(store_path, "nonexistent")
    assert result is False


def test_get_maturity_unknown_returns_none(store_path):
    assert get_maturity(store_path, "ghost") is None


def test_list_by_level_returns_matching_groups(store_path):
    set_maturity(store_path, "a", "stable")
    set_maturity(store_path, "b", "beta")
    set_maturity(store_path, "c", "stable")
    result = list_by_level(store_path, "stable")
    assert sorted(result) == ["a", "c"]


def test_list_by_level_empty_when_none_match(store_path):
    set_maturity(store_path, "a", "beta")
    result = list_by_level(store_path, "deprecated")
    assert result == []


def test_list_by_level_invalid_raises(store_path):
    with pytest.raises(ValueError, match="Invalid maturity level"):
        list_by_level(store_path, "garbage")


def test_all_valid_levels_accepted(store_path):
    for i, level in enumerate(VALID_LEVELS):
        meta = set_maturity(store_path, f"group_{i}", level)
        assert meta["level"] == level
