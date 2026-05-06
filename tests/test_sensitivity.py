"""Tests for featherstore/sensitivity.py and store_sensitivity.py."""

import pytest
from featherstore.sensitivity import (
    load_sensitivity,
    save_sensitivity,
    set_sensitivity,
    remove_sensitivity,
    get_sensitivity,
    list_by_sensitivity_level,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_sensitivity_missing_returns_empty(store_path):
    assert load_sensitivity(store_path) == {}


def test_save_and_load_sensitivity_roundtrip(store_path):
    data = {"features": {"level": "confidential", "note": "", "updated_at": "2024-01-01T00:00:00+00:00"}}
    save_sensitivity(store_path, data)
    loaded = load_sensitivity(store_path)
    assert loaded == data


def test_set_sensitivity_creates_entry(store_path):
    entry = set_sensitivity(store_path, "user_features", "internal")
    assert entry["level"] == "internal"
    assert "updated_at" in entry


def test_set_sensitivity_persists(store_path):
    set_sensitivity(store_path, "pii_data", "restricted", note="Contains PII")
    data = load_sensitivity(store_path)
    assert "pii_data" in data
    assert data["pii_data"]["level"] == "restricted"
    assert data["pii_data"]["note"] == "Contains PII"


def test_set_sensitivity_overwrites_existing(store_path):
    set_sensitivity(store_path, "labels", "public")
    set_sensitivity(store_path, "labels", "confidential")
    entry = get_sensitivity(store_path, "labels")
    assert entry["level"] == "confidential"


def test_set_sensitivity_invalid_level_raises(store_path):
    with pytest.raises(ValueError, match="Invalid sensitivity level"):
        set_sensitivity(store_path, "group", "top_secret")


def test_remove_sensitivity_returns_true(store_path):
    set_sensitivity(store_path, "g1", "internal")
    result = remove_sensitivity(store_path, "g1")
    assert result is True
    assert get_sensitivity(store_path, "g1") is None


def test_remove_sensitivity_missing_returns_false(store_path):
    result = remove_sensitivity(store_path, "nonexistent")
    assert result is False


def test_get_sensitivity_unknown_returns_none(store_path):
    assert get_sensitivity(store_path, "unknown") is None


def test_list_by_sensitivity_level(store_path):
    set_sensitivity(store_path, "a", "public")
    set_sensitivity(store_path, "b", "restricted")
    set_sensitivity(store_path, "c", "public")
    public_groups = list_by_sensitivity_level(store_path, "public")
    assert set(public_groups) == {"a", "c"}


def test_list_by_sensitivity_level_invalid_raises(store_path):
    with pytest.raises(ValueError, match="Invalid sensitivity level"):
        list_by_sensitivity_level(store_path, "ultra_secret")


def test_set_sensitivity_all_valid_levels(store_path):
    for i, level in enumerate(["public", "internal", "confidential", "restricted"]):
        set_sensitivity(store_path, f"group_{i}", level)
        entry = get_sensitivity(store_path, f"group_{i}")
        assert entry["level"] == level
