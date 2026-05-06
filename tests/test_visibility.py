"""Unit tests for featherstore/visibility.py."""

import pytest
from featherstore.visibility import (
    load_visibility,
    save_visibility,
    set_visibility,
    remove_visibility,
    get_visibility,
    list_by_visibility,
    VALID_LEVELS,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_visibility_missing_returns_empty(store_path):
    result = load_visibility(store_path)
    assert result == {}


def test_save_and_load_visibility_roundtrip(store_path):
    data = {"features": {"level": "public", "note": "", "updated_at": "2024-01-01T00:00:00+00:00"}}
    save_visibility(store_path, data)
    loaded = load_visibility(store_path)
    assert loaded == data


def test_set_visibility_creates_entry(store_path):
    entry = set_visibility(store_path, "features", "public")
    assert entry["level"] == "public"
    assert "updated_at" in entry


def test_set_visibility_persists(store_path):
    set_visibility(store_path, "features", "private", note="sensitive")
    data = load_visibility(store_path)
    assert "features" in data
    assert data["features"]["level"] == "private"
    assert data["features"]["note"] == "sensitive"


def test_set_visibility_overwrites_existing(store_path):
    set_visibility(store_path, "features", "public")
    set_visibility(store_path, "features", "internal")
    entry = get_visibility(store_path, "features")
    assert entry["level"] == "internal"


def test_set_visibility_invalid_level_raises(store_path):
    with pytest.raises(ValueError, match="Invalid visibility level"):
        set_visibility(store_path, "features", "secret")


def test_remove_visibility_returns_true(store_path):
    set_visibility(store_path, "features", "public")
    result = remove_visibility(store_path, "features")
    assert result is True


def test_remove_visibility_removes_entry(store_path):
    set_visibility(store_path, "features", "public")
    remove_visibility(store_path, "features")
    assert get_visibility(store_path, "features") is None


def test_remove_visibility_missing_returns_false(store_path):
    result = remove_visibility(store_path, "nonexistent")
    assert result is False


def test_get_visibility_unknown_returns_none(store_path):
    assert get_visibility(store_path, "ghost") is None


def test_list_by_visibility_filters_correctly(store_path):
    set_visibility(store_path, "a", "public")
    set_visibility(store_path, "b", "private")
    set_visibility(store_path, "c", "public")
    public = list_by_visibility(store_path, "public")
    assert set(public) == {"a", "c"}


def test_list_by_visibility_empty_when_none_match(store_path):
    set_visibility(store_path, "a", "public")
    result = list_by_visibility(store_path, "internal")
    assert result == []


def test_list_by_visibility_invalid_level_raises(store_path):
    with pytest.raises(ValueError, match="Invalid visibility level"):
        list_by_visibility(store_path, "top-secret")


def test_valid_levels_constant():
    assert VALID_LEVELS == {"public", "private", "internal"}
