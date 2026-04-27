"""Tests for featherstore/pin.py"""

import pytest
from pathlib import Path
from featherstore.pin import (
    load_pins,
    save_pins,
    pin_group,
    unpin_group,
    get_pin,
    list_pins,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "store")


def test_load_pins_missing_returns_empty(store_path):
    result = load_pins(store_path)
    assert result == {}


def test_save_and_load_pins_roundtrip(store_path):
    pins = {"features": {"group": "features", "version": "v1", "note": "", "pinned_at": "2024-01-01T00:00:00+00:00"}}
    save_pins(store_path, pins)
    loaded = load_pins(store_path)
    assert loaded == pins


def test_pin_group_creates_entry(store_path):
    entry = pin_group(store_path, "features", "v2", note="stable")
    assert entry["group"] == "features"
    assert entry["version"] == "v2"
    assert entry["note"] == "stable"
    assert "pinned_at" in entry


def test_pin_group_persists(store_path):
    pin_group(store_path, "labels", "v1")
    pins = load_pins(store_path)
    assert "labels" in pins
    assert pins["labels"]["version"] == "v1"


def test_pin_group_overwrites_existing(store_path):
    pin_group(store_path, "features", "v1")
    pin_group(store_path, "features", "v3", note="updated")
    pins = load_pins(store_path)
    assert pins["features"]["version"] == "v3"
    assert pins["features"]["note"] == "updated"


def test_get_pin_returns_entry(store_path):
    pin_group(store_path, "raw", "v5")
    result = get_pin(store_path, "raw")
    assert result is not None
    assert result["version"] == "v5"


def test_get_pin_returns_none_for_missing(store_path):
    result = get_pin(store_path, "nonexistent")
    assert result is None


def test_unpin_group_removes_entry(store_path):
    pin_group(store_path, "features", "v1")
    removed = unpin_group(store_path, "features")
    assert removed is True
    assert get_pin(store_path, "features") is None


def test_unpin_group_returns_false_if_not_pinned(store_path):
    result = unpin_group(store_path, "ghost_group")
    assert result is False


def test_list_pins_returns_all(store_path):
    pin_group(store_path, "a", "v1")
    pin_group(store_path, "b", "v2")
    pins = list_pins(store_path)
    groups = {p["group"] for p in pins}
    assert groups == {"a", "b"}


def test_list_pins_empty_store(store_path):
    result = list_pins(store_path)
    assert result == []
