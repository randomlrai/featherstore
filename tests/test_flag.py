"""Tests for featherstore.flag and FlagMixin."""

import pytest
from pathlib import Path

from featherstore.flag import (
    load_flags,
    save_flags,
    set_flag,
    remove_flag,
    get_flags,
    is_flagged,
    list_flagged_groups,
    clear_flags,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------

def test_load_flags_missing_returns_empty(store_path):
    assert load_flags(store_path) == {}


def test_save_and_load_flags_roundtrip(store_path):
    data = {"group_a": {"experimental": True, "priority": 3}}
    save_flags(store_path, data)
    assert load_flags(store_path) == data


# ---------------------------------------------------------------------------
# set_flag
# ---------------------------------------------------------------------------

def test_set_flag_creates_entry(store_path):
    result = set_flag(store_path, "features", "experimental")
    assert result == {"experimental": True}


def test_set_flag_custom_value(store_path):
    set_flag(store_path, "features", "version", 42)
    flags = get_flags(store_path, "features")
    assert flags["version"] == 42


def test_set_flag_persists(store_path):
    set_flag(store_path, "g1", "ready")
    assert load_flags(store_path)["g1"]["ready"] is True


def test_set_flag_overwrites_existing(store_path):
    set_flag(store_path, "g1", "active", True)
    set_flag(store_path, "g1", "active", False)
    assert get_flags(store_path, "g1")["active"] is False


# ---------------------------------------------------------------------------
# remove_flag
# ---------------------------------------------------------------------------

def test_remove_flag_deletes_entry(store_path):
    set_flag(store_path, "g1", "beta")
    remove_flag(store_path, "g1", "beta")
    assert "beta" not in get_flags(store_path, "g1")


def test_remove_flag_cleans_empty_group(store_path):
    set_flag(store_path, "g1", "only")
    remove_flag(store_path, "g1", "only")
    assert "g1" not in load_flags(store_path)


def test_remove_flag_noop_if_missing(store_path):
    # Should not raise
    remove_flag(store_path, "nonexistent", "flag")


# ---------------------------------------------------------------------------
# get_flags / is_flagged
# ---------------------------------------------------------------------------

def test_get_flags_empty_for_unknown_group(store_path):
    assert get_flags(store_path, "ghost") == {}


def test_is_flagged_true(store_path):
    set_flag(store_path, "g1", "enabled")
    assert is_flagged(store_path, "g1", "enabled") is True


def test_is_flagged_false_when_absent(store_path):
    assert is_flagged(store_path, "g1", "missing") is False


def test_is_flagged_false_for_falsy_value(store_path):
    set_flag(store_path, "g1", "disabled", False)
    assert is_flagged(store_path, "g1", "disabled") is False


# ---------------------------------------------------------------------------
# list_flagged_groups
# ---------------------------------------------------------------------------

def test_list_flagged_groups_returns_matching(store_path):
    set_flag(store_path, "a", "experimental")
    set_flag(store_path, "b", "experimental")
    set_flag(store_path, "c", "stable")
    result = list_flagged_groups(store_path, "experimental")
    assert set(result) == {"a", "b"}


def test_list_flagged_groups_excludes_falsy(store_path):
    set_flag(store_path, "a", "ready", False)
    set_flag(store_path, "b", "ready", True)
    assert list_flagged_groups(store_path, "ready") == ["b"]


# ---------------------------------------------------------------------------
# clear_flags
# ---------------------------------------------------------------------------

def test_clear_flags_removes_all(store_path):
    set_flag(store_path, "g1", "x")
    set_flag(store_path, "g1", "y")
    clear_flags(store_path, "g1")
    assert get_flags(store_path, "g1") == {}


def test_clear_flags_noop_for_unknown_group(store_path):
    clear_flags(store_path, "ghost")  # should not raise
