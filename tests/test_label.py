"""Tests for featherstore.label and LabelMixin."""

from __future__ import annotations

import pytest

from featherstore.label import (
    clear_labels,
    find_by_label,
    get_labels,
    load_labels,
    remove_label,
    save_labels,
    set_label,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------

def test_load_labels_missing_returns_empty(store_path):
    assert load_labels(store_path) == {}


def test_save_and_load_labels_roundtrip(store_path):
    data = {"my_group": {"env": "prod", "version": 3}}
    save_labels(store_path, data)
    assert load_labels(store_path) == data


# ---------------------------------------------------------------------------
# set_label
# ---------------------------------------------------------------------------

def test_set_label_creates_entry(store_path):
    result = set_label(store_path, "grp", "owner", "alice")
    assert result == {"owner": "alice"}


def test_set_label_persists(store_path):
    set_label(store_path, "grp", "env", "staging")
    assert get_labels(store_path, "grp") == {"env": "staging"}


def test_set_label_overwrites_existing(store_path):
    set_label(store_path, "grp", "env", "staging")
    set_label(store_path, "grp", "env", "prod")
    assert get_labels(store_path, "grp")["env"] == "prod"


def test_set_label_multiple_keys(store_path):
    set_label(store_path, "grp", "env", "prod")
    set_label(store_path, "grp", "tier", "gold")
    labels = get_labels(store_path, "grp")
    assert labels == {"env": "prod", "tier": "gold"}


# ---------------------------------------------------------------------------
# remove_label
# ---------------------------------------------------------------------------

def test_remove_label_deletes_key(store_path):
    set_label(store_path, "grp", "env", "prod")
    remove_label(store_path, "grp", "env")
    assert "env" not in get_labels(store_path, "grp")


def test_remove_label_noop_if_absent(store_path):
    # Should not raise
    remove_label(store_path, "grp", "nonexistent")


# ---------------------------------------------------------------------------
# clear_labels
# ---------------------------------------------------------------------------

def test_clear_labels_removes_all(store_path):
    set_label(store_path, "grp", "a", 1)
    set_label(store_path, "grp", "b", 2)
    clear_labels(store_path, "grp")
    assert get_labels(store_path, "grp") == {}


def test_clear_labels_noop_for_unknown_group(store_path):
    clear_labels(store_path, "ghost")  # should not raise


# ---------------------------------------------------------------------------
# find_by_label
# ---------------------------------------------------------------------------

def test_find_by_label_key_only(store_path):
    set_label(store_path, "a", "env", "prod")
    set_label(store_path, "b", "env", "staging")
    set_label(store_path, "c", "tier", "gold")
    result = find_by_label(store_path, "env")
    assert sorted(result) == ["a", "b"]


def test_find_by_label_key_and_value(store_path):
    set_label(store_path, "a", "env", "prod")
    set_label(store_path, "b", "env", "staging")
    result = find_by_label(store_path, "env", "prod")
    assert result == ["a"]


def test_find_by_label_no_match_returns_empty(store_path):
    set_label(store_path, "a", "env", "prod")
    assert find_by_label(store_path, "missing_key") == []
