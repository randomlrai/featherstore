"""Tests for featherstore.watchlist and WatchlistMixin."""

import pytest
import pandas as pd

from featherstore.watchlist import (
    load_watchlist,
    save_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist_entry,
    list_watched,
    is_watched,
)
from featherstore.store import FeatherStore


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


# --- unit tests for watchlist.py ---

def test_load_watchlist_missing_returns_empty(store_path):
    assert load_watchlist(store_path) == {}


def test_save_and_load_watchlist_roundtrip(store_path):
    data = {"g1": {"group": "g1", "reason": "test", "tags": [], "added_at": "2024-01-01T00:00:00+00:00"}}
    save_watchlist(store_path, data)
    assert load_watchlist(store_path) == data


def test_add_to_watchlist_creates_entry(store_path):
    entry = add_to_watchlist(store_path, "features")
    assert entry["group"] == "features"
    assert "added_at" in entry


def test_add_to_watchlist_persists(store_path):
    add_to_watchlist(store_path, "features", reason="needs review")
    wl = load_watchlist(store_path)
    assert "features" in wl
    assert wl["features"]["reason"] == "needs review"


def test_add_to_watchlist_with_tags(store_path):
    entry = add_to_watchlist(store_path, "targets", tags=["important", "qa"])
    assert entry["tags"] == ["important", "qa"]


def test_add_to_watchlist_overwrites(store_path):
    add_to_watchlist(store_path, "g", reason="old")
    add_to_watchlist(store_path, "g", reason="new")
    assert load_watchlist(store_path)["g"]["reason"] == "new"


def test_remove_from_watchlist_returns_true(store_path):
    add_to_watchlist(store_path, "g")
    assert remove_from_watchlist(store_path, "g") is True
    assert "g" not in load_watchlist(store_path)


def test_remove_from_watchlist_missing_returns_false(store_path):
    assert remove_from_watchlist(store_path, "nonexistent") is False


def test_get_watchlist_entry_present(store_path):
    add_to_watchlist(store_path, "g", reason="check")
    entry = get_watchlist_entry(store_path, "g")
    assert entry is not None
    assert entry["reason"] == "check"


def test_get_watchlist_entry_missing_returns_none(store_path):
    assert get_watchlist_entry(store_path, "missing") is None


def test_list_watched_returns_all(store_path):
    add_to_watchlist(store_path, "a")
    add_to_watchlist(store_path, "b")
    entries = list_watched(store_path)
    assert len(entries) == 2
    assert {e["group"] for e in entries} == {"a", "b"}


def test_is_watched_true(store_path):
    add_to_watchlist(store_path, "g")
    assert is_watched(store_path, "g") is True


def test_is_watched_false(store_path):
    assert is_watched(store_path, "unknown") is False


# --- integration tests via FeatherStore ---

def test_store_watch_and_is_watched(store, sample_df):
    store.save(sample_df, "features")
    store.watch("features", reason="monitor drift")
    assert store.is_watched("features") is True


def test_store_unwatch(store, sample_df):
    store.save(sample_df, "features")
    store.watch("features")
    result = store.unwatch("features")
    assert result is True
    assert store.is_watched("features") is False


def test_store_list_watched(store, sample_df):
    store.save(sample_df, "a")
    store.save(sample_df, "b")
    store.watch("a")
    store.watch("b", tags=["critical"])
    entries = store.list_watched()
    assert len(entries) == 2


def test_store_get_watch_entry(store, sample_df):
    store.save(sample_df, "features")
    store.watch("features", reason="qa", tags=["review"])
    entry = store.get_watch_entry("features")
    assert entry["reason"] == "qa"
    assert "review" in entry["tags"]


def test_store_get_watch_entry_unknown_returns_none(store):
    assert store.get_watch_entry("ghost") is None
