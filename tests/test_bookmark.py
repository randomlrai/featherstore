"""Tests for featherstore.bookmark and BookmarkMixin integration."""

import pytest
import pandas as pd

from featherstore.bookmark import (
    load_bookmarks,
    add_bookmark,
    remove_bookmark,
    get_bookmark,
    list_bookmarks,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


# ── pure module tests ────────────────────────────────────────────────────────

def test_load_bookmarks_missing_returns_empty(store_path):
    assert load_bookmarks(store_path) == {}


def test_add_bookmark_creates_entry(store_path):
    entry = add_bookmark(store_path, "my_bm", "features/v1")
    assert entry["group"] == "features/v1"
    assert "created_at" in entry


def test_add_bookmark_persists(store_path):
    add_bookmark(store_path, "bm1", "group_a", note="hello")
    bms = load_bookmarks(store_path)
    assert "bm1" in bms
    assert bms["bm1"]["note"] == "hello"


def test_add_bookmark_overwrites_existing(store_path):
    add_bookmark(store_path, "bm", "group_a")
    add_bookmark(store_path, "bm", "group_b")
    assert load_bookmarks(store_path)["bm"]["group"] == "group_b"


def test_remove_bookmark_returns_true(store_path):
    add_bookmark(store_path, "bm", "group_a")
    assert remove_bookmark(store_path, "bm") is True
    assert get_bookmark(store_path, "bm") is None


def test_remove_bookmark_missing_returns_false(store_path):
    assert remove_bookmark(store_path, "nonexistent") is False


def test_get_bookmark_returns_none_when_missing(store_path):
    assert get_bookmark(store_path, "ghost") is None


def test_list_bookmarks_includes_name(store_path):
    add_bookmark(store_path, "alpha", "group_a")
    add_bookmark(store_path, "beta", "group_b")
    names = {bm["name"] for bm in list_bookmarks(store_path)}
    assert names == {"alpha", "beta"}


# ── FeatherStore integration tests ───────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    from featherstore.store import FeatherStore
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})


def test_store_bookmark_requires_existing_group(store, sample_df):
    with pytest.raises(KeyError, match="does not exist"):
        store.bookmark("bm", "missing_group")


def test_store_bookmark_and_list(store, sample_df):
    store.save("features", sample_df)
    store.bookmark("latest", "features", note="current best")
    bms = store.list_bookmarks()
    assert len(bms) == 1
    assert bms[0]["name"] == "latest"
    assert bms[0]["group"] == "features"


def test_store_load_bookmarked(store, sample_df):
    store.save("features", sample_df)
    store.bookmark("ref", "features")
    loaded = store.load_bookmarked("ref")
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), sample_df)


def test_store_load_bookmarked_missing_raises(store):
    with pytest.raises(KeyError, match="Bookmark 'ghost' not found"):
        store.load_bookmarked("ghost")


def test_store_unbookmark(store, sample_df):
    store.save("features", sample_df)
    store.bookmark("tmp", "features")
    assert store.unbookmark("tmp") is True
    assert store.get_bookmark("tmp") is None
