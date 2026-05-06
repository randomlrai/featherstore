"""Tests for featherstore.category and CategoryMixin."""

import pytest
from featherstore.category import (
    load_categories,
    save_categories,
    set_category,
    remove_category,
    get_category,
    list_by_category,
    all_categories,
)


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def test_load_categories_missing_returns_empty(store_path):
    assert load_categories(store_path) == {}


def test_save_and_load_categories_roundtrip(store_path):
    data = {"users": "raw", "features": "processed"}
    save_categories(store_path, data)
    assert load_categories(store_path) == data


def test_set_category_creates_entry(store_path):
    set_category(store_path, "users", "raw")
    assert load_categories(store_path)["users"] == "raw"


def test_set_category_overwrites_existing(store_path):
    set_category(store_path, "users", "raw")
    set_category(store_path, "users", "processed")
    assert get_category(store_path, "users") == "processed"


def test_set_category_strips_whitespace(store_path):
    set_category(store_path, "users", "  raw  ")
    assert get_category(store_path, "users") == "raw"


def test_set_category_empty_string_raises(store_path):
    with pytest.raises(ValueError):
        set_category(store_path, "users", "")


def test_set_category_whitespace_only_raises(store_path):
    with pytest.raises(ValueError):
        set_category(store_path, "users", "   ")


def test_get_category_unknown_returns_none(store_path):
    assert get_category(store_path, "nonexistent") is None


def test_remove_category_returns_true(store_path):
    set_category(store_path, "users", "raw")
    assert remove_category(store_path, "users") is True
    assert get_category(store_path, "users") is None


def test_remove_category_missing_returns_false(store_path):
    assert remove_category(store_path, "ghost") is False


def test_list_by_category_returns_matching_groups(store_path):
    set_category(store_path, "users", "raw")
    set_category(store_path, "events", "raw")
    set_category(store_path, "features", "processed")
    result = list_by_category(store_path, "raw")
    assert sorted(result) == ["events", "users"]


def test_list_by_category_empty_when_none_match(store_path):
    set_category(store_path, "users", "raw")
    assert list_by_category(store_path, "processed") == []


def test_all_categories_returns_sorted_unique(store_path):
    set_category(store_path, "a", "raw")
    set_category(store_path, "b", "processed")
    set_category(store_path, "c", "raw")
    assert all_categories(store_path) == ["processed", "raw"]


def test_all_categories_empty_store(store_path):
    assert all_categories(store_path) == []
