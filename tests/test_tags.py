"""Tests for featherstore.tags module."""

import pytest
from pathlib import Path
from featherstore.tags import (
    add_tag,
    remove_tag,
    get_tags,
    find_groups_by_tag,
    clear_tags,
    load_tags,
    save_tags,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "test_store")


def test_load_tags_missing_returns_empty(store_path):
    result = load_tags(store_path)
    assert result == {}


def test_save_and_load_tags_roundtrip(store_path):
    data = {"features_a": ["raw", "daily"], "features_b": ["processed"]}
    save_tags(store_path, data)
    result = load_tags(store_path)
    assert result == data


def test_add_tag_creates_entry(store_path):
    add_tag(store_path, "user_features", "production")
    assert "production" in get_tags(store_path, "user_features")


def test_add_tag_no_duplicates(store_path):
    add_tag(store_path, "user_features", "production")
    add_tag(store_path, "user_features", "production")
    tags = get_tags(store_path, "user_features")
    assert tags.count("production") == 1


def test_add_multiple_tags(store_path):
    add_tag(store_path, "item_features", "raw")
    add_tag(store_path, "item_features", "daily")
    tags = get_tags(store_path, "item_features")
    assert set(tags) == {"raw", "daily"}


def test_remove_tag(store_path):
    add_tag(store_path, "user_features", "raw")
    add_tag(store_path, "user_features", "experimental")
    remove_tag(store_path, "user_features", "experimental")
    assert "experimental" not in get_tags(store_path, "user_features")
    assert "raw" in get_tags(store_path, "user_features")


def test_remove_nonexistent_tag_is_noop(store_path):
    add_tag(store_path, "user_features", "raw")
    remove_tag(store_path, "user_features", "nonexistent")
    assert get_tags(store_path, "user_features") == ["raw"]


def test_get_tags_unknown_group_returns_empty(store_path):
    result = get_tags(store_path, "no_such_group")
    assert result == []


def test_find_groups_by_tag(store_path):
    add_tag(store_path, "user_features", "production")
    add_tag(store_path, "item_features", "production")
    add_tag(store_path, "order_features", "experimental")
    groups = find_groups_by_tag(store_path, "production")
    assert set(groups) == {"user_features", "item_features"}


def test_find_groups_by_tag_no_match(store_path):
    add_tag(store_path, "user_features", "raw")
    groups = find_groups_by_tag(store_path, "production")
    assert groups == []


def test_clear_tags(store_path):
    add_tag(store_path, "user_features", "raw")
    add_tag(store_path, "user_features", "daily")
    clear_tags(store_path, "user_features")
    assert get_tags(store_path, "user_features") == []


def test_clear_tags_unknown_group_is_noop(store_path):
    clear_tags(store_path, "nonexistent_group")
    assert load_tags(store_path) == {}
