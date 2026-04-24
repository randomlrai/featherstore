"""Tests for featherstore.search module."""

import pytest
import pandas as pd
from featherstore.store import FeatherStore
from featherstore.tags import add_tag
from featherstore.search import search_catalog, list_groups


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path / "test_store"))


@pytest.fixture
def populated_store(store):
    df_users = pd.DataFrame({"user_id": [1, 2], "age": [25, 30]})
    df_items = pd.DataFrame({"item_id": [10, 20], "price": [5.0, 9.99]})
    df_orders = pd.DataFrame({"order_id": [100], "total": [14.99]})
    store.save(df_users, group="user_features")
    store.save(df_items, group="item_features")
    store.save(df_orders, group="order_features")
    add_tag(store.path, "user_features", "production")
    add_tag(store.path, "item_features", "production")
    add_tag(store.path, "item_features", "daily")
    add_tag(store.path, "order_features", "experimental")
    return store


def test_list_groups_returns_all(populated_store):
    results = list_groups(populated_store._catalog, populated_store.path)
    group_names = {r["group"] for r in results}
    assert group_names == {"user_features", "item_features", "order_features"}


def test_list_groups_includes_tags(populated_store):
    results = list_groups(populated_store._catalog, populated_store.path)
    by_name = {r["group"]: r for r in results}
    assert "production" in by_name["user_features"]["tags"]
    assert set(by_name["item_features"]["tags"]) == {"production", "daily"}


def test_search_by_tag(populated_store):
    results = search_catalog(
        populated_store._catalog, populated_store.path, tag="production"
    )
    group_names = {r["group"] for r in results}
    assert group_names == {"user_features", "item_features"}


def test_search_by_tag_no_match(populated_store):
    results = search_catalog(
        populated_store._catalog, populated_store.path, tag="archived"
    )
    assert results == []


def test_search_by_name_contains(populated_store):
    results = search_catalog(
        populated_store._catalog, populated_store.path, name_contains="item"
    )
    assert len(results) == 1
    assert results[0]["group"] == "item_features"


def test_search_by_name_contains_case_insensitive(populated_store):
    results = search_catalog(
        populated_store._catalog, populated_store.path, name_contains="ITEM"
    )
    assert len(results) == 1


def test_search_combined_tag_and_name(populated_store):
    results = search_catalog(
        populated_store._catalog,
        populated_store.path,
        tag="production",
        name_contains="user",
    )
    assert len(results) == 1
    assert results[0]["group"] == "user_features"


def test_search_empty_store(store):
    results = list_groups(store._catalog, store.path)
    assert results == []


def test_search_by_name_contains_no_match(populated_store):
    """Searching for a substring that matches no group name returns an empty list."""
    results = search_catalog(
        populated_store._catalog, populated_store.path, name_contains="nonexistent"
    )
    assert results == []
