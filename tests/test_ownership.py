"""Unit tests for featherstore.ownership."""

import pytest
from featherstore.ownership import (
    load_ownership,
    save_ownership,
    set_owner,
    remove_owner,
    get_owner,
    list_by_owner,
    list_by_team,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_ownership_missing_returns_empty(store_path):
    assert load_ownership(store_path) == {}


def test_save_and_load_ownership_roundtrip(store_path):
    data = {"features": {"owner": "alice", "team": "ml", "email": None, "set_at": "2024-01-01T00:00:00+00:00"}}
    save_ownership(store_path, data)
    assert load_ownership(store_path) == data


def test_set_owner_creates_entry(store_path):
    result = set_owner(store_path, "features", "alice")
    assert result["owner"] == "alice"
    assert result["team"] is None
    assert result["email"] is None
    assert "set_at" in result


def test_set_owner_with_team_and_email(store_path):
    result = set_owner(store_path, "labels", "bob", team="data-eng", email="bob@example.com")
    assert result["team"] == "data-eng"
    assert result["email"] == "bob@example.com"


def test_set_owner_persists(store_path):
    set_owner(store_path, "features", "carol")
    data = load_ownership(store_path)
    assert "features" in data
    assert data["features"]["owner"] == "carol"


def test_set_owner_overwrites_existing(store_path):
    set_owner(store_path, "features", "alice")
    set_owner(store_path, "features", "dave")
    assert get_owner(store_path, "features")["owner"] == "dave"


def test_get_owner_returns_none_for_unknown(store_path):
    assert get_owner(store_path, "nonexistent") is None


def test_remove_owner_returns_true_when_exists(store_path):
    set_owner(store_path, "features", "alice")
    assert remove_owner(store_path, "features") is True


def test_remove_owner_returns_false_when_missing(store_path):
    assert remove_owner(store_path, "ghost") is False


def test_remove_owner_deletes_entry(store_path):
    set_owner(store_path, "features", "alice")
    remove_owner(store_path, "features")
    assert get_owner(store_path, "features") is None


def test_list_by_owner_returns_matching_groups(store_path):
    set_owner(store_path, "features", "alice")
    set_owner(store_path, "labels", "alice")
    set_owner(store_path, "scores", "bob")
    groups = list_by_owner(store_path, "alice")
    assert set(groups) == {"features", "labels"}


def test_list_by_owner_empty_when_none_match(store_path):
    set_owner(store_path, "features", "alice")
    assert list_by_owner(store_path, "nobody") == []


def test_list_by_team_returns_matching_groups(store_path):
    set_owner(store_path, "features", "alice", team="ml")
    set_owner(store_path, "labels", "bob", team="ml")
    set_owner(store_path, "scores", "carol", team="data-eng")
    groups = list_by_team(store_path, "ml")
    assert set(groups) == {"features", "labels"}


def test_list_by_team_empty_when_none_match(store_path):
    set_owner(store_path, "features", "alice", team="ml")
    assert list_by_team(store_path, "unknown-team") == []
