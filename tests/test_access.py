"""Tests for featherstore/access.py"""

import pytest
from featherstore.access import (
    load_access,
    save_access,
    set_access,
    remove_access,
    get_access,
    can_read,
    can_write,
    list_by_principal,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_access_missing_returns_empty(store_path):
    assert load_access(store_path) == {}


def test_save_and_load_access_roundtrip(store_path):
    data = {"features": {"read": ["alice"], "write": ["bob"], "note": "", "updated_at": "2024-01-01T00:00:00+00:00"}}
    save_access(store_path, data)
    loaded = load_access(store_path)
    assert loaded == data


def test_set_access_creates_entry(store_path):
    entry = set_access(store_path, "my_group", read=["alice", "bob"], write=["alice"])
    assert entry["read"] == ["alice", "bob"]
    assert entry["write"] == ["alice"]
    assert "updated_at" in entry


def test_set_access_persists(store_path):
    set_access(store_path, "my_group", read=["alice"], write=["alice"])
    data = load_access(store_path)
    assert "my_group" in data


def test_set_access_deduplicates_principals(store_path):
    entry = set_access(store_path, "g", read=["alice", "alice", "bob"], write=["bob", "bob"])
    assert entry["read"].count("alice") == 1
    assert entry["write"].count("bob") == 1


def test_set_access_with_note(store_path):
    entry = set_access(store_path, "g", read=["*"], write=["admin"], note="public read")
    assert entry["note"] == "public read"


def test_set_access_overwrites_existing(store_path):
    set_access(store_path, "g", read=["alice"], write=["alice"])
    set_access(store_path, "g", read=["bob"], write=["bob"])
    entry = get_access(store_path, "g")
    assert entry["read"] == ["bob"]


def test_get_access_returns_none_for_unknown(store_path):
    assert get_access(store_path, "nonexistent") is None


def test_remove_access_deletes_entry(store_path):
    set_access(store_path, "g", read=["alice"], write=["alice"])
    result = remove_access(store_path, "g")
    assert result is True
    assert get_access(store_path, "g") is None


def test_remove_access_returns_false_for_unknown(store_path):
    assert remove_access(store_path, "ghost") is False


def test_can_read_no_acl_returns_true(store_path):
    assert can_read(store_path, "open_group", "anyone") is True


def test_can_write_no_acl_returns_true(store_path):
    assert can_write(store_path, "open_group", "anyone") is True


def test_can_read_with_explicit_principal(store_path):
    set_access(store_path, "g", read=["alice"], write=[])
    assert can_read(store_path, "g", "alice") is True
    assert can_read(store_path, "g", "bob") is False


def test_can_write_with_wildcard(store_path):
    set_access(store_path, "g", read=["*"], write=["*"])
    assert can_write(store_path, "g", "anyone") is True


def test_list_by_principal_returns_accessible_groups(store_path):
    set_access(store_path, "g1", read=["alice"], write=["alice"])
    set_access(store_path, "g2", read=["bob"], write=["bob"])
    set_access(store_path, "g3", read=["alice", "bob"], write=[])
    result = list_by_principal(store_path, "alice")
    assert "g1" in result
    assert "g2" not in result
    assert "g3" in result
    assert result["g1"]["write"] is True
    assert result["g3"]["write"] is False
