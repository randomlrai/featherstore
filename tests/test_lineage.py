"""Tests for featherstore.lineage module."""

import pytest

from featherstore.lineage import (
    get_ancestors,
    get_lineage,
    load_lineage,
    record_lineage,
    save_lineage,
)


@pytest.fixture
def store_path(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------

def test_load_lineage_missing_returns_empty(store_path):
    assert load_lineage(store_path) == {}


def test_save_and_load_lineage_roundtrip(store_path):
    data = {"features_raw": {"source": "s3://bucket/raw", "parents": []}}
    save_lineage(store_path, data)
    loaded = load_lineage(store_path)
    assert loaded == data


# ---------------------------------------------------------------------------
# record_lineage
# ---------------------------------------------------------------------------

def test_record_lineage_creates_entry(store_path):
    entry = record_lineage(store_path, "features_v1", source="raw_table")
    assert entry["source"] == "raw_table"
    assert entry["parents"] == []
    assert entry["transform"] is None


def test_record_lineage_persists(store_path):
    record_lineage(store_path, "features_v1", source="raw_table")
    lineage = load_lineage(store_path)
    assert "features_v1" in lineage
    assert lineage["features_v1"]["source"] == "raw_table"


def test_record_lineage_stores_parents(store_path):
    record_lineage(store_path, "derived", parents=["features_v1", "features_v2"])
    entry = get_lineage(store_path, "derived")
    assert entry["parents"] == ["features_v1", "features_v2"]


def test_record_lineage_stores_extra(store_path):
    record_lineage(store_path, "g", extra={"pipeline": "etl_v3"})
    entry = get_lineage(store_path, "g")
    assert entry["extra"]["pipeline"] == "etl_v3"


def test_record_lineage_overwrites_existing(store_path):
    record_lineage(store_path, "g", source="old")
    record_lineage(store_path, "g", source="new")
    entry = get_lineage(store_path, "g")
    assert entry["source"] == "new"


# ---------------------------------------------------------------------------
# get_lineage
# ---------------------------------------------------------------------------

def test_get_lineage_returns_none_for_unknown(store_path):
    assert get_lineage(store_path, "nonexistent") is None


# ---------------------------------------------------------------------------
# get_ancestors
# ---------------------------------------------------------------------------

def test_get_ancestors_empty_for_root(store_path):
    record_lineage(store_path, "root")
    assert get_ancestors(store_path, "root") == []


def test_get_ancestors_single_level(store_path):
    record_lineage(store_path, "root")
    record_lineage(store_path, "child", parents=["root"])
    ancestors = get_ancestors(store_path, "child")
    assert "root" in ancestors


def test_get_ancestors_multi_level(store_path):
    record_lineage(store_path, "grandparent")
    record_lineage(store_path, "parent", parents=["grandparent"])
    record_lineage(store_path, "child", parents=["parent"])
    ancestors = get_ancestors(store_path, "child")
    assert "parent" in ancestors
    assert "grandparent" in ancestors


def test_get_ancestors_no_cycle_infinite_loop(store_path):
    """Cyclic references must not cause infinite recursion."""
    record_lineage(store_path, "a", parents=["b"])
    record_lineage(store_path, "b", parents=["a"])
    # Should terminate without error
    result = get_ancestors(store_path, "a")
    assert isinstance(result, list)
