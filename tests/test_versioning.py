"""Tests for featherstore.versioning module."""

import json
from pathlib import Path

import pytest

from featherstore.versioning import (
    get_latest_version,
    get_version_history,
    load_version_manifest,
    record_version,
    save_version_manifest,
)


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "test_store"


def test_load_version_manifest_missing(store_path: Path) -> None:
    """Loading from a non-existent path returns an empty dict."""
    assert load_version_manifest(store_path) == {}


def test_save_and_load_manifest_roundtrip(store_path: Path) -> None:
    manifest = {"features": [{"version": 1, "row_count": 10, "columns": ["a"]}]}
    save_version_manifest(store_path, manifest)
    loaded = load_version_manifest(store_path)
    assert loaded == manifest


def test_record_version_increments(store_path: Path) -> None:
    v1 = record_version(store_path, "user_features", 100, ["age", "spend"])
    v2 = record_version(store_path, "user_features", 120, ["age", "spend", "clicks"])
    assert v1 == 1
    assert v2 == 2


def test_record_version_persists_fields(store_path: Path) -> None:
    record_version(
        store_path,
        "item_features",
        50,
        ["price", "category"],
        metadata={"source": "etl_v2"},
    )
    history = get_version_history(store_path, "item_features")
    assert len(history) == 1
    entry = history[0]
    assert entry["row_count"] == 50
    assert entry["columns"] == ["price", "category"]
    assert entry["metadata"] == {"source": "etl_v2"}
    assert "timestamp" in entry


def test_get_version_history_unknown_group(store_path: Path) -> None:
    assert get_version_history(store_path, "nonexistent") == []


def test_get_latest_version_none_when_empty(store_path: Path) -> None:
    assert get_latest_version(store_path, "ghost_group") is None


def test_get_latest_version_returns_last(store_path: Path) -> None:
    record_version(store_path, "grp", 10, ["x"])
    record_version(store_path, "grp", 20, ["x", "y"])
    latest = get_latest_version(store_path, "grp")
    assert latest is not None
    assert latest["version"] == 2
    assert latest["row_count"] == 20


def test_multiple_groups_are_independent(store_path: Path) -> None:
    record_version(store_path, "alpha", 5, ["a"])
    record_version(store_path, "beta", 15, ["b"])
    record_version(store_path, "alpha", 8, ["a", "c"])
    assert len(get_version_history(store_path, "alpha")) == 2
    assert len(get_version_history(store_path, "beta")) == 1
