"""Tests for featherstore.snapshots."""

from __future__ import annotations

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from featherstore.snapshots import (
    load_snapshots,
    create_snapshot,
    restore_snapshot,
    list_snapshots,
    delete_snapshot,
)


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path


def _write_parquet(store_path: Path, group: str, value: int) -> None:
    """Helper: write a tiny parquet file for *group* with a single 'value' column."""
    group_dir = store_path / group
    group_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"value": [value]})
    pq.write_table(pa.Table.from_pandas(df), group_dir / "data.parquet")


def _read_value(store_path: Path, group: str) -> int:
    tbl = pq.read_table(store_path / group / "data.parquet")
    return tbl.to_pandas()["value"].iloc[0]


# ---------------------------------------------------------------------------
# load_snapshots
# ---------------------------------------------------------------------------

def test_load_snapshots_missing_returns_empty(store_path):
    assert load_snapshots(store_path) == {}


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------

def test_create_snapshot_returns_metadata(store_path):
    _write_parquet(store_path, "features/raw", 42)
    meta = create_snapshot(store_path, "features/raw", "v1")
    assert meta["group"] == "features/raw"
    assert meta["snapshot_name"] == "v1"
    assert "created_at" in meta
    assert "archive_path" in meta


def test_create_snapshot_persists_to_manifest(store_path):
    _write_parquet(store_path, "features/raw", 42)
    create_snapshot(store_path, "features/raw", "v1")
    snaps = load_snapshots(store_path)
    assert "features/raw" in snaps
    assert "v1" in snaps["features/raw"]


def test_create_snapshot_missing_group_raises(store_path):
    with pytest.raises(FileNotFoundError):
        create_snapshot(store_path, "nonexistent", "v1")


# ---------------------------------------------------------------------------
# restore_snapshot
# ---------------------------------------------------------------------------

def test_restore_snapshot_overwrites_live_data(store_path):
    _write_parquet(store_path, "features/raw", 1)
    create_snapshot(store_path, "features/raw", "before")
    _write_parquet(store_path, "features/raw", 99)
    assert _read_value(store_path, "features/raw") == 99

    restore_snapshot(store_path, "features/raw", "before")
    assert _read_value(store_path, "features/raw") == 1


def test_restore_snapshot_unknown_raises(store_path):
    with pytest.raises(KeyError):
        restore_snapshot(store_path, "features/raw", "ghost")


# ---------------------------------------------------------------------------
# list_snapshots
# ---------------------------------------------------------------------------

def test_list_snapshots_empty_for_unknown_group(store_path):
    assert list_snapshots(store_path, "no_such_group") == []


def test_list_snapshots_sorted_by_created_at(store_path):
    _write_parquet(store_path, "feat", 1)
    create_snapshot(store_path, "feat", "alpha")
    create_snapshot(store_path, "feat", "beta")
    snaps = list_snapshots(store_path, "feat")
    assert len(snaps) == 2
    assert snaps[0]["created_at"] <= snaps[1]["created_at"]


# ---------------------------------------------------------------------------
# delete_snapshot
# ---------------------------------------------------------------------------

def test_delete_snapshot_removes_from_manifest(store_path):
    _write_parquet(store_path, "feat", 7)
    create_snapshot(store_path, "feat", "v1")
    delete_snapshot(store_path, "feat", "v1")
    assert list_snapshots(store_path, "feat") == []


def test_delete_snapshot_unknown_raises(store_path):
    with pytest.raises(KeyError):
        delete_snapshot(store_path, "feat", "ghost")
