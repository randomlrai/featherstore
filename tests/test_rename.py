"""Tests for featherstore.rename module."""

from __future__ import annotations

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from featherstore.rename import rename_group, copy_group, group_exists


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path


def _make_group(store_path: Path, name: str) -> None:
    """Create a minimal group directory with a parquet file."""
    group_dir = store_path / name
    group_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"x": [1, 2, 3]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, group_dir / "data.parquet")


# --- group_exists ---

def test_group_exists_true(store_path):
    _make_group(store_path, "alpha")
    assert group_exists(str(store_path), "alpha") is True


def test_group_exists_false(store_path):
    assert group_exists(str(store_path), "ghost") is False


# --- rename_group ---

def test_rename_group_moves_directory(store_path):
    _make_group(store_path, "raw")
    rename_group(str(store_path), "raw", "processed")
    assert not (store_path / "raw").exists()
    assert (store_path / "processed").exists()


def test_rename_group_preserves_files(store_path):
    _make_group(store_path, "raw")
    rename_group(str(store_path), "raw", "processed")
    assert (store_path / "processed" / "data.parquet").exists()


def test_rename_group_missing_source_raises(store_path):
    with pytest.raises(FileNotFoundError, match="'missing'"):
        rename_group(str(store_path), "missing", "target")


def test_rename_group_existing_destination_raises(store_path):
    _make_group(store_path, "src")
    _make_group(store_path, "dst")
    with pytest.raises(FileExistsError, match="'dst'"):
        rename_group(str(store_path), "src", "dst")


# --- copy_group ---

def test_copy_group_creates_destination(store_path):
    _make_group(store_path, "original")
    copy_group(str(store_path), "original", "clone")
    assert (store_path / "clone").exists()


def test_copy_group_source_still_exists(store_path):
    _make_group(store_path, "original")
    copy_group(str(store_path), "original", "clone")
    assert (store_path / "original").exists()


def test_copy_group_files_duplicated(store_path):
    _make_group(store_path, "original")
    copy_group(str(store_path), "original", "clone")
    assert (store_path / "clone" / "data.parquet").exists()


def test_copy_group_missing_source_raises(store_path):
    with pytest.raises(FileNotFoundError, match="'no_such'"):
        copy_group(str(store_path), "no_such", "dest")


def test_copy_group_existing_destination_raises(store_path):
    _make_group(store_path, "src")
    _make_group(store_path, "dst")
    with pytest.raises(FileExistsError, match="'dst'"):
        copy_group(str(store_path), "src", "dst")
