"""Tests for featherstore/checkpoint.py"""

import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from featherstore.checkpoint import (
    load_checkpoints,
    save_checkpoints,
    create_checkpoint,
    restore_checkpoint,
    list_checkpoints,
    delete_checkpoint,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def _make_group(store_path: str, group: str, value: int = 1) -> None:
    group_dir = Path(store_path) / group
    group_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"x": [value, value + 1]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, group_dir / "data.parquet")


def _read_value(store_path: str, group: str) -> int:
    path = Path(store_path) / group / "data.parquet"
    return pq.read_table(path).to_pandas()["x"].iloc[0]


def test_load_checkpoints_missing_returns_empty(store_path):
    assert load_checkpoints(store_path) == {}


def test_save_and_load_checkpoints_roundtrip(store_path):
    data = {"grp": {"v1": {"name": "v1", "created_at": "2024-01-01T00:00:00+00:00"}}}
    save_checkpoints(store_path, data)
    assert load_checkpoints(store_path) == data


def test_create_checkpoint_returns_metadata(store_path):
    _make_group(store_path, "features")
    meta = create_checkpoint(store_path, "features", "v1", description="initial")
    assert meta["group"] == "features"
    assert meta["name"] == "v1"
    assert meta["description"] == "initial"
    assert "created_at" in meta
    assert "path" in meta


def test_create_checkpoint_persists(store_path):
    _make_group(store_path, "features")
    create_checkpoint(store_path, "features", "v1")
    data = load_checkpoints(store_path)
    assert "features" in data
    assert "v1" in data["features"]


def test_create_checkpoint_copies_files(store_path):
    _make_group(store_path, "features")
    meta = create_checkpoint(store_path, "features", "v1")
    checkpoint_dir = Path(meta["path"])
    assert (checkpoint_dir / "data.parquet").exists()


def test_create_checkpoint_nonexistent_group_raises(store_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        create_checkpoint(store_path, "ghost", "v1")


def test_restore_checkpoint_overwrites_group(store_path):
    _make_group(store_path, "features", value=10)
    create_checkpoint(store_path, "features", "v1")
    _make_group(store_path, "features", value=99)  # mutate group
    assert _read_value(store_path, "features") == 99
    restore_checkpoint(store_path, "features", "v1")
    assert _read_value(store_path, "features") == 10


def test_restore_checkpoint_unknown_name_raises(store_path):
    _make_group(store_path, "features")
    with pytest.raises(KeyError, match="not found"):
        restore_checkpoint(store_path, "features", "nonexistent")


def test_list_checkpoints_empty_for_new_group(store_path):
    assert list_checkpoints(store_path, "features") == []


def test_list_checkpoints_sorted_by_created_at(store_path):
    _make_group(store_path, "features")
    create_checkpoint(store_path, "features", "v1")
    create_checkpoint(store_path, "features", "v2")
    entries = list_checkpoints(store_path, "features")
    assert [e["name"] for e in entries] == ["v1", "v2"]


def test_delete_checkpoint_removes_entry(store_path):
    _make_group(store_path, "features")
    create_checkpoint(store_path, "features", "v1")
    result = delete_checkpoint(store_path, "features", "v1")
    assert result is True
    assert list_checkpoints(store_path, "features") == []


def test_delete_checkpoint_removes_directory(store_path):
    _make_group(store_path, "features")
    meta = create_checkpoint(store_path, "features", "v1")
    delete_checkpoint(store_path, "features", "v1")
    assert not Path(meta["path"]).exists()


def test_delete_checkpoint_unknown_returns_false(store_path):
    result = delete_checkpoint(store_path, "features", "ghost")
    assert result is False
