"""Tests for featherstore.archive and ArchiveMixin."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from featherstore.archive import archive_group, restore_group, list_archive_contents


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


def _make_group(store_path: str, group: str) -> Path:
    """Write a minimal parquet file into a group directory."""
    group_dir = Path(store_path) / group
    group_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    df.to_parquet(group_dir / "data.parquet", index=False)
    return group_dir


# ---------------------------------------------------------------------------
# archive_group
# ---------------------------------------------------------------------------

def test_archive_creates_zip(store_path, tmp_path):
    _make_group(store_path, "features")
    dest = str(tmp_path / "features_backup")
    meta = archive_group(store_path, "features", dest)
    assert Path(meta["archive_path"]).exists()
    assert meta["archive_path"].endswith(".zip")


def test_archive_appends_zip_extension(store_path, tmp_path):
    _make_group(store_path, "features")
    dest = str(tmp_path / "no_ext")
    meta = archive_group(store_path, "features", dest)
    assert meta["archive_path"].endswith(".zip")


def test_archive_metadata_fields(store_path, tmp_path):
    _make_group(store_path, "features")
    meta = archive_group(store_path, "features", str(tmp_path / "out.zip"))
    assert meta["group"] == "features"
    assert "archived_at" in meta
    assert isinstance(meta["files"], list)
    assert len(meta["files"]) >= 1
    assert meta["size_bytes"] > 0


def test_archive_contains_meta_json(store_path, tmp_path):
    _make_group(store_path, "features")
    meta = archive_group(store_path, "features", str(tmp_path / "out.zip"))
    with zipfile.ZipFile(meta["archive_path"]) as zf:
        assert "_archive_meta.json" in zf.namelist()


def test_archive_unknown_group_raises(store_path, tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        archive_group(store_path, "ghost", str(tmp_path / "out.zip"))


# ---------------------------------------------------------------------------
# restore_group
# ---------------------------------------------------------------------------

def test_restore_recreates_files(store_path, tmp_path):
    _make_group(store_path, "features")
    archive_path = str(tmp_path / "out.zip")
    archive_group(store_path, "features", archive_path)

    restore_store = str(tmp_path / "restored_store")
    info = restore_group(restore_store, archive_path)
    assert (Path(restore_store) / info["group"] / "data.parquet").exists()


def test_restore_uses_original_group_name(store_path, tmp_path):
    _make_group(store_path, "features")
    archive_path = str(tmp_path / "out.zip")
    archive_group(store_path, "features", archive_path)

    restore_store = str(tmp_path / "rs")
    info = restore_group(restore_store, archive_path)
    assert info["group"] == "features"


def test_restore_override_group_name(store_path, tmp_path):
    _make_group(store_path, "features")
    archive_path = str(tmp_path / "out.zip")
    archive_group(store_path, "features", archive_path)

    restore_store = str(tmp_path / "rs")
    info = restore_group(restore_store, archive_path, group="new_features")
    assert info["group"] == "new_features"
    assert (Path(restore_store) / "new_features" / "data.parquet").exists()


def test_restore_missing_archive_raises(store_path, tmp_path):
    with pytest.raises(FileNotFoundError, match="Archive not found"):
        restore_group(store_path, str(tmp_path / "ghost.zip"))


# ---------------------------------------------------------------------------
# list_archive_contents
# ---------------------------------------------------------------------------

def test_list_archive_contents_returns_files(store_path, tmp_path):
    _make_group(store_path, "features")
    archive_path = str(tmp_path / "out.zip")
    archive_group(store_path, "features", archive_path)

    contents = list_archive_contents(archive_path)
    assert "data.parquet" in contents["files"]
    assert "_archive_meta.json" not in contents["files"]


def test_list_archive_contents_includes_meta(store_path, tmp_path):
    _make_group(store_path, "features")
    archive_path = str(tmp_path / "out.zip")
    archive_group(store_path, "features", archive_path)

    contents = list_archive_contents(archive_path)
    assert contents["meta"]["group"] == "features"


def test_list_archive_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_archive_contents(str(tmp_path / "nope.zip"))
