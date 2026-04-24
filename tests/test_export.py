"""Unit tests for featherstore.export (pure function layer)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from featherstore.export import export_group, export_metadata, SUPPORTED_FORMATS


@pytest.fixture()
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})


def test_export_csv_creates_file(tmp_path, df):
    dest = export_group(df, tmp_path / "out", fmt="csv")
    assert dest.exists()
    assert dest.suffix == ".csv"
    loaded = pd.read_csv(dest)
    pd.testing.assert_frame_equal(loaded, df)


def test_export_json_creates_file(tmp_path, df):
    dest = export_group(df, tmp_path / "out", fmt="json")
    assert dest.exists()
    assert dest.suffix == ".json"
    loaded = pd.read_json(dest, orient="records")
    pd.testing.assert_frame_equal(loaded, df)


def test_export_parquet_creates_file(tmp_path, df):
    dest = export_group(df, tmp_path / "out", fmt="parquet")
    assert dest.exists()
    assert dest.suffix == ".parquet"
    loaded = pd.read_parquet(dest)
    pd.testing.assert_frame_equal(loaded, df)


def test_export_appends_extension_if_missing(tmp_path, df):
    dest = export_group(df, tmp_path / "features", fmt="csv")
    assert dest.name == "features.csv"


def test_export_unsupported_format_raises(tmp_path, df):
    with pytest.raises(ValueError, match="Unsupported format"):
        export_group(df, tmp_path / "out", fmt="xlsx")


def test_export_creates_parent_dirs(tmp_path, df):
    dest = export_group(df, tmp_path / "nested" / "dir" / "out", fmt="csv")
    assert dest.exists()


def test_export_metadata_writes_json(tmp_path):
    meta = {"group": "features", "rows": 100}
    dest = export_metadata(meta, tmp_path / "meta.json")
    assert dest.exists()
    loaded = json.loads(dest.read_text())
    assert loaded == meta


def test_supported_formats_constant():
    assert "csv" in SUPPORTED_FORMATS
    assert "json" in SUPPORTED_FORMATS
    assert "parquet" in SUPPORTED_FORMATS
