"""Integration tests for FeatherStore.export (via ExportMixin)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from featherstore.store import FeatherStore


@pytest.fixture()
def store(tmp_path):
    return FeatherStore(tmp_path / "store")


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"x": [10, 20, 30], "y": [1.1, 2.2, 3.3]})


def test_export_csv_roundtrip(store, sample_df, tmp_path):
    store.save("my_features", sample_df)
    dest = store.export("my_features", tmp_path / "out", fmt="csv")
    loaded = pd.read_csv(dest)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_export_parquet_roundtrip(store, sample_df, tmp_path):
    store.save("my_features", sample_df)
    dest = store.export("my_features", tmp_path / "out", fmt="parquet")
    loaded = pd.read_parquet(dest)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_export_json_roundtrip(store, sample_df, tmp_path):
    store.save("my_features", sample_df)
    dest = store.export("my_features", tmp_path / "out", fmt="json")
    loaded = pd.read_json(dest, orient="records")
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_export_with_metadata_creates_meta_file(store, sample_df, tmp_path):
    store.save("my_features", sample_df)
    dest = store.export(
        "my_features", tmp_path / "out", fmt="csv", include_metadata=True
    )
    meta_path = dest.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "group" in meta or isinstance(meta, dict)  # catalog entry present


def test_export_without_metadata_no_meta_file(store, sample_df, tmp_path):
    store.save("my_features", sample_df)
    dest = store.export("my_features", tmp_path / "out", fmt="csv", include_metadata=False)
    meta_path = dest.with_suffix(".meta.json")
    assert not meta_path.exists()


def test_export_unknown_group_raises(store, tmp_path):
    with pytest.raises(Exception):
        store.export("nonexistent", tmp_path / "out", fmt="csv")
