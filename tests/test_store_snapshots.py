"""Integration tests: FeatherStore snapshot convenience methods."""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from featherstore.store import FeatherStore


@pytest.fixture()
def store(tmp_path: Path) -> FeatherStore:
    return FeatherStore(tmp_path)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [4.0, 5.0, 6.0]})


def test_snapshot_returns_metadata(store, sample_df):
    store.save(sample_df, "features/raw")
    meta = store.snapshot("features/raw", "v1")
    assert meta["group"] == "features/raw"
    assert meta["snapshot_name"] == "v1"


def test_list_snapshots_after_save(store, sample_df):
    store.save(sample_df, "features/raw")
    store.snapshot("features/raw", "v1")
    store.snapshot("features/raw", "v2")
    snaps = store.list_snapshots("features/raw")
    assert len(snaps) == 2
    names = [s["snapshot_name"] for s in snaps]
    assert "v1" in names and "v2" in names


def test_list_snapshots_empty_for_new_group(store, sample_df):
    store.save(sample_df, "features/raw")
    assert store.list_snapshots("features/raw") == []


def test_restore_snapshot_reverts_data(store):
    df_v1 = pd.DataFrame({"x": [10, 20]})
    df_v2 = pd.DataFrame({"x": [99, 88]})

    store.save(df_v1, "feat")
    store.snapshot("feat", "before")
    store.save(df_v2, "feat")

    loaded_v2 = store.load("feat")
    assert list(loaded_v2["x"]) == [99, 88]

    store.restore("feat", "before")
    loaded_restored = store.load("feat")
    assert list(loaded_restored["x"]) == [10, 20]


def test_snapshot_missing_group_raises(store):
    with pytest.raises(FileNotFoundError):
        store.snapshot("nonexistent", "v1")


def test_restore_unknown_snapshot_raises(store, sample_df):
    store.save(sample_df, "feat")
    with pytest.raises(KeyError):
        store.restore("feat", "ghost")
