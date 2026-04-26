"""Tests for featherstore.partition and PartitionMixin."""

from __future__ import annotations

import pytest
import pandas as pd

from featherstore.partition import (
    partition_dataframe,
    load_partition,
    load_partition_meta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        {
            "region": ["east", "west", "east", "west", "north"],
            "value": [10, 20, 30, 40, 50],
            "label": ["a", "b", "c", "d", "e"],
        }
    )


# ---------------------------------------------------------------------------
# partition_dataframe
# ---------------------------------------------------------------------------

def test_partition_creates_files(store_path, sample_df):
    pmap = partition_dataframe(sample_df, "region", store_path, "features")
    assert set(pmap.keys()) == {"east", "west", "north"}


def test_partition_meta_saved(store_path, sample_df):
    partition_dataframe(sample_df, "region", store_path, "features")
    meta = load_partition_meta(store_path, "features")
    assert meta["column"] == "region"
    assert meta["num_partitions"] == 3


def test_partition_invalid_column_raises(store_path, sample_df):
    with pytest.raises(ValueError, match="Partition column"):
        partition_dataframe(sample_df, "nonexistent", store_path, "features")


# ---------------------------------------------------------------------------
# load_partition
# ---------------------------------------------------------------------------

def test_load_single_partition(store_path, sample_df):
    partition_dataframe(sample_df, "region", store_path, "features")
    east = load_partition(store_path, "features", value="east")
    assert list(east["region"].unique()) == ["east"]
    assert len(east) == 2


def test_load_all_partitions(store_path, sample_df):
    partition_dataframe(sample_df, "region", store_path, "features")
    all_data = load_partition(store_path, "features")
    assert len(all_data) == len(sample_df)
    assert set(all_data["region"].unique()) == {"east", "west", "north"}


def test_load_unknown_value_raises(store_path, sample_df):
    partition_dataframe(sample_df, "region", store_path, "features")
    with pytest.raises(KeyError, match="south"):
        load_partition(store_path, "features", value="south")


def test_load_missing_group_raises(store_path):
    with pytest.raises(FileNotFoundError, match="no_group"):
        load_partition(store_path, "no_group")


# ---------------------------------------------------------------------------
# PartitionMixin via FeatherStore
# ---------------------------------------------------------------------------

def test_store_partition_save_and_load(tmp_path, sample_df):
    from featherstore.store import FeatherStore

    store = FeatherStore(str(tmp_path))
    pmap = store.partition_save(sample_df, "features", column="region")
    assert len(pmap) == 3

    result = store.partition_load("features")
    assert len(result) == len(sample_df)


def test_store_partition_info(tmp_path, sample_df):
    from featherstore.store import FeatherStore

    store = FeatherStore(str(tmp_path))
    store.partition_save(sample_df, "features", column="region")
    info = store.partition_info("features")
    assert info["column"] == "region"
    assert info["num_partitions"] == 3


def test_store_partition_registered_in_catalog(tmp_path, sample_df):
    from featherstore.store import FeatherStore

    store = FeatherStore(str(tmp_path))
    store.partition_save(sample_df, "features", column="region")
    assert "features" in store._catalog
    assert store._catalog["features"]["partitioned_by"] == "region"
