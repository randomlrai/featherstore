"""Integration tests verifying lineage tracking via FeatherStore.save()."""

import pandas as pd
import pytest

from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(tmp_path)


@pytest.fixture
def sample_df():
    return pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [4.0, 5.0, 6.0]})


def test_save_without_lineage_returns_none(store, sample_df):
    store.save("raw", sample_df)
    assert store.get_lineage("raw") is None


def test_save_with_source_records_lineage(store, sample_df):
    store.save("raw", sample_df, source="s3://bucket/raw.csv")
    lineage = store.get_lineage("raw")
    assert lineage is not None
    assert lineage["source"] == "s3://bucket/raw.csv"


def test_save_with_transform_records_lineage(store, sample_df):
    store.save("processed", sample_df, transform="StandardScaler")
    lineage = store.get_lineage("processed")
    assert lineage["transform"] == "StandardScaler"


def test_save_with_parents_records_lineage(store, sample_df):
    store.save("raw", sample_df)
    store.save("derived", sample_df, parents=["raw"])
    lineage = store.get_lineage("derived")
    assert "raw" in lineage["parents"]


def test_save_with_lineage_extra(store, sample_df):
    store.save("g", sample_df, lineage_extra={"pipeline_version": "2.1"})
    lineage = store.get_lineage("g")
    assert lineage["extra"]["pipeline_version"] == "2.1"


def test_save_full_lineage(store, sample_df):
    store.save(
        "features_v2",
        sample_df,
        source="postgres://db/table",
        transform="log1p + impute",
        parents=["features_v1"],
        lineage_extra={"author": "alice"},
    )
    lineage = store.get_lineage("features_v2")
    assert lineage["source"] == "postgres://db/table"
    assert lineage["transform"] == "log1p + impute"
    assert lineage["parents"] == ["features_v1"]
    assert lineage["extra"]["author"] == "alice"


def test_lineage_survives_overwrite(store, sample_df):
    store.save("g", sample_df, source="v1")
    store.save("g", sample_df, source="v2")
    lineage = store.get_lineage("g")
    assert lineage["source"] == "v2"


def test_get_lineage_for_unknown_key_returns_none(store):
    """Requesting lineage for a dataset that was never saved should return None."""
    assert store.get_lineage("does_not_exist") is None


def test_save_with_multiple_parents_records_all(store, sample_df):
    """All parent dataset names should be preserved when multiple parents are given."""
    store.save("parent_a", sample_df)
    store.save("parent_b", sample_df)
    store.save("child", sample_df, parents=["parent_a", "parent_b"])
    lineage = store.get_lineage("child")
    assert lineage is not None
    assert "parent_a" in lineage["parents"]
    assert "parent_b" in lineage["parents"]
    assert len(lineage["parents"]) == 2
