"""Tests for featherstore.profile and ProfileMixin integration."""

import pytest
import pandas as pd

from featherstore.profile import (
    compute_profile,
    load_profile,
    record_profile,
    save_profile,
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
            "age": [25, 30, None, 45],
            "name": ["alice", "bob", "carol", None],
            "score": [1.1, 2.2, 3.3, 4.4],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests — compute_profile
# ---------------------------------------------------------------------------

def test_compute_profile_row_col_count(sample_df):
    p = compute_profile(sample_df)
    assert p["row_count"] == 4
    assert p["col_count"] == 3


def test_compute_profile_null_counts(sample_df):
    p = compute_profile(sample_df)
    assert p["columns"]["age"]["null_count"] == 1
    assert p["columns"]["name"]["null_count"] == 1
    assert p["columns"]["score"]["null_count"] == 0


def test_compute_profile_numeric_stats(sample_df):
    p = compute_profile(sample_df)
    age_info = p["columns"]["age"]
    assert age_info["min"] == 25.0
    assert age_info["max"] == 45.0
    assert "mean" in age_info


def test_compute_profile_string_lengths(sample_df):
    p = compute_profile(sample_df)
    name_info = p["columns"]["name"]
    assert name_info["min_length"] == 3   # "bob"
    assert name_info["max_length"] == 5   # "alice" / "carol"


def test_compute_profile_unique_count(sample_df):
    p = compute_profile(sample_df)
    assert p["columns"]["score"]["unique_count"] == 4


# ---------------------------------------------------------------------------
# Unit tests — persistence helpers
# ---------------------------------------------------------------------------

def test_load_profile_missing_returns_empty(store_path):
    assert load_profile(store_path, "no_group") == {}


def test_save_and_load_profile_roundtrip(store_path, sample_df):
    profile = compute_profile(sample_df)
    save_profile(store_path, "grp", profile)
    loaded = load_profile(store_path, "grp")
    assert loaded["row_count"] == profile["row_count"]
    assert set(loaded["columns"].keys()) == set(profile["columns"].keys())


def test_record_profile_persists(store_path, sample_df):
    record_profile(store_path, "grp", sample_df)
    loaded = load_profile(store_path, "grp")
    assert loaded["col_count"] == 3


# ---------------------------------------------------------------------------
# Integration tests — ProfileMixin via FeatherStore
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    from featherstore.store import FeatherStore
    return FeatherStore(str(tmp_path))


def test_get_profile_empty_before_save(store):
    assert store.get_profile("features") == {}


def test_profile_raises_for_unknown_group(store):
    with pytest.raises(KeyError, match="features"):
        store.profile("features")


def test_profile_after_save(store, sample_df):
    store.save(sample_df, "features")
    p = store.profile("features")
    assert p["row_count"] == len(sample_df)
    assert "age" in p["columns"]


def test_get_profile_returns_persisted(store, sample_df):
    store.save(sample_df, "features")
    store.profile("features")
    p = store.get_profile("features")
    assert p["col_count"] == len(sample_df.columns)
