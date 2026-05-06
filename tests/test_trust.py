"""Tests for featherstore.trust and TrustMixin."""

import pytest
import pandas as pd
from featherstore.trust import (
    load_trust,
    save_trust,
    set_trust,
    remove_trust,
    get_trust,
    list_by_trust_level,
    VALID_LEVELS,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


@pytest.fixture
def store(tmp_path):
    from featherstore.store import FeatherStore
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})


def test_load_trust_missing_returns_empty(store_path):
    result = load_trust(store_path)
    assert result == {}


def test_save_and_load_trust_roundtrip(store_path):
    data = {"features_v1": {"level": "trusted", "note": "reviewed", "updated_at": "2024-01-01T00:00:00+00:00"}}
    save_trust(store_path, data)
    loaded = load_trust(store_path)
    assert loaded == data


def test_set_trust_creates_entry(store_path):
    record = set_trust(store_path, "my_group", "provisional")
    assert record["level"] == "provisional"
    assert "updated_at" in record


def test_set_trust_persists(store_path):
    set_trust(store_path, "my_group", "trusted", note="all checks passed")
    trust = load_trust(store_path)
    assert "my_group" in trust
    assert trust["my_group"]["note"] == "all checks passed"


def test_set_trust_invalid_level_raises(store_path):
    with pytest.raises(ValueError, match="Invalid trust level"):
        set_trust(store_path, "g", "legendary")


def test_set_trust_overwrites_existing(store_path):
    set_trust(store_path, "g", "experimental")
    set_trust(store_path, "g", "verified")
    assert get_trust(store_path, "g")["level"] == "verified"


def test_remove_trust_returns_true(store_path):
    set_trust(store_path, "g", "trusted")
    assert remove_trust(store_path, "g") is True
    assert get_trust(store_path, "g") is None


def test_remove_trust_missing_returns_false(store_path):
    assert remove_trust(store_path, "nonexistent") is False


def test_get_trust_returns_none_for_unknown(store_path):
    assert get_trust(store_path, "unknown") is None


def test_list_by_trust_level(store_path):
    set_trust(store_path, "a", "trusted")
    set_trust(store_path, "b", "trusted")
    set_trust(store_path, "c", "experimental")
    result = list_by_trust_level(store_path, "trusted")
    assert set(result) == {"a", "b"}


def test_list_by_trust_level_invalid_raises(store_path):
    with pytest.raises(ValueError):
        list_by_trust_level(store_path, "bogus")


def test_valid_levels_tuple():
    assert "trusted" in VALID_LEVELS
    assert "untrusted" in VALID_LEVELS
    assert "verified" in VALID_LEVELS


def test_store_mixin_set_and_get(store, sample_df):
    store.save(sample_df, "features")
    store.set_trust("features", "provisional", note="initial")
    rec = store.get_trust("features")
    assert rec["level"] == "provisional"
    assert rec["note"] == "initial"


def test_store_mixin_list_by_level(store, sample_df):
    store.save(sample_df, "a")
    store.save(sample_df, "b")
    store.set_trust("a", "verified")
    store.set_trust("b", "experimental")
    assert store.list_by_trust_level("verified") == ["a"]


def test_store_trust_levels_property(store):
    assert "trusted" in store.trust_levels
