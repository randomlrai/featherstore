"""Tests for featherstore.dependency and DependencyMixin."""

import pytest
import pandas as pd
from featherstore.dependency import (
    load_dependencies,
    save_dependencies,
    add_dependency,
    remove_dependency,
    get_dependencies,
    get_dependents,
    get_full_upstream,
    delete_group_dependencies,
)
from featherstore.store import FeatherStore


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "store")


@pytest.fixture
def store(tmp_path):
    s = FeatherStore(str(tmp_path / "store"))
    return s


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


# --- Unit tests for dependency.py ---

def test_load_dependencies_missing_returns_empty(store_path):
    result = load_dependencies(store_path)
    assert result == {}


def test_save_and_load_dependencies_roundtrip(store_path):
    deps = {"features": ["raw", "cleaned"], "model_input": ["features"]}
    save_dependencies(store_path, deps)
    loaded = load_dependencies(store_path)
    assert loaded == deps


def test_add_dependency_creates_entry(store_path):
    add_dependency(store_path, "features", "raw")
    deps = load_dependencies(store_path)
    assert "raw" in deps["features"]


def test_add_dependency_no_duplicates(store_path):
    add_dependency(store_path, "features", "raw")
    add_dependency(store_path, "features", "raw")
    deps = load_dependencies(store_path)
    assert deps["features"].count("raw") == 1


def test_remove_dependency_removes_entry(store_path):
    add_dependency(store_path, "features", "raw")
    add_dependency(store_path, "features", "cleaned")
    remove_dependency(store_path, "features", "raw")
    assert "raw" not in get_dependencies(store_path, "features")
    assert "cleaned" in get_dependencies(store_path, "features")


def test_get_dependents_returns_downstream(store_path):
    add_dependency(store_path, "features", "raw")
    add_dependency(store_path, "model_input", "raw")
    dependents = get_dependents(store_path, "raw")
    assert "features" in dependents
    assert "model_input" in dependents


def test_get_full_upstream_transitive(store_path):
    add_dependency(store_path, "model_input", "features")
    add_dependency(store_path, "features", "raw")
    upstream = get_full_upstream(store_path, "model_input")
    assert "features" in upstream
    assert "raw" in upstream


def test_delete_group_dependencies_clears_all(store_path):
    add_dependency(store_path, "features", "raw")
    add_dependency(store_path, "model_input", "features")
    delete_group_dependencies(store_path, "features")
    deps = load_dependencies(store_path)
    assert "features" not in deps
    assert "features" not in deps.get("model_input", [])


# --- Integration tests via FeatherStore ---

def test_store_add_dependency(store, sample_df):
    store.save(sample_df, "raw")
    store.save(sample_df, "features")
    store.add_dependency("features", "raw")
    assert "raw" in store.get_dependencies("features")


def test_store_add_dependency_unknown_group_raises(store, sample_df):
    store.save(sample_df, "raw")
    with pytest.raises(KeyError):
        store.add_dependency("nonexistent", "raw")


def test_store_self_dependency_raises(store, sample_df):
    store.save(sample_df, "raw")
    with pytest.raises(ValueError):
        store.add_dependency("raw", "raw")


def test_store_get_dependents(store, sample_df):
    store.save(sample_df, "raw")
    store.save(sample_df, "features")
    store.save(sample_df, "model_input")
    store.add_dependency("features", "raw")
    store.add_dependency("model_input", "raw")
    dependents = store.get_dependents("raw")
    assert set(dependents) == {"features", "model_input"}


def test_store_full_upstream(store, sample_df):
    store.save(sample_df, "raw")
    store.save(sample_df, "features")
    store.save(sample_df, "model_input")
    store.add_dependency("features", "raw")
    store.add_dependency("model_input", "features")
    upstream = store.get_full_upstream("model_input")
    assert "features" in upstream
    assert "raw" in upstream
