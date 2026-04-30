"""Tests for featherstore.annotation and AnnotationMixin."""

import pytest
import pandas as pd
from pathlib import Path

from featherstore.annotation import (
    load_annotations,
    save_annotations,
    set_annotation,
    get_annotation,
    remove_annotation,
    list_annotations,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


# --- pure module tests ---

def test_load_annotations_missing_returns_empty(store_path):
    assert load_annotations(store_path) == {}


def test_save_and_load_annotations_roundtrip(store_path):
    data = {"features": {"note": "raw features", "author": "alice"}}
    save_annotations(store_path, data)
    assert load_annotations(store_path) == data


def test_set_annotation_creates_entry(store_path):
    entry = set_annotation(store_path, "features", "raw feature set")
    assert entry["note"] == "raw feature set"
    assert entry["author"] is None


def test_set_annotation_with_author(store_path):
    entry = set_annotation(store_path, "features", "cleaned data", author="bob")
    assert entry["author"] == "bob"


def test_set_annotation_persists(store_path):
    set_annotation(store_path, "features", "my note", author="carol")
    loaded = load_annotations(store_path)
    assert "features" in loaded
    assert loaded["features"]["note"] == "my note"


def test_set_annotation_overwrites_existing(store_path):
    set_annotation(store_path, "features", "first note")
    set_annotation(store_path, "features", "second note")
    ann = get_annotation(store_path, "features")
    assert ann["note"] == "second note"


def test_get_annotation_returns_none_for_missing(store_path):
    assert get_annotation(store_path, "nonexistent") is None


def test_remove_annotation_returns_true_when_exists(store_path):
    set_annotation(store_path, "features", "to remove")
    assert remove_annotation(store_path, "features") is True


def test_remove_annotation_returns_false_when_missing(store_path):
    assert remove_annotation(store_path, "ghost") is False


def test_remove_annotation_deletes_entry(store_path):
    set_annotation(store_path, "features", "bye")
    remove_annotation(store_path, "features")
    assert get_annotation(store_path, "features") is None


def test_list_annotations_returns_all(store_path):
    set_annotation(store_path, "a", "note a")
    set_annotation(store_path, "b", "note b")
    result = list_annotations(store_path)
    assert set(result.keys()) == {"a", "b"}


# --- AnnotationMixin via FeatherStore ---

@pytest.fixture
def store(tmp_path):
    from featherstore.store import FeatherStore
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


def test_store_annotate_and_get(store, sample_df):
    store.save("features", sample_df)
    store.annotate("features", "initial load", author="dave")
    ann = store.get_annotation("features")
    assert ann["note"] == "initial load"
    assert ann["author"] == "dave"


def test_store_remove_annotation(store, sample_df):
    store.save("features", sample_df)
    store.annotate("features", "temp note")
    assert store.remove_annotation("features") is True
    assert store.get_annotation("features") is None


def test_store_list_annotations(store, sample_df):
    store.save("a", sample_df)
    store.save("b", sample_df)
    store.annotate("a", "group a")
    store.annotate("b", "group b")
    result = store.list_annotations()
    assert "a" in result and "b" in result
