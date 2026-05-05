"""Tests for featherstore.score and ScoreMixin integration."""

from __future__ import annotations

import pytest
import pandas as pd

from featherstore.score import (
    get_scores,
    list_top_groups,
    load_scores,
    remove_score,
    set_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def store(tmp_path):
    from featherstore.store import FeatherStore
    return FeatherStore(str(tmp_path))


@pytest.fixture()
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# Unit tests (featherstore.score)
# ---------------------------------------------------------------------------

def test_load_scores_missing_returns_empty(store_path):
    assert load_scores(store_path) == {}


def test_set_score_creates_entry(store_path):
    entry = set_score(store_path, "features", 0.87)
    assert entry["score"] == pytest.approx(0.87)
    assert "updated_at" in entry


def test_set_score_persists(store_path):
    set_score(store_path, "features", 0.91, metric="accuracy")
    scores = get_scores(store_path, "features")
    assert "accuracy" in scores
    assert scores["accuracy"]["score"] == pytest.approx(0.91)


def test_set_score_with_note(store_path):
    set_score(store_path, "features", 0.75, note="initial baseline")
    scores = get_scores(store_path, "features")
    assert scores["quality"]["note"] == "initial baseline"


def test_set_score_invalid_type_raises(store_path):
    with pytest.raises(TypeError):
        set_score(store_path, "features", "high")


def test_set_score_multiple_metrics(store_path):
    set_score(store_path, "g", 0.9, metric="quality")
    set_score(store_path, "g", 0.8, metric="coverage")
    scores = get_scores(store_path, "g")
    assert "quality" in scores
    assert "coverage" in scores


def test_remove_score_returns_true(store_path):
    set_score(store_path, "g", 0.5)
    assert remove_score(store_path, "g") is True


def test_remove_score_missing_returns_false(store_path):
    assert remove_score(store_path, "nonexistent") is False


def test_remove_score_cleans_empty_group(store_path):
    set_score(store_path, "g", 0.5)
    remove_score(store_path, "g")
    assert get_scores(store_path, "g") == {}


def test_list_top_groups_ordering(store_path):
    set_score(store_path, "low", 0.3)
    set_score(store_path, "high", 0.9)
    set_score(store_path, "mid", 0.6)
    top = list_top_groups(store_path)
    assert top[0]["group"] == "high"
    assert top[-1]["group"] == "low"


def test_list_top_groups_respects_n(store_path):
    for i in range(5):
        set_score(store_path, f"g{i}", float(i) / 10)
    top = list_top_groups(store_path, n=3)
    assert len(top) == 3


# ---------------------------------------------------------------------------
# Integration tests (ScoreMixin via FeatherStore)
# ---------------------------------------------------------------------------

def test_store_set_and_get_score(store, sample_df):
    store.save(sample_df, "features")
    store.set_score("features", 0.95)
    scores = store.get_scores("features")
    assert scores["quality"]["score"] == pytest.approx(0.95)


def test_store_remove_score(store, sample_df):
    store.save(sample_df, "features")
    store.set_score("features", 0.8)
    assert store.remove_score("features") is True
    assert store.get_scores("features") == {}


def test_store_top_groups(store, sample_df):
    store.save(sample_df, "a")
    store.save(sample_df, "b")
    store.set_score("a", 0.4)
    store.set_score("b", 0.9)
    top = store.top_groups(n=2)
    assert top[0]["group"] == "b"
