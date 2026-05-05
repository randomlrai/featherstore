"""Integration tests for BadgeMixin via FeatherStore."""

import pytest
import pandas as pd
from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


def test_award_and_get_badge(store, sample_df):
    store.save("features", sample_df)
    store.award_badge("features", "gold")
    assert "gold" in store.get_badges("features")


def test_revoke_badge(store, sample_df):
    store.save("features", sample_df)
    store.award_badge("features", "gold")
    store.award_badge("features", "stable")
    store.revoke_badge("features", "gold")
    badges = store.get_badges("features")
    assert "gold" not in badges
    assert "stable" in badges


def test_get_badges_empty_for_new_group(store, sample_df):
    store.save("features", sample_df)
    assert store.get_badges("features") == []


def test_list_groups_with_badge(store, sample_df):
    store.save("features", sample_df)
    store.save("labels", sample_df)
    store.award_badge("features", "experimental")
    store.award_badge("labels", "experimental")
    result = store.list_groups_with_badge("experimental")
    assert set(result) == {"features", "labels"}


def test_clear_badges(store, sample_df):
    store.save("features", sample_df)
    store.award_badge("features", "gold")
    store.clear_badges("features")
    assert store.get_badges("features") == []


def test_award_invalid_badge_raises(store, sample_df):
    store.save("features", sample_df)
    with pytest.raises(ValueError):
        store.award_badge("features", "diamond")
