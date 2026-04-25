"""Integration tests for MergeMixin via FeatherStore."""

import pandas as pd
import pytest

from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path / "store"))


@pytest.fixture
def sample_users():
    return pd.DataFrame({"user_id": [1, 2, 3], "age": [25, 30, 35]})


@pytest.fixture
def sample_purchases():
    return pd.DataFrame({"user_id": [1, 2, 4], "amount": [50.0, 80.0, 120.0]})


@pytest.fixture
def sample_labels():
    return pd.DataFrame({"user_id": [1, 2, 3], "label": [0, 1, 0]})


def test_merge_two_groups_inner(store, sample_users, sample_purchases):
    store.save("users", sample_users)
    store.save("purchases", sample_purchases)
    result = store.merge(["users", "purchases"], on=["user_id"], how="inner")
    assert len(result) == 2
    assert "age" in result.columns
    assert "amount" in result.columns


def test_merge_two_groups_outer(store, sample_users, sample_purchases):
    store.save("users", sample_users)
    store.save("purchases", sample_purchases)
    result = store.merge(["users", "purchases"], on=["user_id"], how="outer")
    assert len(result) == 4


def test_merge_three_groups(store, sample_users, sample_purchases, sample_labels):
    store.save("users", sample_users)
    store.save("purchases", sample_purchases)
    store.save("labels", sample_labels)
    result = store.merge(
        ["users", "purchases", "labels"], on=["user_id"], how="inner"
    )
    assert set(result.columns) == {"user_id", "age", "amount", "label"}
    assert len(result) == 2


def test_merge_requires_at_least_two_groups(store, sample_users):
    store.save("users", sample_users)
    with pytest.raises(ValueError, match="At least two groups"):
        store.merge(["users"])


def test_merge_with_column_subset(store, sample_users, sample_labels):
    store.save("users", sample_users)
    store.save("labels", sample_labels)
    result = store.merge(
        ["users", "labels"],
        on=["user_id"],
        how="inner",
        columns=["user_id", "age", "label"],
    )
    assert "age" in result.columns
    assert "label" in result.columns
    # 'user_id' used as join key should still be present
    assert "user_id" in result.columns
