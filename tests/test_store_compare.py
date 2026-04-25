"""Integration tests for CompareMixin on FeatherStore."""

import pandas as pd
import pytest

from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_users():
    return pd.DataFrame({"id": [1, 2, 3], "age": [25, 30, 35], "score": [0.9, 0.8, 0.7]})


@pytest.fixture
def sample_users_v2():
    return pd.DataFrame(
        {"id": [1, 2, 3, 4], "age": [25, 30, 35, 40], "score": [0.9, 0.8, 0.7, 0.6]}
    )


def test_compare_two_groups_returns_report(store, sample_users, sample_users_v2):
    store.save("users_v1", sample_users)
    store.save("users_v2", sample_users_v2)
    report = store.compare("users_v1", "users_v2")
    assert "shape" in report
    assert "columns" in report
    assert "diff" in report


def test_compare_row_delta(store, sample_users, sample_users_v2):
    store.save("users_v1", sample_users)
    store.save("users_v2", sample_users_v2)
    report = store.compare("users_v1", "users_v2")
    assert report["shape"]["row_delta"] == 1


def test_compare_identical_groups_no_changes(store, sample_users):
    store.save("users_a", sample_users)
    store.save("users_b", sample_users.copy())
    report = store.compare("users_a", "users_b")
    assert report["columns"]["added"] == []
    assert report["columns"]["removed"] == []


def test_compare_labels_use_group_names(store, sample_users, sample_users_v2):
    store.save("alpha", sample_users)
    store.save("beta", sample_users_v2)
    report = store.compare("alpha", "beta")
    assert report["labels"]["a"] == "alpha"
    assert report["labels"]["b"] == "beta"


def test_compare_stats_returns_dataframe(store, sample_users, sample_users_v2):
    store.save("users_v1", sample_users)
    store.save("users_v2", sample_users_v2)
    result = store.compare_stats("users_v1", "users_v2")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_compare_stats_column_filter(store, sample_users, sample_users_v2):
    store.save("users_v1", sample_users)
    store.save("users_v2", sample_users_v2)
    result = store.compare_stats("users_v1", "users_v2", columns=["age"])
    assert all("age" in col for col in result.columns)
