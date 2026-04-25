"""Integration tests for SamplingMixin via FeatherStore."""

import pandas as pd
import pytest

from featherstore.store import FeatherStore


@pytest.fixture
def store(tmp_path):
    return FeatherStore(str(tmp_path))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": range(80),
            "y": [i * 2.0 for i in range(80)],
            "split": ["train"] * 40 + ["test"] * 40,
        }
    )


def test_sample_frac_returns_correct_rows(store, sample_df):
    store.save("features", sample_df)
    result = store.sample("features", frac=0.25, seed=0)
    assert len(result) == 20


def test_sample_n_returns_correct_rows(store, sample_df):
    store.save("features", sample_df)
    result = store.sample("features", n=10, seed=1)
    assert len(result) == 10


def test_sample_requires_frac_or_n(store, sample_df):
    store.save("features", sample_df)
    with pytest.raises(ValueError):
        store.sample("features")  # neither provided
    with pytest.raises(ValueError):
        store.sample("features", frac=0.5, n=10)  # both provided


def test_sample_stratified(store, sample_df):
    store.save("features", sample_df)
    result = store.sample("features", frac=0.5, seed=42, stratify_col="split")
    counts = result["split"].value_counts()
    assert counts["train"] == counts["test"]


def test_bootstrap_default_size(store, sample_df):
    store.save("features", sample_df)
    result = store.bootstrap("features", seed=0)
    assert len(result) == len(sample_df)


def test_bootstrap_custom_n(store, sample_df):
    store.save("features", sample_df)
    result = store.bootstrap("features", n=200, seed=7)
    assert len(result) == 200
