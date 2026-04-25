"""Unit tests for featherstore/sampling.py."""

import pandas as pd
import pytest

from featherstore.sampling import bootstrap_sample, sample_fraction, sample_n


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": [float(i) * 0.5 for i in range(100)],
            "label": ["cat" if i % 2 == 0 else "dog" for i in range(100)],
        }
    )


# ------------------------------------------------------------------ #
# sample_fraction                                                      #
# ------------------------------------------------------------------ #

def test_sample_fraction_returns_correct_size(base_df):
    result = sample_fraction(base_df, frac=0.2, seed=0)
    assert len(result) == 20


def test_sample_fraction_reproducible(base_df):
    r1 = sample_fraction(base_df, frac=0.3, seed=42)
    r2 = sample_fraction(base_df, frac=0.3, seed=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_sample_fraction_invalid_frac_raises(base_df):
    with pytest.raises(ValueError):
        sample_fraction(base_df, frac=0.0)
    with pytest.raises(ValueError):
        sample_fraction(base_df, frac=1.5)


def test_sample_fraction_stratified_preserves_ratio(base_df):
    result = sample_fraction(base_df, frac=0.5, seed=7, stratify_col="label")
    counts = result["label"].value_counts()
    assert counts["cat"] == counts["dog"]


def test_sample_fraction_bad_stratify_col_raises(base_df):
    with pytest.raises(KeyError):
        sample_fraction(base_df, frac=0.5, stratify_col="nonexistent")


# ------------------------------------------------------------------ #
# sample_n                                                             #
# ------------------------------------------------------------------ #

def test_sample_n_returns_correct_size(base_df):
    result = sample_n(base_df, n=10, seed=1)
    assert len(result) == 10


def test_sample_n_reproducible(base_df):
    r1 = sample_n(base_df, n=15, seed=99)
    r2 = sample_n(base_df, n=15, seed=99)
    pd.testing.assert_frame_equal(r1, r2)


def test_sample_n_too_large_raises(base_df):
    with pytest.raises(ValueError):
        sample_n(base_df, n=200)


def test_sample_n_with_replacement_allows_oversample(base_df):
    result = sample_n(base_df, n=150, replace=True, seed=0)
    assert len(result) == 150


def test_sample_n_negative_raises(base_df):
    with pytest.raises(ValueError):
        sample_n(base_df, n=-1)


# ------------------------------------------------------------------ #
# bootstrap_sample                                                     #
# ------------------------------------------------------------------ #

def test_bootstrap_default_size(base_df):
    result = bootstrap_sample(base_df, seed=0)
    assert len(result) == len(base_df)


def test_bootstrap_custom_n(base_df):
    result = bootstrap_sample(base_df, n=200, seed=0)
    assert len(result) == 200


def test_bootstrap_reproducible(base_df):
    r1 = bootstrap_sample(base_df, seed=5)
    r2 = bootstrap_sample(base_df, seed=5)
    pd.testing.assert_frame_equal(r1, r2)
