"""Sampling utilities for FeatherStore groups."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pandas as pd


def sample_fraction(
    df: pd.DataFrame,
    frac: float,
    seed: Optional[int] = None,
    stratify_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a fractional random sample of *df*.

    Parameters
    ----------
    df:
        Source DataFrame.
    frac:
        Fraction of rows to sample, in (0, 1].
    seed:
        Random seed for reproducibility.
    stratify_col:
        If provided, sample proportionally within each value of this column.
    """
    if not (0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise KeyError(f"stratify_col '{stratify_col}' not found in DataFrame")
        parts = [
            group.sample(frac=frac, random_state=seed)
            for _, group in df.groupby(stratify_col)
        ]
        return pd.concat(parts).sort_index()

    return df.sample(frac=frac, random_state=seed)


def sample_n(
    df: pd.DataFrame,
    n: int,
    seed: Optional[int] = None,
    replace: bool = False,
) -> pd.DataFrame:
    """Return exactly *n* randomly sampled rows from *df*."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not replace and n > len(df):
        raise ValueError(
            f"Cannot sample {n} rows without replacement from DataFrame with {len(df)} rows"
        )
    return df.sample(n=n, random_state=seed, replace=replace)


def bootstrap_sample(
    df: pd.DataFrame,
    n: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return a bootstrap sample (sample with replacement) of *df*.

    Defaults to sampling *len(df)* rows.
    """
    n = n if n is not None else len(df)
    return df.sample(n=n, replace=True, random_state=seed)
