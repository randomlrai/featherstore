"""Merge utilities for combining feature groups in FeatherStore."""

from __future__ import annotations

import pandas as pd
from typing import List, Optional


def merge_groups(
    frames: List[pd.DataFrame],
    on: Optional[List[str]] = None,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge multiple DataFrames on shared key columns.

    Parameters
    ----------
    frames:
        List of DataFrames to merge (at least two required).
    on:
        Column(s) to join on.  If None, the intersection of all
        column names is used as the join key.
    how:
        Join strategy – 'inner', 'left', 'right', or 'outer'.

    Returns
    -------
    pd.DataFrame
        Merged result.
    """
    if len(frames) < 2:
        raise ValueError("merge_groups requires at least two DataFrames.")

    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(f"Unsupported join strategy: {how!r}")

    if on is None:
        key_sets = [set(df.columns) for df in frames]
        shared = key_sets[0].intersection(*key_sets[1:])
        if not shared:
            raise ValueError(
                "No common columns found across DataFrames; specify 'on' explicitly."
            )
        on = sorted(shared)

    result = frames[0]
    for right in frames[1:]:
        result = result.merge(right, on=on, how=how, suffixes=("", "_dup"))
        # Drop any accidental duplicate columns introduced by suffix
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        result = result.drop(columns=dup_cols)

    return result.reset_index(drop=True)


def merge_on_index(
    frames: List[pd.DataFrame],
    how: str = "inner",
) -> pd.DataFrame:
    """Merge DataFrames by aligning on their index.

    Parameters
    ----------
    frames:
        List of DataFrames to merge.
    how:
        Join strategy.

    Returns
    -------
    pd.DataFrame
        Merged result.
    """
    if len(frames) < 2:
        raise ValueError("merge_on_index requires at least two DataFrames.")

    result = frames[0]
    for right in frames[1:]:
        result = result.join(right, how=how, rsuffix="_dup")
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        result = result.drop(columns=dup_cols)

    return result
