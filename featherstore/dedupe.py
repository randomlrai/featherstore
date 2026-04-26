"""Deduplication utilities for FeatherStore groups."""

from __future__ import annotations

import pandas as pd
from typing import List, Optional


def find_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Return rows that are duplicates according to *subset* columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    subset:
        Columns to consider for identifying duplicates.  ``None`` means all
        columns.
    keep:
        Which occurrence to mark as *not* a duplicate – ``'first'``,
        ``'last'``, or ``False`` (mark all duplicates).

    Returns
    -------
    DataFrame containing only the duplicate rows.
    """
    mask = df.duplicated(subset=subset, keep=keep)
    return df[mask].copy()


def drop_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Return a DataFrame with duplicate rows removed.

    Parameters
    ----------
    df:
        Input DataFrame.
    subset:
        Columns to consider.  ``None`` means all columns.
    keep:
        Which occurrence to retain – ``'first'``, ``'last'``, or ``False``
        (drop all duplicates).

    Returns
    -------
    De-duplicated DataFrame with the original index preserved.
    """
    return df.drop_duplicates(subset=subset, keep=keep).copy()


def dedupe_report(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> dict:
    """Return a summary report about duplicate rows in *df*.

    Returns
    -------
    dict with keys:
        - ``total_rows``
        - ``duplicate_rows``
        - ``unique_rows``
        - ``duplicate_rate``  (float 0-1)
        - ``subset``          (list of columns checked, or None)
    """
    total = len(df)
    dupes = int(df.duplicated(subset=subset, keep="first").sum())
    unique = total - dupes
    rate = round(dupes / total, 6) if total > 0 else 0.0
    return {
        "total_rows": total,
        "duplicate_rows": dupes,
        "unique_rows": unique,
        "duplicate_rate": rate,
        "subset": list(subset) if subset is not None else None,
    }
