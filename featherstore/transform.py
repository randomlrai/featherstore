"""Lightweight transform pipeline for feature groups."""

from __future__ import annotations

from typing import Callable, List, Optional
import pandas as pd


class TransformPipeline:
    """Chains a sequence of DataFrame transforms and applies them in order."""

    def __init__(self, name: str):
        self.name = name
        self._steps: List[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = []

    def add_step(self, step_name: str, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "TransformPipeline":
        """Append a named transform step."""
        if not callable(fn):
            raise TypeError(f"Step '{step_name}' must be callable.")
        self._steps.append((step_name, fn))
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all steps sequentially and return the transformed DataFrame."""
        if df is None:
            raise ValueError("Input DataFrame must not be None.")
        result = df.copy()
        for step_name, fn in self._steps:
            try:
                result = fn(result)
            except Exception as exc:
                raise RuntimeError(f"Transform step '{step_name}' failed: {exc}") from exc
        return result

    @property
    def step_names(self) -> List[str]:
        """Return the ordered list of step names."""
        return [name for name, _ in self._steps]

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"TransformPipeline(name={self.name!r}, steps={self.step_names})"


def drop_nulls(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop rows with any null values, optionally restricted to *subset* columns."""
    return df.dropna(subset=subset)


def rename_columns(mapping: dict) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a step function that renames columns according to *mapping*."""
    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=mapping)
    return _rename


def cast_columns(dtypes: dict) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a step function that casts columns to specified dtypes."""
    def _cast(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({col: dtype for col, dtype in dtypes.items() if col in df.columns})
    return _cast


def select_columns(columns: List[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a step function that keeps only the specified columns."""
    def _select(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")
        return df[columns]
    return _select
