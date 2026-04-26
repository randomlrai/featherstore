"""Partition utilities for splitting and storing groups by column values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _partition_meta_path(store_path: str, group: str) -> Path:
    return Path(store_path) / group / "_partition_meta.json"


def load_partition_meta(store_path: str, group: str) -> dict:
    path = _partition_meta_path(store_path, group)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_partition_meta(store_path: str, group: str, meta: dict) -> None:
    path = _partition_meta_path(store_path, group)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def partition_dataframe(
    df: pd.DataFrame,
    column: str,
    store_path: str,
    group: str,
) -> dict[str, str]:
    """Split df by unique values of `column`, write each partition as parquet.

    Returns a mapping of partition_value -> relative parquet path.
    """
    if column not in df.columns:
        raise ValueError(f"Partition column '{column}' not found in DataFrame")

    group_dir = Path(store_path) / group
    group_dir.mkdir(parents=True, exist_ok=True)

    partition_map: dict[str, str] = {}
    for value, subset in df.groupby(column):
        safe_val = str(value).replace("/", "_").replace(" ", "_")
        filename = f"part_{column}_{safe_val}.parquet"
        file_path = group_dir / filename
        subset.reset_index(drop=True).to_parquet(file_path, index=False)
        partition_map[str(value)] = filename

    meta = {
        "column": column,
        "partitions": partition_map,
        "num_partitions": len(partition_map),
    }
    save_partition_meta(store_path, group, meta)
    return partition_map


def load_partition(
    store_path: str,
    group: str,
    value: Optional[str] = None,
) -> pd.DataFrame:
    """Load one or all partitions for a group. If value is None, loads all."""
    meta = load_partition_meta(store_path, group)
    if not meta:
        raise FileNotFoundError(f"No partition metadata found for group '{group}'")

    partition_map: dict[str, str] = meta["partitions"]
    group_dir = Path(store_path) / group

    if value is not None:
        key = str(value)
        if key not in partition_map:
            raise KeyError(f"Partition value '{value}' not found in group '{group}'")
        return pd.read_parquet(group_dir / partition_map[key])

    frames = [
        pd.read_parquet(group_dir / filename)
        for filename in partition_map.values()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
