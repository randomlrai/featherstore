"""Compression utilities for FeatherStore — supports rewriting Parquet files
with different compression codecs and reporting size savings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CompressionCodec = Literal["snappy", "gzip", "brotli", "zstd", "none"]

_VALID_CODECS = {"snappy", "gzip", "brotli", "zstd", "none"}


def _group_path(store_path: str | Path, group: str) -> Path:
    return Path(store_path) / f"{group}.parquet"


def get_compression_info(store_path: str | Path, group: str) -> dict:
    """Return file size and current compression codec for a stored group."""
    path = _group_path(store_path, group)
    if not path.exists():
        raise FileNotFoundError(f"Group '{group}' not found at {path}")

    pf = pq.ParquetFile(path)
    meta = pf.metadata
    row_group = meta.row_group(0) if meta.num_row_groups > 0 else None
    codec = (
        row_group.column(0).compression.lower()
        if row_group is not None
        else "unknown"
    )
    size_bytes = path.stat().st_size
    return {
        "group": group,
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 2),
        "codec": codec,
        "num_rows": meta.num_rows,
        "num_columns": meta.num_columns,
    }


def recompress_group(
    store_path: str | Path,
    group: str,
    codec: CompressionCodec = "zstd",
) -> dict:
    """Rewrite a group's Parquet file using the specified compression codec.

    Returns a dict with before/after size info.
    """
    if codec not in _VALID_CODECS:
        raise ValueError(f"Invalid codec '{codec}'. Choose from {_VALID_CODECS}.")

    path = _group_path(store_path, group)
    if not path.exists():
        raise FileNotFoundError(f"Group '{group}' not found at {path}")

    size_before = path.stat().st_size
    table = pq.read_table(path)

    pq_codec = None if codec == "none" else codec
    pq.write_table(table, path, compression=pq_codec)

    size_after = path.stat().st_size
    saved = size_before - size_after
    return {
        "group": group,
        "codec": codec,
        "size_before_bytes": size_before,
        "size_after_bytes": size_after,
        "saved_bytes": saved,
        "saved_pct": round(saved / size_before * 100, 2) if size_before else 0.0,
    }
