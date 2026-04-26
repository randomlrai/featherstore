"""Tests for featherstore/compress.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from featherstore.compress import get_compression_info, recompress_group


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def sample_group(store_path: Path) -> str:
    group = "features"
    df = pd.DataFrame(
        {
            "id": range(200),
            "value": [float(i) * 1.5 for i in range(200)],
            "label": [f"label_{i % 5}" for i in range(200)],
        }
    )
    table = pa.Table.from_pandas(df)
    pq.write_table(table, store_path / f"{group}.parquet", compression="snappy")
    return group


# --- get_compression_info ---

def test_get_compression_info_returns_dict(store_path, sample_group):
    info = get_compression_info(store_path, sample_group)
    assert isinstance(info, dict)


def test_get_compression_info_keys(store_path, sample_group):
    info = get_compression_info(store_path, sample_group)
    for key in ("group", "size_bytes", "size_kb", "codec", "num_rows", "num_columns"):
        assert key in info


def test_get_compression_info_group_name(store_path, sample_group):
    info = get_compression_info(store_path, sample_group)
    assert info["group"] == sample_group


def test_get_compression_info_size_positive(store_path, sample_group):
    info = get_compression_info(store_path, sample_group)
    assert info["size_bytes"] > 0
    assert info["size_kb"] > 0


def test_get_compression_info_row_count(store_path, sample_group):
    info = get_compression_info(store_path, sample_group)
    assert info["num_rows"] == 200


def test_get_compression_info_missing_group_raises(store_path):
    with pytest.raises(FileNotFoundError, match="ghost"):
        get_compression_info(store_path, "ghost")


# --- recompress_group ---

def test_recompress_returns_dict(store_path, sample_group):
    result = recompress_group(store_path, sample_group, codec="zstd")
    assert isinstance(result, dict)


def test_recompress_result_keys(store_path, sample_group):
    result = recompress_group(store_path, sample_group, codec="gzip")
    for key in ("group", "codec", "size_before_bytes", "size_after_bytes", "saved_bytes", "saved_pct"):
        assert key in result


def test_recompress_codec_recorded(store_path, sample_group):
    result = recompress_group(store_path, sample_group, codec="zstd")
    assert result["codec"] == "zstd"


def test_recompress_file_still_readable(store_path, sample_group):
    recompress_group(store_path, sample_group, codec="zstd")
    table = pq.read_table(store_path / f"{sample_group}.parquet")
    assert table.num_rows == 200


def test_recompress_invalid_codec_raises(store_path, sample_group):
    with pytest.raises(ValueError, match="Invalid codec"):
        recompress_group(store_path, sample_group, codec="lzo")  # type: ignore


def test_recompress_missing_group_raises(store_path):
    with pytest.raises(FileNotFoundError):
        recompress_group(store_path, "nonexistent", codec="snappy")


def test_recompress_none_codec_writes_uncompressed(store_path, sample_group):
    result = recompress_group(store_path, sample_group, codec="none")
    assert result["size_after_bytes"] > 0
