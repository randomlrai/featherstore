"""Export feature groups to various formats (CSV, JSON, Parquet)."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd


SUPPORTED_FORMATS = ("csv", "json", "parquet")


def export_group(
    df: pd.DataFrame,
    dest: str | Path,
    fmt: str = "csv",
    **kwargs,
) -> Path:
    """Export a DataFrame to *dest* in the requested format.

    Parameters
    ----------
    df:   DataFrame to export.
    dest: Destination file path (extension may be omitted; will be appended).
    fmt:  One of 'csv', 'json', 'parquet'.
    **kwargs: Forwarded to the underlying pandas writer.

    Returns
    -------
    Path of the written file.
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. Choose from: {SUPPORTED_FORMATS}"
        )

    dest = Path(dest)
    # Ensure the correct extension
    if dest.suffix.lstrip(".").lower() != fmt:
        dest = dest.with_suffix(f".{fmt}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        df.to_csv(dest, index=False, **kwargs)
    elif fmt == "json":
        df.to_json(dest, orient="records", indent=2, **kwargs)
    elif fmt == "parquet":
        df.to_parquet(dest, index=False, **kwargs)

    return dest


def export_metadata(metadata: dict, dest: str | Path) -> Path:
    """Write a metadata dict as a pretty-printed JSON file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(metadata, indent=2, default=str))
    return dest
