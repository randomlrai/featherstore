"""Rename and copy group utilities for FeatherStore."""

from __future__ import annotations

import shutil
from pathlib import Path


def _group_path(store_path: str, group: str) -> Path:
    return Path(store_path) / group


def rename_group(store_path: str, old_name: str, new_name: str) -> None:
    """Rename an existing group within the store.

    Raises FileNotFoundError if the source group does not exist.
    Raises FileExistsError if the destination group already exists.
    """
    src = _group_path(store_path, old_name)
    dst = _group_path(store_path, new_name)

    if not src.exists():
        raise FileNotFoundError(f"Group '{old_name}' does not exist in store.")
    if dst.exists():
        raise FileExistsError(f"Group '{new_name}' already exists in store.")

    src.rename(dst)


def copy_group(store_path: str, src_name: str, dst_name: str) -> None:
    """Copy an existing group to a new name within the store.

    Raises FileNotFoundError if the source group does not exist.
    Raises FileExistsError if the destination group already exists.
    """
    src = _group_path(store_path, src_name)
    dst = _group_path(store_path, dst_name)

    if not src.exists():
        raise FileNotFoundError(f"Group '{src_name}' does not exist in store.")
    if dst.exists():
        raise FileExistsError(f"Group '{dst_name}' already exists in store.")

    shutil.copytree(src, dst)


def group_exists(store_path: str, group: str) -> bool:
    """Return True if the given group directory exists in the store."""
    return _group_path(store_path, group).exists()
