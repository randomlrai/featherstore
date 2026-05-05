"""Checkpoint support for FeatherStore groups.

Allows marking a group at a specific state as a named checkpoint,
listing checkpoints, and restoring a group to a prior checkpoint.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _checkpoints_path(store_path: str) -> Path:
    return Path(store_path) / ".feather" / "checkpoints.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoints(store_path: str) -> dict:
    path = _checkpoints_path(store_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_checkpoints(store_path: str, data: dict) -> None:
    path = _checkpoints_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_checkpoint(
    store_path: str,
    group: str,
    name: str,
    description: str = "",
) -> dict:
    """Copy the current group data into a checkpoint directory and record metadata."""
    group_dir = Path(store_path) / group
    if not group_dir.exists():
        raise FileNotFoundError(f"Group '{group}' does not exist in store.")

    checkpoint_dir = Path(store_path) / ".feather" / "checkpoints" / group / name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    shutil.copytree(group_dir, checkpoint_dir)

    meta = {
        "group": group,
        "name": name,
        "description": description,
        "created_at": _now_iso(),
        "path": str(checkpoint_dir),
    }

    data = load_checkpoints(store_path)
    data.setdefault(group, {})[name] = meta
    save_checkpoints(store_path, data)
    return meta


def restore_checkpoint(store_path: str, group: str, name: str) -> None:
    """Overwrite the current group directory with the named checkpoint."""
    data = load_checkpoints(store_path)
    group_checkpoints = data.get(group, {})
    if name not in group_checkpoints:
        raise KeyError(f"Checkpoint '{name}' not found for group '{group}'.")

    checkpoint_dir = Path(group_checkpoints[name]["path"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint data missing at {checkpoint_dir}.")

    group_dir = Path(store_path) / group
    if group_dir.exists():
        shutil.rmtree(group_dir)
    shutil.copytree(checkpoint_dir, group_dir)


def list_checkpoints(store_path: str, group: str) -> list[dict]:
    """Return all checkpoint metadata entries for a group, sorted by creation time."""
    data = load_checkpoints(store_path)
    entries = list(data.get(group, {}).values())
    return sorted(entries, key=lambda e: e["created_at"])


def delete_checkpoint(store_path: str, group: str, name: str) -> bool:
    """Remove a named checkpoint. Returns True if deleted, False if not found."""
    data = load_checkpoints(store_path)
    group_checkpoints = data.get(group, {})
    if name not in group_checkpoints:
        return False

    checkpoint_dir = Path(group_checkpoints[name]["path"])
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    del group_checkpoints[name]
    data[group] = group_checkpoints
    save_checkpoints(store_path, data)
    return True
