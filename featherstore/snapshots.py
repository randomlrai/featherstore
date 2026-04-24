"""Snapshot support for FeatherStore — capture and restore named states of feature groups."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone


def _snapshots_path(store_path: str | Path) -> Path:
    return Path(store_path) / ".featherstore" / "snapshots.json"


def load_snapshots(store_path: str | Path) -> dict:
    """Load the snapshots manifest; returns empty dict if missing."""
    path = _snapshots_path(store_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_snapshots(store_path: str | Path, snapshots: dict) -> None:
    """Persist the snapshots manifest to disk."""
    path = _snapshots_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(snapshots, fh, indent=2)


def create_snapshot(store_path: str | Path, group: str, snapshot_name: str) -> dict:
    """Copy current parquet file for *group* into a snapshot archive.

    Returns the snapshot metadata dict.
    """
    store_path = Path(store_path)
    src = store_path / group / "data.parquet"
    if not src.exists():
        raise FileNotFoundError(f"No data found for group '{group}' at {src}")

    archive_dir = store_path / ".featherstore" / "snapshot_data" / group / snapshot_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / "data.parquet"
    shutil.copy2(src, dest)

    snapshots = load_snapshots(store_path)
    snapshots.setdefault(group, {})
    meta = {
        "snapshot_name": snapshot_name,
        "group": group,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(dest.relative_to(store_path)),
    }
    snapshots[group][snapshot_name] = meta
    save_snapshots(store_path, snapshots)
    return meta


def restore_snapshot(store_path: str | Path, group: str, snapshot_name: str) -> None:
    """Overwrite the live parquet file for *group* with the named snapshot."""
    store_path = Path(store_path)
    snapshots = load_snapshots(store_path)
    if group not in snapshots or snapshot_name not in snapshots[group]:
        raise KeyError(f"Snapshot '{snapshot_name}' not found for group '{group}'")

    archive_path = store_path / snapshots[group][snapshot_name]["archive_path"]
    dest = store_path / group / "data.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_path, dest)


def list_snapshots(store_path: str | Path, group: str) -> list[dict]:
    """Return all snapshot metadata entries for *group*, sorted by creation time."""
    snapshots = load_snapshots(store_path)
    group_snaps = snapshots.get(group, {})
    return sorted(group_snaps.values(), key=lambda s: s["created_at"])


def delete_snapshot(store_path: str | Path, group: str, snapshot_name: str) -> None:
    """Remove a snapshot from the manifest and delete its archived data."""
    store_path = Path(store_path)
    snapshots = load_snapshots(store_path)
    if group not in snapshots or snapshot_name not in snapshots[group]:
        raise KeyError(f"Snapshot '{snapshot_name}' not found for group '{group}'")

    archive_path = store_path / snapshots[group].pop(snapshot_name)["archive_path"]
    if archive_path.exists():
        archive_path.unlink()
    save_snapshots(store_path, snapshots)
