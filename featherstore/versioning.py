"""Version tracking for feature groups stored in FeatherStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION_FILE = "_versions.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_version_manifest(store_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the version manifest for a store, returning an empty dict if absent."""
    manifest_path = store_path / VERSION_FILE
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_version_manifest(
    store_path: Path, manifest: dict[str, list[dict[str, Any]]]
) -> None:
    """Persist the version manifest to disk."""
    store_path.mkdir(parents=True, exist_ok=True)
    manifest_path = store_path / VERSION_FILE
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def record_version(
    store_path: Path,
    group: str,
    row_count: int,
    columns: list[str],
    metadata: dict[str, Any] | None = None,
) -> int:
    """Append a new version entry for *group* and return the new version number."""
    manifest = load_version_manifest(store_path)
    history = manifest.setdefault(group, [])
    version_number = len(history) + 1
    entry: dict[str, Any] = {
        "version": version_number,
        "timestamp": _now_iso(),
        "row_count": row_count,
        "columns": columns,
    }
    if metadata:
        entry["metadata"] = metadata
    history.append(entry)
    save_version_manifest(store_path, manifest)
    return version_number


def get_version_history(
    store_path: Path, group: str
) -> list[dict[str, Any]]:
    """Return the full version history for *group*, oldest first."""
    manifest = load_version_manifest(store_path)
    return manifest.get(group, [])


def get_latest_version(store_path: Path, group: str) -> dict[str, Any] | None:
    """Return the most recent version entry for *group*, or None if no history."""
    history = get_version_history(store_path, group)
    return history[-1] if history else None
