"""Feature lineage tracking for FeatherStore.

Records the origin and transformation history of feature groups,
enabling reproducibility and audit trails for ML experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_LINEAGE_FILE = "lineage.json"


def _lineage_path(store_path: str | Path) -> Path:
    return Path(store_path) / _LINEAGE_FILE


def load_lineage(store_path: str | Path) -> dict[str, Any]:
    """Load the lineage registry from disk. Returns empty dict if missing."""
    path = _lineage_path(store_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_lineage(store_path: str | Path, lineage: dict[str, Any]) -> None:
    """Persist the lineage registry to disk."""
    path = _lineage_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(lineage, fh, indent=2)


def record_lineage(
    store_path: str | Path,
    group: str,
    source: str | None = None,
    transform: str | None = None,
    parents: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record lineage metadata for a feature group.

    Args:
        store_path: Root directory of the store.
        group: Name of the feature group.
        source: Human-readable description of the data source.
        transform: Description of transformations applied.
        parents: List of parent feature group names this group was derived from.
        extra: Arbitrary additional metadata.

    Returns:
        The lineage entry that was recorded.
    """
    lineage = load_lineage(store_path)
    entry: dict[str, Any] = {
        "source": source,
        "transform": transform,
        "parents": parents or [],
        "extra": extra or {},
    }
    lineage[group] = entry
    save_lineage(store_path, lineage)
    return entry


def get_lineage(store_path: str | Path, group: str) -> dict[str, Any] | None:
    """Retrieve lineage metadata for a specific feature group.

    Returns None if no lineage has been recorded for the group.
    """
    lineage = load_lineage(store_path)
    return lineage.get(group)


def get_ancestors(
    store_path: str | Path, group: str, _visited: set[str] | None = None
) -> list[str]:
    """Recursively collect all ancestor group names for a given group."""
    if _visited is None:
        _visited = set()
    if group in _visited:
        return []
    _visited.add(group)
    entry = get_lineage(store_path, group)
    if entry is None:
        return []
    ancestors: list[str] = []
    for parent in entry.get("parents", []):
        ancestors.append(parent)
        ancestors.extend(get_ancestors(store_path, parent, _visited))
    return ancestors
