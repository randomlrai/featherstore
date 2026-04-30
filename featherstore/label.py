"""Group labeling: attach arbitrary key-value metadata labels to groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _labels_path(store_path: str) -> Path:
    return Path(store_path) / "_labels.json"


def load_labels(store_path: str) -> dict[str, dict[str, Any]]:
    """Return all labels, keyed by group name."""
    path = _labels_path(store_path)
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def save_labels(store_path: str, labels: dict[str, dict[str, Any]]) -> None:
    """Persist the labels mapping to disk."""
    path = _labels_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(labels, fh, indent=2)


def set_label(store_path: str, group: str, key: str, value: Any) -> dict[str, Any]:
    """Set a single label key-value pair for *group*. Returns updated labels for group."""
    labels = load_labels(store_path)
    group_labels = labels.get(group, {})
    group_labels[key] = value
    labels[group] = group_labels
    save_labels(store_path, labels)
    return group_labels


def remove_label(store_path: str, group: str, key: str) -> dict[str, Any]:
    """Remove a label key from *group*. No-op if key does not exist."""
    labels = load_labels(store_path)
    group_labels = labels.get(group, {})
    group_labels.pop(key, None)
    labels[group] = group_labels
    save_labels(store_path, labels)
    return group_labels


def get_labels(store_path: str, group: str) -> dict[str, Any]:
    """Return all labels for *group*, or an empty dict."""
    return load_labels(store_path).get(group, {})


def clear_labels(store_path: str, group: str) -> None:
    """Remove all labels for *group*."""
    labels = load_labels(store_path)
    labels.pop(group, None)
    save_labels(store_path, labels)


def find_by_label(store_path: str, key: str, value: Any | None = None) -> list[str]:
    """Return group names that have *key* (optionally matching *value*)."""
    labels = load_labels(store_path)
    results = []
    for group, kv in labels.items():
        if key in kv:
            if value is None or kv[key] == value:
                results.append(group)
    return sorted(results)
