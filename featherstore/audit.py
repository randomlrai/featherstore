"""Audit log for tracking read/write operations on feature groups."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _audit_path(store_path: str) -> Path:
    return Path(store_path) / "_audit_log.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_audit_log(store_path: str) -> list[dict[str, Any]]:
    """Load the audit log; returns empty list if missing."""
    path = _audit_path(store_path)
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_audit_log(store_path: str, log: list[dict[str, Any]]) -> None:
    """Persist the audit log to disk."""
    path = _audit_path(store_path)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def record_event(
    store_path: str,
    group: str,
    operation: str,
    user: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an audit event and return the recorded entry."""
    if operation not in {"read", "write", "delete", "export", "snapshot"}:
        raise ValueError(f"Unsupported operation: {operation!r}")
    log = load_audit_log(store_path)
    entry: dict[str, Any] = {
        "timestamp": _now_iso(),
        "group": group,
        "operation": operation,
        "user": user,
        "details": details or {},
    }
    log.append(entry)
    save_audit_log(store_path, log)
    return entry


def get_audit_history(
    store_path: str,
    group: str | None = None,
    operation: str | None = None,
) -> list[dict[str, Any]]:
    """Return filtered audit log entries."""
    log = load_audit_log(store_path)
    if group is not None:
        log = [e for e in log if e["group"] == group]
    if operation is not None:
        log = [e for e in log if e["operation"] == operation]
    return log


def clear_audit_log(store_path: str) -> None:
    """Remove all audit log entries."""
    save_audit_log(store_path, [])
