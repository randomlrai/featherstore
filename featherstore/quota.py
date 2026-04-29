"""Disk quota management for FeatherStore groups."""

import json
import os
from pathlib import Path
from typing import Optional


def _quota_path(store_path: str) -> Path:
    return Path(store_path) / ".featherstore" / "quotas.json"


def load_quotas(store_path: str) -> dict:
    path = _quota_path(store_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_quotas(store_path: str, quotas: dict) -> None:
    path = _quota_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(quotas, f, indent=2)


def set_quota(store_path: str, group: str, max_bytes: int) -> dict:
    """Set a disk quota (in bytes) for a group."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    quotas = load_quotas(store_path)
    quotas[group] = {"max_bytes": max_bytes}
    save_quotas(store_path, quotas)
    return quotas[group]


def remove_quota(store_path: str, group: str) -> bool:
    """Remove a quota for a group. Returns True if removed, False if not found."""
    quotas = load_quotas(store_path)
    if group not in quotas:
        return False
    del quotas[group]
    save_quotas(store_path, quotas)
    return True


def get_quota(store_path: str, group: str) -> Optional[dict]:
    """Return quota info for a group, or None if not set."""
    quotas = load_quotas(store_path)
    return quotas.get(group)


def get_group_size(store_path: str, group: str) -> int:
    """Return total size in bytes of all files in a group directory."""
    group_dir = Path(store_path) / group
    if not group_dir.exists():
        return 0
    total = 0
    for entry in group_dir.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def check_quota(store_path: str, group: str) -> dict:
    """Check quota usage for a group. Returns usage report dict."""
    quota = get_quota(store_path, group)
    used = get_group_size(store_path, group)
    report = {"group": group, "used_bytes": used, "quota_bytes": None, "exceeded": False}
    if quota is not None:
        report["quota_bytes"] = quota["max_bytes"]
        report["exceeded"] = used > quota["max_bytes"]
    return report


def enforce_quota(store_path: str, group: str) -> None:
    """Raise an error if the group exceeds its quota."""
    report = check_quota(store_path, group)
    if report["exceeded"]:
        raise QuotaExceededError(
            f"Group '{group}' exceeds quota: "
            f"{report['used_bytes']} bytes used, "
            f"{report['quota_bytes']} bytes allowed."
        )


class QuotaExceededError(Exception):
    pass
