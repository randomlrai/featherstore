"""Advisory file-based locking for FeatherStore groups."""

import json
import os
import time
from pathlib import Path
from typing import Optional


def _lock_path(store_path: str, group: str) -> Path:
    return Path(store_path) / group / ".lock"


def acquire_lock(store_path: str, group: str, timeout: float = 10.0, poll: float = 0.05) -> bool:
    """Attempt to acquire an advisory lock for a group.

    Returns True if the lock was acquired, False if it timed out.
    """
    lock_file = _lock_path(store_path, group)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    pid = os.getpid()

    while time.monotonic() < deadline:
        if not lock_file.exists():
            try:
                lock_file.write_text(json.dumps({"pid": pid, "acquired_at": time.time()}))
                return True
            except OSError:
                pass
        else:
            # Check if the lock is stale (holder process no longer running)
            try:
                data = json.loads(lock_file.read_text())
                holder_pid = data.get("pid")
                if holder_pid and not _pid_alive(holder_pid):
                    lock_file.unlink(missing_ok=True)
                    continue
            except (json.JSONDecodeError, OSError):
                lock_file.unlink(missing_ok=True)
                continue
        time.sleep(poll)

    return False


def release_lock(store_path: str, group: str) -> None:
    """Release the advisory lock for a group."""
    lock_file = _lock_path(store_path, group)
    lock_file.unlink(missing_ok=True)


def is_locked(store_path: str, group: str) -> bool:
    """Return True if the group currently holds a lock."""
    lock_file = _lock_path(store_path, group)
    if not lock_file.exists():
        return False
    try:
        data = json.loads(lock_file.read_text())
        holder_pid = data.get("pid")
        if holder_pid and not _pid_alive(holder_pid):
            lock_file.unlink(missing_ok=True)
            return False
    except (json.JSONDecodeError, OSError):
        return False
    return True


def lock_info(store_path: str, group: str) -> Optional[dict]:
    """Return lock metadata or None if not locked."""
    lock_file = _lock_path(store_path, group)
    if not lock_file.exists():
        return None
    try:
        return json.loads(lock_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
