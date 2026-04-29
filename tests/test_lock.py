"""Tests for featherstore.lock and LockMixin."""

import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from featherstore.lock import (
    acquire_lock,
    release_lock,
    is_locked,
    lock_info,
    _pid_alive,
)


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------

def test_pid_alive_current_process():
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_nonexistent_pid():
    # PID 0 is never a valid user process on POSIX
    assert _pid_alive(99999999) is False


# ---------------------------------------------------------------------------
# acquire / release
# ---------------------------------------------------------------------------

def test_acquire_lock_creates_file(store_path):
    acquired = acquire_lock(store_path, "features")
    assert acquired is True
    assert (Path(store_path) / "features" / ".lock").exists()


def test_acquire_lock_returns_true(store_path):
    assert acquire_lock(store_path, "features") is True


def test_release_removes_lock_file(store_path):
    acquire_lock(store_path, "features")
    release_lock(store_path, "features")
    assert not (Path(store_path) / "features" / ".lock").exists()


def test_release_nonexistent_lock_does_not_raise(store_path):
    release_lock(store_path, "features")  # should not raise


# ---------------------------------------------------------------------------
# is_locked
# ---------------------------------------------------------------------------

def test_is_locked_true_after_acquire(store_path):
    acquire_lock(store_path, "features")
    assert is_locked(store_path, "features") is True


def test_is_locked_false_before_acquire(store_path):
    assert is_locked(store_path, "features") is False


def test_is_locked_false_after_release(store_path):
    acquire_lock(store_path, "features")
    release_lock(store_path, "features")
    assert is_locked(store_path, "features") is False


def test_is_locked_clears_stale_lock(store_path, tmp_path):
    """A lock whose holder PID is dead should be treated as not locked."""
    lock_file = Path(store_path) / "features" / ".lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    lock_file.write_text(json.dumps({"pid": 99999999, "acquired_at": time.time()}))
    assert is_locked(store_path, "features") is False
    assert not lock_file.exists()


# ---------------------------------------------------------------------------
# lock_info
# ---------------------------------------------------------------------------

def test_lock_info_returns_none_when_not_locked(store_path):
    assert lock_info(store_path, "features") is None


def test_lock_info_returns_dict_when_locked(store_path):
    acquire_lock(store_path, "features")
    info = lock_info(store_path, "features")
    assert isinstance(info, dict)
    assert "pid" in info
    assert info["pid"] == os.getpid()


def test_lock_info_contains_acquired_at(store_path):
    acquire_lock(store_path, "features")
    info = lock_info(store_path, "features")
    assert "acquired_at" in info


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------

def test_acquire_times_out_when_locked_by_live_pid(store_path):
    """Simulate a live-process lock so the second acquire should time out."""
    import json
    lock_file = Path(store_path) / "features" / ".lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    # Use current PID so _pid_alive returns True
    lock_file.write_text(json.dumps({"pid": os.getpid(), "acquired_at": time.time()}))

    # We can't actually block ourselves, so patch acquire to skip its own write
    with patch("featherstore.lock.os.getpid", return_value=os.getpid() + 1):
        result = acquire_lock(store_path, "features", timeout=0.1, poll=0.02)
    assert result is False
