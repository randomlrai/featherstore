"""LockMixin — adds advisory locking helpers to FeatherStore."""

from typing import Optional
from featherstore.lock import acquire_lock, release_lock, is_locked, lock_info


class LockMixin:
    """Mixin that adds group-level advisory locking to FeatherStore."""

    def lock(self, group: str, timeout: float = 10.0) -> bool:
        """Acquire an advisory lock for *group*.

        Parameters
        ----------
        group:
            Name of the feature group to lock.
        timeout:
            Maximum seconds to wait before giving up.

        Returns
        -------
        bool
            True if the lock was acquired, False on timeout.
        """
        acquired = acquire_lock(self.path, group, timeout=timeout)
        if acquired:
            self.log_event(group, "lock_acquired")  # type: ignore[attr-defined]
        return acquired

    def unlock(self, group: str) -> None:
        """Release the advisory lock for *group*."""
        release_lock(self.path, group)
        self.log_event(group, "lock_released")  # type: ignore[attr-defined]

    def is_locked(self, group: str) -> bool:
        """Return True if *group* is currently locked."""
        return is_locked(self.path, group)

    def lock_info(self, group: str) -> Optional[dict]:
        """Return lock metadata dict or None if the group is not locked."""
        return lock_info(self.path, group)
