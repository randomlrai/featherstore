"""TTL (time-to-live) mixin for FeatherStore.

Provides methods to set, remove, check, and enforce expiry
policies on feature groups stored in the catalog.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from featherstore.ttl import (
    load_ttl,
    save_ttl,
    set_ttl,
    remove_ttl,
    is_expired,
    list_expired,
)


class TTLMixin:
    """Mixin that adds TTL / expiry management to FeatherStore."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_ttl(self, group: str, days: float) -> dict:
        """Attach a TTL policy to *group*.

        Parameters
        ----------
        group:
            Name of the feature group.
        days:
            Number of days from now until the group is considered expired.
            Must be a positive number.

        Returns
        -------
        dict
            The TTL record that was persisted, including ``expires_at``.

        Raises
        ------
        ValueError
            If *days* is not positive or *group* does not exist in the catalog.
        """
        if days <= 0:
            raise ValueError(f"'days' must be positive, got {days}")
        if group not in self._catalog:
            raise KeyError(f"Group '{group}' not found in catalog.")

        ttl_data = load_ttl(self._store_path)
        record = set_ttl(ttl_data, group, days)
        save_ttl(self._store_path, ttl_data)
        return record

    def remove_ttl(self, group: str) -> bool:
        """Remove the TTL policy for *group*.

        Returns ``True`` if a policy existed and was removed, ``False``
        if the group had no TTL policy.
        """
        ttl_data = load_ttl(self._store_path)
        removed = remove_ttl(ttl_data, group)
        if removed:
            save_ttl(self._store_path, ttl_data)
        return removed

    def is_expired(self, group: str) -> bool:
        """Return ``True`` if *group* has an active TTL policy that has lapsed."""
        ttl_data = load_ttl(self._store_path)
        return is_expired(ttl_data, group)

    def get_ttl(self, group: str) -> Optional[dict]:
        """Return the TTL record for *group*, or ``None`` if none is set."""
        ttl_data = load_ttl(self._store_path)
        return ttl_data.get(group)

    def list_expired(self) -> list[str]:
        """Return a list of group names whose TTL has lapsed."""
        ttl_data = load_ttl(self._store_path)
        return list_expired(ttl_data)

    def purge_expired(self, dry_run: bool = False) -> list[str]:
        """Delete all groups whose TTL has lapsed.

        Parameters
        ----------
        dry_run:
            When ``True`` the method returns the list of expired groups
            without actually deleting them.

        Returns
        -------
        list[str]
            Names of the groups that were (or would be) purged.
        """
        expired = self.list_expired()
        if not dry_run:
            for group in expired:
                try:
                    self.delete(group)
                except Exception:  # noqa: BLE001
                    # Best-effort: skip groups that fail to delete
                    pass
                # Also clean up the TTL record so it doesn't linger
                ttl_data = load_ttl(self._store_path)
                remove_ttl(ttl_data, group)
                save_ttl(self._store_path, ttl_data)
        return expired
