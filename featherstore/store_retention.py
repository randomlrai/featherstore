"""RetentionMixin — integrates retention policies into FeatherStore."""

from featherstore.retention import (
    set_retention,
    remove_retention,
    get_retention,
    is_expired,
    list_expired,
)


class RetentionMixin:
    """Mixin that adds retention-policy methods to FeatherStore."""

    def set_retention(self, group: str, days: int) -> dict:
        """Set a retention policy for *group* (expires after *days* days).

        Returns the stored policy metadata dict.
        """
        return set_retention(self.store_path, group, days)

    def remove_retention(self, group: str) -> bool:
        """Remove the retention policy for *group*.

        Returns True if a policy existed and was removed, False otherwise.
        """
        return remove_retention(self.store_path, group)

    def get_retention(self, group: str):
        """Return the retention policy dict for *group*, or None if not set."""
        return get_retention(self.store_path, group)

    def is_expired(self, group: str) -> bool:
        """Return True if *group*'s retention period has elapsed."""
        return is_expired(self.store_path, group)

    def list_expired(self) -> list:
        """Return a list of group names whose retention period has elapsed."""
        return list_expired(self.store_path)

    def purge_expired(self) -> list:
        """Delete all groups whose retention period has elapsed.

        Returns the list of group names that were purged.
        """
        expired = list_expired(self.store_path)
        for group in expired:
            try:
                self.delete(group)
            except Exception:
                pass
            remove_retention(self.store_path, group)
        return expired
