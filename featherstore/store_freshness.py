"""FreshnessMixin — integrates freshness tracking into FeatherStore."""

from featherstore.freshness import (
    record_freshness,
    get_freshness,
    is_stale,
    remove_freshness,
    load_freshness,
)


class FreshnessMixin:
    """Mixin that adds freshness / staleness tracking to FeatherStore."""

    def touch(self, group: str) -> dict:
        """Manually mark a group as freshly updated right now."""
        return record_freshness(self.store_path, group)

    def get_freshness(self, group: str) -> dict | None:
        """Return the freshness metadata for *group*, or None if unknown."""
        return get_freshness(self.store_path, group)

    def is_stale(self, group: str, max_age_seconds: float) -> bool:
        """Return True if *group* has not been updated within *max_age_seconds*."""
        return is_stale(self.store_path, group, max_age_seconds)

    def remove_freshness(self, group: str) -> bool:
        """Delete the freshness record for *group*. Returns True if it existed."""
        return remove_freshness(self.store_path, group)

    def list_freshness(self) -> dict:
        """Return the full freshness registry for this store."""
        return load_freshness(self.store_path)
