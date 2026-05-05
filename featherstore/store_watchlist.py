"""WatchlistMixin: integrate watchlist functionality into FeatherStore."""

from featherstore.watchlist import (
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist_entry,
    list_watched,
    is_watched,
)


class WatchlistMixin:
    """Mixin that adds watchlist management to FeatherStore."""

    def watch(self, group: str, reason: str = "", tags: list = None) -> dict:
        """Add *group* to the watchlist.

        Parameters
        ----------
        group:  name of the feature group to watch.
        reason: optional human-readable note explaining why it is watched.
        tags:   optional list of string labels for filtering.

        Returns the created watchlist entry.
        """
        return add_to_watchlist(self.store_path, group, reason=reason, tags=tags)

    def unwatch(self, group: str) -> bool:
        """Remove *group* from the watchlist.

        Returns True if the group was present, False if it was not watched.
        """
        return remove_from_watchlist(self.store_path, group)

    def is_watched(self, group: str) -> bool:
        """Return True if *group* is currently on the watchlist."""
        return is_watched(self.store_path, group)

    def get_watch_entry(self, group: str) -> dict | None:
        """Return the watchlist entry for *group*, or None if not watched."""
        return get_watchlist_entry(self.store_path, group)

    def list_watched(self) -> list[dict]:
        """Return all watchlist entries ordered by the time they were added."""
        return list_watched(self.store_path)
