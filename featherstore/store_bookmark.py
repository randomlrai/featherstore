"""BookmarkMixin — integrates bookmark functionality into FeatherStore."""

from featherstore.bookmark import (
    add_bookmark,
    remove_bookmark,
    get_bookmark,
    list_bookmarks,
)


class BookmarkMixin:
    """Mixin that adds bookmark methods to FeatherStore."""

    def bookmark(self, name: str, group: str, note: str = "") -> dict:
        """Save a named bookmark pointing to *group*.

        Parameters
        ----------
        name:  Unique bookmark label.
        group: The feature group this bookmark references.
        note:  Optional human-readable description.

        Returns the bookmark metadata dict.
        """
        if group not in self._catalog:
            raise KeyError(f"Group '{group}' does not exist in the store.")
        return add_bookmark(self.store_path, name, group, note=note)

    def unbookmark(self, name: str) -> bool:
        """Remove a bookmark by *name*. Returns True if it existed."""
        return remove_bookmark(self.store_path, name)

    def get_bookmark(self, name: str) -> dict | None:
        """Return the bookmark metadata for *name*, or None."""
        return get_bookmark(self.store_path, name)

    def list_bookmarks(self) -> list[dict]:
        """Return all bookmarks as a list of metadata dicts."""
        return list_bookmarks(self.store_path)

    def load_bookmarked(self, name: str, **load_kwargs):
        """Load the DataFrame for the group referenced by bookmark *name*."""
        entry = get_bookmark(self.store_path, name)
        if entry is None:
            raise KeyError(f"Bookmark '{name}' not found.")
        return self.load(entry["group"], **load_kwargs)
