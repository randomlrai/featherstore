"""ArchiveMixin — adds archive/restore capabilities to FeatherStore."""

from featherstore.archive import archive_group, restore_group, list_archive_contents


class ArchiveMixin:
    """Mixin providing archive and restore methods for FeatherStore."""

    def archive(self, group: str, dest: str) -> dict:
        """Archive a group to a zip file at *dest*.

        Parameters
        ----------
        group:
            Name of the group to archive.
        dest:
            Destination path for the zip file. A ``.zip`` extension is
            appended automatically if missing.

        Returns
        -------
        dict
            Metadata about the created archive (path, size, files, timestamp).
        """
        return archive_group(self.path, group, dest)

    def restore(self, archive_path: str, group: str | None = None) -> dict:
        """Restore a group from a zip archive.

        Parameters
        ----------
        archive_path:
            Path to the ``.zip`` archive produced by :meth:`archive`.
        group:
            Override the group name to restore into. If *None*, the original
            group name stored in the archive metadata is used.

        Returns
        -------
        dict
            Metadata about the restore operation.
        """
        return restore_group(self.path, archive_path, group)

    def inspect_archive(self, archive_path: str) -> dict:
        """Return the contents of an archive without extracting it."""
        return list_archive_contents(archive_path)
