"""FeatherStore mixin / helpers that expose export functionality on the store."""

from pathlib import Path
from typing import Optional

from featherstore.export import export_group, export_metadata


class ExportMixin:
    """Mixin that adds export_group / export_metadata helpers to FeatherStore."""

    def export(
        self,
        group: str,
        dest: str | Path,
        fmt: str = "csv",
        include_metadata: bool = False,
        **kwargs,
    ) -> Path:
        """Export a saved feature group to *dest*.

        Parameters
        ----------
        group:            Name of the feature group to export.
        dest:             Destination file path.
        fmt:              Output format – 'csv', 'json', or 'parquet'.
        include_metadata: When True, also write a ``<dest>.meta.json`` file
                          containing the catalog entry for the group.
        **kwargs:         Forwarded to the underlying pandas writer.

        Returns
        -------
        Path of the written data file.
        """
        df = self.load(group)
        data_path = export_group(df, dest, fmt=fmt, **kwargs)

        if include_metadata:
            catalog = self._load_catalog()
            meta = catalog.get(group, {})
            meta_path = data_path.with_suffix(".meta.json")
            export_metadata(meta, meta_path)

        return data_path
