"""Core FeatherStore class — lightweight local feature store backed by DuckDB + Parquet."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from featherstore.lineage import get_lineage, record_lineage
from featherstore.tags import add_tag, list_tags, remove_tag
from featherstore.versioning import get_version_history, record_version

_CATALOG_DB = "catalog.duckdb"


class FeatherStore:
    """Lightweight local feature store for rapid ML experimentation."""

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self.store_path / _CATALOG_DB
        self._con = duckdb.connect(str(self._db_path))
        self._init_catalog()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_catalog(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog (
                group_name TEXT PRIMARY KEY,
                parquet_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    def _group_dir(self, group: str) -> Path:
        return self.store_path / group

    def _parquet_path(self, group: str) -> Path:
        return self._group_dir(group) / "data.parquet"

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def save(
        self,
        group: str,
        df: pd.DataFrame,
        *,
        tags: list[str] | None = None,
        source: str | None = None,
        transform: str | None = None,
        parents: list[str] | None = None,
        lineage_extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist a DataFrame as a named feature group."""
        group_dir = self._group_dir(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = self._parquet_path(group)
        df.to_parquet(parquet_path, index=False)

        rel_path = str(parquet_path.relative_to(self.store_path))
        self._con.execute(
            """
            INSERT INTO catalog (group_name, parquet_path, created_at)
            VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%S', 'now'))
            ON CONFLICT (group_name) DO UPDATE SET
                parquet_path = excluded.parquet_path,
                created_at = excluded.created_at
            """,
            [group, rel_path],
        )

        record_version(self.store_path, group, {"rows": len(df), "columns": list(df.columns)})

        if tags:
            for tag in tags:
                add_tag(self.store_path, group, tag)

        if any(v is not None for v in (source, transform, parents, lineage_extra)):
            record_lineage(
                self.store_path,
                group,
                source=source,
                transform=transform,
                parents=parents,
                extra=lineage_extra,
            )

    def load(self, group: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Load a feature group by name, optionally selecting columns."""
        row = self._con.execute(
            "SELECT parquet_path FROM catalog WHERE group_name = ?", [group]
        ).fetchone()
        if row is None:
            raise KeyError(f"Feature group '{group}' not found in store.")
        parquet_path = self.store_path / row[0]
        df = pd.read_parquet(parquet_path, columns=columns)
        return df

    def delete(self, group: str) -> None:
        """Remove a feature group and all associated data."""
        self._con.execute("DELETE FROM catalog WHERE group_name = ?", [group])
        group_dir = self._group_dir(group)
        if group_dir.exists():
            shutil.rmtree(group_dir)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def get_tags(self, group: str) -> list[str]:
        return list_tags(self.store_path, group)

    def tag(self, group: str, tag: str) -> None:
        add_tag(self.store_path, group, tag)

    def untag(self, group: str, tag: str) -> None:
        remove_tag(self.store_path, group, tag)

    def get_lineage(self, group: str) -> dict[str, Any] | None:
        """Return recorded lineage metadata for a feature group."""
        return get_lineage(self.store_path, group)

    def get_history(self, group: str) -> list[dict[str, Any]]:
        """Return version history for a feature group."""
        return get_version_history(self.store_path, group)
