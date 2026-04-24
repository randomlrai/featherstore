"""Core FeatherStore — save/load Pandas DataFrames as Parquet with DuckDB queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from featherstore.versioning import record_version
from featherstore.lineage import record_lineage
from featherstore.snapshots import create_snapshot, restore_snapshot, list_snapshots


class FeatherStore:
    """Lightweight local feature store backed by Parquet files."""

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self._meta_dir = self.store_path / ".featherstore"
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = self._meta_dir / "catalog.json"
        self._init_catalog()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_catalog(self) -> None:
        if not self._catalog_path.exists():
            self._catalog_path.write_text(json.dumps({}), encoding="utf-8")

    def _load_catalog(self) -> dict:
        return json.loads(self._catalog_path.read_text(encoding="utf-8"))

    def _save_catalog(self, catalog: dict) -> None:
        self._catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    def _group_dir(self, group: str) -> Path:
        return self.store_path / group

    def _parquet_path(self, group: str) -> Path:
        return self._group_dir(group) / "data.parquet"

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(
        self,
        df: pd.DataFrame,
        group: str,
        *,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        transform: Optional[str] = None,
    ) -> None:
        """Persist *df* under *group*, updating catalog, versioning, and lineage."""
        dest = self._parquet_path(group)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, dest)

        catalog = self._load_catalog()
        catalog[group] = {
            "columns": df.columns.tolist(),
            "rows": len(df),
            "tags": tags or [],
        }
        self._save_catalog(catalog)

        record_version(self.store_path, group, {"rows": len(df), "columns": df.columns.tolist()})

        if source or transform:
            record_lineage(
                self.store_path,
                group,
                source=source,
                transform=transform,
            )

    def load(
        self,
        group: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load features for *group*, optionally selecting *columns*."""
        path = self._parquet_path(group)
        if not path.exists():
            raise KeyError(f"Group '{group}' not found in store at {self.store_path}")
        table = pq.read_table(path, columns=columns)
        return table.to_pandas()

    def query(self, sql: str) -> pd.DataFrame:
        """Run an arbitrary DuckDB SQL query against all parquet files in the store."""
        con = duckdb.connect()
        catalog = self._load_catalog()
        for group in catalog:
            parquet_file = str(self._parquet_path(group))
            safe_name = group.replace("/", "__")
            con.execute(f"CREATE VIEW {safe_name} AS SELECT * FROM read_parquet('{parquet_file}')")
        return con.execute(sql).df()

    # ------------------------------------------------------------------
    # Snapshot convenience wrappers
    # ------------------------------------------------------------------

    def snapshot(self, group: str, snapshot_name: str) -> dict:
        """Create a named snapshot of the current state of *group*."""
        return create_snapshot(self.store_path, group, snapshot_name)

    def restore(self, group: str, snapshot_name: str) -> None:
        """Restore *group* to a previously created snapshot."""
        restore_snapshot(self.store_path, group, snapshot_name)

    def list_snapshots(self, group: str) -> list[dict]:
        """Return all snapshots for *group* sorted by creation time."""
        return list_snapshots(self.store_path, group)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def groups(self) -> List[str]:
        """Return all registered group names."""
        return list(self._load_catalog().keys())
