"""Core FeatherStore class – lightweight local feature store."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from featherstore.versioning import record_version, get_version_history
from featherstore.tags import add_tag, load_tags
from featherstore.lineage import record_lineage, get_lineage
from featherstore.snapshots import create_snapshot, load_snapshots
from featherstore.stats import compute_stats, record_stats, load_stats
from featherstore.store_export import ExportMixin

_CATALOG_FILE = "catalog.json"


class FeatherStore(ExportMixin):
    """Lightweight local feature store backed by Parquet files."""

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._init_catalog()

    # ------------------------------------------------------------------
    # Catalog helpers
    # ------------------------------------------------------------------

    def _init_catalog(self) -> None:
        if not self._catalog_path.exists():
            self._save_catalog({})

    @property
    def _catalog_path(self) -> Path:
        return self.store_path / _CATALOG_FILE

    def _load_catalog(self) -> dict:
        return json.loads(self._catalog_path.read_text())

    def _save_catalog(self, catalog: dict) -> None:
        self._catalog_path.write_text(json.dumps(catalog, indent=2, default=str))

    # ------------------------------------------------------------------
    # Core save / load
    # ------------------------------------------------------------------

    def save(
        self,
        group: str,
        df: pd.DataFrame,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        transform: Optional[str] = None,
    ) -> dict:
        """Persist *df* as feature group *group*."""
        group_dir = self.store_path / group
        group_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = group_dir / "data.parquet"
        df.to_parquet(parquet_path, index=False)

        catalog = self._load_catalog()
        catalog[group] = {
            "group": group,
            "path": str(parquet_path),
            "rows": len(df),
            "columns": list(df.columns),
        }
        self._save_catalog(catalog)

        version_meta = record_version(self.store_path, group, parquet_path)

        if tags:
            for tag in tags:
                add_tag(self.store_path, group, tag)
            catalog[group]["tags"] = tags
            self._save_catalog(catalog)

        lineage_entry = None
        if source or transform:
            lineage_entry = record_lineage(
                self.store_path, group, source=source, transform=transform
            )

        record_stats(self.store_path, group, df)

        return version_meta

    def load(
        self,
        group: str,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Load feature group *group* from the store."""
        catalog = self._load_catalog()
        if group not in catalog:
            raise KeyError(f"Group '{group}' not found in the store.")

        parquet_path = Path(catalog[group]["path"])
        df = pd.read_parquet(parquet_path, columns=columns)
        return df

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, group: str, label: Optional[str] = None) -> dict:
        """Create a snapshot of the current state of *group*."""
        catalog = self._load_catalog()
        if group not in catalog:
            raise KeyError(f"Group '{group}' not found in the store.")
        parquet_path = Path(catalog[group]["path"])
        return create_snapshot(self.store_path, group, parquet_path, label=label)

    def list_snapshots(self, group: str) -> list[dict]:
        """Return snapshot history for *group*."""
        snaps = load_snapshots(self.store_path)
        return snaps.get(group, [])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, group: str) -> Optional[dict]:
        """Return recorded stats for *group*, or None if not yet saved."""
        all_stats = load_stats(self.store_path)
        return all_stats.get(group)

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def lineage(self, group: str) -> Optional[dict]:
        """Return lineage record for *group*, or None."""
        return get_lineage(self.store_path, group)
