"""FeatherStore — core store class."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from featherstore.versioning import record_version, get_version_history
from featherstore.tags import add_tag, load_tags
from featherstore.lineage import record_lineage, get_lineage
from featherstore.snapshots import create_snapshot, load_snapshots
from featherstore.stats import compute_stats, record_stats, load_stats
from featherstore.store_export import ExportMixin
from featherstore.store_merge import MergeMixin


class FeatherStore(ExportMixin, MergeMixin):
    """Lightweight local feature store backed by Parquet files."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._init_catalog()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_catalog(self) -> None:
        if not os.path.exists(self._catalog_path()):
            self._save_catalog({})

    def _catalog_path(self) -> str:
        return os.path.join(self.path, "catalog.json")

    def _load_catalog(self) -> Dict:
        with open(self._catalog_path(), "r") as fh:
            return json.load(fh)

    def _save_catalog(self, catalog: Dict) -> None:
        with open(self._catalog_path(), "w") as fh:
            json.dump(catalog, fh, indent=2)

    def _group_path(self, group: str) -> str:
        return os.path.join(self.path, f"{group}.parquet")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        group: str,
        df: pd.DataFrame,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        transform: Optional[str] = None,
    ) -> None:
        """Persist a DataFrame as a named feature group."""
        df.to_parquet(self._group_path(group), index=False)

        catalog = self._load_catalog()
        catalog[group] = {
            "path": self._group_path(group),
            "columns": list(df.columns),
            "tags": tags or [],
        }
        self._save_catalog(catalog)

        record_version(self.path, group, {"rows": len(df), "cols": len(df.columns)})

        if tags:
            for tag in tags:
                add_tag(self.path, group, tag)

        if source or transform:
            record_lineage(
                self.path,
                group,
                source=source,
                transform=transform,
            )

        stats = compute_stats(df)
        record_stats(self.path, group, stats)

    def load(
        self,
        group: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load a feature group, optionally selecting columns."""
        catalog = self._load_catalog()
        if group not in catalog:
            raise KeyError(f"Group {group!r} not found in store.")

        df = pd.read_parquet(self._group_path(group))
        if columns is not None:
            df = df[columns]
        return df

    def delete(self, group: str) -> None:
        """Remove a feature group from the store."""
        catalog = self._load_catalog()
        if group not in catalog:
            raise KeyError(f"Group {group!r} not found in store.")
        os.remove(self._group_path(group))
        del catalog[group]
        self._save_catalog(catalog)

    def list_groups(self) -> List[str]:
        """Return names of all stored feature groups."""
        return list(self._load_catalog().keys())

    def version_history(self, group: str) -> List[Dict]:
        return get_version_history(self.path, group)

    def lineage(self, group: str):
        return get_lineage(self.path, group)

    def snapshot(self, group: str, label: Optional[str] = None) -> Dict:
        df = self.load(group)
        return create_snapshot(self.path, group, df, label=label)

    def list_snapshots(self, group: str) -> List[Dict]:
        return load_snapshots(self.path, group)

    def stats(self, group: str) -> Optional[Dict]:
        all_stats = load_stats(self.path, group)
        return all_stats[-1] if all_stats else None

    def tags(self, group: str) -> List[str]:
        return load_tags(self.path, group).get(group, [])
