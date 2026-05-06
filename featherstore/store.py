"""FeatherStore — core store class assembling all mixins."""

import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from featherstore.store_export import ExportMixin
from featherstore.store_merge import MergeMixin
from featherstore.store_profile import ProfileMixin
from featherstore.store_validate import ValidateMixin
from featherstore.store_sampling import SamplingMixin
from featherstore.store_compare import CompareMixin
from featherstore.store_schedule import ScheduleMixin
from featherstore.store_partition import PartitionMixin
from featherstore.store_pin import PinMixin
from featherstore.store_ttl import TTLMixin
from featherstore.store_audit import AuditMixin
from featherstore.store_notify import NotifyMixin
from featherstore.store_lock import LockMixin
from featherstore.store_bookmark import BookmarkMixin
from featherstore.store_quota import QuotaMixin
from featherstore.store_dependency import DependencyMixin
from featherstore.store_freshness import FreshnessMixin
from featherstore.store_archive import ArchiveMixin
from featherstore.store_retention import RetentionMixin
from featherstore.store_label import LabelMixin
from featherstore.store_alert import AlertMixin
from featherstore.store_annotation import AnnotationMixin
from featherstore.store_watchlist import WatchlistMixin
from featherstore.store_badge import BadgeMixin
from featherstore.store_category import CategoryMixin
from featherstore.store_trust import TrustMixin
from featherstore.store_score import ScoreMixin
from featherstore.store_flag import FlagMixin
from featherstore.store_visibility import VisibilityMixin


class FeatherStore(
    ExportMixin,
    MergeMixin,
    ProfileMixin,
    ValidateMixin,
    SamplingMixin,
    CompareMixin,
    ScheduleMixin,
    PartitionMixin,
    PinMixin,
    TTLMixin,
    AuditMixin,
    NotifyMixin,
    LockMixin,
    BookmarkMixin,
    QuotaMixin,
    DependencyMixin,
    FreshnessMixin,
    ArchiveMixin,
    RetentionMixin,
    LabelMixin,
    AlertMixin,
    AnnotationMixin,
    WatchlistMixin,
    BadgeMixin,
    CategoryMixin,
    TrustMixin,
    ScoreMixin,
    FlagMixin,
    VisibilityMixin,
):
    """Lightweight local feature store backed by DuckDB and Parquet."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        Path(store_path).mkdir(parents=True, exist_ok=True)
        self._catalog = self._load_catalog()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _catalog_path(self) -> Path:
        return Path(self.store_path) / "_catalog.json"

    def _load_catalog(self) -> dict:
        p = self._catalog_path()
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    def _save_catalog(self) -> None:
        with open(self._catalog_path(), "w") as f:
            json.dump(self._catalog, f, indent=2)

    def _group_dir(self, group: str) -> Path:
        return Path(self.store_path) / group

    def _parquet_path(self, group: str) -> Path:
        return self._group_dir(group) / "data.parquet"

    # ------------------------------------------------------------------
    # Core save / load
    # ------------------------------------------------------------------

    def save(
        self,
        group: str,
        df: pd.DataFrame,
        source: str | None = None,
        transform: str | None = None,
        compression: str = "snappy",
    ) -> None:
        """Persist a DataFrame as a named feature group."""
        group_dir = self._group_dir(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self._parquet_path(group), compression=compression)
        self._catalog[group] = {
            "columns": list(df.columns),
            "rows": len(df),
            "compression": compression,
        }
        self._save_catalog()

        # Optional integrations
        from featherstore import lineage as _lineage
        from featherstore import stats as _stats
        from featherstore import versioning as _versioning

        if source or transform:
            _lineage.record_lineage(self.store_path, group, source=source, transform=transform)
        _stats.record_stats(self.store_path, group, df)
        _versioning.record_version(self.store_path, group, metadata={"rows": len(df), "columns": len(df.columns)})

    def load(
        self,
        group: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load a feature group from the store."""
        if group not in self._catalog:
            raise KeyError(f"Group '{group}' not found in store.")
        path = str(self._parquet_path(group))
        query = f"SELECT {', '.join(columns) if columns else '*'} FROM read_parquet('{path}')"
        con = duckdb.connect()
        return con.execute(query).df()

    def delete(self, group: str) -> None:
        """Remove a feature group from the store."""
        import shutil
        if group not in self._catalog:
            raise KeyError(f"Group '{group}' not found in store.")
        shutil.rmtree(self._group_dir(group))
        del self._catalog[group]
        self._save_catalog()

    def list_groups(self) -> list[str]:
        """Return all saved group names."""
        return list(self._catalog.keys())

    def _init_catalog(self) -> None:
        self._catalog = {}
        self._save_catalog()
