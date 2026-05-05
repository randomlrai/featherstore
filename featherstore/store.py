"""FeatherStore — core store class (updated to include BadgeMixin)."""

import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from featherstore.versioning import record_version, get_version_history
from featherstore.tags import add_tag, load_tags
from featherstore.lineage import record_lineage, get_lineage
from featherstore.snapshots import create_snapshot, load_snapshots
from featherstore.stats import record_stats, load_stats
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
):
    def __init__(self, store_path: str):
        self.store_path = store_path
        Path(store_path).mkdir(parents=True, exist_ok=True)
        self._init_catalog()

    def _init_catalog(self):
        meta_dir = Path(self.store_path) / ".featherstore"
        meta_dir.mkdir(exist_ok=True)
        if not self._catalog_path.exists():
            self._catalog_path.write_text(json.dumps({}))

    @property
    def _catalog_path(self) -> Path:
        return Path(self.store_path) / ".featherstore" / "catalog.json"

    def _load_catalog(self) -> dict:
        return json.loads(self._catalog_path.read_text())

    def _save_catalog(self, catalog: dict) -> None:
        self._catalog_path.write_text(json.dumps(catalog, indent=2))

    def _group_path(self, group: str) -> Path:
        return Path(self.store_path) / group

    def _parquet_path(self, group: str) -> Path:
        return self._group_path(group) / "data.parquet"

    def save(self, group: str, df: pd.DataFrame, tags: list = None,
             source: str = None, transform: str = None) -> None:
        group_dir = self._group_path(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self._parquet_path(group))
        catalog = self._load_catalog()
        catalog[group] = {"columns": list(df.columns), "rows": len(df)}
        self._save_catalog(catalog)
        record_version(self.store_path, group, metadata={"rows": len(df), "cols": len(df.columns)})
        record_stats(self.store_path, group, df)
        if tags:
            for tag in tags:
                add_tag(self.store_path, group, tag)
        if source or transform:
            record_lineage(self.store_path, group, source=source, transform=transform)

    def load(self, group: str, columns: list = None) -> pd.DataFrame:
        if not self._parquet_path(group).exists():
            raise KeyError(f"Group '{group}' not found in store.")
        con = duckdb.connect()
        path = str(self._parquet_path(group))
        if columns:
            cols = ", ".join(f'"{c}"' for c in columns)
            query = f"SELECT {cols} FROM read_parquet('{path}')"
        else:
            query = f"SELECT * FROM read_parquet('{path}')"
        return con.execute(query).df()

    def delete(self, group: str) -> None:
        group_dir = self._group_path(group)
        if group_dir.exists():
            shutil.rmtree(group_dir)
        catalog = self._load_catalog()
        catalog.pop(group, None)
        self._save_catalog(catalog)

    def list_groups(self) -> list:
        return list(self._load_catalog().keys())

    def get_version_history(self, group: str) -> list:
        return get_version_history(self.store_path, group)

    def get_lineage(self, group: str):
        return get_lineage(self.store_path, group)

    def snapshot(self, group: str, label: str = None) -> dict:
        return create_snapshot(self.store_path, group, label=label)

    def list_snapshots(self, group: str) -> list:
        snaps = load_snapshots(self.store_path)
        return snaps.get(group, [])

    def get_stats(self, group: str):
        stats = load_stats(self.store_path)
        return stats.get(group)
