"""FeatherStore — core store class with all mixins."""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from featherstore.versioning import record_version, get_version_history
from featherstore.tags import add_tag, remove_tag, load_tags
from featherstore.lineage import record_lineage, get_lineage
from featherstore.snapshots import create_snapshot, restore_snapshot, load_snapshots
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
):
    def __init__(self, store_path: str):
        self.store_path = store_path
        Path(store_path).mkdir(parents=True, exist_ok=True)
        self._init_catalog()

    def _init_catalog(self):
        meta_dir = Path(self.store_path) / ".featherstore"
        meta_dir.mkdir(exist_ok=True)
        if not self._catalog_path.exists():
            self._save_catalog({})

    @property
    def _catalog_path(self) -> Path:
        return Path(self.store_path) / ".featherstore" / "catalog.json"

    def _load_catalog(self) -> dict:
        with open(self._catalog_path, "r") as f:
            return json.load(f)

    def _save_catalog(self, catalog: dict) -> None:
        with open(self._catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

    def _group_path(self, group: str) -> Path:
        return Path(self.store_path) / group

    def _parquet_path(self, group: str) -> Path:
        return self._group_path(group) / "data.parquet"

    def save(self, df: pd.DataFrame, group: str, source: str = None, transform: str = None) -> None:
        group_dir = self._group_path(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, self._parquet_path(group))
        catalog = self._load_catalog()
        catalog[group] = {"columns": list(df.columns), "rows": len(df)}
        self._save_catalog(catalog)
        record_version(self.store_path, group, {"rows": len(df), "columns": list(df.columns)})
        record_stats(self.store_path, group, df)
        if source or transform:
            record_lineage(self.store_path, group, source=source, transform=transform)

    def load(self, group: str, columns: List[str] = None) -> pd.DataFrame:
        catalog = self._load_catalog()
        if group not in catalog:
            raise KeyError(f"Group '{group}' not found in store.")
        path = self._parquet_path(group)
        table = pq.read_table(path, columns=columns)
        return table.to_pandas()

    def delete(self, group: str) -> None:
        import shutil
        catalog = self._load_catalog()
        if group not in catalog:
            raise KeyError(f"Group '{group}' not found.")
        shutil.rmtree(self._group_path(group))
        del catalog[group]
        self._save_catalog(catalog)

    def list_groups(self) -> List[str]:
        return list(self._load_catalog().keys())

    def query(self, group: str, sql_filter: str) -> pd.DataFrame:
        path = str(self._parquet_path(group))
        con = duckdb.connect()
        result = con.execute(f"SELECT * FROM read_parquet('{path}') WHERE {sql_filter}").fetchdf()
        return result

    def version_history(self, group: str) -> list:
        return get_version_history(self.store_path, group)

    def tag(self, group: str, tag: str) -> None:
        add_tag(self.store_path, group, tag)

    def untag(self, group: str, tag: str) -> None:
        remove_tag(self.store_path, group, tag)

    def get_tags(self, group: str) -> list:
        tags = load_tags(self.store_path)
        return tags.get(group, {}).get("tags", [])

    def get_lineage(self, group: str):
        return get_lineage(self.store_path, group)

    def snapshot(self, group: str, label: str = None) -> dict:
        return create_snapshot(self.store_path, group, label=label)

    def restore(self, group: str, snapshot_id: str) -> pd.DataFrame:
        return restore_snapshot(self.store_path, group, snapshot_id)

    def list_snapshots(self, group: str) -> list:
        snaps = load_snapshots(self.store_path)
        return snaps.get(group, [])

    def get_stats(self, group: str):
        stats = load_stats(self.store_path)
        return stats.get(group)
