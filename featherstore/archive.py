"""Archive and restore groups to/from compressed zip archives."""

import json
import zipfile
from pathlib import Path
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _group_path(store_path: str, group: str) -> Path:
    return Path(store_path) / group


def archive_group(store_path: str, group: str, dest: str) -> dict:
    """Archive a group directory to a zip file. Returns archive metadata."""
    group_dir = _group_path(store_path, group)
    if not group_dir.exists():
        raise FileNotFoundError(f"Group '{group}' does not exist in store.")

    dest_path = Path(dest)
    if dest_path.suffix != ".zip":
        dest_path = dest_path.with_suffix(".zip")

    files_archived = []
    with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(group_dir.rglob("*")):
            if file.is_file():
                arcname = file.relative_to(group_dir)
                zf.write(file, arcname)
                files_archived.append(str(arcname))

        meta = {
            "group": group,
            "archived_at": _now_iso(),
            "store_path": str(store_path),
            "files": files_archived,
        }
        zf.writestr("_archive_meta.json", json.dumps(meta, indent=2))

    meta["archive_path"] = str(dest_path)
    meta["size_bytes"] = dest_path.stat().st_size
    return meta


def restore_group(store_path: str, archive_path: str, group: str | None = None) -> dict:
    """Restore a group from a zip archive. Returns restore metadata."""
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
        if "_archive_meta.json" in names:
            meta = json.loads(zf.read("_archive_meta.json"))
            target_group = group or meta.get("group", "restored_group")
        else:
            target_group = group or "restored_group"
            meta = {}

        target_dir = _group_path(store_path, target_group)
        target_dir.mkdir(parents=True, exist_ok=True)

        for name in names:
            if name == "_archive_meta.json":
                continue
            zf.extract(name, target_dir)

    return {
        "group": target_group,
        "restored_at": _now_iso(),
        "archive_path": str(archive),
        "files_restored": [n for n in names if n != "_archive_meta.json"],
    }


def list_archive_contents(archive_path: str) -> dict:
    """Inspect the contents of an archive without extracting."""
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
        meta = {}
        if "_archive_meta.json" in names:
            meta = json.loads(zf.read("_archive_meta.json"))

    return {
        "archive_path": str(archive),
        "files": [n for n in names if n != "_archive_meta.json"],
        "meta": meta,
    }
