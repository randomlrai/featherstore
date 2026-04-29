"""Tests for featherstore.audit and AuditMixin."""

import pytest

from featherstore.audit import (
    clear_audit_log,
    get_audit_history,
    load_audit_log,
    record_event,
    save_audit_log,
)


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


def test_load_audit_log_missing_returns_empty(store_path):
    assert load_audit_log(store_path) == []


def test_save_and_load_audit_log_roundtrip(store_path):
    entries = [{"timestamp": "t", "group": "g", "operation": "read", "user": None, "details": {}}]
    save_audit_log(store_path, entries)
    assert load_audit_log(store_path) == entries


def test_record_event_appends_entry(store_path):
    record_event(store_path, group="features", operation="write")
    log = load_audit_log(store_path)
    assert len(log) == 1
    assert log[0]["group"] == "features"
    assert log[0]["operation"] == "write"


def test_record_event_increments(store_path):
    record_event(store_path, group="a", operation="read")
    record_event(store_path, group="a", operation="write")
    assert len(load_audit_log(store_path)) == 2


def test_record_event_stores_user_and_details(store_path):
    entry = record_event(
        store_path, group="g", operation="export",
        user="alice", details={"format": "csv"}
    )
    assert entry["user"] == "alice"
    assert entry["details"] == {"format": "csv"}


def test_record_event_invalid_operation_raises(store_path):
    with pytest.raises(ValueError, match="Unsupported operation"):
        record_event(store_path, group="g", operation="unknown")


def test_get_audit_history_filter_by_group(store_path):
    record_event(store_path, group="a", operation="read")
    record_event(store_path, group="b", operation="write")
    result = get_audit_history(store_path, group="a")
    assert all(e["group"] == "a" for e in result)
    assert len(result) == 1


def test_get_audit_history_filter_by_operation(store_path):
    record_event(store_path, group="a", operation="read")
    record_event(store_path, group="b", operation="delete")
    result = get_audit_history(store_path, operation="delete")
    assert len(result) == 1
    assert result[0]["operation"] == "delete"


def test_get_audit_history_no_filter_returns_all(store_path):
    for op in ("read", "write", "snapshot"):
        record_event(store_path, group="g", operation=op)
    assert len(get_audit_history(store_path)) == 3


def test_clear_audit_log(store_path):
    record_event(store_path, group="g", operation="read")
    clear_audit_log(store_path)
    assert load_audit_log(store_path) == []
