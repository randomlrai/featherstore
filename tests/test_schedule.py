"""Tests for featherstore.schedule and ScheduleMixin."""

from __future__ import annotations

import pytest

from featherstore.schedule import (
    list_schedules,
    load_schedules,
    mark_run,
    register_schedule,
    remove_schedule,
)


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# load_schedules
# ---------------------------------------------------------------------------

def test_load_schedules_missing_returns_empty(store_path):
    assert load_schedules(store_path) == {}


# ---------------------------------------------------------------------------
# register_schedule
# ---------------------------------------------------------------------------

def test_register_schedule_creates_entry(store_path):
    entry = register_schedule(store_path, "users", "0 2 * * *")
    assert entry["group"] == "users"
    assert entry["cron"] == "0 2 * * *"
    assert entry["last_run"] is None


def test_register_schedule_persists(store_path):
    register_schedule(store_path, "users", "0 2 * * *", description="nightly")
    schedules = load_schedules(store_path)
    assert "users" in schedules
    assert schedules["users"]["description"] == "nightly"


def test_register_schedule_overwrites_existing(store_path):
    register_schedule(store_path, "users", "0 2 * * *")
    register_schedule(store_path, "users", "0 4 * * *")
    schedules = load_schedules(store_path)
    assert schedules["users"]["cron"] == "0 4 * * *"


def test_register_multiple_groups(store_path):
    register_schedule(store_path, "users", "0 1 * * *")
    register_schedule(store_path, "orders", "0 2 * * *")
    assert len(load_schedules(store_path)) == 2


# ---------------------------------------------------------------------------
# remove_schedule
# ---------------------------------------------------------------------------

def test_remove_schedule_returns_true_when_exists(store_path):
    register_schedule(store_path, "users", "0 1 * * *")
    assert remove_schedule(store_path, "users") is True
    assert "users" not in load_schedules(store_path)


def test_remove_schedule_returns_false_when_missing(store_path):
    assert remove_schedule(store_path, "nonexistent") is False


# ---------------------------------------------------------------------------
# mark_run
# ---------------------------------------------------------------------------

def test_mark_run_updates_last_run(store_path):
    register_schedule(store_path, "users", "0 1 * * *")
    mark_run(store_path, "users")
    schedules = load_schedules(store_path)
    assert schedules["users"]["last_run"] is not None


def test_mark_run_raises_for_unknown_group(store_path):
    with pytest.raises(KeyError, match="No schedule registered"):
        mark_run(store_path, "ghost")


# ---------------------------------------------------------------------------
# list_schedules
# ---------------------------------------------------------------------------

def test_list_schedules_empty(store_path):
    assert list_schedules(store_path) == []


def test_list_schedules_returns_all(store_path):
    register_schedule(store_path, "a", "* * * * *")
    register_schedule(store_path, "b", "* * * * *")
    result = list_schedules(store_path)
    groups = {e["group"] for e in result}
    assert groups == {"a", "b"}
