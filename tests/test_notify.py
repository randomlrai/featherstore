"""Tests for featherstore.notify and NotifyMixin."""

import pytest

from featherstore.notify import (
    fire_hooks,
    list_hooks,
    load_hooks,
    register_hook,
    remove_hook,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_hooks_missing_returns_empty(store_path):
    assert load_hooks(store_path) == {}


def test_register_hook_creates_entry(store_path):
    entry = register_hook(store_path, "on_save", "my_hook", "mymod.fn")
    assert entry["label"] == "my_hook"
    assert entry["callback_ref"] == "mymod.fn"


def test_register_hook_persists(store_path):
    register_hook(store_path, "on_save", "hook1", "mymod.fn")
    hooks = load_hooks(store_path)
    assert any(h["label"] == "hook1" for h in hooks["on_save"])


def test_register_hook_no_duplicates(store_path):
    register_hook(store_path, "on_save", "hook1", "mymod.fn")
    register_hook(store_path, "on_save", "hook1", "mymod.fn2")
    hooks = load_hooks(store_path)
    labels = [h["label"] for h in hooks["on_save"]]
    assert labels.count("hook1") == 1


def test_register_hook_multiple_events(store_path):
    register_hook(store_path, "on_save", "h1", "mod.a")
    register_hook(store_path, "on_delete", "h2", "mod.b")
    hooks = load_hooks(store_path)
    assert "on_save" in hooks
    assert "on_delete" in hooks


def test_remove_hook_returns_true(store_path):
    register_hook(store_path, "on_save", "hook1", "mod.fn")
    result = remove_hook(store_path, "on_save", "hook1")
    assert result is True


def test_remove_hook_missing_returns_false(store_path):
    result = remove_hook(store_path, "on_save", "nonexistent")
    assert result is False


def test_remove_hook_removes_entry(store_path):
    register_hook(store_path, "on_save", "hook1", "mod.fn")
    remove_hook(store_path, "on_save", "hook1")
    hooks = load_hooks(store_path)
    assert all(h["label"] != "hook1" for h in hooks.get("on_save", []))


def test_list_hooks_all(store_path):
    register_hook(store_path, "on_save", "h1", "mod.a")
    register_hook(store_path, "on_load", "h2", "mod.b")
    result = list_hooks(store_path)
    assert "on_save" in result
    assert "on_load" in result


def test_list_hooks_filtered_by_event(store_path):
    register_hook(store_path, "on_save", "h1", "mod.a")
    register_hook(store_path, "on_load", "h2", "mod.b")
    result = list_hooks(store_path, event="on_save")
    assert "on_save" in result
    assert "on_load" not in result


def test_fire_hooks_calls_callable(store_path):
    called = []
    register_hook(store_path, "on_save", "recorder", "test.recorder")
    fired = fire_hooks(
        store_path,
        "on_save",
        context={"group": "features"},
        registry={"test.recorder": lambda event, context: called.append(context)},
    )
    assert "recorder" in fired
    assert called[0]["group"] == "features"


def test_fire_hooks_returns_empty_for_unknown_event(store_path):
    fired = fire_hooks(store_path, "on_unknown")
    assert fired == []


def test_fire_hooks_skips_bad_callable(store_path):
    register_hook(store_path, "on_save", "bad", "nonexistent.module.fn")
    fired = fire_hooks(store_path, "on_save")
    assert fired == []
