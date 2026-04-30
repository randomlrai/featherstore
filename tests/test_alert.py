"""Tests for featherstore.alert and AlertMixin."""

import pytest
from featherstore.alert import (
    load_alerts,
    add_alert,
    remove_alert,
    evaluate_alert,
    evaluate_all_alerts,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_alerts_missing_returns_empty(store_path):
    assert load_alerts(store_path) == {}


def test_add_alert_creates_entry(store_path):
    entry = add_alert(store_path, "users", "row_count", ">", 1000)
    assert entry["group"] == "users"
    assert entry["metric"] == "row_count"
    assert entry["op"] == ">"
    assert entry["threshold"] == 1000


def test_add_alert_persists(store_path):
    add_alert(store_path, "users", "row_count", ">", 500, label="big_users")
    alerts = load_alerts(store_path)
    assert "big_users" in alerts


def test_add_alert_custom_label(store_path):
    entry = add_alert(store_path, "g", "col_count", ">=", 5, label="my_alert")
    assert entry["label"] == "my_alert"


def test_add_alert_auto_label(store_path):
    entry = add_alert(store_path, "g", "row_count", "<", 10)
    assert "g.row_count" in entry["label"]


def test_add_alert_invalid_op_raises(store_path):
    with pytest.raises(ValueError, match="Invalid operator"):
        add_alert(store_path, "g", "row_count", "~", 10)


def test_add_alert_overwrites_same_label(store_path):
    add_alert(store_path, "g", "row_count", ">", 10, label="lbl")
    add_alert(store_path, "g", "row_count", ">", 999, label="lbl")
    alerts = load_alerts(store_path)
    assert alerts["lbl"]["threshold"] == 999


def test_remove_alert_returns_true(store_path):
    add_alert(store_path, "g", "row_count", ">", 10, label="x")
    assert remove_alert(store_path, "x") is True
    assert "x" not in load_alerts(store_path)


def test_remove_alert_missing_returns_false(store_path):
    assert remove_alert(store_path, "nonexistent") is False


def test_evaluate_alert_fires_when_condition_met():
    alert = {"label": "a", "metric": "row_count", "op": ">", "threshold": 100}
    result = evaluate_alert(alert, {"row_count": 200})
    assert result["fired"] is True
    assert result["value"] == 200


def test_evaluate_alert_does_not_fire_when_not_met():
    alert = {"label": "a", "metric": "row_count", "op": ">", "threshold": 100}
    result = evaluate_alert(alert, {"row_count": 50})
    assert result["fired"] is False


def test_evaluate_alert_missing_metric():
    alert = {"label": "a", "metric": "missing", "op": ">", "threshold": 0}
    result = evaluate_alert(alert, {})
    assert result["fired"] is False
    assert "not found" in result["reason"]


def test_evaluate_all_alerts_filters_by_group(store_path):
    add_alert(store_path, "users", "row_count", ">", 10, label="u1")
    add_alert(store_path, "orders", "row_count", "<", 5, label="o1")
    results = evaluate_all_alerts(store_path, "users", {"row_count": 100})
    labels = [r["label"] for r in results]
    assert "u1" in labels
    assert "o1" not in labels
