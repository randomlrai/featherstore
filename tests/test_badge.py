"""Unit tests for featherstore/badge.py."""

import pytest
from featherstore.badge import (
    load_badges,
    save_badges,
    award_badge,
    revoke_badge,
    get_badges,
    list_groups_with_badge,
    clear_badges,
    VALID_BADGES,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_badges_missing_returns_empty(store_path):
    assert load_badges(store_path) == {}


def test_save_and_load_badges_roundtrip(store_path):
    data = {"my_group": {"group": "my_group", "badges": ["gold"]}}
    save_badges(store_path, data)
    assert load_badges(store_path) == data


def test_award_badge_creates_entry(store_path):
    entry = award_badge(store_path, "features", "gold")
    assert "gold" in entry["badges"]
    assert entry["group"] == "features"


def test_award_badge_persists(store_path):
    award_badge(store_path, "features", "stable")
    badges = load_badges(store_path)
    assert "stable" in badges["features"]["badges"]


def test_award_badge_no_duplicates(store_path):
    award_badge(store_path, "features", "gold")
    award_badge(store_path, "features", "gold")
    badges = get_badges(store_path, "features")
    assert badges.count("gold") == 1


def test_award_multiple_badges(store_path):
    award_badge(store_path, "features", "gold")
    award_badge(store_path, "features", "stable")
    badges = get_badges(store_path, "features")
    assert "gold" in badges
    assert "stable" in badges


def test_award_invalid_badge_raises(store_path):
    with pytest.raises(ValueError, match="Invalid badge"):
        award_badge(store_path, "features", "platinum")


def test_revoke_badge_removes_it(store_path):
    award_badge(store_path, "features", "gold")
    award_badge(store_path, "features", "silver")
    revoke_badge(store_path, "features", "gold")
    badges = get_badges(store_path, "features")
    assert "gold" not in badges
    assert "silver" in badges


def test_revoke_badge_unknown_group_raises(store_path):
    with pytest.raises(KeyError):
        revoke_badge(store_path, "nonexistent", "gold")


def test_get_badges_unknown_group_returns_empty(store_path):
    assert get_badges(store_path, "ghost") == []


def test_list_groups_with_badge(store_path):
    award_badge(store_path, "features", "gold")
    award_badge(store_path, "labels", "gold")
    award_badge(store_path, "meta", "silver")
    result = list_groups_with_badge(store_path, "gold")
    assert set(result) == {"features", "labels"}


def test_list_groups_with_badge_none_match(store_path):
    award_badge(store_path, "features", "silver")
    result = list_groups_with_badge(store_path, "gold")
    assert result == []


def test_clear_badges_removes_group(store_path):
    award_badge(store_path, "features", "gold")
    clear_badges(store_path, "features")
    assert get_badges(store_path, "features") == []


def test_clear_badges_unknown_group_is_noop(store_path):
    clear_badges(store_path, "ghost")  # should not raise


def test_award_badge_records_awarded_by(store_path):
    entry = award_badge(store_path, "features", "gold", awarded_by="alice")
    assert entry["awarded_by"] == "alice"


def test_valid_badges_set_is_nonempty():
    assert len(VALID_BADGES) > 0
