"""Tests for featherstore/comment.py"""

import pytest

from featherstore.comment import (
    add_comment,
    clear_comments,
    get_comments,
    load_comments,
    remove_comment,
    save_comments,
)


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path)


def test_load_comments_missing_returns_empty(store_path):
    result = load_comments(store_path)
    assert result == {}


def test_save_and_load_comments_roundtrip(store_path):
    data = {"features": [{"id": "abc", "text": "hello", "author": None, "created_at": "2024-01-01"}]}
    save_comments(store_path, data)
    loaded = load_comments(store_path)
    assert loaded == data


def test_add_comment_creates_entry(store_path):
    entry = add_comment(store_path, "my_group", "looks good")
    comments = get_comments(store_path, "my_group")
    assert len(comments) == 1
    assert comments[0]["text"] == "looks good"
    assert comments[0]["id"] == entry["id"]


def test_add_comment_with_author(store_path):
    add_comment(store_path, "grp", "nice feature", author="alice")
    comments = get_comments(store_path, "grp")
    assert comments[0]["author"] == "alice"


def test_add_comment_no_author_is_none(store_path):
    add_comment(store_path, "grp", "no author")
    comments = get_comments(store_path, "grp")
    assert comments[0]["author"] is None


def test_add_multiple_comments(store_path):
    add_comment(store_path, "grp", "first")
    add_comment(store_path, "grp", "second")
    add_comment(store_path, "grp", "third")
    comments = get_comments(store_path, "grp")
    assert len(comments) == 3
    assert [c["text"] for c in comments] == ["first", "second", "third"]


def test_add_comment_has_created_at(store_path):
    entry = add_comment(store_path, "grp", "timestamped")
    assert "created_at" in entry
    assert entry["created_at"] != ""


def test_add_comment_unique_ids(store_path):
    e1 = add_comment(store_path, "grp", "a")
    e2 = add_comment(store_path, "grp", "b")
    assert e1["id"] != e2["id"]


def test_remove_comment_returns_true(store_path):
    entry = add_comment(store_path, "grp", "to remove")
    result = remove_comment(store_path, "grp", entry["id"])
    assert result is True
    assert get_comments(store_path, "grp") == []


def test_remove_comment_unknown_id_returns_false(store_path):
    add_comment(store_path, "grp", "keep me")
    result = remove_comment(store_path, "grp", "nonexistent-id")
    assert result is False
    assert len(get_comments(store_path, "grp")) == 1


def test_remove_comment_unknown_group_returns_false(store_path):
    result = remove_comment(store_path, "no_such_group", "some-id")
    assert result is False


def test_get_comments_unknown_group_returns_empty(store_path):
    assert get_comments(store_path, "ghost") == []


def test_clear_comments_removes_group(store_path):
    add_comment(store_path, "grp", "one")
    add_comment(store_path, "grp", "two")
    clear_comments(store_path, "grp")
    assert get_comments(store_path, "grp") == []


def test_clear_comments_does_not_affect_other_groups(store_path):
    add_comment(store_path, "grp_a", "keep")
    add_comment(store_path, "grp_b", "remove")
    clear_comments(store_path, "grp_b")
    assert len(get_comments(store_path, "grp_a")) == 1
