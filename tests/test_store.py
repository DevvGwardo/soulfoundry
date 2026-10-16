"""Tests for the project store: persistence, scoping, cross-project import."""

from __future__ import annotations

import pytest

from nodes.core.store import ELEMENT_KINDS, Element, ProjectStore, Shot


@pytest.fixture()
def store(tmp_path):
    with ProjectStore(tmp_path / "proj") as s:
        yield s


def test_element_requires_a_valid_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        Element(name="x", kind="vehicle")


def test_element_name_cannot_be_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        Element(name="   ")


def test_element_name_cannot_contain_whitespace():
    with pytest.raises(ValueError, match="whitespace"):
        Element(name="Old Man River")


def test_element_strips_leading_at_sign():
    assert Element(name="@santiago").name == "santiago"
    assert Element(name="@santiago").handle == "@santiago"


def test_save_and_get_round_trip(store):
    store.save_element(Element(name="santiago", kind="character", bible_md="bible"))
    found = store.get_element("santiago")
    assert found is not None
    assert found.bible_md == "bible"
    assert found.kind == "character"


def test_get_is_case_insensitive_and_accepts_at_sign(store):
    store.save_element(Element(name="santiago"))
    assert store.get_element("SANTIAGO") is not None
    assert store.get_element("@santiago") is not None


def test_get_missing_returns_none(store):
    assert store.get_element("nobody") is None


def test_save_is_upsert_and_preserves_id(store):
    first = store.save_element(Element(name="santiago", bible_md="v1"))
    second = store.save_element(Element(name="santiago", bible_md="v2"))
    assert first.id == second.id
    assert store.get_element("santiago").bible_md == "v2"
    assert len(store.list_elements()) == 1


def test_list_filters_by_kind(store):
    store.save_element(Element(name="a", kind="character"))
    store.save_element(Element(name="b", kind="location"))
    store.save_element(Element(name="c", kind="prop"))
    assert len(store.list_elements()) == 3
    assert [e.name for e in store.list_elements("character")] == ["a"]
    assert [e.name for e in store.list_elements("location")] == ["b"]


def test_list_rejects_bad_kind(store):
    with pytest.raises(ValueError, match="kind must be one of"):
        store.list_elements("vehicle")


def test_pin_and_pinned_only(store):
    store.save_element(Element(name="a"))
    store.save_element(Element(name="b"))
    assert store.pin("a") is True
    pinned = store.list_elements(pinned_only=True)
    assert [e.name for e in pinned] == ["a"]


def test_pin_missing_returns_false(store):
    assert store.pin("nobody") is False


def test_delete(store):
    store.save_element(Element(name="a"))
    assert store.delete_element("a") is True
    assert store.delete_element("a") is False
    assert store.list_elements() == []


def test_library_maps_handle_to_bible(store):
    store.save_element(Element(name="santiago", bible_md="the bible"))
    store.save_element(Element(name="silent", bible_md=None))
    lib = store.library()
    assert lib == {"santiago": "the bible"}


def test_cross_project_import_records_provenance(tmp_path):
    a = ProjectStore(tmp_path / "filmA")
    try:
        a.save_element(Element(name="santiago", bible_md="bible", kind="character"))
        with ProjectStore(tmp_path / "filmB") as b:
            imported = b.import_element(a, "santiago")
            assert imported.origin_project == "filmA"
            assert imported.bible_md == "bible"
            assert b.get_element("santiago") is not None
    finally:
        a.close()


def test_cross_project_import_missing_raises(tmp_path):
    a = ProjectStore(tmp_path / "filmA")
    try:
        with ProjectStore(tmp_path / "filmB") as b:
            with pytest.raises(KeyError):
                b.import_element(a, "nobody")
    finally:
        a.close()


def test_imported_element_keeps_a_distinct_id(tmp_path):
    a = ProjectStore(tmp_path / "filmA")
    try:
        original = a.save_element(Element(name="santiago"))
        with ProjectStore(tmp_path / "filmB") as b:
            imported = b.import_element(a, "santiago")
            assert imported.id != original.id
    finally:
        a.close()


def test_shot_upsert_by_seq(store):
    store.save_shot(Shot(seq=1, prompt="one"))
    store.save_shot(Shot(seq=1, prompt="one revised"))
    shots = store.list_shots()
    assert len(shots) == 1
    assert shots[0].prompt == "one revised"


def test_shots_are_ordered_by_seq(store):
    store.save_shot(Shot(seq=3, prompt="three"))
    store.save_shot(Shot(seq=1, prompt="one"))
    store.save_shot(Shot(seq=2, prompt="two"))
    assert [s.seq for s in store.list_shots()] == [1, 2, 3]


def test_record_drift_and_curve(store):
    for seq in (1, 2, 3):
        store.save_shot(Shot(seq=seq, prompt=str(seq)))
    assert store.record_drift(1, 0.95) is True
    store.record_drift(2, 0.93)
    assert store.record_drift(9, 0.5) is False
    curve = store.drift_curve()
    assert curve == [(1, 0.95), (2, 0.93)]


def test_summary_counts(store):
    store.save_element(Element(name="a", kind="character"))
    store.save_element(Element(name="b", kind="location"))
    store.save_shot(Shot(seq=1, prompt="x"))
    summary = store.summary()
    assert summary["total_elements"] == 2
    assert summary["elements"]["character"] == 1
    assert summary["shots"] == 1
    assert summary["scored_shots"] == 0


def test_export_writes_a_readable_copy(tmp_path):
    import sqlite3

    with ProjectStore(tmp_path / "proj") as s:
        s.save_element(Element(name="santiago", bible_md="bible"))
        dest = s.export(tmp_path / "backup.sqlite")
    conn = sqlite3.connect(dest)
    rows = conn.execute("SELECT name FROM elements").fetchall()
    conn.close()
    assert rows == [("santiago",)]


def test_memory_store_works():
    with ProjectStore(":memory:") as s:
        s.save_element(Element(name="a"))
        assert s.get_element("a") is not None


def test_store_creates_missing_directories(tmp_path):
    nested = tmp_path / "deep" / "project"
    with ProjectStore(nested) as s:
        s.save_element(Element(name="a"))
    assert (nested / "project.sqlite").exists()


def test_all_kinds_are_usable(store):
    for kind in ELEMENT_KINDS:
        store.save_element(Element(name=f"thing_{kind}", kind=kind))
    assert len(store.list_elements()) == 3
