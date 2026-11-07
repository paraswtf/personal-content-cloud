"""Regression tests for search review findings: contrast gate in small or one-subject libraries,
colour words next to digits or underscores, stable paging while the list changes, snippets made
for the returned page only, and no matrix copy on visibility-only search_version bumps."""
from __future__ import annotations

from app.db import bump_search_version, tx
from app.search import service
from tests.helpers import make_face
from tests.test_search_support import (  # noqa: F401
    add_embedding,
    add_fts,
    fake_engine,
    hit_ids,
    image_vec,
    item,
    no_engine,
    pad,
)


# ------------------------------------------------------------------ contrast gate
def test_one_item_library_gets_its_semantic_hit(conn, fake_engine):
    dog = item(conn)
    add_embedding(conn, dog["id"], image_vec("dog", seed="only"))
    assert hit_ids(service.search(conn, "dog", mode="semantic")) == [dog["id"]]
    assert service.search(conn, "qwzx blorp", mode="semantic").total == 0


def test_subject_that_fills_the_library_is_still_found(conn, fake_engine):
    pad(conn, 2)
    dogs = []
    for k in range(10):
        r = item(conn)
        add_embedding(conn, r["id"], image_vec("dog", seed=f"dog{k}"))
        dogs.append(r["id"])
    assert sorted(hit_ids(service.search(conn, "dog", mode="semantic"))) == sorted(dogs)
    assert service.search(conn, "qwzx blorp", mode="semantic").total == 0


def test_person_crops_survive_when_people_fill_the_library(conn, fake_engine):
    ids = []
    for k in range(6):
        r = item(conn)
        add_embedding(conn, r["id"], image_vec("shirt", seed=f"whole{k}", weight=0.3))
        make_face(conn, r["id"], torso=image_vec("shirt", seed=f"torso{k}", weight=0.5), write_crop=False)
        ids.append(r["id"])
    assert sorted(hit_ids(service.search(conn, "shirt", mode="semantic"))) == sorted(ids)


# ------------------------------------------------------------------ colour words
def test_colour_variants_replace_the_word_found_by_query_color():
    assert service.query_color("blue_car") == ("blue", "blue")
    assert service.color_variants("blue_car", "blue")[:2] == ["red_car", "orange_car"]
    assert service.color_variants("blue2", "blue")[:2] == ["red2", "orange2"]
    assert service.color_variants("Navy car", "Navy")[4] == "blue car"
    assert len(set(service.color_variants("café blue", "blue"))) == len(service.COLORS)


def test_colour_word_next_to_digit_or_underscore(conn, fake_engine):
    pad(conn)
    red, blue = item(conn), item(conn)
    add_embedding(conn, red["id"], image_vec("car", "red", seed="red", weight=0.5))
    add_embedding(conn, blue["id"], image_vec("car", "blue", seed="blue", weight=0.5))
    assert hit_ids(service.search(conn, "blue_car", mode="semantic")) == [blue["id"]]
    assert hit_ids(service.search(conn, "blue2 car", mode="semantic")) == [blue["id"]]
    assert hit_ids(service.search(conn, "red2 car", mode="semantic")) == [red["id"]]


# ------------------------------------------------------------------ paging
def test_later_pages_keep_the_page_one_order(conn, fake_engine):
    pad(conn)
    dogs, labelled = [], []
    for k in range(3):
        d = item(conn)
        add_embedding(conn, d["id"], image_vec("dog", seed=f"d{k}"))
        dogs.append(d["id"])
        t = item(conn)
        add_fts(conn, t["id"], labels="dog")  # labels only: not a strong text hit
        labelled.append(t["id"])

    first = service.search(conn, "dog", limit=3)
    assert set(hit_ids(first)) == set(dogs) and first.total == 6 and first.next_offset == 3

    # a text extraction finishes between the pages: a strong text hit flips auto mode to
    # text-first, and an item of page 2 goes to the trash
    late = item(conn, kind="pdf")
    add_fts(conn, late["id"], text="dog licence renewal")
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-11-07T00:00:00.000Z' WHERE id = ?", (labelled[0],))
        bump_search_version(conn)

    second = service.search(conn, "dog", limit=3, offset=3)
    assert set(hit_ids(second)) == set(labelled[1:])  # nothing skipped, the trashed one dropped
    assert second.total == 6 and second.next_offset is None

    # a new search (offset 0) sees the new list
    fresh = service.search(conn, "dog", limit=3)
    assert set(hit_ids(fresh)) == {late["id"], *labelled[1:]} and fresh.total == 6  # text first now


# ------------------------------------------------------------------ snippets
def test_snippets_are_made_for_the_page_only(conn, monkeypatch):
    for k in range(10):
        r = item(conn, kind="pdf")
        add_fts(conn, r["id"], text=f"monthly bank statement number {k}")
    calls = []
    real = service._render_snippet

    def counting(raw):
        calls.append(raw)
        return real(raw)

    monkeypatch.setattr(service, "_render_snippet", counting)
    first = service.search(conn, "bank", mode="text", limit=2)
    assert first.total == 10 and len(first.hits) == 2
    assert all(h["snippet"] and "<b>bank</b>" in h["snippet"] for h in first.hits)
    assert len(calls) <= 2 * 2
    second = service.search(conn, "bank", mode="text", limit=2, offset=2)
    assert all(h["snippet"] and "<b>bank</b>" in h["snippet"] for h in second.hits)
    assert len(calls) <= 4 * 2


def test_relaxed_query_snippets_still_highlight(conn):
    r = item(conn, kind="pdf")
    add_fts(conn, r["id"], text="the invoice is attached")
    res = service.search(conn, "invoice zebra", mode="text")
    assert hit_ids(res) == [r["id"]] and "<b>invoice</b>" in res.hits[0]["snippet"]


# ------------------------------------------------------------------ matrix refresh
def test_visibility_only_bump_keeps_the_matrix_without_a_copy(conn, fake_engine):
    ids = pad(conn)
    service.search(conn, "dog", mode="semantic")
    before = service.get_matrix().snapshot()
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-11-07T00:00:00.000Z' WHERE id = ?", (ids[0],))
        bump_search_version(conn)
    service.search(conn, "dog", mode="semantic")
    after = service.get_matrix().snapshot()
    assert after.frames is before.frames and after.frames.emb is before.frames.emb
    assert after.crops is before.crops

    dog = item(conn)
    add_embedding(conn, dog["id"], image_vec("dog", seed="new"))
    assert hit_ids(service.search(conn, "dog", mode="semantic")) == [dog["id"]]
    assert service.get_matrix().snapshot().frames.size == before.frames.size + 1
