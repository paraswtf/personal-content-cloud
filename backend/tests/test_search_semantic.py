"""Semantic search of app.search.service with a deterministic fake SigLIP engine: gates, colour
check, video frames, person crops, matrix refresh, and the merge with full-text hits."""
from __future__ import annotations

import pytest

from app.db import bump_search_version, tx
from app.search import service
from tests.helpers import make_face
from tests.test_search_support import (  # noqa: F401
    MODEL,
    add_embedding,
    add_fts,
    fake_engine,
    hit_ids,
    image_vec,
    item,
    no_engine,
    pad,
    unit,
)


def test_semantic_hit_and_gibberish(conn, fake_engine):
    pad(conn)
    dog = item(conn, filename="dog.jpg")
    cat = item(conn, filename="cat.jpg")
    add_embedding(conn, dog["id"], image_vec("dog", seed="dog"))
    add_embedding(conn, cat["id"], image_vec("cat", seed="cat"))

    res = service.search(conn, "dog", mode="semantic")
    assert hit_ids(res) == [dog["id"]]
    hit = res.hits[0]
    assert hit["source"] == "semantic" and hit["score"] == 1.0 and hit["snippet"] is None
    assert hit["match_time_s"] is None  # not a video
    assert res.semantic_used is True and res.indexed == 14

    assert service.search(conn, "qwzx blorp", mode="semantic").total == 0


def test_contrast_gate_uses_settings(conn, fake_engine, monkeypatch):
    from app.config import settings

    a, b = item(conn), item(conn)
    add_embedding(conn, a["id"], image_vec("dog", seed="a", weight=0.07))
    add_embedding(conn, b["id"], image_vec("dog", seed="b", weight=0.07))
    # a weak lift (relevance about 0.06) of the whole library: 20 x median is above every score,
    # and the gate (capped at CONTRAST_CAP) still rejects it
    assert service.search(conn, "dog", mode="semantic").total == 0
    monkeypatch.setattr(settings, "search_contrast", 0.0)
    assert sorted(hit_ids(service.search(conn, "dog", mode="semantic"))) == sorted([a["id"], b["id"]])


def test_colour_check(conn, fake_engine):
    pad(conn)
    red = item(conn)
    blue = item(conn)
    add_embedding(conn, red["id"], image_vec("car", "red", seed="red", weight=0.5))
    add_embedding(conn, blue["id"], image_vec("car", "blue", seed="blue", weight=0.5))
    assert hit_ids(service.search(conn, "red car", mode="semantic")) == [red["id"]]
    assert hit_ids(service.search(conn, "navy car", mode="semantic")) == [blue["id"]]
    assert sorted(hit_ids(service.search(conn, "car", mode="semantic"))) == sorted([red["id"], blue["id"]])


def test_colour_check_can_be_disabled(conn, fake_engine, monkeypatch):
    from app.config import settings

    pad(conn)
    red, blue = item(conn), item(conn)
    add_embedding(conn, red["id"], image_vec("car", "red", seed="red", weight=0.5))
    add_embedding(conn, blue["id"], image_vec("car", "blue", seed="blue", weight=0.5))
    monkeypatch.setattr(settings, "search_color_check", False)
    assert set(hit_ids(service.search(conn, "red car", mode="semantic"))) == {red["id"], blue["id"]}


def test_video_hit_reports_best_frame_time(conn, fake_engine):
    pad(conn)
    video = item(conn, kind="video", ext=".mp4", mime="video/mp4", duration_s=4.0)
    add_embedding(conn, video["id"], image_vec(seed="poster"), frame_index=0, frame_time_s=0.0)
    add_embedding(conn, video["id"], image_vec(seed="f1"), frame_index=1, frame_time_s=1.0)
    add_embedding(conn, video["id"], image_vec("beach", seed="f2"), frame_index=2, frame_time_s=2.0)
    add_embedding(conn, video["id"], image_vec(seed="f3"), frame_index=3, frame_time_s=3.0)
    res = service.search(conn, "beach", mode="semantic")
    assert hit_ids(res) == [video["id"]]
    assert res.hits[0]["match_time_s"] == 2.0


def test_person_crops_find_items_the_whole_image_misses(conn, fake_engine):
    pad(conn)
    crowd = item(conn)
    add_embedding(conn, crowd["id"], image_vec(seed="crowd"))
    make_face(conn, crowd["id"], torso=image_vec("green", "shirt", seed="torso", weight=0.5), write_crop=False)
    # a faceless body counts too; a body that has a face is covered by the face's crop
    other = item(conn)
    add_embedding(conn, other["id"], image_vec(seed="other"))
    now = "2026-11-07T00:00:00.000Z"
    with tx(conn):
        conn.execute(
            "INSERT INTO bodies(item_id, x, y, w, h, det_score, torso_embedding, created_at) "
            "VALUES (?, 0.1, 0.1, 0.3, 0.8, 0.9, ?, ?)",
            (other["id"], image_vec("red", "shirt", seed="body", weight=0.5).tobytes(), now),
        )
        bump_search_version(conn)

    assert hit_ids(service.search(conn, "green shirt", mode="semantic")) == [crowd["id"]]
    assert hit_ids(service.search(conn, "red shirt", mode="semantic")) == [other["id"]]
    assert set(hit_ids(service.search(conn, "shirt", mode="semantic"))) == {crowd["id"], other["id"]}


def test_person_crop_floor(conn, fake_engine, monkeypatch):
    from app.config import settings

    pad(conn)
    crowd = item(conn)
    add_embedding(conn, crowd["id"], image_vec(seed="crowd"))
    make_face(conn, crowd["id"], torso=image_vec("shirt", seed="torso", weight=0.5), write_crop=False)
    assert hit_ids(service.search(conn, "shirt", mode="semantic")) == [crowd["id"]]
    monkeypatch.setattr(settings, "search_person_min_relevance", 1.01)
    assert service.search(conn, "shirt", mode="semantic").total == 0


def test_matrix_refreshes_on_new_rows_deletes_and_versions(conn, fake_engine):
    pad(conn)
    assert service.search(conn, "mountain", mode="semantic").total == 0
    m = item(conn)
    add_embedding(conn, m["id"], image_vec("mountain", seed="m"))
    assert hit_ids(service.search(conn, "mountain", mode="semantic")) == [m["id"]]
    size = service.get_matrix().size

    # a newer item version without rows: the stale rows no longer count
    with tx(conn):
        conn.execute("UPDATE items SET version = 2 WHERE id = ?", (m["id"],))
        bump_search_version(conn)
    assert service.search(conn, "mountain", mode="semantic").total == 0
    assert service.get_matrix().size == size - 1

    add_embedding(conn, m["id"], image_vec("mountain", seed="m2"), version=2)
    assert hit_ids(service.search(conn, "mountain", mode="semantic")) == [m["id"]]

    # a purge drops rows by cascade; the row count change alone refreshes the matrix
    with tx(conn):
        conn.execute("DELETE FROM items WHERE id = ?", (m["id"],))
    assert service.search(conn, "mountain", mode="semantic").total == 0

    service.reset_matrix()
    assert service.get_matrix().size == 0
    assert service.search(conn, "dog", mode="semantic").total == 0
    assert service.get_matrix().size == 12


def test_rows_of_other_models_are_ignored(conn, fake_engine):
    pad(conn)
    old = item(conn)
    add_embedding(conn, old["id"], image_vec("dog", seed="old"), model="some-old-model")
    assert service.search(conn, "dog", mode="semantic").total == 0
    assert service.indexed_count(conn) == 12


def test_semantic_hits_respect_vault_trash_and_filters(conn, fake_engine):
    pad(conn)
    shown = item(conn, category="pets", kind="image")
    private = item(conn, nsfw=1, category="nsfw")
    trashed = item(conn, deleted_at="2026-01-01T00:00:00.000Z")
    video = item(conn, kind="video", category="pets")
    for r in (shown, private, trashed, video):
        add_embedding(conn, r["id"], image_vec("dog", seed=f"v{r['id']}"))
    assert set(hit_ids(service.search(conn, "dog", mode="semantic"))) == {shown["id"], video["id"]}
    assert hit_ids(service.search(conn, "dog", mode="semantic", vault=True)) == [private["id"]]
    assert hit_ids(service.search(conn, "dog", mode="semantic", kind="video")) == [video["id"]]
    assert service.search(conn, "dog", mode="semantic", category="food").total == 0
    assert service.indexed_count(conn, vault=True) == 1
    assert service.indexed_count(conn, vault=False) == 12 + 2


# ------------------------------------------------------------------ merge
def test_auto_merges_both_engines(conn, fake_engine):
    pad(conn)
    both = item(conn, filename="dog-park.jpg")
    sem_only = item(conn, filename="IMG_0001.jpg")
    text_only = item(conn, kind="pdf", filename="vet.pdf")
    add_embedding(conn, both["id"], image_vec("dog", seed="both"))
    add_embedding(conn, sem_only["id"], image_vec("dog", seed="sem", weight=0.55))
    add_fts(conn, both["id"], filename="dog-park.jpg", labels="dog")
    add_fts(conn, text_only["id"], filename="vet.pdf", text="dog vaccination record")

    res = service.search(conn, "dog")
    by_id = {h["id"]: h for h in res.hits}
    assert set(by_id) == {both["id"], sem_only["id"], text_only["id"]}
    assert by_id[both["id"]]["source"] == "both"
    assert by_id[sem_only["id"]]["source"] == "semantic"
    assert by_id[text_only["id"]]["source"] == "text"
    assert by_id[text_only["id"]]["snippet"] and "<b>dog</b>" in by_id[text_only["id"]]["snippet"]
    # strong text hit (file name / text): text order first, then the semantic-only item
    assert hit_ids(res)[-1] == sem_only["id"]
    # the item found both ways keeps the higher of its normalised scores
    assert by_id[both["id"]]["score"] == 1.0
    assert res.total == 3


def test_auto_puts_semantic_first_when_only_labels_match(conn, fake_engine):
    pad(conn)
    sem = item(conn)
    labelled = item(conn)
    add_embedding(conn, sem["id"], image_vec("beach", seed="sem"))
    add_fts(conn, labelled["id"], labels="beach")
    res = service.search(conn, "beach")
    assert hit_ids(res) == [sem["id"], labelled["id"]]
    assert [h["source"] for h in res.hits] == ["semantic", "text"]


def test_text_mode_ignores_the_engine_and_semantic_mode_ignores_fts(conn, fake_engine):
    pad(conn)
    a = item(conn)
    b = item(conn, kind="pdf")
    add_embedding(conn, a["id"], image_vec("cat", seed="a"))
    add_fts(conn, b["id"], text="cat food receipt")
    assert hit_ids(service.search(conn, "cat", mode="text")) == [b["id"]]
    assert fake_engine.texts == []
    assert hit_ids(service.search(conn, "cat", mode="semantic")) == [a["id"]]


def test_engine_failure_falls_back_to_text_in_auto_mode(conn, fake_engine):
    pad(conn)
    a = item(conn, kind="pdf")
    add_fts(conn, a["id"], text="salary slip")
    fake_engine.fail = True
    res = service.search(conn, "salary")
    assert hit_ids(res) == [a["id"]] and res.semantic_used is False
    with pytest.raises(service.SearchUnavailable):
        service.search(conn, "salary", mode="semantic")


def test_both_engines_failing_raises(conn, fake_engine, monkeypatch):
    import sqlite3

    def broken(*a, **kw):
        raise sqlite3.OperationalError("fts5 broke")

    monkeypatch.setattr(service, "_text_hits", broken)
    fake_engine.fail = True
    with pytest.raises(service.SearchUnavailable):
        service.search(conn, "anything")
    with pytest.raises(service.SearchUnavailable):
        service.search(conn, "anything", mode="text")
    fake_engine.fail = False
    pad(conn)
    assert service.search(conn, "anything").total == 0  # semantic answered, so no error


def test_mixed_embedding_sizes_keep_the_common_one(conn, fake_engine):
    import numpy as np

    pad(conn)
    odd = item(conn)
    with tx(conn):
        conn.execute(
            "INSERT INTO item_embeddings(item_id, version, model, frame_index, embedding, created_at) "
            "VALUES (?, 1, ?, 0, ?, '2026')",
            (odd["id"], MODEL, unit(np.ones(512)).tobytes()),
        )
        bump_search_version(conn)
    service.search(conn, "dog", mode="semantic")
    assert service.get_matrix().snapshot().frames.dim == 768
    assert service.get_matrix().size == 12


def test_wiring_through_the_real_ml_siglip_module(conn, monkeypatch):
    """The lazy import of app.ml.siglip and its available() / get_engine() / MODEL_ID names; no
    model is loaded (both functions are patched)."""
    siglip = pytest.importorskip("app.ml.siglip")
    from app.config import settings
    from tests.test_search_support import FakeEngine

    engine = FakeEngine()
    monkeypatch.setattr(settings, "search_enabled", True)
    monkeypatch.setattr(siglip, "available", lambda: True, raising=False)
    monkeypatch.setattr(siglip, "get_engine", lambda: engine, raising=False)
    assert service.current_model() == MODEL == siglip.MODEL_ID
    pad(conn)
    dog = item(conn)
    add_embedding(conn, dog["id"], image_vec("dog", seed="dog"))
    assert service.engine_available() is True
    assert hit_ids(service.search(conn, "dog", mode="semantic")) == [dog["id"]]
    assert engine.texts and engine.texts[0] == "dog"
