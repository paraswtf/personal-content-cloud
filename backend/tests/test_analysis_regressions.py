"""Regression tests for the analysis review findings: vault checks in the people API, reprocessing
without models, one guess per person per photo, rejections that survive a reprocess, burst groups,
FTS refreshes, guess tags, the guess window and the face index bookkeeping."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import storage
from app.analysis import bodies, cluster, guess, pipeline, quality
from app.analysis.index import get_index
from app.api import people
from app.api.deps import require_user
from app.auth.sessions import SessionInfo
from app.db import faces_version, tx
from app.util import now_iso
from tests.helpers import jpeg_bytes, make_face, make_item, make_person, noise_image

H = {"X-Requested-With": "pcc"}


def _client(unlocked: bool) -> TestClient:
    app = FastAPI()
    app.include_router(people.router)
    fake = SessionInfo(id=1, created_at="", last_seen_at="", expires_at="",
                       vault_until=now_iso(3600) if unlocked else None, user_agent=None, ip=None)
    app.dependency_overrides[require_user] = lambda: fake
    c = TestClient(app)
    c.headers.update(H)
    return c


@pytest.fixture()
def locked(data_dir):
    with _client(False) as c:
        yield c


@pytest.fixture()
def unlocked(data_dir):
    with _client(True) as c:
        yield c


@pytest.fixture()
def no_models(monkeypatch):
    monkeypatch.setattr(pipeline, "_model", lambda name: None)
    monkeypatch.setattr(bodies, "available", lambda: False)


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, seed: int, noise: float = 0.1) -> np.ndarray:
    v = base + noise * _vec(seed, base.shape[0])
    return (v / np.linalg.norm(v)).astype(np.float32)


def _item(conn, **kw):
    return make_item(conn, write_files=False, data=None, **kw)


def _body(conn, item_id: int, torso: np.ndarray | None = None, person_id: int | None = None,
          assigned_by: str = "auto", face_id: int | None = None, **kw) -> int:
    cols = dict(item_id=item_id, face_id=face_id, person_id=person_id, assigned_by=assigned_by, x=0.1, y=0.1,
                w=0.3, h=0.7, det_score=0.9, torso_embedding=None if torso is None else torso.tobytes(),
                created_at=now_iso())
    cols.update(kw)
    with tx(conn):
        bid = int(conn.execute(f"INSERT INTO bodies({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                               tuple(cols.values())).lastrowid)
    p = storage.body_crop_path(bid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return bid


def _scene(conn, item_id: int, vec: np.ndarray) -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding, created_at) "
            "VALUES (?, 1, 'test', 0, NULL, ?, ?)", (item_id, vec.astype(np.float32).tobytes(), now_iso()))


def _tag(conn, item_id: int, person_id: int, source: str = "user") -> None:
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, ?, ?)",
                     (item_id, person_id, source, now_iso()))


def _fts_people(conn, item_id: int) -> str | None:
    row = conn.execute("SELECT people FROM items_fts WHERE rowid = ?", (item_id,)).fetchone()
    return None if row is None else row[0]


# ------------------------------------------------------------------ 1. vault checks in the people API
def test_locked_vault_blocks_nsfw_boxes_tags_and_hidden_people(conn, locked, unlocked):
    hidden = make_person(conn, "Hidden Person")
    secret = _item(conn, nsfw=1)
    face = make_face(conn, secret["id"], hidden, embedding=_vec(1))
    body = _body(conn, secret["id"], torso=_vec(2, 768))
    _tag(conn, secret["id"], hidden)
    assert [p["id"] for p in locked.get("/api/people").json()["items"]] == []

    assert locked.get(f"/api/people/{hidden}").status_code == 404
    assert locked.get(f"/api/people/{hidden}/faces").status_code == 404
    assert locked.patch(f"/api/people/{hidden}", json={"name": "Peek"}).status_code == 404
    assert locked.post(f"/api/items/{secret['uid']}/tags", json={"new_person_name": "X"}).status_code == 403
    r = locked.delete(f"/api/items/{secret['uid']}/tags/0")
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    assert locked.post("/api/faces/assign", json={"face_ids": [face], "person_id": None}).status_code == 403
    assert locked.post("/api/bodies/assign", json={"body_ids": [body], "new_person_name": "X"}).status_code == 403
    assert locked.delete(f"/api/bodies/{body}").status_code == 403
    assert locked.delete(f"/api/faces/{face}").status_code == 403
    assert locked.post("/api/guesses/confirm", json={"faces": [], "bodies": [body]}).status_code == 403
    assert locked.post("/api/guesses/reject", json={"faces": [face], "bodies": []}).status_code == 403
    assert locked.post("/api/people", json={"name": "New", "face_ids": [face]}).status_code == 403
    # A library photo cannot be tagged with the hidden person either (the reply would carry the name).
    lib = _item(conn)
    assert locked.post(f"/api/items/{lib['uid']}/tags", json={"person_id": hidden}).status_code == 404
    # Nothing changed.
    assert conn.execute("SELECT person_id FROM faces WHERE id = ?", (face,)).fetchone()[0] == hidden
    assert conn.execute("SELECT COUNT(*) FROM bodies WHERE id = ?", (body,)).fetchone()[0] == 1

    # Unlocked, the same calls work.
    assert unlocked.get(f"/api/people/{hidden}").json()["name"] == "Hidden Person"
    tags = unlocked.post(f"/api/items/{secret['uid']}/tags", json={"new_person_name": "Meera"}).json()
    assert {t["name"] for t in tags} == {"Hidden Person", "Meera"}
    assert unlocked.delete(f"/api/bodies/{body}").json() == {"ok": True}

    # People without any items, and people seen in the library, stay visible when locked.
    empty = make_person(conn, "Newcomer")
    assert locked.get(f"/api/people/{empty}").status_code == 200
    make_face(conn, lib["id"], hidden, embedding=_vec(3))
    assert locked.get(f"/api/people/{hidden}").status_code == 200


# ------------------------------------------------------------------ 2. reprocess without models
def test_reprocess_without_models_keeps_faces_bodies_embeddings_and_labels(conn, no_models, monkeypatch):
    asha, ravi = make_person(conn, "Asha"), make_person(conn, "Ravi")
    img = noise_image(800, 600, seed=7)
    item = make_item(conn, data=jpeg_bytes(img), width=800, height=600, analysis_status="pending",
                     category="personal")
    face = make_face(conn, item["id"], asha, embedding=_vec(4), assigned_by="user", size_px=120, det_score=0.9)
    body = _body(conn, item["id"], torso=_vec(5, 768), person_id=ravi, assigned_by="user")
    _scene(conn, item["id"], _vec(6, 768))
    with tx(conn):
        conn.execute("INSERT INTO item_labels(item_id, label, score) VALUES (?, 'beach', 0.4)", (item["id"],))
        conn.execute("UPDATE persons SET cover_face_id = ? WHERE id = ?", (face, asha))

    pipeline.analyze_item(conn, item["id"], 1)

    row = conn.execute("SELECT * FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert row["analysis_status"] == "done"
    assert {"faces", "siglip", "bodies"} <= set(json.loads(row["quality_scores"])["skipped"])
    assert [tuple(r) for r in conn.execute("SELECT id, person_id FROM faces WHERE item_id = ?", (item["id"],))] == [
        (face, asha)]
    assert storage.face_crop_path(face).exists()
    assert [tuple(r) for r in conn.execute("SELECT id, person_id FROM bodies WHERE item_id = ?", (item["id"],))] == [
        (body, ravi)]
    assert conn.execute("SELECT COUNT(*) FROM item_embeddings WHERE item_id = ?", (item["id"],)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM item_labels WHERE item_id = ?", (item["id"],)).fetchone()[0] == 1
    assert row["category"] == "personal" and row["face_count"] == 1 and row["body_count"] == 1
    assert conn.execute("SELECT cover_face_id FROM persons WHERE id = ?", (asha,)).fetchone()[0] == face
    # The FTS row lists the people (finding 12: refreshed even without labels).
    assert set((_fts_people(conn, item["id"]) or "").split()) == {"Asha", "Ravi"}
    # Finding 13: the committed analysis leaves the face index current.
    idx = get_index()
    assert idx._tentative_conn is None and idx._version == faces_version(conn)


# ------------------------------------------------------------------ 3. one person per photo
def _guess_world(conn):
    scene, torso = _vec(10, 768), _vec(11, 768)
    p = make_person(conn, "Asha")
    a = _item(conn, taken_at="2026-05-01T10:00:00.000Z")
    b = _item(conn, taken_at="2026-05-01T11:00:00.000Z")
    make_face(conn, a["id"], p, embedding=_vec(12), assigned_by="user", torso=torso)
    _scene(conn, a["id"], scene)
    _scene(conn, b["id"], scene)
    return p, a, b, torso


def test_one_guess_per_person_per_photo(conn):
    p, _, b, torso = _guess_world(conn)
    b1 = _body(conn, b["id"], _near(torso, 13, 0.05))
    b2 = _body(conn, b["id"], _near(torso, 14, 0.1))
    assert guess.guess_item(conn, b["id"]) == 1
    rows = dict(conn.execute("SELECT id, guess_person_id FROM bodies WHERE item_id = ?", (b["id"],)).fetchall())
    assert rows == {b1: p, b2: None}


def test_confirm_skips_a_person_already_in_the_photo(conn):
    p, _, b, torso = _guess_world(conn)
    b1 = _body(conn, b["id"], _near(torso, 15), assigned_by="guess", guess_person_id=p, guess_score=0.9)
    b2 = _body(conn, b["id"], _near(torso, 16), assigned_by="guess", guess_person_id=p, guess_score=0.8)
    assert guess.confirm(conn, [], [b1, b2]) == 1
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM bodies WHERE item_id = ?", (b["id"],))}
    assert rows[b1]["person_id"] == p
    assert rows[b2]["person_id"] is None and rows[b2]["guess_person_id"] is None and rows[b2]["assigned_by"] == "auto"


def test_propagate_clears_a_stale_guess_on_the_anchor(conn):
    p, _, b, torso = _guess_world(conn)
    b1 = _body(conn, b["id"], _near(torso, 17), person_id=p, assigned_by="user")
    b2 = _body(conn, b["id"], _near(torso, 18), assigned_by="guess", guess_person_id=p, guess_score=0.8)
    assert b1
    guess.propagate(conn, p, b["id"])
    assert conn.execute("SELECT guess_person_id FROM bodies WHERE id = ?", (b2,)).fetchone()[0] is None


# ------------------------------------------------------------------ 4. rejections survive a reprocess
def test_carry_over_moves_rejections_of_unassigned_faces(conn):
    p = make_person(conn, "Asha")
    item = _item(conn)
    emb = _vec(20)
    old = make_face(conn, item["id"], None, embedding=emb)
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (old, p, now_iso()))
    old_rows = conn.execute("SELECT * FROM faces WHERE id = ?", (old,)).fetchall()
    new = make_face(conn, item["id"], None, embedding=_near(emb, 21, 0.1))
    assert cluster.carry_over(conn, old_rows, [new]) == 0  # no person to carry
    assert [tuple(r) for r in conn.execute("SELECT ref_id FROM guess_rejections")] == [(new,)]
    assert conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (new,)).fetchone()[:] == (None, "auto")


def test_carry_bodies_moves_rejections_and_delete_drops_them(conn, locked):
    p = make_person(conn, "Asha")
    item = _item(conn)
    old = _body(conn, item["id"], _vec(22, 768))
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('body', ?, ?, ?)",
                     (old, p, now_iso()))
    old_rows = conn.execute("SELECT * FROM bodies WHERE id = ?", (old,)).fetchall()
    new = _body(conn, item["id"], _vec(23, 768))
    with tx(conn):
        pipeline._carry_bodies(conn, old_rows, [new])
    assert [tuple(r) for r in conn.execute("SELECT kind, ref_id FROM guess_rejections")] == [("body", new)]
    assert locked.delete(f"/api/bodies/{new}").json() == {"ok": True}
    assert conn.execute("SELECT COUNT(*) FROM guess_rejections").fetchone()[0] == 0


# ------------------------------------------------------------------ 5, 9, 15. burst groups
def _burst_item(conn, **kw):
    return _item(conn, taken_at=kw.pop("taken_at", "2026-05-01T10:00:00.000Z"), camera="Pixel", phash="ff00ff00ff00ff00",
                 quality_scores=json.dumps({"blur_var": kw.pop("blur", 500.0)}), **kw)


def test_burst_group_skips_documents_screenshots_and_the_other_side_of_the_vault(conn):
    shot = _burst_item(conn, analysis_status="pending")
    doc = _burst_item(conn, taken_at="2026-05-01T10:00:01.000Z", category="documents")
    _burst_item(conn, taken_at="2026-05-01T10:00:01.000Z", category="screenshots")
    vault = _burst_item(conn, taken_at="2026-05-01T09:59:59.000Z", nsfw=1, blur=900.0)
    twin = _burst_item(conn, taken_at="2026-05-01T10:00:02.000Z")
    ids = {m.id for m in quality.burst_group(conn, shot, 300.0)}
    assert ids == {shot["id"], twin["id"]} and doc["id"] not in ids
    # The analysed item's new nsfw decision decides the side.
    assert {m.id for m in quality.burst_group(conn, shot, 300.0, nsfw=1)} == {shot["id"], vault["id"]}


def test_range_queries_use_the_taken_at_index(conn):
    plans = [
        (quality.BURST_SQL, ("a", "b", 1, None, 0)),
        (guess.WINDOW_BEFORE_SQL, ("a", "b", 1, 5)),
        (guess.WINDOW_AFTER_SQL, ("a", "b", 1, 5)),
    ]
    for sql, args in plans:
        detail = " ".join(r[-1] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, args))
        assert "items_taken" in detail, detail


# ------------------------------------------------------------------ 6, 12. recluster refreshes FTS
def test_recluster_refreshes_people_in_fts(conn):
    asha = make_person(conn, "Asha")
    emb = _vec(30)
    for k in range(3):
        make_face(conn, _item(conn)["id"], asha, embedding=_near(emb, 31 + k, 0.05), assigned_by="user")
    late = _item(conn)
    make_face(conn, late["id"], None, embedding=_near(emb, 40, 0.05))
    assert cluster.recluster_unassigned(conn) == 1
    assert _fts_people(conn, late["id"]) == "Asha"


# ------------------------------------------------------------------ 7. tag from a confirmed faceless body
def test_confirmed_body_tag_follows_the_body(conn, locked):
    p, _, b, torso = _guess_world(conn)
    body = _body(conn, b["id"], _near(torso, 50))
    guess.guess_item(conn, b["id"])
    assert guess.confirm(conn, [], [body]) == 1
    assert conn.execute("SELECT source FROM item_tags WHERE item_id = ? AND person_id = ?",
                        (b["id"], p)).fetchone()[0] == "guess"
    meera = make_person(conn, "Meera")
    assert locked.post("/api/bodies/assign", json={"body_ids": [body], "person_id": meera}).status_code == 200
    assert conn.execute("SELECT person_id FROM item_tags WHERE item_id = ?", (b["id"],)).fetchall() == []
    # A manual tag stays when the body goes.
    _tag(conn, b["id"], p)
    assert locked.delete(f"/api/bodies/{body}").json() == {"ok": True}
    assert [tuple(r) for r in conn.execute("SELECT person_id, source FROM item_tags WHERE item_id = ?",
                                           (b["id"],))] == [(p, "user")]


def test_deleting_a_confirmed_body_lets_an_unnamed_person_go(conn, locked):
    unnamed = make_person(conn, None)
    item = _item(conn)
    body = _body(conn, item["id"], _vec(51, 768), assigned_by="guess", guess_person_id=unnamed, guess_score=0.8)
    assert guess.confirm(conn, [], [body]) == 1
    assert locked.delete(f"/api/bodies/{body}").json() == {"ok": True}
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (unnamed,)).fetchone()[0] == 0


# ------------------------------------------------------------------ 8. vault items take no part in guessing
def test_vault_items_are_not_references_and_get_no_guesses(conn):
    scene, torso = _vec(60, 768), _vec(61, 768)
    p = make_person(conn, "Asha")
    v = _item(conn, taken_at="2026-05-01T10:00:00.000Z", nsfw=1)
    lib = _item(conn, taken_at="2026-05-01T11:00:00.000Z")
    make_face(conn, v["id"], p, embedding=_vec(62), assigned_by="user", torso=torso)
    _scene(conn, v["id"], scene)
    _scene(conn, lib["id"], scene)
    body = _body(conn, lib["id"], _near(torso, 63))
    assert guess.guess_item(conn, lib["id"]) == 0
    assert conn.execute("SELECT guess_person_id FROM bodies WHERE id = ?", (body,)).fetchone()[0] is None

    # A vault item gets no guesses either, and a stale one is cleared.
    lib2 = _item(conn, taken_at="2026-05-01T10:30:00.000Z")
    make_face(conn, lib2["id"], p, embedding=_vec(64), assigned_by="user", torso=torso)
    _scene(conn, lib2["id"], scene)
    v2 = _item(conn, taken_at="2026-05-01T10:40:00.000Z", nsfw=1)
    _scene(conn, v2["id"], scene)
    vb = _body(conn, v2["id"], _near(torso, 65), assigned_by="guess", guess_person_id=p, guess_score=0.9)
    guess.guess_item(conn, v2["id"])
    assert conn.execute("SELECT guess_person_id FROM bodies WHERE id = ?", (vb,)).fetchone()[0] is None


# ------------------------------------------------------------------ 10. merge keeps rejections
def test_merge_does_not_repoint_a_guess_to_a_rejected_person(conn):
    target, source = make_person(conn, "Asha"), make_person(conn)
    item = _item(conn)
    face = make_face(conn, item["id"], None, embedding=_vec(70), assigned_by="guess", guess_person_id=source,
                     guess_score=0.8)
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (face, target, now_iso()))
    cluster.merge_persons(conn, target, [source])
    row = conn.execute("SELECT guess_person_id, assigned_by FROM faces WHERE id = ?", (face,)).fetchone()
    assert row["guess_person_id"] is None and row["assigned_by"] == "auto"


# ------------------------------------------------------------------ 11. dense days
def test_window_keeps_the_nearest_items_on_both_sides(conn, monkeypatch):
    monkeypatch.setattr(guess, "MAX_WINDOW_ITEMS", 3)
    early = [_item(conn, taken_at=f"2026-05-01T0{h}:00:00.000Z") for h in range(1, 9)]
    anchor = _item(conn, taken_at="2026-05-01T09:00:00.000Z")
    after = _item(conn, taken_at="2026-05-01T09:30:00.000Z")
    got = [r["id"] for r in guess.window_items(conn, anchor)]
    assert got == [early[6]["id"], early[7]["id"], after["id"]]


# ------------------------------------------------------------------ 13, 14. face index bookkeeping
def test_guess_writes_do_not_force_a_full_index_reload(conn):
    p, _, b, torso = _guess_world(conn)
    _body(conn, b["id"], _near(torso, 80))
    idx = get_index()
    idx.sync(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    assert idx._version == faces_version(conn) and idx._tentative_conn is None


def test_sync_item_appends_and_replaces_rows(conn):
    a, b = _item(conn), _item(conn)
    fa = make_face(conn, a["id"], None, embedding=_vec(90))
    idx = get_index()
    idx.sync(conn)
    with tx(conn):
        base = faces_version(conn)
        fb = make_face(conn, b["id"], None, embedding=_vec(91))
        idx.sync_item(conn, b["id"], base)
        assert idx._ids.tolist() == [fa, fb] and idx._pos == {fa: 0, fb: 1}
        np.testing.assert_allclose(idx.embeddings_of([fb])[0], _vec(91), atol=1e-6)
        # Replace a's face with a newer one: rows stay sorted by id.
        conn.execute("DELETE FROM faces WHERE id = ?", (fa,))
        fa2 = make_face(conn, a["id"], None, embedding=_vec(92))
        idx.sync_item(conn, a["id"], faces_version(conn))
    idx.sync(conn)
    assert idx._ids.tolist() == [fb, fa2] and idx._pos == {fb: 0, fa2: 1}


# ------------------------------------------------------------------ 16. people list
def test_people_list_total_and_covers_on_every_page(conn, locked):
    ids = [make_person(conn, f"P{k}") for k in range(3)]
    faces = [make_face(conn, _item(conn)["id"], pid, embedding=_vec(100 + k)) for k, pid in enumerate(ids)]
    secret = make_face(conn, _item(conn, nsfw=1)["id"], ids[0], embedding=_vec(110), size_px=160, det_score=1.0)
    with tx(conn):
        conn.execute("UPDATE persons SET cover_face_id = ? WHERE id = ?", (secret, ids[0]))
    first = locked.get("/api/people", params={"sort": "name", "limit": 2}).json()
    rest = locked.get("/api/people", params={"sort": "name", "limit": 2, "cursor": first["next_cursor"]}).json()
    assert first["total"] == rest["total"] == 3
    covers = {p["id"]: p["cover_url"] for p in first["items"] + rest["items"]}
    assert covers == {pid: f"/files/face/{fid}" for pid, fid in zip(ids, faces)}


def test_reprocess_without_models_keeps_the_vault_flag_and_scene_category(conn, no_models):
    img = noise_image(800, 600, seed=8)
    vault = make_item(conn, data=jpeg_bytes(img), width=800, height=600, analysis_status="pending",
                      category="nsfw", nsfw=1, nsfw_score=0.93)
    food = make_item(conn, data=jpeg_bytes(noise_image(800, 600, seed=9)), width=800, height=600,
                     analysis_status="pending", category="food",
                     category_scores=json.dumps({"food": 0.81, "places": 0.05}))
    _scene(conn, food["id"], _vec(120, 768))
    for it in (vault, food):
        pipeline.analyze_item(conn, it["id"], 1)
    v = conn.execute("SELECT nsfw, nsfw_score, category FROM items WHERE id = ?", (vault["id"],)).fetchone()
    assert (v["nsfw"], v["nsfw_score"], v["category"]) == (1, 0.93, "nsfw")
    f = conn.execute("SELECT category, category_scores FROM items WHERE id = ?", (food["id"],)).fetchone()
    assert f["category"] == "food"
    scores = json.loads(f["category_scores"])
    assert scores["food"] == 0.81 and "siglip" in scores["skipped"]
    # A first analysis without models still decides from what it has.
    fresh = make_item(conn, data=jpeg_bytes(noise_image(800, 600, seed=10)), width=800, height=600,
                      analysis_status="pending", category="food")
    pipeline.analyze_item(conn, fresh["id"], 1)
    assert conn.execute("SELECT category, nsfw FROM items WHERE id = ?", (fresh["id"],)).fetchone()[:] == ("misc", 0)
