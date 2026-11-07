"""People API routes (app.api.people) on a bare FastAPI app with the session dependency overridden."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import storage
from app.api import people
from app.api.deps import require_user
from app.auth.sessions import SessionInfo
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person

H = {"X-Requested-With": "pcc"}


@pytest.fixture()
def api(data_dir):
    app = FastAPI()
    app.include_router(people.router)
    fake = SessionInfo(id=1, created_at="", last_seen_at="", expires_at="", vault_until=None, user_agent=None, ip=None)
    app.dependency_overrides[require_user] = lambda: fake
    with TestClient(app) as c:
        c.headers.update(H)
        yield c


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _body(conn, item_id, person_id=None, face_id=None, **kw) -> int:
    cols = dict(item_id=item_id, person_id=person_id, face_id=face_id, assigned_by="auto", x=0.1, y=0.1, w=0.3, h=0.7,
                det_score=0.9, created_at=now_iso())
    cols.update(kw)
    with tx(conn):
        bid = int(conn.execute(f"INSERT INTO bodies({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                               tuple(cols.values())).lastrowid)
    p = storage.body_crop_path(bid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return bid


def _item(conn, **kw):
    return make_item(conn, write_files=False, data=None, **kw)


def test_list_people_filters_sorts_and_pages(api, conn):
    asha, ravi, unnamed, secret = make_person(conn, "Asha"), make_person(conn, "Ravi"), make_person(conn), make_person(conn, "Hidden")
    for k in range(3):
        make_face(conn, _item(conn)["id"], asha, embedding=_vec(k))
    make_face(conn, _item(conn)["id"], ravi, embedding=_vec(10))
    make_face(conn, _item(conn)["id"], unnamed, embedding=_vec(11))
    make_face(conn, _item(conn, nsfw=1)["id"], secret, embedding=_vec(12))
    empty_named = make_person(conn, "Newcomer")

    r = api.get("/api/people").json()
    ids = [p["id"] for p in r["items"]]
    assert secret not in ids and r["total"] == 4
    assert ids[0] == asha and r["items"][0]["item_count"] == 3 and r["items"][0]["face_count"] == 3
    assert empty_named in ids

    assert {p["id"] for p in api.get("/api/people?filter=unnamed").json()["items"]} == {unnamed}
    assert {p["id"] for p in api.get("/api/people?q=rav").json()["items"]} == {ravi}
    assert {p["id"] for p in api.get(f"/api/people?q=unnamed%20%23{unnamed}").json()["items"]} == {unnamed}

    for sort in ("items", "name", "recent"):
        seen, cursor = [], None
        while True:
            page = api.get("/api/people", params={"sort": sort, "limit": 1, **({"cursor": cursor} if cursor else {})}).json()
            seen += [p["id"] for p in page["items"]]
            cursor = page["next_cursor"]
            if not cursor:
                break
        assert sorted(seen) == sorted(ids) and len(seen) == len(set(seen)), sort
        if sort == "name":
            assert seen[:3] == [asha, empty_named, ravi] and seen[-1] == unnamed
    assert api.get("/api/people?cursor=bad").status_code == 400


def test_create_rename_merge_delete(api, conn):
    i1, i2 = _item(conn), _item(conn)
    f1 = make_face(conn, i1["id"], None, embedding=_vec(1))
    f2 = make_face(conn, i2["id"], None, embedding=_vec(2))
    r = api.post("/api/people", json={"name": "  Asha  ", "face_ids": [f1]})
    assert r.status_code == 200, r.text
    asha = r.json()["id"]
    assert r.json()["name"] == "Asha" and r.json()["face_count"] == 1
    row = conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (f1,)).fetchone()
    assert (row["person_id"], row["assigned_by"]) == (asha, "user")
    # Same name again reuses the person.
    assert api.post("/api/people", json={"name": "asha", "face_ids": [f2]}).json()["id"] == asha
    assert api.post("/api/people", json={"name": "   "}).status_code in (400, 422)

    other = make_person(conn)
    f3 = make_face(conn, _item(conn)["id"], other, embedding=_vec(3))
    r = api.patch(f"/api/people/{other}", json={"name": "ASHA"})
    assert r.status_code == 200 and r.json()["id"] == asha and r.json()["face_count"] == 3
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (other,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE type = 'guess'").fetchone()[0] >= 1

    r = api.patch(f"/api/people/{asha}", json={"cover_face_id": f3})
    assert r.status_code == 200 and r.json()["cover_url"] == f"/files/face/{f3}"
    stranger_face = make_face(conn, _item(conn)["id"], None, embedding=_vec(4))
    assert api.patch(f"/api/people/{asha}", json={"cover_face_id": stranger_face}).status_code == 400
    assert api.patch(f"/api/people/{asha}", json={"name": "Asha K"}).json()["name"] == "Asha K"

    b = make_person(conn, "Ravi")
    make_face(conn, _item(conn)["id"], b, embedding=_vec(5))
    r = api.post("/api/people/merge", json={"target_id": asha, "source_ids": [b]})
    assert r.status_code == 200 and r.json()["face_count"] == 4
    assert api.post("/api/people/merge", json={"target_id": asha, "source_ids": [asha]}).status_code == 400
    assert api.post("/api/people/merge", json={"target_id": asha, "source_ids": [424242]}).status_code == 404

    assert api.delete(f"/api/people/{asha}").json() == {"ok": True}
    assert api.get(f"/api/people/{asha}").status_code == 404
    left = conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (f1,)).fetchone()
    assert left["person_id"] is None and left["assigned_by"] == "user"


def test_person_faces_and_items(api, conn):
    p = make_person(conn, "Asha")
    visible, secret, trashed = _item(conn, taken_at="2026-01-02T10:00:00.000Z"), _item(conn, nsfw=1), _item(conn, deleted_at=now_iso())
    tagged = _item(conn, taken_at="2026-01-01T10:00:00.000Z")
    fv = make_face(conn, visible["id"], p, embedding=_vec(1))
    make_face(conn, secret["id"], p, embedding=_vec(2))
    make_face(conn, trashed["id"], p, embedding=_vec(3))
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                     (tagged["id"], p, now_iso()))
    faces = api.get(f"/api/people/{p}/faces").json()
    assert [f["id"] for f in faces["items"]] == [fv] and faces["total"] == 1
    assert faces["items"][0]["url"] == f"/files/face/{fv}" and faces["items"][0]["item_uid"] == visible["uid"]
    items = api.get(f"/api/people/{p}/items").json()
    assert [i["uid"] for i in items["items"]] == [visible["uid"], tagged["uid"]] and items["total"] == 2
    page = api.get(f"/api/people/{p}/items?limit=1").json()
    nxt = api.get(f"/api/people/{p}/items", params={"limit": 1, "cursor": page["next_cursor"]}).json()
    assert [i["uid"] for i in nxt["items"]] == [tagged["uid"]] and nxt["next_cursor"] is None
    assert api.get(f"/api/people/{p}/items?vault=1").json()["detail"] == "vault_locked"
    assert api.get("/api/people/9999/faces").status_code == 404


def test_unassigned_assign_and_delete_faces(api, conn):
    i1 = _item(conn)
    f1 = make_face(conn, i1["id"], None, embedding=_vec(1))
    f2 = make_face(conn, i1["id"], None, embedding=_vec(2), box=(0.6, 0.1, 0.2, 0.3))
    make_face(conn, _item(conn, nsfw=1)["id"], None, embedding=_vec(3))
    body = _body(conn, i1["id"], face_id=f1)
    page = api.get("/api/faces/unassigned").json()
    assert [f["id"] for f in page["items"]] == [f2, f1] and page["total"] == 2
    p1 = api.get("/api/faces/unassigned?limit=1").json()
    assert [f["id"] for f in api.get("/api/faces/unassigned", params={"limit": 1, "cursor": p1["next_cursor"]}).json()["items"]] == [f1]

    r = api.post("/api/faces/assign", json={"face_ids": [f1], "new_person_name": "Meera"})
    assert r.status_code == 200 and r.json()["updated"] == 1
    pid = r.json()["person"]["id"]
    assert conn.execute("SELECT person_id, assigned_by FROM bodies WHERE id = ?", (body,)).fetchone()[:] == (pid, "user")
    assert api.post("/api/faces/assign", json={"face_ids": [f2], "person_id": pid, "new_person_name": "X"}).status_code == 400
    assert api.post("/api/faces/assign", json={"face_ids": [12345], "person_id": pid}).status_code == 404
    r = api.post("/api/faces/assign", json={"face_ids": [f1]})
    assert r.json() == {"updated": 1, "person": None, "followed": 0}
    # Meera had nothing else: still there (named people are kept).
    assert api.get(f"/api/people/{pid}").status_code == 200

    crop = storage.face_crop_path(f2)
    assert crop.exists()
    v0 = conn.execute("SELECT value FROM kv WHERE key = 'search_version'").fetchone()[0]
    assert api.delete(f"/api/faces/{f2}").json() == {"ok": True}
    assert not crop.exists()
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (i1["id"],)).fetchone()[0] == 1
    assert conn.execute("SELECT value FROM kv WHERE key = 'search_version'").fetchone()[0] != v0
    assert api.delete(f"/api/faces/{f2}").status_code == 404


def test_bodies_assign_and_delete(api, conn):
    item = _item(conn)
    b = _body(conn, item["id"])
    r = api.post("/api/bodies/assign", json={"body_ids": [b], "new_person_name": "Kiran"})
    assert r.status_code == 200 and r.json()["person"]["item_count"] == 1
    assert api.delete(f"/api/bodies/{b}").json() == {"ok": True}
    assert not storage.body_crop_path(b).exists()
    assert api.delete(f"/api/bodies/{b}").status_code == 404


def test_guess_review_confirm_reject(api, conn):
    p = make_person(conn, "Asha")
    make_face(conn, _item(conn)["id"], p, embedding=_vec(1))
    item = _item(conn)
    ev = json.dumps({"clothes": 0.9, "scene": 0.85, "face": None})
    b1 = _body(conn, item["id"], assigned_by="guess", guess_person_id=p, guess_score=0.8, guess_evidence=ev)
    # One guess per photo for a person (guess._compute), so the other two boxes are in other photos.
    b2 = _body(conn, _item(conn)["id"], assigned_by="guess", guess_person_id=p, guess_score=0.7, guess_evidence=ev)
    f1 = make_face(conn, _item(conn)["id"], None, embedding=_vec(2), assigned_by="guess", guess_person_id=p,
                   guess_score=0.75, guess_evidence=ev)
    hidden = _item(conn, nsfw=1)
    _body(conn, hidden["id"], assigned_by="guess", guess_person_id=p, guess_score=0.9, guess_evidence=ev)

    page = api.get("/api/review/guesses").json()
    assert page["total"] == 3
    assert [(g["kind"], g["id"]) for g in page["items"]] == [("body", b1), ("face", f1), ("body", b2)]
    first = page["items"][0]
    assert first["guess"]["person"]["id"] == p and first["item"]["uid"] == item["uid"]
    assert first["box"]["id"] == b1 and first["guess"]["evidence"]["clothes"] == 0.9
    p1 = api.get("/api/review/guesses?limit=2").json()
    rest = api.get("/api/review/guesses", params={"limit": 2, "cursor": p1["next_cursor"]}).json()
    assert [(g["kind"], g["id"]) for g in rest["items"]] == [("body", b2)]
    assert api.get(f"/api/people/{p}").json()["guess_count"] == 3

    assert api.post("/api/guesses/confirm", json={"faces": [f1], "bodies": [b1]}).json() == {"updated": 2}
    assert conn.execute("SELECT person_id FROM faces WHERE id = ?", (f1,)).fetchone()[0] == p
    assert conn.execute("SELECT COUNT(*) FROM item_tags WHERE item_id = ? AND person_id = ?", (item["id"], p)).fetchone()[0] == 1
    assert api.post("/api/guesses/reject", json={"bodies": [b2]}).json() == {"updated": 1}
    assert conn.execute("SELECT COUNT(*) FROM guess_rejections").fetchone()[0] == 1
    assert api.get("/api/review/guesses").json()["total"] == 0


def test_suggestions_and_dismiss(api, conn):
    base = _vec(50)
    a, b = make_person(conn), make_person(conn)
    near = lambda s: (lambda v: v / np.linalg.norm(v))(base + 0.5 * _vec(s))  # noqa: E731
    make_face(conn, _item(conn)["id"], a, embedding=near(51))
    make_face(conn, _item(conn)["id"], b, embedding=near(52))
    items = api.get("/api/suggestions").json()["items"]
    assert len(items) == 1 and {items[0]["a"]["id"], items[0]["b"]["id"]} == {a, b}
    assert items[0]["a_samples"] and items[0]["similarity"] >= 0.35
    assert api.post("/api/suggestions/dismiss", json={"a": b, "b": a}).json() == {"ok": True}
    assert api.get("/api/suggestions").json()["items"] == []
    assert api.post("/api/suggestions/dismiss", json={"a": a, "b": a}).status_code == 400
    assert api.post("/api/suggestions/dismiss", json={"a": a, "b": 9999}).status_code == 404


def test_tags_and_recluster(api, conn):
    item = _item(conn)
    r = api.post(f"/api/items/{item['uid']}/tags", json={"new_person_name": "Dev"})
    assert r.status_code == 200 and [t["name"] for t in r.json()] == ["Dev"]
    dev = r.json()[0]["id"]
    assert api.post(f"/api/items/{item['uid']}/tags", json={"person_id": dev}).status_code == 200
    assert api.post(f"/api/items/{item['uid']}/tags", json={}).status_code == 400
    assert api.post("/api/items/nope/tags", json={"person_id": dev}).status_code == 404
    job = conn.execute("SELECT payload FROM jobs WHERE type = 'guess'").fetchone()
    assert json.loads(job[0]) == {"person_id": dev, "item_id": item["id"]}
    assert api.delete(f"/api/items/{item['uid']}/tags/{dev}").json() == []

    assert api.post("/api/recluster").json() == {"ok": True}
    assert api.post("/api/recluster").json() == {"ok": True}
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE type = 'recluster'").fetchone()[0] == 1
