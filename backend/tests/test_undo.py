"""Undo (section 33, 2026-11-09): every owner write is an action whose changes are journaled and can be
put back exactly, with its crop files, and undoing the undo is a redo."""
from __future__ import annotations

import numpy as np
import pytest

from app import storage
from tests.helpers import make_face, make_item, make_person


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _face(conn, face_id):
    return conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (face_id,)).fetchone()


def test_a_face_named_by_hand_goes_back_with_undo_and_comes_again_with_redo(client, conn):
    item = make_item(conn)
    fid = make_face(conn, item["id"], None, embedding=_unit(1))
    r = client.post("/api/faces/assign", json={"face_ids": [fid], "new_person_name": "Meera"})
    assert r.status_code == 200, r.text
    action = r.headers.get("x-action-id")
    assert action
    pid = r.json()["person"]["id"]
    assert tuple(_face(conn, fid)) == (pid, "user")

    u = client.post(f"/api/actions/{action}/undo")
    assert u.status_code == 200, u.text
    body = u.json()
    assert body["changes"] > 0 and body["items"] == [item["uid"]]
    assert tuple(_face(conn, fid)) == (None, "auto")
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (pid,)).fetchone()[0] == 0
    assert client.post(f"/api/actions/{action}/undo").json()["detail"] == "already_undone"

    redo = client.post(f"/api/actions/{body['undo_id']}/undo")
    assert redo.status_code == 200, redo.text
    assert tuple(_face(conn, fid)) == (pid, "user")
    assert conn.execute("SELECT name FROM persons WHERE id = ?", (pid,)).fetchone()[0] == "Meera"


def test_not_a_face_comes_back_with_its_crop_and_its_lookalikes(client, conn):
    item, other = make_item(conn), make_item(conn)
    weak = {"det_score": 0.6, "size_px": 40}
    marked = make_face(conn, item["id"], None, embedding=_unit(2), **weak)
    lookalike = make_face(conn, other["id"], None, embedding=_unit(2), **weak)
    r = client.delete(f"/api/faces/{marked}")
    assert r.status_code == 200 and r.json()["followed"] == 1
    for fid in (marked, lookalike):
        assert not storage.face_crop_path(fid).exists()
    assert conn.execute("SELECT COUNT(*) FROM not_faces").fetchone()[0] == 1

    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    for fid, it in ((marked, item), (lookalike, other)):
        assert conn.execute("SELECT item_id FROM faces WHERE id = ?", (fid,)).fetchone()[0] == it["id"]
        assert storage.face_crop_path(fid).exists()
        assert conn.execute("SELECT face_count FROM items WHERE id = ?", (it["id"],)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM not_faces").fetchone()[0] == 0  # nothing learned from it any more


def test_a_merge_and_a_person_delete_come_back_with_their_faces(client, conn):
    item = make_item(conn)
    a, b = make_person(conn, "Asha"), make_person(conn, "Bina")
    fa = make_face(conn, item["id"], a, embedding=_unit(3), assigned_by="user")
    fb = make_face(conn, item["id"], b, embedding=_unit(4), assigned_by="user", box=(0.6, 0.2, 0.2, 0.3))
    r = client.post("/api/people/merge", json={"target_id": a, "source_ids": [b]})
    assert r.status_code == 200, r.text
    assert _face(conn, fb)["person_id"] == a
    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    assert _face(conn, fb)["person_id"] == b
    assert conn.execute("SELECT name FROM persons WHERE id = ?", (b,)).fetchone()[0] == "Bina"

    r = client.delete(f"/api/people/{a}")  # ON DELETE SET NULL clears the faces: journaled too
    assert r.status_code == 200, r.text
    assert _face(conn, fa)["person_id"] is None
    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    assert _face(conn, fa)["person_id"] == a


def test_a_drawn_person_box_goes_away_and_comes_back(client, conn):
    item = make_item(conn)
    r = client.post(f"/api/items/{item['uid']}/people-boxes",
                    json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.6, "new_person_name": "Ira"})
    assert r.status_code == 201, r.text
    body_id = r.json()["body"]["id"]
    crop = storage.body_crop_path(body_id)
    assert crop.exists()
    u = client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
    assert u.status_code == 200
    assert conn.execute("SELECT COUNT(*) FROM bodies WHERE id = ?", (body_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE name = 'Ira'").fetchone()[0] == 0
    assert not crop.exists()
    assert client.post(f"/api/actions/{u.json()['undo_id']}/undo").status_code == 200  # redo
    assert conn.execute("SELECT person_id FROM bodies WHERE id = ?", (body_id,)).fetchone()[0] is not None
    assert crop.exists()


def test_category_trash_and_album_changes_go_back(client, conn):
    item = make_item(conn, category="misc")
    r = client.patch(f"/api/items/{item['uid']}", json={"category": "documents"})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    row = conn.execute("SELECT category, category_source FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert row["category"] == "misc"

    r = client.delete(f"/api/items/{item['uid']}")
    assert conn.execute("SELECT deleted_at FROM items WHERE id = ?", (item["id"],)).fetchone()[0] is not None
    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    assert conn.execute("SELECT deleted_at FROM items WHERE id = ?", (item["id"],)).fetchone()[0] is None

    r = client.post("/api/albums", json={"name": "Goa", "item_uids": [item["uid"]]})
    assert r.status_code == 201, r.text
    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    assert conn.execute("SELECT COUNT(*) FROM albums").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM album_items").fetchone()[0] == 0


def test_an_undo_waits_for_later_changes_to_the_same_thing(client, conn):
    item = make_item(conn)
    fid = make_face(conn, item["id"], None, embedding=_unit(5))
    first = client.post("/api/faces/assign", json={"face_ids": [fid], "new_person_name": "Dev"})
    second = client.post("/api/faces/assign", json={"face_ids": [fid], "new_person_name": "Kabir"})
    a1, a2 = first.headers["x-action-id"], second.headers["x-action-id"]
    r = client.post(f"/api/actions/{a1}/undo")
    assert r.status_code == 409 and r.json() == {"detail": "later_changes", "blocking": [a2]}
    assert client.post(f"/api/actions/{a2}/undo").status_code == 200
    assert client.post(f"/api/actions/{a1}/undo").status_code == 200
    assert tuple(_face(conn, fid)) == (None, "auto")


def test_what_the_analysis_changed_since_stays(client, conn):
    from app.db import tx

    item = make_item(conn, category="misc")
    r = client.patch(f"/api/items/{item['uid']}", json={"category": "documents", "favorite": True})
    with tx(conn):  # no action: like the worker
        conn.execute("UPDATE items SET category = 'screenshots' WHERE id = ?", (item["id"],))
    u = client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
    assert u.status_code == 200 and u.json()["partial"] is True
    row = conn.execute("SELECT category, favorite FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert row["category"] == "screenshots" and not row["favorite"]


def test_requests_with_one_action_id_undo_together(client, conn):
    item = make_item(conn)
    f1 = make_face(conn, item["id"], None, embedding=_unit(6))
    f2 = make_face(conn, item["id"], None, embedding=_unit(7), box=(0.6, 0.2, 0.2, 0.3))
    pid = make_person(conn, "Noor")
    head = {"X-Action-Id": "bulk-0001-noor"}
    for fid in (f1, f2):
        r = client.post("/api/faces/assign", json={"face_ids": [fid], "person_id": pid}, headers=head)
        assert r.headers["x-action-id"] == "bulk-0001-noor"
    assert client.post("/api/actions/bulk-0001-noor/undo").status_code == 200
    assert _face(conn, f1)["person_id"] is None and _face(conn, f2)["person_id"] is None


def test_only_owner_changes_are_actions(client, conn):
    item = make_item(conn)
    assert "x-action-id" not in client.get(f"/api/items/{item['uid']}").headers
    assert "x-action-id" not in client.post("/api/auth/vault/lock").headers
    r = client.post("/api/items/people", json={"uids": [item["uid"]]})  # a read sent as a POST
    assert r.status_code == 200 and "x-action-id" not in r.headers
    client.delete(f"/api/items/{item['uid']}")
    r = client.post("/api/items/bulk", json={"uids": [item["uid"]], "action": "purge"})
    assert r.status_code == 200 and "x-action-id" not in r.headers  # the files are gone: no undo
    assert client.post("/api/actions/nope-nope-nope/undo").status_code == 404
    assert client.post("/api/actions/bad id/undo").status_code == 404


def test_a_vault_item_is_undone_only_with_the_vault_open(vault_client, conn):
    item = make_item(conn, nsfw=1)
    r = vault_client.patch(f"/api/items/{item['uid']}", json={"favorite": True})
    assert r.status_code == 200, r.text
    vault_client.post("/api/auth/vault/lock")
    u = vault_client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
    assert u.status_code == 403 and u.json()["detail"] == "vault_locked"


def test_old_actions_expire_and_are_pruned(client, conn):
    from datetime import datetime, timedelta, timezone

    from app import undo
    from app.db import tx

    item = make_item(conn)
    fid = make_face(conn, item["id"], None, embedding=_unit(8))
    r = client.delete(f"/api/faces/{fid}")
    action = r.headers["x-action-id"]
    assert undo.stash_dir(action).exists()
    with tx(conn):
        conn.execute("UPDATE undo_actions SET created_at = '2020-01-01T00:00:00.000Z' WHERE id = ?", (action,))
    assert client.post(f"/api/actions/{action}/undo").status_code == 410
    assert undo.prune(conn, now=datetime.now(timezone.utc) + timedelta(days=1)) >= 1
    assert conn.execute("SELECT COUNT(*) FROM undo_log WHERE action_id = ?", (action,)).fetchone()[0] == 0
    assert not undo.stash_dir(action).exists()


def test_every_journaled_table_has_its_triggers(conn):
    from app import undo

    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    for table in undo.JOURNALED:
        for op in ("insert", "update", "delete"):
            assert f"undo_{table}_{op}" in names
    assert conn.execute("SELECT COUNT(*) FROM undo_context").fetchone()[0] == 0  # never left behind


def test_counters_never_block_an_undo(client, conn):
    """Two "Not a face" on one photo both change its face count: the first can still be undone, and the count
    is counted again."""
    item = make_item(conn)
    a = make_face(conn, item["id"], None, embedding=_unit(9))
    b = make_face(conn, item["id"], None, embedding=_unit(10), box=(0.6, 0.2, 0.2, 0.3))
    make_face(conn, item["id"], None, embedding=_unit(11), box=(0.1, 0.6, 0.2, 0.3))
    first = client.delete(f"/api/faces/{a}")
    client.delete(f"/api/faces/{b}")
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 1
    assert client.post(f"/api/actions/{first.headers['x-action-id']}/undo").status_code == 200
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM faces WHERE id IN (?, ?)", (a, b)).fetchone()[0] == 1


def _call_args(sql: str, start: int) -> int:
    """Top-level arguments of the call whose '(' is at start."""
    depth, count, quote = 0, 1, None
    for ch in sql[start:]:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return count
        elif ch == "," and depth == 1:
            count += 1
    raise AssertionError("unbalanced call")


def test_older_sqlite_can_still_open_the_database(conn):
    """SQLite before 3.48 allows 127 arguments per call, and a trigger over that makes the whole database
    unreadable to it: the row JSON of wide tables (items has over 60 columns) is built in chunks."""
    import re

    widest = 0
    for (sql,) in conn.execute("SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'undo_%'"):
        for m in re.finditer(r"json_(object|insert)\(", sql):
            widest = max(widest, _call_args(sql, m.end() - 1))
    assert 0 < widest <= 127


def test_the_journal_keeps_only_what_changed(client, conn):
    """Moving a face logs its person, not its embeddings; a whole row is kept only where an undo needs it."""
    item = make_item(conn)
    fid = make_face(conn, item["id"], None, embedding=_unit(12), torso=_unit(13)[:512])
    r = client.post("/api/faces/assign", json={"face_ids": [fid], "new_person_name": "Lina"})
    rows = conn.execute("SELECT tbl, op, LENGTH(old) + LENGTH(new) AS n FROM undo_log WHERE action_id = ? AND tbl = 'faces'",
                        (r.headers["x-action-id"],)).fetchall()
    assert rows and all(row["op"] == "update" and row["n"] < 1000 for row in rows)  # was about 21 KB with both embeddings


def test_blobs_come_back_on_undo_and_redo(client, conn):
    """An insert's undo and redo bring the row back whole, and a changed BLOB is put back."""
    from app import undo
    from app.db import ActionContext, current_action, tx

    item = make_item(conn)
    emb = _unit(14)
    token = current_action.set(ActionContext("test-insert-0001", "POST", "/api/test"))
    try:
        fid = make_face(conn, item["id"], None, embedding=emb)
    finally:
        current_action.reset(token)
    out = undo.undo(conn, "test-insert-0001")
    assert conn.execute("SELECT COUNT(*) FROM faces WHERE id = ?", (fid,)).fetchone()[0] == 0
    undo.undo(conn, out["undo_id"])  # redo
    back = conn.execute("SELECT embedding FROM faces WHERE id = ?", (fid,)).fetchone()[0]
    assert np.frombuffer(back, dtype=np.float32) == pytest.approx(emb, abs=1e-7)

    other = _unit(15)
    token = current_action.set(ActionContext("test-update-0001", "POST", "/api/test"))
    try:
        with tx(conn):
            conn.execute("UPDATE faces SET embedding = ? WHERE id = ?", (other.tobytes(), fid))
    finally:
        current_action.reset(token)
    undo.undo(conn, "test-update-0001")
    back = conn.execute("SELECT embedding FROM faces WHERE id = ?", (fid,)).fetchone()[0]
    assert np.frombuffer(back, dtype=np.float32) == pytest.approx(emb, abs=1e-7)
