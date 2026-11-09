"""/api/system/learning: status, the switch, training on demand, applying to existing items, reset."""
from __future__ import annotations

from app import db
from app.util import now_iso
from tests.helpers import make_item

H = {"X-Requested-With": "pcc"}


def test_learning_routes_round_trip(client):
    r = client.get("/api/system/learning")
    assert r.status_code == 200
    st = r.json()
    assert st["enabled"] is True and st["trained_at"] is None
    assert set(st["learners"]) == {"doc_type", "category", "vault", "misclick", "people", "tags", "forwards", "not_faces",
                                    "traits"}

    r = client.patch("/api/system/learning", json={"enabled": False}, headers=H)
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert client.get("/api/system/learning").json()["enabled"] is False
    client.patch("/api/system/learning", json={"enabled": True}, headers=H)

    with db.connect() as conn:
        make_item(conn, filename="user.jpg", category="food", category_source="user", write_files=False)
        auto = make_item(conn, filename="auto.jpg", category="objects", write_files=False)
        doc = make_item(conn, kind="pdf", filename="doc.pdf", text_status="done", write_files=False)
        now = now_iso()
        with db.tx(conn):
            conn.execute("INSERT INTO documents(item_id, doc_type, doc_group, confidence, fields, source, created_at, updated_at) "
                         "VALUES (?, 'other', 'other', 0.5, '{}', 'auto', ?, ?)", (doc["id"], now, now))
    r = client.post("/api/system/learning/train", headers=H)
    assert r.status_code == 200 and r.json()["trained_at"] and r.json()["pending"] == 0

    r = client.post("/api/system/learning/apply", headers=H)
    assert r.status_code == 200
    out = r.json()
    assert out["items"] == 2 and out["documents"] == 1  # the user-labelled item has an automatic vault flag
    with db.connect() as conn:
        kinds = [row[0] for row in conn.execute("SELECT type FROM jobs WHERE status = 'queued'")]
    assert kinds.count("analyze") == 2 and kinds.count("text") == 1
    assert auto["id"]

    r = client.delete("/api/system/learning", headers=H)
    assert r.status_code == 200
    assert client.get("/api/system/learning").json()["trained_at"] is None


def test_learning_routes_need_a_session(anon_client):
    assert anon_client.get("/api/system/learning").status_code == 401
