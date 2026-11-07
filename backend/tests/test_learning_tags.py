"""The tag learner (section 19): learns a tag from three look-alikes, respects a removed suggestion,
reads documents by text, applies to existing items, and reports its status."""
from __future__ import annotations

import numpy as np

from app.db import tx
from app.learning import tags as learner
from app.learning import train
from app.util import now_iso
from tests.helpers import make_item
from tests.test_learning_core import add_embedding, near, unit


def tag(conn, name: str) -> int:
    with tx(conn):
        conn.execute("INSERT INTO tags(name, name_key, created_at) VALUES (?, ?, ?)", (name, name.lower(), now_iso()))
        return int(conn.execute("SELECT id FROM tags WHERE name_key = ?", (name.lower(),)).fetchone()[0])


def put(conn, item_id: int, tag_id: int, source: str = "user") -> None:
    with tx(conn):
        conn.execute("INSERT INTO item_user_tags(item_id, tag_id, source, created_at) VALUES (?, ?, ?, ?)",
                     (item_id, tag_id, source, now_iso()))


def test_image_tag_learned_from_three_look_alikes_and_blocked_by_a_rejection(conn):
    cake, beach = tag(conn, "cake"), tag(conn, "beach")
    base = unit(11)
    for k in range(3):
        it = make_item(conn, filename=f"c{k}.jpg", write_files=False)
        add_embedding(conn, it["id"], near(base, 50 + k))
        put(conn, it["id"], cake)
    for k in range(2):  # two examples are not enough
        it = make_item(conn, filename=f"b{k}.jpg", write_files=False)
        add_embedding(conn, it["id"], near(unit(12), 60 + k))
        put(conn, it["id"], beach)
    assert learner.predict(near(base, 99), None, "image") == []  # untrained
    st = learner.train(conn)
    assert st["tags"] == 1 and st["examples"] == 5 and st["names"] == ["cake"]
    hits = learner.predict(near(base, 99), None, "image")
    assert len(hits) == 1 and hits[0][0] == cake and hits[0][1] >= learner.IMG_MIN
    assert learner.predict(unit(77), None, "image") == []
    # the owner removed the suggestion on a look-alike: that neighbourhood stays quiet
    rej = make_item(conn, filename="rej.jpg", write_files=False)
    add_embedding(conn, rej["id"], near(base, 98))
    with tx(conn):
        conn.execute("INSERT INTO tag_rejections(item_id, tag_id, created_at) VALUES (?, ?, ?)", (rej["id"], cake, now_iso()))
    learner.train(conn)
    assert learner.predict(near(base, 98, 0.01), None, "image") == []
    assert learner.status()["rejections"] == 1


def test_document_tag_learned_from_text(conn):
    shop = tag(conn, "shop")
    texts = ["Sharma Electronics receipt number 1 mixer grinder total 2100 paid by UPI thanks",
             "Sharma Electronics receipt number 2 kettle total 400 paid by cash thanks",
             "Sharma Electronics receipt number 3 fan total 1800 paid by card thanks"]
    for k, text in enumerate(texts):
        it = make_item(conn, kind="pdf", filename=f"r{k}.pdf", write_files=False)
        with tx(conn):
            conn.execute("INSERT INTO item_text(item_id, text, source, lang, char_count, created_at) VALUES (?, ?, 'pdf', NULL, ?, ?)",
                         (it["id"], text, len(text), now_iso()))
        put(conn, it["id"], shop)
    learner.train(conn)
    hits = learner.predict(None, "Sharma Electronics receipt number 4 washing machine total 18500 paid by card thanks", "pdf")
    assert hits and hits[0][0] == shop and hits[0][1] >= learner.TEXT_MIN
    assert learner.predict(None, "a poem about monsoon hills and rivers", "pdf") == []
    assert learner.predict(None, "Sharma Electronics receipt", "image") == []  # wrong kind of evidence


def test_apply_item_and_apply_all_insert_suggestions_once(conn, monkeypatch):
    cake = tag(conn, "cake")
    a = make_item(conn, filename="a.jpg", write_files=False)
    b = make_item(conn, filename="b.jpg", write_files=False)
    add_embedding(conn, a["id"], unit(1))
    add_embedding(conn, b["id"], unit(2))
    put(conn, b["id"], cake)  # already tagged by hand
    monkeypatch.setattr(learner, "predict", lambda emb, text, kind: [(cake, 0.93)])
    with tx(conn):
        assert learner.apply_item(conn, a["id"], emb=unit(1), text=None, kind="image") == 1
        assert learner.apply_item(conn, a["id"], emb=unit(1), text=None, kind="image") == 0  # present
        assert learner.apply_item(conn, b["id"], emb=unit(2), text=None, kind="image") == 0  # user row stays
    row = conn.execute("SELECT source, score FROM item_user_tags WHERE item_id = ?", (a["id"],)).fetchone()
    assert row["source"] == "auto" and row["score"] == 0.93
    with tx(conn):
        conn.execute("DELETE FROM item_user_tags WHERE item_id = ?", (a["id"],))
        conn.execute("INSERT INTO tag_rejections(item_id, tag_id, created_at) VALUES (?, ?, ?)", (a["id"], cake, now_iso()))
        assert learner.apply_item(conn, a["id"], emb=unit(1), text=None, kind="image") == 0  # rejected pair
    # apply_all needs a trained model with learned tags; fake one
    monkeypatch.setattr(learner.store, "load", lambda name: {"learned": [cake]} if name == learner.MODEL else None)
    c = make_item(conn, filename="c.jpg", write_files=False)
    add_embedding(conn, c["id"], unit(3))
    assert learner.apply_all(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM item_user_tags WHERE item_id = ? AND source = 'auto'", (c["id"],)).fetchone()[0] == 1
    assert "cake" in conn.execute("SELECT labels FROM items_fts WHERE rowid = ?", (c["id"],)).fetchone()[0]


def test_training_status_includes_tags(conn):
    st = train.train_all(conn)
    assert st["learners"]["tags"] == {"tags": 0, "examples": 0, "rejections": 0, "names": []}
