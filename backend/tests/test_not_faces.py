""""Not a face" (sections 10 and 18): a removal is kept for the learner, its unclear unassigned look-alikes
go with it, and detection drops what looks like a marking (never a clear face, never one the owner
assigned on the item). The models are off here (PCC_NO_MODELS=1); detections are faked where they matter."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app import storage
from app.analysis import faces as faces_mod
from app.analysis import pipeline
from app.analysis.faces import CLEAR_WEIGHT, face_weight
from app.db import tx
from app.learning import not_faces, store, train
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person

UNCLEAR = dict(det_score=0.6, size_px=80, clarity=12.0, sharpness=6.0)   # face weight about 0
CLEAR = dict(det_score=0.95, size_px=150, clarity=25.0, sharpness=60.0)  # face weight 1


def _unit(v: np.ndarray) -> np.ndarray:
    return (np.asarray(v, dtype=np.float32) / np.linalg.norm(v)).astype(np.float32)


def _rand(seed: int) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(512))


def _alike(base: np.ndarray, sim: float, seed: int) -> np.ndarray:
    """A unit vector at about `sim` cosine to base."""
    other = _rand(seed)
    other = _unit(other - float(other @ base) * base)
    return _unit(sim * base + math.sqrt(max(0.0, 1.0 - sim * sim)) * other)


def _rec(emb: np.ndarray, box=(0.3, 0.2, 0.2, 0.3), **kw) -> SimpleNamespace:
    vals = dict(UNCLEAR, **kw)
    return SimpleNamespace(x=box[0], y=box[1], w=box[2], h=box[3], embedding=emb, **vals)


def _mark(conn, emb: np.ndarray, item_id: int | None = None, **kw) -> None:
    vals = dict(UNCLEAR, **kw)
    with tx(conn):
        conn.execute(
            "INSERT INTO not_faces(item_id, det_score, size_px, clarity, sharpness, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (item_id, vals["det_score"], vals["size_px"], vals["clarity"], vals["sharpness"], emb.tobytes(), now_iso()))


def test_the_test_faces_weigh_as_intended():
    assert face_weight(**UNCLEAR) < 0.05 and face_weight(**CLEAR) == pytest.approx(1.0)


# ------------------------------------------------------------------ the removal
def test_not_a_face_is_kept_and_its_unclear_unassigned_lookalikes_go(client, conn):
    a, b = make_item(conn), make_item(conn)
    base = _rand(1)
    marked = make_face(conn, a["id"], None, embedding=base, **UNCLEAR)
    follower_a = make_face(conn, a["id"], None, embedding=_alike(base, 0.8, 2), box=(0.6, 0.1, 0.2, 0.3), **UNCLEAR)
    follower_b = make_face(conn, b["id"], None, embedding=_alike(base, 0.7, 3), **UNCLEAR)
    clear = make_face(conn, b["id"], None, embedding=_alike(base, 0.9, 4), box=(0.6, 0.5, 0.2, 0.3), **CLEAR)
    pid = make_person(conn, "Asha")
    assigned = make_face(conn, b["id"], pid, embedding=_alike(base, 0.9, 5), box=(0.1, 0.5, 0.2, 0.3), **UNCLEAR)
    locked = make_face(conn, b["id"], None, embedding=_alike(base, 0.9, 6), box=(0.1, 0.1, 0.1, 0.1),
                       assigned_by="user", **UNCLEAR)
    guessed = make_face(conn, b["id"], None, embedding=_alike(base, 0.9, 7), box=(0.4, 0.6, 0.1, 0.1),
                        assigned_by="guess", guess_person_id=pid, guess_score=0.6, **UNCLEAR)
    unlike = make_face(conn, b["id"], None, embedding=_alike(base, 0.2, 8), box=(0.8, 0.8, 0.1, 0.1), **UNCLEAR)

    r = client.delete(f"/api/faces/{marked}")
    assert r.status_code == 200 and r.json() == {"ok": True, "followed": 2}
    row = conn.execute("SELECT * FROM not_faces").fetchone()
    assert row["item_id"] == a["id"] and row["det_score"] == pytest.approx(0.6) and row["size_px"] == 80
    assert row["clarity"] == pytest.approx(12.0) and row["sharpness"] == pytest.approx(6.0)
    assert np.allclose(np.frombuffer(row["embedding"], dtype=np.float32), base)
    assert conn.execute("SELECT COUNT(*) FROM not_faces").fetchone()[0] == 1   # look-alikes are no markings
    left = {int(x[0]) for x in conn.execute("SELECT id FROM faces")}
    assert left == {clear, assigned, locked, guessed, unlike}
    for gone in (marked, follower_a, follower_b):
        assert not storage.face_crop_path(gone).exists()
    counts = dict(conn.execute("SELECT id, face_count FROM items").fetchall())
    assert counts[a["id"]] == 0 and counts[b["id"]] == 5
    assert client.delete(f"/api/faces/{marked}").status_code == 404


def test_a_face_without_lookalikes_follows_nothing(client, conn):
    item = make_item(conn)
    face = make_face(conn, item["id"], None, embedding=_rand(11), **CLEAR)
    make_face(conn, item["id"], None, embedding=_rand(12), box=(0.6, 0.1, 0.2, 0.3), **UNCLEAR)
    assert client.delete(f"/api/faces/{face}").json() == {"ok": True, "followed": 0}
    assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM not_faces").fetchone()[0] == 1


# ------------------------------------------------------------------ the filter
def test_filter_drops_what_looks_like_a_marking_but_never_a_clear_face():
    base = _rand(20)
    f = not_faces.Filter(marks=np.stack([base]))
    like = _rec(_alike(base, 0.8, 21))
    clear_like = _rec(_alike(base, 0.99, 22), **CLEAR)
    unlike = _rec(_alike(base, 0.3, 23))
    assert f.nearest(like.embedding) == pytest.approx(0.8, abs=1e-3)
    assert f.drops(like) and not f.drops(clear_like) and not f.drops(unlike)
    assert f.keep([like, clear_like, unlike]) == [clear_like, unlike]
    assert not not_faces.Filter(marks=np.zeros((0, 512), dtype=np.float32)).drops(like)


def test_filter_keeps_a_face_the_owner_assigned_on_the_item():
    base = _rand(30)
    owners = _alike(base, 0.75, 31)   # the owner's face on the item, itself rather like the marking
    f = not_faces.Filter(marks=np.stack([base]), protect=[((0.3, 0.2, 0.2, 0.3), owners)])
    by_box = _rec(_alike(base, 0.6, 32), box=(0.31, 0.2, 0.2, 0.3))     # over it, not alike
    by_look = _rec(_alike(owners, 0.97, 33), box=(0.7, 0.6, 0.1, 0.1))  # elsewhere, the same face
    other = _rec(_alike(base, 0.6, 34), box=(0.05, 0.6, 0.1, 0.1))
    assert float(by_box.embedding @ owners) < not_faces.PROTECT_SIM
    assert float(other.embedding @ owners) < not_faces.PROTECT_SIM
    assert min(f.nearest(r.embedding) for r in (by_box, by_look, other)) >= not_faces.SIM
    assert not f.drops(by_box) and not f.drops(by_look) and f.drops(other)


def test_the_learned_rule_drops_what_it_calls_no_face():
    base = _rand(40)
    model = {"w": {"det_score": 0.0, "clarity": 0.0, "sharpness": 0.0, "size": 0.0, "nearest": 20.0}, "b": -6.0}
    f = not_faces.Filter(marks=np.stack([base]), model=model)
    near = _rec(_alike(base, 0.45, 41))          # below SIM: only the rule can drop it
    far = _rec(_alike(base, 0.1, 42))
    assert f.nearest(near.embedding) < not_faces.SIM
    assert f.probability(near.det_score, near.size_px, near.clarity, near.sharpness,
                         f.nearest(near.embedding)) >= not_faces.DROP_P
    assert f.drops(near) and not f.drops(far)
    assert not f.drops(_rec(near.embedding, **CLEAR))   # a clear face is never dropped
    # the rule needs the measures: a face analysed before them is left alone
    assert not f.drops(_rec(near.embedding, clarity=None, sharpness=None, det_score=0.55, size_px=30))


def _train_set(conn, n: int) -> tuple[np.ndarray, int]:
    item = make_item(conn)
    pid = make_person(conn, "Asha")
    base = _rand(50)
    for i in range(n):
        _mark(conn, _unit(base + 0.6 * _rand(100 + i)), item["id"], det_score=0.55, size_px=60, clarity=9.0,
              sharpness=3.0)
        make_face(conn, item["id"], pid, embedding=_rand(200 + i), assigned_by="user", write_crop=False, **CLEAR)
    return base, int(item["id"])


def test_the_rule_is_trained_once_enough_markings_and_faces_exist(conn):
    _train_set(conn, not_faces.MIN_EACH - 1)
    assert not_faces.train(conn)["rule"] is False and not_faces.load(conn).model is None
    base, item_id = _train_set(conn, 2)
    st = not_faces.train(conn)
    assert st["rule"] is True and st["markings"] == not_faces.MIN_EACH + 1
    f = not_faces.load(conn)
    assert f.model is not None and set(f.model["w"]) == set(not_faces.FEATURES)
    mark_p = f.probability(0.55, 60, 9.0, 3.0, f.nearest(_unit(base + 0.6 * _rand(999))))
    face_p = f.probability(0.95, 150, 25.0, 60.0, f.nearest(_rand(998)))
    assert mark_p > 0.5 > face_p


def test_learning_off_drops_the_rule_but_keeps_the_markings(conn):
    base, _ = _train_set(conn, not_faces.MIN_EACH)
    not_faces.train(conn)
    assert not_faces.load(conn).model is not None
    store.set_enabled(conn, False)
    f = not_faces.load(conn)
    assert f.model is None
    assert f.drops(_rec(_alike(base, 0.95, 60)))


def test_load_protects_the_owners_faces_on_the_item(conn):
    item, other = make_item(conn), make_item(conn)
    pid = make_person(conn, "Asha")
    mine = make_face(conn, item["id"], pid, embedding=_rand(70), assigned_by="user", box=(0.3, 0.2, 0.2, 0.3))
    make_face(conn, item["id"], pid, embedding=_rand(71), box=(0.6, 0.2, 0.2, 0.3))          # auto: not protected
    make_face(conn, other["id"], pid, embedding=_rand(72), assigned_by="user")              # another item
    f = not_faces.load(conn, item["id"])
    assert len(f.protect) == 1 and f.protect[0][0] == pytest.approx((0.3, 0.2, 0.2, 0.3))
    assert mine  # the protected one is the owner's face on this item


def test_learning_status_lists_the_not_a_face_learner(conn):
    _mark(conn, _rand(80))
    st = train.status(conn)
    assert "not_faces" in st["learners"]
    train.train_all(conn)
    assert train.status(conn)["learners"]["not_faces"]["markings"] == 1


# ------------------------------------------------------------------ the analysis
def test_analysis_drops_detections_that_look_like_a_marking(conn, monkeypatch):
    base = _rand(90)
    _mark(conn, base)
    item = make_item(conn, width=800, height=600, analysis_status="pending")
    crop = Image.new("RGB", (192, 192), (100, 100, 100))

    def record(emb, box, **kw):
        vals = dict(UNCLEAR, **kw)
        return faces_mod.FaceRecord(x=box[0], y=box[1], w=box[2], h=box[3], det_score=vals["det_score"],
                                    size_px=vals["size_px"], quality=0.5, frame_time_s=None, embedding=emb, crop=crop,
                                    clarity=vals["clarity"], sharpness=vals["sharpness"])

    detected = [record(_alike(base, 0.85, 91), (0.1, 0.1, 0.1, 0.1)),
                record(_alike(base, 0.85, 92), (0.4, 0.1, 0.2, 0.3), **CLEAR),
                record(_alike(base, 0.1, 93), (0.7, 0.1, 0.1, 0.1))]
    monkeypatch.setattr(pipeline, "_model", lambda name: SimpleNamespace() if name == "faces" else None)
    monkeypatch.setattr(faces_mod, "analyze_image", lambda img, source_scale=1.0, img_bgr=None: list(detected))
    pipeline.analyze_item(conn, item["id"], 1)
    xs = sorted(round(r[0], 2) for r in conn.execute("SELECT x FROM faces WHERE item_id = ?", (item["id"],)))
    assert xs == [0.4, 0.7]
    row = conn.execute("SELECT analysis_status, face_count FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert row["analysis_status"] == "done" and row["face_count"] == 2
