"""Bringing stored faces up to the current face analysis (section 10): analysed videos are queued once for
the new tracker below new uploads, and the faces of analysed photos are measured for clarity in place. The
models are off here (PCC_NO_MODELS=1); the face engine and the photo loader are faked."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app import jobs
from app.analysis import animals as animals_mod
from app.analysis import face_upkeep
from app.analysis.faces import TRACKER_VERSION
from app.db import faces_version, kv_get, tx
from app.util import now_iso
from tests.helpers import make_face, make_item


def _video(conn, **kw):
    return make_item(conn, kind="video", ext=".mp4", mime="video/mp4", write_files=False, data=None, **kw)


def _status(conn, item_id: int) -> str:
    return conn.execute("SELECT analysis_status FROM items WHERE id = ?", (item_id,)).fetchone()[0]


def _analyze_jobs(conn) -> dict[int, int]:
    return {json.loads(r["payload"])["item_id"]: int(r["priority"])
            for r in conn.execute("SELECT payload, priority FROM jobs WHERE type = 'analyze'")}


# ------------------------------------------------------------------ videos
def test_analysed_videos_are_queued_once_below_new_uploads(conn, monkeypatch):
    monkeypatch.setattr(face_upkeep, "_faces_ready", lambda: True)
    still = make_item(conn)
    done = _video(conn)
    clip = _video(conn, companion_of=still["id"])
    trashed = _video(conn, deleted_at=now_iso())
    pending = _video(conn, analysis_status="pending")
    photo = make_item(conn)

    assert face_upkeep.retrack_videos(conn) == 1
    assert _status(conn, done["id"]) == "pending"
    assert {i: _status(conn, i) for i in (clip["id"], trashed["id"], photo["id"])} == {
        clip["id"]: "done", trashed["id"]: "done", photo["id"]: "done"}
    assert _status(conn, pending["id"]) == "pending"
    assert _analyze_jobs(conn) == {done["id"]: face_upkeep.REANALYSE_PRIORITY}
    assert kv_get(conn, face_upkeep.VIDEO_TRACKER_KEY) == str(TRACKER_VERSION)
    # once per tracker version
    with tx(conn):
        conn.execute("UPDATE items SET analysis_status = 'done' WHERE id = ?", (done["id"],))
    assert face_upkeep.retrack_videos(conn) == 0
    assert _status(conn, done["id"]) == "done"


def test_videos_wait_while_the_face_models_are_off(conn):
    done = _video(conn)
    assert face_upkeep.retrack_videos(conn) == 0
    assert _status(conn, done["id"]) == "done" and _analyze_jobs(conn) == {}
    assert kv_get(conn, face_upkeep.VIDEO_TRACKER_KEY) is None


# ------------------------------------------------------------------ photos
class FakeEngine:
    """Detects a face at each of `boxes` (normalised) and measures them with `norm` and `sharpness`."""

    def __init__(self, boxes_by_item: dict[int, list[tuple[float, float, float, float]]], size=(800, 600),
                 norm: float = 21.5, sharpness: float = 42.0):
        self.boxes = boxes_by_item
        self.size = size
        self.norm, self.sharpness = norm, sharpness
        self.current: int | None = None
        self.calls = 0

    def detect_multiscale(self, bgr, sizes, threshold):
        self.calls += 1
        W, H = self.size
        return [SimpleNamespace(bbox=np.array([x * W, y * H, (x + w) * W, (y + h) * H], dtype=np.float32),
                                norm=None, sharpness=None)
                for x, y, w, h in self.boxes.get(self.current, [])]

    def embed(self, bgr, faces):
        for f in faces:
            f.norm, f.sharpness = self.norm, self.sharpness


def _fake_photos(monkeypatch, engine: FakeEngine) -> None:
    def open_photo(item):
        engine.current = int(item["id"])
        W, H = engine.size
        return Image.new("RGB", (W, H)), np.zeros((H, W, 3), dtype=np.uint8)

    monkeypatch.setattr(face_upkeep, "_faces_ready", lambda: True)
    monkeypatch.setattr(animals_mod, "open_photo", open_photo)
    from app.ml import faces as ml_faces

    monkeypatch.setattr(ml_faces, "get_engine", lambda: engine)


BOX_A = (0.1, 0.1, 0.2, 0.25)
BOX_B = (0.6, 0.1, 0.15, 0.2)


def test_photo_faces_get_their_clarity_in_place(conn, monkeypatch):
    one, two = make_item(conn, width=800, height=600), make_item(conn, width=800, height=600)
    seen = make_face(conn, one["id"], None, box=BOX_A)
    unseen = make_face(conn, one["id"], None, box=BOX_B)   # the detector finds nothing there
    other = make_face(conn, two["id"], None, box=BOX_B)
    measured = make_face(conn, two["id"], None, box=BOX_A, clarity=19.0, sharpness=33.0)
    clip = _video(conn)
    video_face = make_face(conn, clip["id"], None, box=BOX_A)
    waiting = make_item(conn, width=800, height=600, analysis_status="pending")
    not_yet = make_face(conn, waiting["id"], None, box=BOX_A)
    before = {fid: bytes(conn.execute("SELECT embedding FROM faces WHERE id = ?", (fid,)).fetchone()[0])
              for fid in (seen, other)}
    engine = FakeEngine({one["id"]: [BOX_A], two["id"]: [BOX_B, BOX_A]})
    _fake_photos(monkeypatch, engine)
    v0 = faces_version(conn)

    out = face_upkeep.measure_photo_faces(conn)
    assert out == {"items": 2, "faces": 2, "complete": True}

    def measure(fid):
        return tuple(conn.execute("SELECT clarity, sharpness FROM faces WHERE id = ?", (fid,)).fetchone())

    assert measure(seen) == pytest.approx((21.5, 42.0)) and measure(other) == pytest.approx((21.5, 42.0))
    assert measure(unseen) == (None, None)
    assert measure(measured) == pytest.approx((19.0, 33.0))       # already measured: left as it is
    assert measure(video_face) == (None, None) and measure(not_yet) == (None, None)
    for fid, emb in before.items():
        assert bytes(conn.execute("SELECT embedding FROM faces WHERE id = ?", (fid,)).fetchone()[0]) == emb
    assert kv_get(conn, face_upkeep.CURSOR_KEY) == str(two["id"])
    assert faces_version(conn) != v0
    # the cursor moved past the photo with the unseen face: nothing is looked at again
    calls = engine.calls
    assert face_upkeep.measure_photo_faces(conn) == {"items": 0, "faces": 0, "complete": True}
    assert engine.calls == calls


def test_a_budget_stops_the_pass_early(conn, monkeypatch):
    one, two = make_item(conn, width=800, height=600), make_item(conn, width=800, height=600)
    first = make_face(conn, one["id"], None, box=BOX_A)
    second = make_face(conn, two["id"], None, box=BOX_A)
    _fake_photos(monkeypatch, FakeEngine({one["id"]: [BOX_A], two["id"]: [BOX_A]}))
    ticks = iter(range(100))
    monkeypatch.setattr(face_upkeep, "time", SimpleNamespace(monotonic=lambda: float(next(ticks))))
    # the deadline is read at 0 (+1.5), the first photo at 1, the second at 2: only the first is measured
    out = face_upkeep.measure_photo_faces(conn, budget_s=1.5)
    assert out == {"items": 1, "faces": 1, "complete": False}
    assert conn.execute("SELECT clarity FROM faces WHERE id = ?", (first,)).fetchone()[0] == pytest.approx(21.5)
    assert conn.execute("SELECT clarity FROM faces WHERE id = ?", (second,)).fetchone()[0] is None
    assert kv_get(conn, face_upkeep.CURSOR_KEY) == str(one["id"])


def test_nothing_is_measured_while_the_face_models_are_off(conn):
    item = make_item(conn)
    face = make_face(conn, item["id"], None, box=BOX_A)
    assert face_upkeep.measure_photo_faces(conn) == {"items": 0, "faces": 0, "complete": False}
    assert conn.execute("SELECT clarity FROM faces WHERE id = ?", (face,)).fetchone()[0] is None
    assert kv_get(conn, face_upkeep.CURSOR_KEY) is None


def test_an_unreadable_photo_is_passed_over(conn, monkeypatch):
    item = make_item(conn, width=800, height=600)
    face = make_face(conn, item["id"], None, box=BOX_A)
    _fake_photos(monkeypatch, FakeEngine({}))

    def broken(row):
        raise animals_mod.NoImage()

    monkeypatch.setattr(animals_mod, "open_photo", broken)
    assert face_upkeep.measure_photo_faces(conn) == {"items": 1, "faces": 0, "complete": True}
    assert conn.execute("SELECT clarity FROM faces WHERE id = ?", (face,)).fetchone()[0] is None


# ------------------------------------------------------------------ job priority
def test_a_job_can_be_queued_below_its_usual_priority(conn):
    low = jobs.enqueue(conn, "analyze", {"item_id": 1, "version": 1}, priority=face_upkeep.REANALYSE_PRIORITY)
    usual = jobs.enqueue(conn, "analyze", {"item_id": 2, "version": 1})
    prio = dict(conn.execute("SELECT id, priority FROM jobs").fetchall())
    assert prio == {low: 80, usual: 50}
    assert int(jobs.claim_next(conn, "main")["id"]) == usual
    assert int(jobs.claim_next(conn, "main")["id"]) == low
    added = jobs.enqueue_many(conn, "analyze", [{"item_id": 3, "version": 1}], priority=85)
    assert added == 1 and conn.execute("SELECT priority FROM jobs WHERE payload LIKE '%\"item_id\":3%'").fetchone()[0] == 85
