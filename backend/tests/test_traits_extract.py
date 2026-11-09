"""Body traits extraction, storage and runner (section 34, app.analysis.traits / trait_regions / trait_nails).

Fake engines stand in for the models (PCC_NO_MODELS=1): a pose that draws a standing person in every box, a
handpose that returns fixed landmarks, a DINOv2 that hashes the crop, and a SigLIP engine that checks it is given
PIL RGB images and answers with the probabilities a test sets. SIFT is real."""
from __future__ import annotations

import functools
import json
import threading

import cv2
import numpy as np
import pytest
from PIL import Image

from app import storage
from app.analysis import trait_nails, trait_regions, trait_sift, traits
from app.analysis.trait_regions import LIMB_GROUPS
from app.config import settings
from app.db import kv_get, kv_set, tx
from app.util import now_iso
from tests.helpers import (READY_REV, make_body, make_face, make_item, make_person, make_scan, make_trait,
                           set_traits_ready)

SK = (196, 150, 122)
INK = (38, 42, 52)
W, H = 1200, 1600
BOX = (0.25, 0.05, 0.5, 0.9)
TATTOO = {"tattoo": 0.95, "bare": 0.02, "cloth": 0.01, "henna": 0.01, "jewel": 0.01}
BARE = {"tattoo": 0.05, "bare": 0.9, "cloth": 0.02, "henna": 0.02, "jewel": 0.01}
# Normalised box coordinates of a standing person facing the camera (their left is the image's right).
SKELETON = {5: (0.65, 0.25), 6: (0.35, 0.25), 7: (0.75, 0.45), 8: (0.25, 0.45), 9: (0.78, 0.62), 10: (0.22, 0.62),
            13: (0.6, 0.7), 14: (0.4, 0.7), 15: (0.6, 0.95), 16: (0.4, 0.95)}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    trait_sift.reset()
    traits._set_error(None)
    with traits._state_lock:
        traits._state["calib_tried"] = None
    yield
    trait_sift.reset()


# ------------------------------------------------------------------ fakes
def skeleton(box_px, *, hands: float = 0.1, hand_span: float = 0.18, moves: dict | None = None) -> np.ndarray:
    x0, y0, x1, y1 = box_px
    kp = np.zeros((133, 3), np.float32)
    kp[:, 2] = 0.1
    for k, (x, y) in {**SKELETON, **(moves or {})}.items():
        kp[k] = (x0 + x * (x1 - x0), y0 + y * (y1 - y0), 0.9)
    for lo, (wx, wy) in ((91, SKELETON[9]), (112, SKELETON[10])):
        cx, cy = x0 + wx * (x1 - x0), y0 + (wy + 0.07) * (y1 - y0)
        span = hand_span * (x1 - x0)
        rng = np.random.default_rng(lo)
        pts = rng.uniform(-0.5, 0.5, (21, 2)) * span
        pts[0], pts[1] = (-0.5 * span, -0.5 * span), (0.5 * span, 0.5 * span)
        kp[lo:lo + 21, 0], kp[lo:lo + 21, 1], kp[lo:lo + 21, 2] = cx + pts[:, 0], cy + pts[:, 1], hands
        kp[lo + 9] = (cx, cy + 0.3 * span, hands)  # middle knuckle below the wrist point: the hand hangs down
        kp[lo] = (cx, cy - 0.4 * span, hands)
    return kp


class FakePose:
    def __init__(self, **kw):
        self.calls: list[int] = []
        self.kw = kw
        self.per_box = None  # callable(i, box_px) -> (133, 3), overrides

    def estimate(self, bgr, boxes):
        assert bgr.ndim == 3 and bgr.dtype == np.uint8
        self.calls.append(len(boxes))
        return [self.per_box(i, b) if self.per_box else skeleton(b, **self.kw) for i, b in enumerate(boxes)]


class FakeHands:
    def __init__(self):
        self.calls: list[int] = []

    def landmarks(self, crops):
        self.calls.append(len(crops))
        for c in crops:
            assert c.shape == (224, 224, 3) and c.dtype == np.uint8
        return [(hand_landmarks(), 0.95, 0.9) for _ in crops]


class FakeDino:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[int] = []

    def embed(self, crops):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.calls.append(len(crops))
        out = []
        for c in crops:
            assert c.shape == (224, 224, 3) and c.dtype == np.uint8
            v = np.random.default_rng(int(c.astype(np.int64).sum()) % (2 ** 31)).normal(size=384).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class FakeSiglip:
    """embed_images takes PIL RGB images only; zero_shot answers with `limb` for LIMB_GROUPS (or a callable of the
    image), and with fixed answers for the nail groups."""

    def __init__(self, limb=None):
        self.limb = limb or TATTOO
        self.images: list[Image.Image] = []
        self.groups_seen: list[tuple[str, ...]] = []

    def embed_images(self, images):
        out = []
        for im in images:
            assert isinstance(im, Image.Image) and im.mode == "RGB", "SigLIP must get PIL RGB images"
            self.images.append(im)
            out.append([float(len(self.images) - 1)])
        return np.array(out, np.float32)

    def zero_shot(self, embs, groups):
        self.groups_seen.append(tuple(groups))
        rows = []
        for e in np.atleast_2d(embs):
            im = self.images[int(e[0])]
            if tuple(groups) == tuple(LIMB_GROUPS):
                p = self.limb(im) if callable(self.limb) else self.limb
            elif "polish" in groups:
                p = {"polish": 0.9, "natural": 0.1}
            else:
                p = {k: (0.9 if k == "plain" else 0.1 / (len(groups) - 1)) for k in groups}
            rows.append([p[k] for k in groups])
        return np.array(rows, np.float32)


def fake_engines(monkeypatch, *, pose=None, hands=None, dino=None, siglip=None, missing: tuple[str, ...] = ()):
    e = traits.Engines(pose or FakePose(), hands or FakeHands(), dino or FakeDino(), siglip or FakeSiglip())
    e = e._replace(**{k: None for k in missing})
    monkeypatch.setattr(traits, "engines", lambda: e)
    monkeypatch.setattr(traits, "missing",
                        lambda e2=None: [k for k in traits.Engines._fields if getattr(e2 or e, k) is None])
    return e


def hand_landmarks() -> np.ndarray:
    """An upright right hand seen from the back in a 224 px crop: wrist at the bottom, thumb on the left."""
    lm = np.zeros((21, 2), np.float32)
    lm[0] = (112, 200)
    lm[1:5] = [(80, 175), (60, 150), (45, 125), (35, 105)]
    for f, x in zip(range(5, 21, 4), (80, 105, 130, 152)):
        lm[f:f + 4] = [(x, 120), (x, 90), (x, 65), (x, 45)]
    return lm


# ------------------------------------------------------------------ pixels and rows
@functools.lru_cache(maxsize=8)
def _inked_skin(w: int, h: int, seed: int, ink: bool) -> np.ndarray:
    a = _draw_skin(w, h, seed, ink)
    a.flags.writeable = False
    return a


def inked_skin(w: int, h: int, seed: int = 0, ink: bool = True) -> np.ndarray:
    """Skin with faint texture and (ink=True) blotches of dark ink everywhere; cached, read-only."""
    return _inked_skin(w, h, seed, ink)


def _draw_skin(w: int, h: int, seed: int, ink: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.empty((h, w, 3), np.float32)
    b[:] = SK
    b += rng.normal(0, 7, (h, w, 1))
    if ink:
        f = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6)
        a = (f > f.std()).astype(np.float32)
        b = b * (1 - a[..., None]) + np.array(INK, np.float32) * a[..., None]
    return np.clip(b, 0, 255).astype(np.uint8)


def photo(conn, pixels: np.ndarray | None = None, *, original: np.ndarray | None = None, **kw):
    """An analysed photo whose preview is `pixels` (lossless) and whose original is `original` (or the same)."""
    pixels = inked_skin(W, H) if pixels is None else pixels
    h, w = (original if original is not None else pixels).shape[:2]
    item = make_item(conn, width=w, height=h, write_files=False, **kw)
    pv = storage.preview_path(item["uid"], 1)
    pv.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(pv, "WEBP", lossless=True, method=0)
    op = storage.original_path(item["uid"], 1, ".jpg")
    op.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(original if original is not None else pixels).save(op, "JPEG", quality=95)
    return item


def scans(conn, item_id) -> dict:
    return {int(r["body_id"]): r for r in conn.execute("SELECT * FROM trait_scans WHERE item_id = ?", (item_id,))}


def rows(conn, item_id=None):
    q = "SELECT * FROM body_traits" + (" WHERE item_id = ?" if item_id is not None else "") + " ORDER BY id"
    return conn.execute(q, () if item_id is None else (item_id,)).fetchall()


def pending(conn) -> list[int]:
    return traits.pending_items(conn, 100)


# ------------------------------------------------------------------ models missing or failing
def test_no_models_writes_nothing_and_progress_says_paused(conn):
    item = photo(conn)
    b = make_body(conn, item["id"], box=BOX)
    other = make_body(conn, item["id"], box=(0.6, 0.1, 0.3, 0.8))
    old = make_trait(conn, other, "tattoo", "forearm_l", vec=np.ones(384))
    with tx(conn):
        conn.execute("UPDATE trait_scans SET rev = 'old' WHERE body_id = ?", (other,))
    out = traits.run_pending(conn)
    assert out["items"] == 0 and not out["complete"] and set(out["paused"]) == {"pose", "hands", "dino", "siglip"}
    assert [r["id"] for r in rows(conn)] == [old]
    assert set(scans(conn, item["id"])) == {other}
    p = traits.progress(conn)
    assert set(p["paused"]) == {"pose", "hands", "dino", "siglip"} and p["eligible"] == 2 and p["scanned"] == 0
    assert p["ready"] is False and p["last_error"] is None
    assert b in [int(r[0]) for r in conn.execute("SELECT id FROM bodies")]


def test_a_model_missing_after_ready_keeps_the_rows_and_the_ready_rev(conn, monkeypatch):
    item = photo(conn)
    b = make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch)
    out = traits.run_pending(conn)
    assert out["complete"] and out["items"] == 1
    assert traits.ready_rev(conn) == READY_REV
    before = [r["id"] for r in rows(conn)]
    assert before
    fake_engines(monkeypatch, missing=("dino",))
    out = traits.run_pending(conn)
    assert out["paused"] == ["dino"] and out["items"] == 0
    assert [r["id"] for r in rows(conn)] == before and traits.ready_rev(conn) == READY_REV
    fake_engines(monkeypatch)  # back: nothing is measured again
    assert traits.run_pending(conn)["items"] == 0
    assert [r["id"] for r in rows(conn)] == before
    assert scans(conn, item["id"])[b]["status"] == "ok"


def test_an_engine_failure_aborts_without_error_rows(conn, monkeypatch):
    item = photo(conn)
    make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch, dino=FakeDino(fail=True))
    out = traits.run_pending(conn)
    assert out["aborted"] and out["items"] == 0
    assert not rows(conn) and not scans(conn, item["id"])
    assert pending(conn) == [item["id"]]
    assert "CUDA out of memory" in traits.progress(conn)["last_error"]
    fake_engines(monkeypatch)
    assert traits.run_pending(conn)["complete"]
    assert traits.progress(conn)["last_error"] is None


# ------------------------------------------------------------------ a measured photo
def test_a_tattooed_photo_gets_rows_with_native_detail(conn, monkeypatch):
    item = photo(conn)
    b = make_body(conn, item["id"], box=BOX)
    e = fake_engines(monkeypatch)
    out = traits.run_pending(conn)
    assert out == {"items": 1, "bodies": 1, "complete": True, "aborted": False}
    got = rows(conn, item["id"])
    parts = sorted(r["part"] for r in got)
    assert parts == ["calf_l", "calf_r", "forearm_l", "forearm_r", "upper_arm_l", "upper_arm_r"]
    for r in got:
        assert r["kind"] == "tattoo" and r["body_id"] == b
        a = json.loads(r["attrs"])
        assert a["src"] == "preview"  # the original is the preview's size: s < 1.2
        assert a["share"] == traits.share_for(r["part"]) and 224 <= a["out"] <= 512
        assert a["kp"] == len(trait_sift.describe(r["detail"], a["share"]).plain[0])
        assert a["small"] is (a["kp"] < 60) and set(a["p"]) == set(LIMB_GROUPS)
        assert traits.decode_vec(r["vec"]).shape == (768,)
        assert Image.open(__import__("io").BytesIO(r["crop"])).size[0] <= 192
        assert 0 <= r["x"] <= 1 and 0 < r["w"] <= 1 and 0 < r["quality"] <= 1
    s = scans(conn, item["id"])[b]
    assert s["status"] == "ok" and s["rev"] == READY_REV and s["box"] == traits.body_box(conn, b)
    assert s["item_boxes"] == traits.item_fingerprint(conn, item["id"])
    assert e.pose.calls == [1] and e.dino.calls == [12]  # one pose batch, crop + mirror per row
    assert traits.rows_for_bodies(conn, [b])[b]


def test_a_bare_arm_is_recorded_on_the_scan(conn, monkeypatch):
    item = photo(conn, inked_skin(W, H, ink=False))
    b = make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch, siglip=FakeSiglip(BARE))
    traits.run_pending(conn)
    assert not rows(conn)
    bare = json.loads(scans(conn, item["id"])[b]["bare"])
    assert set(bare) == {"forearm_l", "forearm_r", "upper_arm_l", "upper_arm_r", "calf_l", "calf_r"}
    assert bare["forearm_l"]["p"] == 0.9 and bare["forearm_l"]["len"] > 128


def test_the_original_is_used_for_the_detail_crop(conn, monkeypatch):
    big = inked_skin(2 * W, 2 * H, seed=3)
    small = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    item = photo(conn, small, original=big)
    make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch)
    traits.run_pending(conn)
    got = rows(conn, item["id"])
    assert got
    for r in got:
        a = json.loads(r["attrs"])
        assert a["src"] == "original"
        assert a["out"] > 400  # about 1.15 x 2 x the preview segment, up to 512
        assert max(trait_sift.decode_detail(r["detail"]).shape[:2]) == a["out"]


def test_siglip_gets_pil_rgb_and_red_stays_red(monkeypatch):
    seen = []

    class Sig(FakeSiglip):
        def embed_images(self, images):
            seen.extend(np.asarray(im).reshape(-1, 3).mean(0) for im in images)
            return super().embed_images(images)

    red = np.zeros((224, 224, 3), np.uint8)
    red[..., 0] = 220
    trait_regions.siglip_probs(Sig(), [red], LIMB_GROUPS)
    r, g, b = seen[0]
    assert r > 200 and g < 10 and b < 10


# ------------------------------------------------------------------ statuses
def test_statuses_without_a_decode(conn, monkeypatch):
    e = fake_engines(monkeypatch)
    video = make_item(conn, kind="video", write_files=False, width=W, height=H)
    vb = make_body(conn, video["id"], box=BOX)
    vault = photo(conn, nsfw=1)
    vab = make_body(conn, vault["id"], box=BOX)
    doc = photo(conn, category="documents")
    db_ = make_body(conn, doc["id"], box=BOX)
    doclike = photo(conn, doc_likeness=0.9)
    dlb = make_body(conn, doclike["id"], box=BOX)
    tiny = photo(conn)
    tb = make_body(conn, tiny["id"], box=(0.1, 0.1, 0.05, 0.1))  # 160 px long side at 2048
    reads = []
    monkeypatch.setattr(trait_regions, "load_preview", lambda item: reads.append(item["id"]))
    traits.run_pending(conn)
    assert reads == [] and e.pose.calls == []
    got = {b: s["status"] for item in (video, vault, doc, doclike, tiny) for b, s in scans(conn, item["id"]).items()}
    assert got == {vb: "video", vab: "vault", db_: "doc", dlb: "doc", tb: "small"}


def test_a_child_is_not_measured(conn, monkeypatch):
    from app.analysis import ages

    item = photo(conn)
    f = make_face(conn, item["id"], box=(0.3, 0.06, 0.1, 0.08), age_groups=b"\0" * 36)
    b = make_body(conn, item["id"], box=BOX, face_id=f)
    monkeypatch.setattr(ages, "ready", lambda conn: True)
    monkeypatch.setattr(ages, "in_child_band", lambda groups, weight: True)
    fake_engines(monkeypatch)
    traits.run_pending(conn)
    assert scans(conn, item["id"])[b]["status"] == "child" and not rows(conn)


def test_no_pixels_and_error(conn, monkeypatch):
    gone = photo(conn)
    gb = make_body(conn, gone["id"], box=BOX)
    storage.preview_path(gone["uid"], 1).unlink()
    storage.original_path(gone["uid"], 1, ".jpg").unlink()
    item = photo(conn)
    ok_b = make_body(conn, item["id"], box=BOX)
    bad_b = make_body(conn, item["id"], box=(0.7, 0.05, 0.25, 0.9))
    real = trait_regions.limb_regions

    def broken(body_id, kp, min_px):
        if body_id == bad_b:
            raise ZeroDivisionError("our own bug")
        return real(body_id, kp, min_px)

    monkeypatch.setattr(trait_regions, "limb_regions", broken)
    fake_engines(monkeypatch)
    out = traits.run_pending(conn)
    assert out["complete"] and not out["aborted"]
    assert scans(conn, gone["id"])[gb]["status"] == "no_pixels"
    s = scans(conn, item["id"])
    assert s[bad_b]["status"] == "error" and s[ok_b]["status"] == "ok"
    assert {r["body_id"] for r in rows(conn)} == {ok_b}


def test_gated_statuses_are_checked_again_after_a_day(conn):
    by_item = {}
    for status in ("doc", "child", "no_pixels", "error", "small", "ok", "video"):
        item_id = make_item(conn, write_files=False)["id"]
        make_scan(conn, make_body(conn, item_id, box=BOX), status=status, scanned_at=now_iso(-3600))
        by_item[item_id] = status
    assert pending(conn) == []
    with tx(conn):
        conn.execute("UPDATE trait_scans SET scanned_at = ?", (now_iso(-2 * 86400),))
    assert sorted(by_item[i] for i in pending(conn)) == ["child", "doc", "error", "no_pixels"]


def test_leaving_the_vault_measures_again(conn, monkeypatch):
    item = photo(conn, nsfw=1)
    b = make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch)
    traits.run_pending(conn)
    assert scans(conn, item["id"])[b]["status"] == "vault" and not pending(conn)
    with tx(conn):
        conn.execute("UPDATE items SET nsfw = 0 WHERE id = ?", (item["id"],))
    assert pending(conn) == [item["id"]]
    traits.run_pending(conn)
    assert scans(conn, item["id"])[b]["status"] == "ok" and rows(conn, item["id"])


# ------------------------------------------------------------------ the item fingerprint
def test_box_changes_anywhere_in_the_photo_measure_every_body_again(conn, monkeypatch):
    item = photo(conn)
    a = make_body(conn, item["id"], box=BOX)
    b = make_body(conn, item["id"], box=(0.7, 0.05, 0.25, 0.9))
    fake_engines(monkeypatch)
    traits.run_pending(conn)
    assert not pending(conn)
    first = {r["id"] for r in rows(conn)}
    with tx(conn):  # a moved neighbour
        conn.execute("UPDATE bodies SET x = 0.71 WHERE id = ?", (b,))
    assert pending(conn) == [item["id"]]
    traits.run_pending(conn)
    assert not pending(conn) and not first & {r["id"] for r in rows(conn)}  # every row of the item is new
    assert {s["item_boxes"] for s in scans(conn, item["id"]).values()} == {traits.item_fingerprint(conn, item["id"])}
    drawn = make_body(conn, item["id"], box=(0.0, 0.5, 0.1, 0.2), source="user")  # a drawn box
    assert pending(conn) == [item["id"]]
    traits.run_pending(conn)
    assert set(scans(conn, item["id"])) == {a, b, drawn}
    with tx(conn):
        conn.execute("DELETE FROM bodies WHERE id = ?", (drawn,))  # a deleted box
    assert pending(conn) == [item["id"]]


def test_a_box_change_during_the_measurement_is_skipped(conn, monkeypatch):
    item = photo(conn)
    b = make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch)
    real = trait_regions.measure_item

    def measure_then_move(c, item_id, e):
        m = real(c, item_id, e)
        with tx(c):
            c.execute("UPDATE bodies SET w = 0.3 WHERE id = ?", (b,))
        return m

    monkeypatch.setattr(trait_regions, "measure_item", measure_then_move)
    out = traits.run_pending(conn)
    assert out["items"] == 0 and not out["complete"]
    assert not rows(conn) and not scans(conn, item["id"]) and pending(conn) == [item["id"]]


def test_move_and_undo_of_the_move_make_the_item_pending(conn, monkeypatch):
    from app import undo
    from app.db import ActionContext, current_action

    item = photo(conn)
    b = make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch)
    traits.run_pending(conn)
    token = current_action.set(ActionContext("act-move-1", "PUT", f"/api/bodies/{b}/box"))
    try:
        with tx(conn):
            conn.execute("UPDATE bodies SET x = 0.3 WHERE id = ?", (b,))
    finally:
        current_action.reset(token)
    assert pending(conn) == [item["id"]]
    traits.run_pending(conn)
    assert not pending(conn)
    heard = []
    monkeypatch.setattr(traits, "kick", lambda conn=None: heard.append("kick"))
    undo.undo(conn, "act-move-1")
    assert heard == ["kick"]
    assert pending(conn) == [item["id"]]


def test_undo_of_a_box_delete_kicks_and_measures_it(conn, monkeypatch):
    from app import undo
    from app.db import ActionContext, current_action

    item = photo(conn)
    b = make_body(conn, item["id"], box=BOX)
    fake_engines(monkeypatch)
    traits.run_pending(conn)
    token = current_action.set(ActionContext("act-del-1", "DELETE", f"/api/bodies/{b}"))
    try:
        with tx(conn):
            conn.execute("DELETE FROM bodies WHERE id = ?", (b,))
    finally:
        current_action.reset(token)
    assert not scans(conn, item["id"]) and not rows(conn)
    seen = []
    monkeypatch.setattr(traits, "after_undo", lambda c, tables, items: seen.append((set(tables), set(items))))
    undo.undo(conn, "act-del-1")
    assert seen == [({"bodies"}, {item["id"]})]
    assert pending(conn) == [item["id"]]


def test_after_undo_routes_hides_and_boxes(conn, monkeypatch):
    """An undo of hides only queues their photos for the runner (look-alikes and the re-guess verify, never inside
    the undo request); boxes put back kick the runner; anything else does nothing."""
    from app.analysis import trait_match

    calls = []
    monkeypatch.setattr(trait_match, "after_hide_change", lambda c, ids: calls.append(("hides", list(ids))))
    monkeypatch.setattr(traits, "kick", lambda conn=None: calls.append(("kick",)))
    one, three = photo(conn), photo(conn)
    traits.after_undo(conn, {"trait_hides"}, {three["id"], one["id"]})
    assert calls == [("kick",)]
    assert sorted(r[0] for r in conn.execute("SELECT item_id FROM trait_requeue")) == sorted([one["id"], three["id"]])
    traits.after_undo(conn, {"bodies", "faces"}, {2})
    traits.after_undo(conn, {"persons"}, {2})
    assert calls == [("kick",), ("kick",)]
    # the runner's drain does the look-alikes for photos with hides, and only for those
    with tx(conn):
        conn.execute("INSERT INTO trait_hides(item_id, kind, verdict, x, y, w, h, created_at) "
                     "VALUES (?, 'tattoo', 'not_trait', 0.1, 0.1, 0.1, 0.1, ?)", (three["id"], now_iso()))
    assert traits.drain_requeue(conn) == 2
    assert ("hides", [three["id"]]) in calls and ("hides", [one["id"]]) not in calls


# ------------------------------------------------------------------ geometry, ownership, gate
def test_the_forearm_crop_ends_at_80_percent_of_the_way_to_the_wrist():
    kp = skeleton((0, 0, 600, 1440))
    cands, _ = trait_regions.limb_regions(1, kp, 64)
    fa = next(r for r in cands if r.part == "forearm_l")
    elbow, wrist = kp[7, :2], kp[9, :2]
    assert np.allclose(fa.a, elbow) and np.allclose(fa.b, elbow + 0.8 * (wrist - elbow))
    assert fa.length == pytest.approx(float(np.hypot(*(wrist - elbow))))
    assert fa.crop_len == pytest.approx(0.8 * fa.length)
    ua = next(r for r in cands if r.part == "upper_arm_l")
    assert np.allclose(ua.b, kp[7, :2])  # the full segment


def test_low_scores_and_short_limbs_give_no_region():
    kp = skeleton((0, 0, 600, 1440))
    kp[9, 2] = 0.45  # a left wrist below 0.5: no forearm, but still a claim
    cands, claims = trait_regions.limb_regions(1, kp, 64)
    assert "forearm_l" not in {r.part for r in cands} and "forearm_l" in {r.part for r in claims}
    cands, _ = trait_regions.limb_regions(1, skeleton((0, 0, 60, 144)), 64)
    assert not [r for r in cands if not r.is_hand]


def test_ownership_overlap_double_claim_and_plausibility():
    box = (0, 0, 600, 1440)
    cands, _ = trait_regions.limb_regions(1, skeleton(box), 64)
    fa = next(r for r in cands if r.part == "forearm_l")
    assert trait_regions.owned(fa, box, [], [])
    # another box over the forearm: more than half of it goes; half or less stays unless another pose claims it
    x0, y0, x1, y1 = fa.bound
    assert not trait_regions.owned(fa, box, [(x0, y0, x1, y0 + 0.75 * (y1 - y0))], [])
    assert trait_regions.owned(fa, box, [(x0, y0, x1, (y0 + y1) / 2)], [])
    # outside its own box
    assert not trait_regions.owned(fa, (0, 0, 300, 1440), [], [])
    # another body's forearm right there (Leela carrying Kabir's arm): dropped
    _, other = trait_regions.limb_regions(2, skeleton(box), 64)
    their = [r for r in other if r.part == "forearm_r"]
    their[0].a, their[0].b = fa.a + 5, fa.b + 5
    assert not trait_regions.owned(fa, box, [], their)
    assert trait_regions.owned(fa, box, [], [r for r in other if r.part == "calf_l"])
    # a forearm far longer than its upper arm: both go
    kp = skeleton(box, moves={9: (0.95, 0.99)})
    kept = {r.part for r in trait_regions.plausible(trait_regions.limb_regions(1, kp, 64)[0])}
    assert "forearm_l" not in kept and "upper_arm_l" not in kept and "forearm_r" in kept


def test_a_double_claim_in_a_photo_drops_the_arm_for_both(conn, monkeypatch):
    item = photo(conn)
    a = make_body(conn, item["id"], box=(0.05, 0.05, 0.45, 0.9))
    b = make_body(conn, item["id"], box=(0.5, 0.05, 0.45, 0.9))
    pose = FakePose()

    def per_box(i, box):
        kp = skeleton(box)
        if i == 1:  # the second body's right forearm lies on the first body's left forearm
            first = skeleton((0.05 * W, 0.05 * H, 0.5 * W, 0.95 * H))
            kp[8], kp[10] = first[7], first[9]
        return kp

    pose.per_box = per_box
    fake_engines(monkeypatch, pose=pose)
    traits.run_pending(conn)
    have = {(r["body_id"], r["part"]) for r in rows(conn, item["id"])}
    assert (a, "forearm_l") not in have and (b, "forearm_r") not in have
    assert (a, "calf_l") in have and (b, "calf_l") in have


@pytest.mark.parametrize("p, family, want, mehndi_on", [
    (TATTOO, "forearm", "tattoo", True),
    ({**TATTOO, "tattoo": 0.7}, "forearm", "unclear", True),
    ({**TATTOO, "tattoo": 0.5}, "forearm", None, True),
    ({**TATTOO, "cloth": 0.2}, "forearm", None, True),
    ({**TATTOO, "henna": 0.6}, "forearm", "mehndi", True),
    ({**TATTOO, "henna": 0.6}, "forearm", None, False),   # mehndi off (the default): henna is no evidence at all
    ({**TATTOO, "henna": 0.6}, "hand", None, False),
    ({**TATTOO, "henna": 0.6}, "calf", "tattoo", True),   # mehndi on forearms and hands only
    ({**TATTOO, "henna": 0.6}, "calf", "tattoo", False),
    ({**TATTOO, "jewel": 0.3}, "forearm", "unclear", True),  # a watch: unclear, not a tattoo and not bare
    (BARE, "forearm", "bare", True),
])
def test_the_gate_rules(p, family, want, mehndi_on, monkeypatch):
    monkeypatch.setattr(settings, "traits_mehndi", mehndi_on)
    got = trait_regions.decide(family, p, skin=0.8, ink=0.15 if want != "bare" else 0.0, sharp=500,
                               henna_hue=False, score=0.9, length=300)
    assert got == want


def test_the_tattoo_gate_needs_sharpness_not_skin_or_ink(monkeypatch):
    """A tattoo is SigLIP's verdict on a sharp enough crop: dense sleeves leave little skin, so the skin and ink
    fractions only decide bare limbs (they dropped most real sleeves on the owner's library)."""
    monkeypatch.setattr(settings, "traits_mehndi", True)
    d = trait_regions.decide
    sharp = trait_regions.TATTOO_SHARP_MIN
    assert d("forearm", TATTOO, 0.2, 0.15, 500, False, 0.9, 300) == "tattoo"   # a dense sleeve: little skin
    assert d("forearm", TATTOO, 0.8, 0.01, 500, False, 0.9, 300) == "tattoo"   # faint ink
    assert d("forearm", TATTOO, 0.8, 0.15, sharp, False, 0.9, 300) == "tattoo"
    assert d("forearm", TATTOO, 0.8, 0.15, sharp - 1, False, 0.9, 300) is None  # blurred
    assert d("forearm", {**TATTOO, "jewel": 0.3}, 0.8, 0.15, sharp - 1, False, 0.9, 300) is None  # blurred watch
    assert d("forearm", TATTOO, 0.8, 0.15, 500, True, 0.9, 300) == "mehndi"  # brown ink
    assert d("upper_arm", TATTOO, 0.8, 0.15, 500, True, 0.9, 300) == "tattoo"
    monkeypatch.setattr(settings, "traits_mehndi", False)
    assert d("forearm", TATTOO, 0.8, 0.15, 500, True, 0.9, 300) is None  # brown ink with mehndi off: never a tattoo
    assert d("upper_arm", TATTOO, 0.8, 0.15, 500, True, 0.9, 300) == "tattoo"
    # bare limbs still need skin, no ink, sharpness, keypoints at 0.6 and a long enough segment
    assert d("forearm", BARE, 0.8, 0.0, 500, False, 0.9, 300) == "bare"
    assert d("forearm", BARE, 0.5, 0.0, 500, False, 0.9, 300) is None
    assert d("forearm", BARE, 0.8, 0.05, 500, False, 0.9, 300) is None
    assert d("forearm", BARE, 0.8, 0.0, trait_regions.SHARP_MIN - 1, False, 0.9, 300) is None
    assert d("forearm", BARE, 0.8, 0.0, 500, False, 0.55, 300) is None
    assert d("forearm", BARE, 0.8, 0.0, 500, False, 0.9, 100) is None


def test_skin_and_ink_on_a_strip():
    skin, ink, sharp, henna = trait_regions.skin_ink(inked_skin(130, 224), None)
    assert skin > 0.6 and 0.05 < ink < 0.4 and sharp > 60 and not henna
    skin, ink, _s, _h = trait_regions.skin_ink(inked_skin(130, 224, ink=False), None)
    assert skin > 0.95 and ink < 0.01
    brown = np.array(inked_skin(130, 224))
    brown[np.all(brown < 80, axis=2)] = (110, 45, 15)  # henna-coloured ink
    assert trait_regions.skin_ink(brown, None)[3]


# ------------------------------------------------------------------ hands and nails
def test_an_inked_hand_or_a_phone_gives_no_nails(conn, monkeypatch):
    item = photo(conn)
    make_body(conn, item["id"], box=BOX)
    hands = FakeHands()
    fake_engines(monkeypatch, pose=FakePose(hands=0.9), hands=hands)  # every crop reads as a tattoo
    traits.run_pending(conn)
    assert hands.calls == []
    assert {r["part"] for r in rows(conn, item["id"]) if r["kind"] == "tattoo"} >= {"hand_l", "hand_r"}
    item2 = photo(conn)
    make_body(conn, item2["id"], box=BOX)
    phone = {"tattoo": 0.1, "bare": 0.2, "cloth": 0.05, "henna": 0.05, "jewel": 0.6}
    fake_engines(monkeypatch, pose=FakePose(hands=0.9), hands=hands, siglip=FakeSiglip(phone))
    traits.run_pending(conn)
    assert hands.calls == [] and not rows(conn, item2["id"])


def test_a_polished_hand_gives_a_nails_row(conn, monkeypatch):
    item = photo(conn, inked_skin(W, H, ink=False))
    b = make_body(conn, item["id"], box=BOX)
    seen = []

    def nails(crop, lm, conf, side, probs, *, src):
        seen.append((crop.shape, side, src))
        return {"lab": [40.0, 60.0, 40.0], "color": "red", "hex": "#c81e28", "finish": "plain", "finish_p": 0.9,
                "length": "medium", "r": 0.2, "fingers": 4, "p_tip": 0.9, "back": True, "sharp": 300.0, "src": src}

    monkeypatch.setattr(trait_nails, "describe_hand", nails)
    hands = FakeHands()
    fake_engines(monkeypatch, pose=FakePose(hands=0.9), hands=hands, siglip=FakeSiglip(BARE))
    traits.run_pending(conn)
    assert hands.calls == [2] and sorted(s[1] for s in seen) == ["l", "r"]
    got = [r for r in rows(conn, item["id"]) if r["kind"] == "nails"]
    assert sorted(r["part"] for r in got) == ["hand_l", "hand_r"]
    for r in got:
        assert r["vec"] is None and r["detail"] is None and r["body_id"] == b
        assert json.loads(r["attrs"])["color"] == "red"


def draw_hand(colors, *, run: float = 0.2, held: bool = False) -> np.ndarray:
    """A 224 px hand crop for hand_landmarks(): skin fingers on a dark backdrop, nails painted in `colors`
    (thumb first) from 0.4 of the way from DIP to TIP up to `run` past the tip; held: the colour runs on."""
    rng = np.random.default_rng(7)
    img = np.empty((224, 224, 3), np.float32)
    img[:] = (60, 60, 64)
    lm = hand_landmarks()
    for (t, d, p), color in zip(trait_nails.FINGERS, colors):
        for a, b in ((0, 1 if t == 4 else t - 3), (t - 3, p), (p, d), (d, t)):
            cv2.line(img, tuple(int(v) for v in lm[a]), tuple(int(v) for v in lm[b]), SK, 14)
        if color is not None:  # a flat-ended strip, so the run past the tip is what was asked for
            u = lm[t] - lm[d]
            n = np.array([-u[1], u[0]]) / np.linalg.norm(u) * 6
            a, b = lm[d] + 0.4 * u, lm[t] + (1.5 if held else run) * u
            cv2.fillConvexPoly(img, np.rint([a - n, a + n, b + n, b - n]).astype(np.int32), color)
    img += rng.normal(0, 7, (224, 224, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def polish_probs(crops, groups):
    if "polish" in groups:
        return np.array([[0.9, 0.1]] * len(crops), np.float32)
    return np.array([[0.9, 0.04, 0.03, 0.03]] * len(crops), np.float32)


RED = (200, 30, 40)


def test_nails_polished_red_medium():
    a = trait_nails.describe_hand(draw_hand([RED] * 5), hand_landmarks(), 0.95, "r", polish_probs, src="preview")
    assert a is not None
    assert a["color"] in ("red", "maroon") and a["finish"] == "plain" and a["fingers"] >= 4
    assert a["length"] == "medium" and 0.15 <= a["r"] < 0.35 and a["back"] is True and a["src"] == "preview"


def test_nails_palm_low_confidence_unpolished_and_held_objects():
    lm = hand_landmarks()
    d = trait_nails.describe_hand
    assert d(draw_hand([RED] * 5), lm, 0.95, "l", polish_probs, src="p") is None  # the palm of a left hand
    assert d(draw_hand([RED] * 5), lm, 0.5, "r", polish_probs, src="p") is None
    assert d(draw_hand([None] * 5), lm, 0.95, "r", polish_probs, src="p") is None
    assert d(draw_hand([RED] * 5, held=True), lm, 0.95, "r", polish_probs, src="p") is None
    natural = lambda crops, groups: np.array([[0.2, 0.8]] * len(crops)) if "polish" in groups else None  # noqa: E731
    assert d(draw_hand([RED] * 5), lm, 0.95, "r", natural, src="p") is None


def test_nails_spread_versus_multi():
    lm = hand_landmarks()
    blue = (30, 60, 200)
    multi = trait_nails.describe_hand(draw_hand([RED, RED, RED, blue, blue]), lm, 0.95, "r", polish_probs, src="p")
    assert multi is not None and multi["finish"] == "multi" and multi["color"] in ("red", "maroon", "blue")
    mixed = [RED, blue, (40, 160, 60), (240, 210, 60), (240, 130, 170)]
    assert trait_nails.describe_hand(draw_hand(mixed), lm, 0.95, "r", polish_probs, src="p") is None


def test_nail_length_and_white_balance():
    lm = hand_landmarks()
    short = trait_nails.describe_hand(draw_hand([RED] * 5, run=0.0), lm, 0.95, "r", polish_probs, src="p")
    long_ = trait_nails.describe_hand(draw_hand([RED] * 5, run=0.5), lm, 0.95, "r", polish_probs, src="p")
    assert short["length"] == "short" and long_["length"] == "long"
    out = trait_nails.white_balance(np.array([150.0, 120.0, 140.0]), np.array([150.0, 120.0, 140.0]))
    assert np.allclose(out, trait_nails.SKIN_TARGET, atol=0.5)  # the skin itself maps onto the target
    warm = trait_nails.white_balance(np.array([200.0, 200.0, 200.0]), np.array([230.0, 150.0, 90.0]))
    assert warm[2] > warm[0]  # under warm light a grey nail is bluer than it looks


# ------------------------------------------------------------------ after the write
def test_new_rows_are_checked_against_hides_and_named_rows_queue_look_alikes(conn, monkeypatch):
    from app.analysis import guess, trait_match

    person = make_person(conn, "Kabir")
    item = photo(conn)
    b = make_body(conn, item["id"], person, box=BOX, assigned_by="user")
    other = photo(conn)
    calls = {"hits": [], "targets": [], "guess": []}
    monkeypatch.setattr(trait_match, "hide_hits_for_rows", lambda c, ids: calls["hits"].append(list(ids)) or 0)
    monkeypatch.setattr(trait_match, "targets_for_rows",
                        lambda c, ids, limit: calls["targets"].append((list(ids), limit)) or [other["id"]])
    monkeypatch.setattr(guess, "guess_item", lambda c, item_id, cache=None: calls["guess"].append(item_id) or 0)
    fake_engines(monkeypatch)
    set_traits_ready(conn)
    traits.run_pending(conn)
    ids = [r["id"] for r in rows(conn, item["id"])]
    assert calls["hits"] == [ids]
    assert calls["guess"] == [item["id"]]
    assert calls["targets"] == [(ids, trait_match.REGUESS_LIMIT)]
    assert [int(r[0]) for r in conn.execute("SELECT item_id FROM trait_requeue")] == [other["id"]]
    assert b


def test_no_re_guess_before_ready_or_with_matching_off(conn, monkeypatch):
    from app.analysis import guess
    from app.config import settings

    item = photo(conn)
    make_body(conn, item["id"], box=BOX)
    seen = []
    monkeypatch.setattr(guess, "guess_item", lambda c, item_id, cache=None: seen.append(item_id) or 0)
    fake_engines(monkeypatch)
    traits.run_pending(conn)  # marks ready at the end, after the item was written
    assert seen == [] and traits.ready(conn)
    monkeypatch.setattr(settings, "trait_matching", False)
    with tx(conn):
        conn.execute("UPDATE bodies SET x = 0.26")
    traits.run_pending(conn)
    assert seen == []


def test_the_requeue_drains_without_models(conn, monkeypatch):
    from app.analysis import guess

    a, b, c = (make_item(conn, write_files=False)["id"] for _ in range(3))
    seen = []
    monkeypatch.setattr(guess, "guess_item", lambda conn, item_id, cache=None: seen.append((item_id, id(cache))) or 0)
    assert traits.requeue(conn, [a, b, c, 999999]) == 3
    assert traits.requeue(conn, [a]) == 0
    assert traits.drain_requeue(conn) == 3
    assert [i for i, _ in seen] == [c, b, a] and len({k for _, k in seen}) == 1  # newest first, one cache
    assert conn.execute("SELECT COUNT(*) FROM trait_requeue").fetchone()[0] == 0


# ------------------------------------------------------------------ the runner loop
def test_kick_does_nothing_with_workers_off(conn, monkeypatch):
    from app import worker

    started = []
    monkeypatch.setattr(worker, "run_in_background", lambda *a: started.append(a))
    photo_id = photo(conn)["id"]
    traits.requeue(conn, [photo_id])
    traits.kick(conn)
    assert started == []


def test_kick_starts_the_runner_only_with_work(conn, monkeypatch):
    from app import worker
    from app.config import settings

    monkeypatch.setattr(settings, "workers_enabled", True)
    started = []
    monkeypatch.setattr(worker, "run_in_background", lambda key, fn, *a: started.append(key))
    set_traits_ready(conn)
    with tx(conn):
        kv_set(conn, traits.SWEPT_KEY, READY_REV)
    traits.kick(conn)
    assert started == []  # nothing queued, swept, no models: no thread
    traits.requeue(conn, [photo(conn)["id"]])
    traits.kick(conn)
    assert started == ["traits"]
    monkeypatch.setattr(worker, "background_running", lambda key: True)
    traits.kick(conn)
    assert started == ["traits"]  # already running: the loop asks again before it exits


def test_the_loop_asks_again_before_it_exits(data_dir, conn, monkeypatch):
    from app.analysis import guess

    monkeypatch.setattr(traits, "LOOP_SLEEP_S", 0.0)
    set_traits_ready(conn)
    with tx(conn):
        kv_set(conn, traits.SWEPT_KEY, READY_REV)
    late = photo(conn)["id"]
    first = photo(conn)["id"]
    seen = []

    def guess_item(c, item_id, cache=None):
        seen.append(item_id)
        if item_id == first:  # a requeue that arrives while the runner works (its kick is dropped)
            traits.requeue(c, [late])
        return 0

    monkeypatch.setattr(guess, "guess_item", guess_item)
    traits.requeue(conn, [first])
    traits._loop()
    assert seen == [first, late]


def test_ready_then_sweep_on_the_runner_then_a_new_rev(conn, monkeypatch):
    from app.analysis import trait_match

    monkeypatch.setattr(traits, "LOOP_SLEEP_S", 0.0)
    item = photo(conn)
    b = make_body(conn, item["id"], box=BOX)
    swept = []
    monkeypatch.setattr(trait_match, "sweep_open", lambda c, budget_s=None: swept.append(traits.ready_rev(c)) or 0)
    fake_engines(monkeypatch)
    traits.start(conn)  # decides only: not ready (a body is pending), no sweep here
    assert not traits.ready(conn) and swept == []
    traits._loop()
    assert traits.ready_rev(conn) == READY_REV and swept == [READY_REV]
    assert kv_get(conn, traits.SWEPT_KEY) == READY_REV
    old = {r["id"] for r in rows(conn)}
    monkeypatch.setattr(traits, "TRAITS_REV", "3")
    assert traits.wanted_rev() == "3:mehndi,nails,tattoo" and pending(conn) == [item["id"]]
    assert traits.ready_rev(conn) == READY_REV and {r["id"] for r in traits.rows_for_bodies(conn, [b])[b]} == old
    fake_engines(monkeypatch, missing=("pose",))
    traits._loop()  # paused: the old rows and the old ready rev stay
    assert traits.ready_rev(conn) == READY_REV and {r["id"] for r in rows(conn)} == old
    fake_engines(monkeypatch)
    traits._loop()
    assert traits.ready_rev(conn) == "3:mehndi,nails,tattoo" and swept[-1] == "3:mehndi,nails,tattoo"
    # switching mehndi off is a new rev too: the library is read again without it
    monkeypatch.setattr(settings, "traits_mehndi", False)
    assert traits.wanted_rev() == "3:nails,tattoo" and pending(conn) == [item["id"]]
    traits._loop()
    assert traits.ready_rev(conn) == "3:nails,tattoo"


def test_an_empty_library_is_ready_at_start(conn):
    traits.start(conn)
    assert traits.ready_rev(conn) == READY_REV


def test_the_loop_stops_when_memory_grows(conn, monkeypatch):
    monkeypatch.setattr(traits, "LOOP_SLEEP_S", 0.0)
    for _ in range(3):
        make_body(conn, photo(conn)["id"], box=BOX)
    fake_engines(monkeypatch)
    monkeypatch.setattr(traits, "BATCH_ITEMS", 1)
    sizes = iter([1_000, 1_000 + traits.RSS_GROWTH + 1] + [10 ** 12] * 50)
    monkeypatch.setattr(traits, "_rss", lambda: next(sizes))
    real = traits.run_pending
    monkeypatch.setattr(traits, "run_pending", lambda c, budget_s=None, limit=None: real(c, limit=1))
    traits._loop()
    assert conn.execute("SELECT COUNT(DISTINCT item_id) FROM trait_scans").fetchone()[0] == 1
    assert not traits.ready(conn)


def test_calibration_runs_on_the_runner_when_due(conn, monkeypatch):
    from app.analysis import trait_match

    monkeypatch.setattr(traits, "LOOP_SLEEP_S", 0.0)
    taken = []
    monkeypatch.setattr(trait_match, "calibration_sample", lambda c: taken.append(threading.current_thread().name) or {})
    item = photo(conn)
    make_trait(conn, make_body(conn, item["id"], box=BOX), "tattoo", "forearm_l")
    assert not traits._calib_due(conn)  # not ready
    set_traits_ready(conn)
    with tx(conn):
        kv_set(conn, traits.SWEPT_KEY, READY_REV)
    assert traits._calib_due(conn)
    traits._loop()
    assert len(taken) == 1
    assert not traits._calib_due(conn)  # tried: waits CALIB_MIN_S even though the stub wrote nothing
    with traits._state_lock:
        traits._state["calib_tried"] = None
    with tx(conn):
        kv_set(conn, traits.CALIB_AT_KEY, now_iso(-7 * 3600))
        kv_set(conn, traits.CALIB_KEY, json.dumps({"max_id": rows(conn)[-1]["id"]}))
    assert not traits._calib_due(conn)  # fewer than 200 new rows since


# ------------------------------------------------------------------ prune and progress
def test_prune_keeps_descriptors_inside_the_undo_window(conn):
    item = photo(conn)
    old = now_iso(-40 * 86400)
    with tx(conn):
        for hide_id, alive, made in ((1, True, old), (2, False, now_iso()), (3, False, old), (4, False, old)):
            if alive:
                conn.execute("INSERT INTO trait_hides(id, item_id, kind, verdict, x, y, w, h, created_at) "
                             "VALUES (?, ?, 'tattoo', 'not_trait', 0, 0, 1, 1, ?)", (hide_id, item["id"], made))
            conn.execute("INSERT INTO trait_hide_desc(hide_id, vec, detail, created_at) VALUES (?, x'00', x'00', ?)",
                         (hide_id, made))
        conn.execute("INSERT INTO undo_actions(id, method, path, created_at) VALUES ('a1', 'DELETE', '/x', ?)",
                     (now_iso(),))
        conn.execute("INSERT INTO undo_log(action_id, tbl, op, old, new) VALUES ('a1', 'trait_hides', 'delete', ?, NULL)",
                     (json.dumps({"id": 4, "item_id": item["id"]}),))
    assert traits.prune(conn) == 1
    assert sorted(int(r[0]) for r in conn.execute("SELECT hide_id FROM trait_hide_desc")) == [1, 2, 4]


def test_progress_counts_scanned_bodies_of_the_wanted_rev(conn, monkeypatch):
    item = photo(conn)
    a = make_body(conn, item["id"], box=BOX)
    make_body(conn, item["id"], box=(0.6, 0.1, 0.3, 0.8))
    make_scan(conn, a)
    gone = photo(conn, deleted_at=now_iso())
    make_scan(conn, make_body(conn, gone["id"], box=BOX))
    p = traits.progress(conn)
    assert (p["scanned"], p["eligible"], p["ready"], p["rev"]) == (1, 2, False, None)
    from app.config import settings

    monkeypatch.setattr(settings, "traits_enabled", False)
    assert traits.progress(conn)["paused"] is None
