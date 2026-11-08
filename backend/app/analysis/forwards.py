"""Forwards: media received through messaging apps (docs/ARCHITECTURE.md section 29).

CONTRACT.
- name_app(filename) -> str | None: the app a file name names (whatsapp, telegram, facebook, snapchat,
  messenger), else None.
- jpeg_quality(path) -> int | None: the IJG quality a JPEG was written at, estimated from its luma
  quantisation table (the header only is read); None for anything else or an unreadable file.
- file_signals(kind=, ext=, width=, height=, camera=, exif_datetime=, gps_lat=, bitrate=, filename=, path=)
  -> dict: what the file alone says: name_app, stripped, small, side, wa_size, jpeg_quality, video_small.
  Computed at ingest and by the backfill.
- model_signals(meme_p=, doc_likeness=, screenshot_p=, people=) -> dict: meme_p, text, people (the analyze
  stage; meme_p is None without SigLIP).
- decide(signals, *, learned=None, vote=None) -> Decision(received_via, forward, reasons). Step 1
  (received_via) is a rule. Step 2 (forward) is a labelled look-alike's answer when one is close
  (`vote`), else the learner's probability when it has learned (`learned`), else the rule of section 29.
- apply(conn, item_id, signals, *, emb=None) -> Decision: decide for one item (asking the learners) and
  write received_via, forward, forward_reasons, forward_signals, forward_checked. A row whose
  forward_source is 'user' keeps its forward and reasons; only its signals and received_via are written.
- housekeeping(conn, budget_s) -> int: the backfill for items analysed before this section
  (forward_checked = 0), BACKFILL_BATCH at a time within the budget; the worker calls it.
- status(conn) -> {unchecked, running, forwards, received} for GET /api/system.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.db import tx
from app.util import json_dumps, json_loads, now_iso

log = logging.getLogger(__name__)

APPS = ("whatsapp", "telegram", "facebook", "snapchat", "messenger")
APP_UNKNOWN = "app"  # the file carries a messaging app's marks, the app is not known
NAME_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(IMG|VID|AUD|PTT|DOC)-\d{8}-WA\d{4}", re.IGNORECASE), "whatsapp"),
    (re.compile(r"^WhatsApp (Image|Video|Audio|Unknown)\b", re.IGNORECASE), "whatsapp"),
    (re.compile(r"^(photo|video)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", re.IGNORECASE), "telegram"),
    (re.compile(r"^FB_IMG_\d+", re.IGNORECASE), "facebook"),
    (re.compile(r"^Snapchat-\d+", re.IGNORECASE), "snapchat"),
    (re.compile(r"^received_\d+", re.IGNORECASE), "messenger"),
)
STRIPPED_EXTS = (".jpg", ".jpeg", ".webp")  # PNG is screenshots, HEIC from a phone always carries EXIF
WA_SIDES = (1600, 1280, 1024, 960, 800)  # the longest sides WhatsApp writes
VIDEO_MAX_SIDE = 1280
VIDEO_MAX_BPS = 2_500_000
MEME_STRONG = 0.7  # a forward whatever is in the picture (posters with faces)
SCREENSHOT_MIN = 0.6  # categories.SCREENSHOT_P_MIN
BACKFILL_BATCH = 200
# The JPEG standard's luminance quantisation table (Annex K.1). A file written at IJG quality q holds
# it scaled by 5000 / q percent (q < 50) or 200 - 2q percent, so the table's mean gives q back.
STD_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
STD_LUMA_MEAN = sum(STD_LUMA) / len(STD_LUMA)

_lock = threading.Lock()
_running = False


@dataclass
class Decision:
    received_via: str | None
    forward: bool
    reasons: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ signals
def name_app(filename: str | None) -> str | None:
    name = (filename or "").strip()
    for pattern, app in NAME_PATTERNS:
        if pattern.search(name):
            return app
    return None


def jpeg_quality(path: Path | str | None) -> int | None:
    if path is None:
        return None
    try:
        with Image.open(path) as im:
            if (im.format or "").upper() != "JPEG":
                return None
            tables = getattr(im, "quantization", None) or {}
    except Exception:  # noqa: BLE001 - an unreadable file is simply no signal
        return None
    table = tables.get(0) or (next(iter(tables.values())) if tables else None)
    if not table:
        return None
    scale = (sum(table) / len(table)) / STD_LUMA_MEAN
    q = 50.0 / scale if scale > 1.0 else 100.0 - 50.0 * scale
    return int(round(max(1.0, min(100.0, q))))


def file_signals(*, kind: str, ext: str | None, width: int | None, height: int | None, camera: str | None,
                 exif_datetime: str | None, gps_lat: float | None, bitrate: int | None, filename: str | None,
                 path: Path | str | None = None) -> dict[str, Any]:
    ext = (ext or "").lower()
    side = max(int(width or 0), int(height or 0))
    image = kind == "image"
    return {
        "name_app": name_app(filename),
        "stripped": bool(image and ext in STRIPPED_EXTS and not camera and not exif_datetime),
        "small": bool(image and 0 < side <= int(settings.forward_max_side)),
        "side": side or None,
        "wa_size": bool(image and side in WA_SIDES),
        "jpeg_quality": jpeg_quality(path) if image and ext in (".jpg", ".jpeg") else None,
        "video_small": bool(kind == "video" and 0 < side <= VIDEO_MAX_SIDE and gps_lat is None
                            and (bitrate is None or int(bitrate) <= VIDEO_MAX_BPS)),
    }


def model_signals(*, meme_p: float | None, doc_likeness: float | None, screenshot_p: float | None,
                  people: bool) -> dict[str, Any]:
    text = (doc_likeness is not None and float(doc_likeness) >= float(settings.doc_likeness_min)) or (
        screenshot_p is not None and float(screenshot_p) >= SCREENSHOT_MIN)
    return {"meme_p": None if meme_p is None else round(float(meme_p), 4), "text": bool(text), "people": bool(people)}


# ------------------------------------------------------------------ decision
def decide(sig: dict[str, Any], *, learned: float | None = None, vote: bool | None = None) -> Decision:
    """Section 29: received_via by rule, then forward by a look-alike's answer, the learner, or the rule."""
    reasons: dict[str, Any] = {}
    app = sig.get("name_app")
    received: str | None = None
    if app:
        received = str(app)
        reasons["name"] = received
    else:
        q = sig.get("jpeg_quality")
        quality_ok = q is not None and float(q) <= float(settings.forward_jpeg_quality_max)
        if sig.get("stripped") and sig.get("small") and (quality_ok or sig.get("wa_size")):
            received = APP_UNKNOWN
            reasons["stripped"] = True
            reasons["small"] = sig.get("side")
            if quality_ok:
                reasons["jpeg_quality"] = int(q)
            elif sig.get("wa_size"):
                reasons["wa_size"] = True
        elif sig.get("video_small"):
            received = APP_UNKNOWN
            reasons["video_small"] = True
    if received is None:
        return Decision(None, False, {})
    meme = sig.get("meme_p")
    people = bool(sig.get("people"))
    text = bool(sig.get("text"))
    if vote is not None:
        forward = bool(vote)
        reasons["lookalike"] = True
    elif learned is not None:
        forward = float(learned) >= 0.5
        reasons["learned"] = round(float(learned), 3)
    else:
        forward = (text and not people) or (
            meme is not None and ((float(meme) >= float(settings.forward_meme_min) and not people)
                                  or float(meme) >= MEME_STRONG))
    if forward:
        if text:
            reasons["text"] = True
        if meme is not None:
            reasons["meme_p"] = round(float(meme), 3)
        return Decision(received, True, reasons)
    return Decision(received, False, {})


def _learned(sig: dict[str, Any]) -> float | None:
    try:
        from app.learning import forwards as learned_forwards

        return learned_forwards.probability(sig)
    except Exception:  # noqa: BLE001 - learning never blocks a decision
        log.exception("learning: forwards learner failed")
        return None


def _vote(emb: np.ndarray | None) -> bool | None:
    if emb is None:
        return None
    try:
        from app.learning import images as learned_images

        return learned_images.forward_vote(emb)
    except Exception:  # noqa: BLE001
        log.exception("learning: forward look-alike vote failed")
        return None


def apply(conn: sqlite3.Connection, item_id: int, sig: dict[str, Any], *, emb: np.ndarray | None = None) -> Decision:
    """Decide for one item and write the columns (inside the caller's transaction, or one of its own).
    The owner's decision (forward_source 'user') stays; everything else follows the decision."""
    row = conn.execute("SELECT forward, forward_reasons, forward_source FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return Decision(None, False, {})
    if not settings.forwards_enabled:
        dec = Decision(None, False, {})
    else:
        dec = decide(sig, learned=_learned(sig), vote=_vote(emb))
    now = now_iso()
    with tx(conn):
        if row["forward_source"] == "user" or not settings.forwards_enabled:
            kept = Decision(dec.received_via, bool(row["forward"]), json_loads(row["forward_reasons"], {}) or {})
            conn.execute(
                "UPDATE items SET received_via = ?, forward_signals = ?, forward_checked = 1, updated_at = ? WHERE id = ?",
                (dec.received_via, json_dumps(sig), now, item_id),
            )
            return kept if row["forward_source"] == "user" else dec
        conn.execute(
            "UPDATE items SET received_via = ?, forward = ?, forward_reasons = ?, forward_signals = ?, "
            "forward_checked = 1, updated_at = ? WHERE id = ?",
            (dec.received_via, int(dec.forward), json_dumps(dec.reasons) if dec.reasons else None,
             json_dumps(sig), now, item_id),
        )
    return dec


# ------------------------------------------------------------------ backfill
def _engine() -> Any | None:
    """The SigLIP engine when the worker may load it (None with the models off)."""
    try:
        from app.analysis.pipeline import _model

        siglip = _model("siglip")
        return siglip.get_engine() if siglip is not None else None
    except Exception:  # noqa: BLE001 - no engine, no meme signal
        return None


def _embedding(conn: sqlite3.Connection, item_id: int) -> np.ndarray | None:
    r = conn.execute(
        "SELECT embedding FROM item_embeddings WHERE item_id = ? AND frame_index = 0 ORDER BY version DESC LIMIT 1",
        (item_id,),
    ).fetchone()
    if r is None or not r["embedding"]:
        return None
    v = np.frombuffer(r["embedding"], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def meme_probability(engine: Any, emb: np.ndarray) -> float | None:
    from app.analysis import categories

    try:
        return round(float(categories.binary_p(engine, emb.reshape(1, -1), categories.MEME_PROMPTS,
                                               categories.NOT_MEME_PROMPTS)), 4)
    except Exception:  # noqa: BLE001
        log.exception("forwards: meme zero-shot failed")
        return None


def _backfill_row(conn: sqlite3.Connection, r: sqlite3.Row, engine: Any | None) -> None:
    from app import storage

    path = None
    if r["kind"] == "image" and (r["ext"] or "").lower() in (".jpg", ".jpeg"):
        p = storage.original_path(r["uid"], int(r["version"]), r["ext"])
        path = p if p.exists() else None
    sig = file_signals(kind=r["kind"], ext=r["ext"], width=r["width"], height=r["height"], camera=r["camera"],
                       exif_datetime=r["exif_datetime"], gps_lat=r["gps_lat"], bitrate=r["bitrate"],
                       filename=r["original_filename"], path=path)
    scores = json_loads(r["category_scores"], {}) or {}
    emb = _embedding(conn, int(r["id"])) if r["kind"] in ("image", "video") else None
    meme = scores.get("meme_p")
    if meme is None and emb is not None and engine is not None:
        meme = meme_probability(engine, emb)
        if meme is not None:
            scores["meme_p"] = meme
            with tx(conn):
                conn.execute("UPDATE items SET category_scores = ? WHERE id = ?", (json_dumps(scores), int(r["id"])))
    sig.update(model_signals(meme_p=meme, doc_likeness=r["doc_likeness"], screenshot_p=scores.get("screenshot_p"),
                             people=int(r["face_count"] or 0) + int(r["body_count"] or 0) > 0))
    apply(conn, int(r["id"]), sig, emb=emb)


def housekeeping(conn: sqlite3.Connection, budget_s: float = 5.0) -> int:
    """The worker's entry: compute the signals and decide for items that have none yet, within the
    time budget. Items still waiting for analysis are left to the analyze stage. Returns how many
    items were done."""
    global _running
    if not settings.forwards_enabled:
        return 0
    if not _lock.acquire(blocking=False):
        return 0
    deadline = time.monotonic() + float(budget_s)
    touched = 0
    _running = True
    try:
        engine: Any | None = None
        engine_asked = False
        while time.monotonic() < deadline:
            rows = conn.execute(
                "SELECT id, uid, version, kind, ext, width, height, camera, exif_datetime, gps_lat, bitrate, "
                "original_filename, category_scores, doc_likeness, face_count, body_count FROM items "
                "WHERE forward_checked = 0 AND deleted_at IS NULL "
                "AND (analysis_status != 'pending' OR kind NOT IN ('image', 'video')) ORDER BY id LIMIT ?",
                (BACKFILL_BATCH,),
            ).fetchall()
            if not rows:
                break
            if not engine_asked:
                engine, engine_asked = _engine(), True
            for r in rows:
                if time.monotonic() >= deadline:
                    break
                try:
                    _backfill_row(conn, r, engine)
                except Exception:  # noqa: BLE001 - one bad row must not stop the backfill
                    log.exception("forwards: backfill of item %s failed", r["uid"])
                    with tx(conn):
                        conn.execute("UPDATE items SET forward_checked = 1 WHERE id = ?", (int(r["id"]),))
                touched += 1
            if len(rows) < BACKFILL_BATCH:
                break
    finally:
        _running = False
        _lock.release()
    if touched:
        log.info("forwards: backfill decided %d item(s)", touched)
    return touched


def running() -> bool:
    return _running


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Counts over the library's normal side (live, not a companion, not in the vault)."""
    row = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN forward_checked = 0 AND (analysis_status != 'pending' OR kind NOT IN ('image', 'video')) "
        "THEN 1 ELSE 0 END), 0) AS unchecked, "
        "COALESCE(SUM(CASE WHEN forward = 1 THEN 1 ELSE 0 END), 0) AS forwards, "
        "COALESCE(SUM(CASE WHEN received_via IS NOT NULL THEN 1 ELSE 0 END), 0) AS received "
        "FROM items WHERE deleted_at IS NULL AND companion_of IS NULL AND nsfw = 0"
    ).fetchone()
    return {
        "enabled": bool(settings.forwards_enabled),
        "unchecked": int(row["unchecked"]),
        "forwards": int(row["forwards"]),
        "received": int(row["received"]),
        "running": running(),
    }
