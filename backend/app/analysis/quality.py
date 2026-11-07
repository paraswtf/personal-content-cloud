"""Mis-click detection: blur, exposure, pocket and ground shots, bursts (docs/ARCHITECTURE.md section 9).

CONTRACT.
- blur_variance(img) -> float: variance of the Laplacian on the grey image scaled down to 1024 px
  on the long side (PIL image or BGR array).
- exposure(img) -> (mean, p95, clipped_frac): luminance 0..255; clipped = share of pixels >= 250.
- measure(img) -> {blur_var, mean_lum, p95_lum, clipped_frac}.
- accident_probability(engine, embs) / motion_blur_probability(engine, embs) -> np.ndarray (n,):
  SigLIP zero-shot of ACCIDENT_PROMPTS vs NORMAL_PROMPTS and MOTION_BLUR_PROMPTS vs SHARP_PROMPTS.
- burst_group(conn, item, blur_var) -> list[BurstMember]: analysed images of the same camera taken
  within settings.burst_window_s with pHash distance <= BURST_PHASH_MAX (the item itself included).
- burst_roles(members) -> {item_id: None | best_uid}: None for the sharpest, else the best's uid.
- decide(kind, category, scores, *, good_face, burst) -> (misclick, reasons, scores).
- with_burst(reasons, burst) -> reasons with the burst entries replaced; is_misclick(reasons).
Nothing here writes to the database.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.util import json_loads, parse_iso

log = logging.getLogger(__name__)

BLUR_SIDE = 1024
DARK_MEAN = 28.0
DARK_P95 = 70.0
BRIGHT_MEAN = 235.0
CLIPPED_LEVEL = 250
CLIPPED_FRAC = 0.60
ACCIDENT_MIN = 0.6
MOTION_BLUR_MIN = 0.6
BURST_PHASH_MAX = 12
# Reasons that put an item into the review bin. 'burst_best' is informational only.
FLAG_REASONS = ("blurry", "dark", "blown_out", "pocket", "burst")

ACCIDENT_PROMPTS = [
    "an accidental photo of the floor",
    "an accidental photo of the ceiling",
    "a photo taken inside a pocket",
    "a dark blurry photo of the inside of a bag",
    "a finger covering the camera lens",
    "a hand blocking the camera",
    "a photo of only the sky taken by accident",
    "an accidental photo of the ground",
    "a blurry accidental photo of nothing",
]
NORMAL_PROMPTS = [
    "a photo of people",
    "a photo of a place",
    "a photo of an object",
    "a photo of food",
    "a photo of an animal",
    "a photo of a landscape",
    "a photo of a room",
    "a picture of a document",
    "a selfie",
]
MOTION_BLUR_PROMPTS = [
    "a photo with heavy motion blur",
    "a shaky blurry photo",
    "an out of focus photo",
    "a blurred photo taken while moving",
]
SHARP_PROMPTS = [
    "a sharp photo",
    "a clear photo in focus",
    "a crisp detailed photo",
]


# ------------------------------------------------------------------ pixel measures
def _grey(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, Image.Image):
        if img.mode == "L":
            return np.asarray(img, dtype=np.uint8)
        return np.asarray(img.convert("L"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGR2GRAY)


def _scaled(grey: np.ndarray, side: int) -> np.ndarray:
    h, w = grey.shape[:2]
    longer = max(h, w)
    if longer <= side:
        return grey
    s = side / longer
    return cv2.resize(grey, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def blur_variance(img: Image.Image | np.ndarray) -> float:
    """Variance of the Laplacian of the grey image, downscaled to BLUR_SIDE px on the long side."""
    g = _scaled(_grey(img), BLUR_SIDE)
    if g.size == 0:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def exposure(img: Image.Image | np.ndarray) -> tuple[float, float, float]:
    """(mean luminance, 95th percentile, share of pixels >= CLIPPED_LEVEL), all on 0..255 / 0..1."""
    g = _scaled(_grey(img), BLUR_SIDE)
    if g.size == 0:
        return 0.0, 0.0, 0.0
    return float(g.mean()), float(np.percentile(g, 95)), float((g >= CLIPPED_LEVEL).mean())


def exposure_many(frames: list[np.ndarray]) -> tuple[float, float, float]:
    """exposure() over several frames taken together (videos)."""
    greys = [_scaled(_grey(f), 512).ravel() for f in frames if f is not None and np.asarray(f).size]
    if not greys:
        return 0.0, 0.0, 0.0
    g = np.concatenate(greys)
    return float(g.mean()), float(np.percentile(g, 95)), float((g >= CLIPPED_LEVEL).mean())


def measure(img: Image.Image | np.ndarray) -> dict[str, float]:
    mean, p95, clipped = exposure(img)
    return {
        "blur_var": round(blur_variance(img), 3),
        "mean_lum": round(mean, 3),
        "p95_lum": round(p95, 3),
        "clipped_frac": round(clipped, 4),
    }


# ------------------------------------------------------------------ zero-shot measures
def _binary(engine: Any, embs: np.ndarray, pos: list[str], neg: list[str]) -> np.ndarray:
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim == 1:
        embs = embs[None, :]
    if embs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    probs = np.asarray(engine.zero_shot(embs, {"pos": pos, "neg": neg}), dtype=np.float32)
    return probs[:, 0]


def accident_probability(engine: Any, embs: np.ndarray) -> np.ndarray:
    """p(accidental shot: floor, ceiling, pocket, finger on the lens, sky only) per embedding."""
    return _binary(engine, embs, ACCIDENT_PROMPTS, NORMAL_PROMPTS)


def motion_blur_probability(engine: Any, embs: np.ndarray) -> np.ndarray:
    return _binary(engine, embs, MOTION_BLUR_PROMPTS, SHARP_PROMPTS)


# ------------------------------------------------------------------ bursts
@dataclass
class BurstMember:
    id: int
    uid: str
    blur_var: float
    misclick_source: str


def _hamming(a: str | None, b: str | None) -> int:
    if not a or not b or len(a) != len(b):
        return 64
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return 64


def burst_group(conn: sqlite3.Connection, item: sqlite3.Row, blur_var: float) -> list[BurstMember]:
    """Images of the same camera (both without one counts as the same) taken within
    settings.burst_window_s of `item` with pHash distance <= BURST_PHASH_MAX, analysed already, live,
    plus `item` itself (with blur_var). Only direct neighbours of `item` count. Empty when `item` has no
    capture time or pHash."""
    if item["kind"] != "image" or not item["taken_at"] or not item["phash"]:
        return []
    t0 = parse_iso(item["taken_at"])
    if t0 is None:
        return []
    from datetime import timedelta

    window = float(settings.burst_window_s)
    lo = (t0 - timedelta(seconds=window + 1)).isoformat().replace("+00:00", "Z")
    hi = (t0 + timedelta(seconds=window + 1)).isoformat().replace("+00:00", "Z")
    rows = conn.execute(
        "SELECT id, uid, taken_at, phash, quality_scores, misclick_source FROM items "
        "WHERE id != ? AND kind = 'image' AND deleted_at IS NULL AND analysis_status = 'done' "
        "AND camera IS ? AND taken_at IS NOT NULL AND taken_at >= ? AND taken_at <= ? AND phash IS NOT NULL",
        (item["id"], item["camera"], lo[:19], hi),
    ).fetchall()
    members = [BurstMember(int(item["id"]), item["uid"], float(blur_var), item["misclick_source"])]
    for r in rows:
        t = parse_iso(r["taken_at"])
        if t is None or abs((t - t0).total_seconds()) > window:
            continue
        if _hamming(r["phash"], item["phash"]) > BURST_PHASH_MAX:
            continue
        scores = json_loads(r["quality_scores"], {}) or {}
        bv = scores.get("blur_var")
        if not isinstance(bv, (int, float)):
            continue
        members.append(BurstMember(int(r["id"]), r["uid"], float(bv), r["misclick_source"]))
    return members if len(members) > 1 else []


def burst_roles(members: list[BurstMember]) -> dict[int, str | None]:
    """{item_id: None} for the sharpest member (ties: lowest id), {item_id: best_uid} for the rest."""
    if len(members) < 2:
        return {}
    best = max(members, key=lambda m: (m.blur_var, -m.id))
    return {m.id: (None if m.id == best.id else best.uid) for m in members}


def with_burst(reasons: dict[str, Any] | None, burst: dict[str, Any] | None) -> dict[str, Any]:
    """reasons with 'burst' / 'burst_best' replaced by `burst` ({'burst': {...}} or {'burst_best': {...}} or None)."""
    out = {k: v for k, v in (reasons or {}).items() if k not in ("burst", "burst_best")}
    if burst:
        out.update(burst)
    return out


def is_misclick(reasons: dict[str, Any] | None) -> bool:
    return any(k in FLAG_REASONS for k in (reasons or {}))


# ------------------------------------------------------------------ decision
def decide(kind: str, category: str | None, scores: dict[str, Any], *, good_face: bool,
           burst: dict[str, Any] | None = None) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Mis-click decision. scores holds blur_var, mean_lum, p95_lum, clipped_frac and optionally
    accident_p, motion_blur_p. Images only (documents and screenshots never); a video only gets
    'dark'. burst: {'burst': {'best_uid': ...}} or {'burst_best': {...}} from burst_roles.
    Returns (misclick, reasons, scores)."""
    reasons: dict[str, Any] = {}
    if kind not in ("image", "video") or category in ("documents", "screenshots"):
        return False, reasons, scores
    mean = scores.get("mean_lum")
    p95 = scores.get("p95_lum")
    if isinstance(mean, (int, float)) and isinstance(p95, (int, float)) and mean < DARK_MEAN and p95 < DARK_P95:
        reasons["dark"] = {"mean_lum": round(float(mean), 1), "p95_lum": round(float(p95), 1)}
    if kind == "video":
        return bool(reasons), reasons, scores
    clipped = scores.get("clipped_frac")
    if (isinstance(mean, (int, float)) and mean > BRIGHT_MEAN) or (
            isinstance(clipped, (int, float)) and clipped > CLIPPED_FRAC):
        reasons["blown_out"] = {"mean_lum": round(float(mean or 0), 1), "clipped_frac": round(float(clipped or 0), 3)}
    bv = scores.get("blur_var")
    motion = scores.get("motion_blur_p")
    if isinstance(bv, (int, float)):
        limit = float(settings.blur_var_min)
        if bv < limit:
            reasons["blurry"] = {"blur_var": round(float(bv), 1)}
        elif isinstance(motion, (int, float)) and motion >= MOTION_BLUR_MIN and bv < 2 * limit:
            reasons["blurry"] = {"blur_var": round(float(bv), 1), "motion_blur_p": round(float(motion), 3)}
    accident = scores.get("accident_p")
    if isinstance(accident, (int, float)) and accident >= ACCIDENT_MIN and not good_face:
        reasons["pocket"] = {"accident_p": round(float(accident), 3)}
    reasons = with_burst(reasons, burst)
    return is_misclick(reasons), reasons, scores
