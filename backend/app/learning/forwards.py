"""Forwards learner (section 29): a logistic regression over the forward signals, fitted on the
owner's choices (items whose forward_source is 'user', and learning_events of kind 'forward'), used
in place of the rule's second step once MIN_EVENTS examples with both labels exist. Step 1 of the
decision (received_via) stays a rule; the look-alike vote lives in learning/images.py."""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from app.learning import events, store
from app.util import json_loads, now_iso

MODEL = "forwards"
MIN_EVENTS = 10
CUT = 0.5
FEATURES = ("name_app", "stripped", "small", "wa_size", "jpeg_quality", "meme_p", "text", "people", "video_small")


def features(sig: dict[str, Any] | None) -> list[float] | None:
    """The learner's input from a signals dict, in FEATURES order."""
    if not isinstance(sig, dict):
        return None
    try:
        q = sig.get("jpeg_quality")
        meme = sig.get("meme_p")
        return [
            1.0 if sig.get("name_app") else 0.0,
            1.0 if sig.get("stripped") else 0.0,
            1.0 if sig.get("small") else 0.0,
            1.0 if sig.get("wa_size") else 0.0,
            float(q) / 100.0 if q is not None else 1.0,
            float(meme) if meme is not None else 0.0,
            1.0 if sig.get("text") else 0.0,
            1.0 if sig.get("people") else 0.0,
            1.0 if sig.get("video_small") else 0.0,
        ]
    except (TypeError, ValueError):
        return None


def examples(conn: sqlite3.Connection) -> list[tuple[list[float], int]]:
    out: list[tuple[list[float], int]] = []
    for r in conn.execute(
        "SELECT forward, forward_signals FROM items WHERE forward_source = 'user' AND forward_signals IS NOT NULL"
    ):
        f = features(json_loads(r["forward_signals"], None))
        if f is not None:
            out.append((f, int(r["forward"])))
    for r in events.rows(conn, "forward"):
        if r["label"] is None:
            continue
        f = features(json_loads(r["features"], None))
        if f is not None:
            out.append((f, int(r["label"])))
    return out


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    from app.learning.people import fit_logistic

    rows = examples(conn)
    ys = [y for _, y in rows]
    weights = None
    if len(ys) >= MIN_EVENTS and 0 < sum(ys) < len(ys):
        w, b = fit_logistic(np.asarray([x for x, _ in rows], dtype=np.float64), np.asarray(ys, dtype=np.float64),
                            steps=800, lr=0.5, l2=0.01)
        weights = {"w": dict(zip(FEATURES, w)), "b": b}
    store.save(MODEL, {"weights": weights, "examples": len(ys), "positives": int(sum(ys)), "trained_at": now_iso()})
    return status()


def probability(sig: dict[str, Any]) -> float | None:
    """The learned probability that a received item is a forward, or None before the learner has
    enough examples (the rule decides then)."""
    if not store.enabled():
        return None
    model = store.load(MODEL) or {}
    weights = model.get("weights")
    f = features(sig)
    if not weights or f is None:
        return None
    w = np.asarray([float(weights["w"].get(k, 0.0)) for k in FEATURES], dtype=np.float64)
    z = float(np.asarray(f, dtype=np.float64) @ w + float(weights.get("b", 0.0)))
    return round(float(1.0 / (1.0 + np.exp(-z))), 4)


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    weights = model.get("weights")
    return {
        "examples": int(model.get("examples", 0)),
        "positives": int(model.get("positives", 0)),
        "learned": bool(weights),
        "weights": weights["w"] if weights else None,
        "min_events": MIN_EVENTS,
    }
