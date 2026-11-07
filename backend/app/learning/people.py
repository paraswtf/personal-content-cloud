"""People learner (section 18).

Per-person face thresholds: a person the owner confirmed faces for is matched more readily (the
threshold drops towards the 10th percentile of those faces' similarity to the person, floor FLOOR),
and one the owner rejected a guess for is matched more carefully (the threshold rises above the
closest rejected face). Guess weights: a logistic regression on the evidence terms of confirmed and
rejected guesses (learning_events kind 'guess'), used instead of the fixed 0.5/0.3/0.2 mix once
MIN_GUESS_EVENTS outcomes with both labels exist.
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from app.config import settings
from app.learning import events, store
from app.util import json_loads, now_iso

MODEL = "people"
MIN_GUESS_EVENTS = 30
MIN_USER_FACES = 3
FLOOR = 0.30
CEIL = 0.90
LOOSEN = 0.02
TIGHTEN = 0.02
FEATURES = ("clothes", "scene_term", "face_term", "same_day", "has_gps")


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def person_thresholds(conn: sqlite3.Connection) -> dict[int, float]:
    """{person_id: match threshold} for the people (faces) and the pets (animal crops, section 22)
    whose owner-assigned examples say the default is too loose or too tight."""
    by_person: dict[int, list[np.ndarray]] = {}
    for r in conn.execute(
        "SELECT person_id, embedding FROM faces WHERE assigned_by = 'user' AND person_id IS NOT NULL"
    ):
        v = _vec(r["embedding"])
        if v is not None:
            by_person.setdefault(int(r["person_id"]), []).append(v)
    rejected: dict[int, list[np.ndarray]] = {}
    for r in conn.execute(
        "SELECT g.person_id, f.embedding FROM guess_rejections g JOIN faces f ON f.id = g.ref_id WHERE g.kind = 'face'"
    ):
        v = _vec(r["embedding"])
        if v is not None:
            rejected.setdefault(int(r["person_id"]), []).append(v)
    out = _thresholds(by_person, rejected, float(settings.cluster_match_threshold))
    by_pet: dict[int, list[np.ndarray]] = {}
    for r in conn.execute(
        "SELECT a.person_id, a.embedding FROM animals a JOIN persons p ON p.id = a.person_id "
        "WHERE a.assigned_by = 'user' AND a.embedding IS NOT NULL AND p.kind = 'pet'"
    ):
        v = _vec(r["embedding"])
        if v is not None:
            by_pet.setdefault(int(r["person_id"]), []).append(v)
    out.update(_thresholds(by_pet, {}, float(settings.pet_match_threshold)))
    return out


def _thresholds(by_person: dict[int, list[np.ndarray]], rejected: dict[int, list[np.ndarray]],
                default: float) -> dict[int, float]:
    out: dict[int, float] = {}
    for pid in set(by_person) | set(rejected):
        confirmed = by_person.get(pid, [])
        t: float | None = None
        centroid: np.ndarray | None = None
        if confirmed:
            x = np.stack(confirmed)
            centroid = x.mean(axis=0)
            n = float(np.linalg.norm(centroid))
            centroid = centroid / n if n > 0 else None
        if centroid is not None and len(confirmed) >= MIN_USER_FACES:
            x = np.stack(confirmed)
            if len(confirmed) >= 2:  # leave-one-out: each face against the centroid of the others
                sims = []
                total = x.sum(axis=0)
                for i in range(len(confirmed)):
                    other = total - x[i]
                    n = float(np.linalg.norm(other))
                    sims.append(float(x[i] @ (other / n)) if n > 0 else 1.0)
            else:
                sims = [1.0]
            p10 = float(np.percentile(sims, 10))
            t = max(FLOOR, min(default, p10 - LOOSEN))
        if centroid is not None and rejected.get(pid):
            worst = max(float(v @ centroid) for v in rejected[pid])
            t = min(CEIL, max(t if t is not None else default, worst + TIGHTEN))
        if t is not None and abs(t - default) > 1e-6:
            out[pid] = round(t, 4)
    return out


def _features(row_features: dict[str, Any]) -> list[float] | None:
    try:
        return [
            float(row_features.get("clothes") or 0.0),
            float(row_features.get("scene_term") or 0.0),
            float(row_features.get("face_term") if row_features.get("face_term") is not None else 0.5),
            1.0 if row_features.get("same_day") else 0.0,
            1.0 if row_features.get("has_gps") else 0.0,
        ]
    except (TypeError, ValueError):
        return None


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.5, l2: float = 0.01) -> tuple[list[float], float]:
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    n = len(y)
    for _ in range(steps):
        z = x @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (x.T @ g / n + l2 * w)
        b -= lr * float(g.mean())
    return [round(float(v), 5) for v in w], round(float(b), 5)


def guess_model(conn: sqlite3.Connection) -> dict[str, Any] | None:
    rows = events.rows(conn, "guess")
    xs, ys = [], []
    for r in rows:
        if r["label"] is None:
            continue
        f = _features(json_loads(r["features"], {}) or {})
        if f is None:
            continue
        xs.append(f)
        ys.append(int(r["label"]))
    if len(ys) < MIN_GUESS_EVENTS or len(set(ys)) < 2:
        return None
    w, b = fit_logistic(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return {"w": dict(zip(FEATURES, w)), "b": b, "events": len(ys)}


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    thresholds = person_thresholds(conn)
    weights = guess_model(conn)
    n_events = sum(1 for r in events.rows(conn, "guess") if r["label"] is not None)
    store.save(MODEL, {
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "weights": weights,
        "persons": len(thresholds),
        "events": n_events,
        "trained_at": now_iso(),
    })
    return status()


def person_threshold(person_id: int) -> float | None:
    if not store.enabled():
        return None
    model = store.load(MODEL)
    if not model:
        return None
    v = (model.get("thresholds") or {}).get(str(int(person_id)))
    return float(v) if isinstance(v, (int, float)) else None


def guess_probability(features: dict[str, Any]) -> float | None:
    if not store.enabled():
        return None
    model = store.load(MODEL)
    weights = (model or {}).get("weights")
    if not weights:
        return None
    f = _features(features)
    if f is None:
        return None
    w = np.asarray([float(weights["w"].get(k, 0.0)) for k in FEATURES], dtype=np.float64)
    z = float(np.asarray(f, dtype=np.float64) @ w + float(weights.get("b", 0.0)))
    return round(float(1.0 / (1.0 + np.exp(-z))), 4)


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    weights = model.get("weights")
    return {
        "persons": int(model.get("persons", 0)),
        "events": int(model.get("events", 0)),
        "weights": (weights or {}).get("w") if weights else None,
        "min_events": MIN_GUESS_EVENTS,
    }
