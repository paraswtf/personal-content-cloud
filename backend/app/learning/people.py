"""People learner (section 18).

Per-person face thresholds: a person the owner confirmed faces for is matched more readily (the
threshold drops towards the weighted 10th percentile of those faces' similarity to the person, floor
FLOOR),
a person the owner merged others into (or moved faces to, section 10) is matched down to the similarity
they accepted in the least alike of those merges (learning_events kind 'merge', features.accepted, minus
LOOSEN, floor FLOOR; a merge under cluster.PULL_FLOOR teaches nothing, a cross-age link has no value), and one the
owner rejected a guess for or took faces away from is matched more carefully (the threshold rises above
the closest rejected face, which wins over any loosening). Faces in the child band (app.analysis.ages,
once the library's ages are read) are left out: a baby photo says nothing about the adult. Every
face counts by its face weight (app.analysis.faces.face_weight, section 10): a blurred, masked or
turned-away face moves a threshold little or not at all, so unclear faces cannot bias it. Pets
learn the same way from their crops (section 22): the owner's crops set the threshold, and a crop the
owner took away from a pet (guess_rejections kind 'animal') raises it above that crop, up to PET_CEIL.
Guess weights: a logistic regression on the evidence terms of confirmed and rejected guesses
(learning_events kind 'guess'), used instead of the fixed 0.5/0.3/0.2 mix once MIN_GUESS_EVENTS outcomes
with both labels exist.
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from app.config import settings
from app.learning import events, store
from app.util import json_loads, now_iso

MODEL = "people"
RULES_REV = 2  # 2 (2026-11-09): merges under PULL_FLOOR teach nothing; childhood photos left out (train.signature)
MIN_GUESS_EVENTS = 30
MIN_USER_FACES = 3
FLOOR = 0.30
CEIL = 0.90
PET_CEIL = 0.97  # pet crops of one animal sit near 0.86 to 0.95 (SigLIP), so a pet's threshold may rise higher
LOOSEN = 0.02
TIGHTEN = 0.02
FEATURES = ("clothes", "scene_term", "face_term", "same_day", "has_gps")


def _weight(row: sqlite3.Row) -> float:
    from app.analysis.faces import face_weight

    return face_weight(row["det_score"], row["size_px"], row["clarity"], row["sharpness"])


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def person_thresholds(conn: sqlite3.Connection) -> dict[int, float]:
    """{person_id: match threshold} for the people (faces) and the pets (animal crops, section 22)
    whose owner-assigned examples say the default is too loose or too tight."""
    from app.analysis import ages
    from app.analysis.cluster import PULL_FLOOR

    # Faces in the child band (section 10) say nothing about the person as an adult, whose threshold this is.
    young = ages.ready(conn)

    def adult(r: sqlite3.Row, w: float) -> bool:
        return not (young and ages.in_child_band(r["age_groups"], w))

    by_person: dict[int, list[np.ndarray]] = {}
    weights: dict[int, list[float]] = {}
    for r in conn.execute(
        "SELECT person_id, embedding, det_score, size_px, clarity, sharpness, age_groups FROM faces "
        "WHERE assigned_by = 'user' AND person_id IS NOT NULL"
    ):
        v, w = _vec(r["embedding"]), _weight(r)
        if v is not None and w > 0 and adult(r, w):
            by_person.setdefault(int(r["person_id"]), []).append(v)
            weights.setdefault(int(r["person_id"]), []).append(w)
    rejected: dict[int, list[np.ndarray]] = {}
    rejected_weights: dict[int, list[float]] = {}
    for r in conn.execute(
        "SELECT g.person_id, f.embedding, f.det_score, f.size_px, f.clarity, f.sharpness, f.age_groups "
        "FROM guess_rejections g JOIN faces f ON f.id = g.ref_id WHERE g.kind = 'face' "
        "AND f.person_id IS NOT g.person_id"  # a face that is theirs again (a merge) is no rejection
    ):
        v, w = _vec(r["embedding"]), _weight(r)
        if v is not None and w > 0 and adult(r, w):
            rejected.setdefault(int(r["person_id"]), []).append(v)
            rejected_weights.setdefault(int(r["person_id"]), []).append(w)
    merged: dict[int, float] = {}
    for r in conn.execute("SELECT person_id, features FROM learning_events WHERE kind = 'merge' AND person_id IS NOT NULL"):
        f = json_loads(r["features"], {}) or {}
        a = f.get("accepted")
        # A merge the matching saw as under PULL_FLOOR (recorded before the merge itself checked, 2026-11-09)
        # teaches nothing: it would loosen the person to the floor for good. A link across the child band
        # records no adult value (cluster.learn_accepted), so it teaches nothing either.
        if isinstance(a, (int, float)) and a >= PULL_FLOOR:
            pid = int(r["person_id"])
            merged[pid] = min(merged.get(pid, 1.0), float(a))
    # A person without confirmed faces still needs a reference for the rejection rule: their faces.
    reference: dict[int, np.ndarray] = {}
    for pid in (set(rejected) | set(merged)) - set(by_person):
        vs, ws = [], []
        for r in conn.execute("SELECT embedding, det_score, size_px, clarity, sharpness, age_groups FROM faces "
                              "WHERE person_id = ? ORDER BY quality DESC LIMIT 200", (pid,)):
            v, w = _vec(r["embedding"]), _weight(r)
            if v is not None and w > 0 and adult(r, w):
                vs.append(v)
                ws.append(w)
        if vs:
            c = (np.stack(vs) * np.asarray(ws, dtype=np.float32)[:, None]).sum(axis=0)
            n = float(np.linalg.norm(c))
            if n > 0:
                reference[pid] = (c / n).astype(np.float32)
    out = _thresholds(by_person, rejected, float(settings.cluster_match_threshold), merged, reference,
                      weights=weights, rejected_weights=rejected_weights)
    by_pet: dict[int, list[np.ndarray]] = {}
    for r in conn.execute(
        "SELECT a.person_id, a.embedding FROM animals a JOIN persons p ON p.id = a.person_id "
        "WHERE a.assigned_by = 'user' AND a.embedding IS NOT NULL AND p.kind = 'pet'"
    ):
        v = _vec(r["embedding"])
        if v is not None:
            by_pet.setdefault(int(r["person_id"]), []).append(v)
    rejected_pets: dict[int, list[np.ndarray]] = {}
    for r in conn.execute(
        "SELECT g.person_id, a.embedding FROM guess_rejections g JOIN animals a ON a.id = g.ref_id "
        "JOIN persons p ON p.id = g.person_id WHERE g.kind = 'animal' AND p.kind = 'pet' AND a.embedding IS NOT NULL "
        "AND a.person_id IS NOT g.person_id"
    ):
        v = _vec(r["embedding"])
        if v is not None:
            rejected_pets.setdefault(int(r["person_id"]), []).append(v)
    # A pet without crops of the owner's still needs a reference for the rejection rule: its crops.
    pet_reference: dict[int, np.ndarray] = {}
    for pid in set(rejected_pets) - set(by_pet):
        vs = [v for v in (_vec(r["embedding"]) for r in conn.execute(
            "SELECT embedding FROM animals WHERE person_id = ? AND embedding IS NOT NULL ORDER BY id LIMIT 200",
            (pid,))) if v is not None]
        if vs:
            c = np.stack(vs).mean(axis=0)
            n = float(np.linalg.norm(c))
            if n > 0:
                pet_reference[pid] = (c / n).astype(np.float32)
    out.update(_thresholds(by_pet, rejected_pets, float(settings.pet_match_threshold),
                           reference=pet_reference, ceil=PET_CEIL))
    return out


def _weighted_percentile(values: list[float], weights: list[float], q: float) -> float:
    """The value below which q percent of the total weight lies (np.percentile when all weigh 1)."""
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    if v.size == 1 or float(w.sum()) <= 0:
        return float(np.percentile(v, q))
    # Midpoints of each value's weight span, scaled to 0..100, as np.percentile's linear rule does
    # for equal weights.
    cum = np.cumsum(w) - 0.5 * w
    pos = (cum - cum[0]) / max(float(cum[-1] - cum[0]), 1e-12) * 100.0
    return float(np.interp(q, pos, v))


def _thresholds(by_person: dict[int, list[np.ndarray]], rejected: dict[int, list[np.ndarray]],
                default: float, merged: dict[int, float] | None = None,
                reference: dict[int, np.ndarray] | None = None, ceil: float = CEIL,
                weights: dict[int, list[float]] | None = None,
                rejected_weights: dict[int, list[float]] | None = None) -> dict[int, float]:
    """weights / rejected_weights: the face weight of each example (1 when not given). The centroid
    and the 10th percentile are weighted, MIN_USER_FACES counts the total weight, and a rejected
    example raises the threshold by its weight's share of the way to its similarity + TIGHTEN."""
    out: dict[int, float] = {}
    merged = merged or {}
    reference = reference or {}
    weights = weights or {}
    rejected_weights = rejected_weights or {}
    for pid in set(by_person) | set(rejected) | set(merged):
        confirmed = by_person.get(pid, [])
        cw = np.asarray(weights.get(pid) or [1.0] * len(confirmed), dtype=np.float64)
        t: float | None = None
        centroid: np.ndarray | None = reference.get(pid)
        if confirmed:
            x = np.stack(confirmed)
            centroid = (x * cw[:, None]).sum(axis=0)
            n = float(np.linalg.norm(centroid))
            centroid = centroid / n if n > 0 else None
        if centroid is not None and float(cw.sum()) >= MIN_USER_FACES - 1e-9:
            x = np.stack(confirmed)
            if len(confirmed) >= 2:  # leave-one-out: each face against the weighted centroid of the others
                sims = []
                total = (x * cw[:, None]).sum(axis=0)
                for i in range(len(confirmed)):
                    other = total - x[i] * cw[i]
                    n = float(np.linalg.norm(other))
                    sims.append(float(x[i] @ (other / n)) if n > 0 else 1.0)
            else:
                sims = [1.0]
            p10 = _weighted_percentile(sims, cw.tolist(), 10)
            t = max(FLOOR, min(default, p10 - LOOSEN))
        if pid in merged:
            loose = max(FLOOR, min(default, merged[pid] - LOOSEN))
            t = loose if t is None else min(t, loose)
        if centroid is not None and rejected.get(pid):
            rw = rejected_weights.get(pid) or [1.0] * len(rejected[pid])
            base = t if t is not None else default
            raised = [base + w * (float(v @ centroid) + TIGHTEN - base)
                      for v, w in zip(rejected[pid], rw) if v.shape == centroid.shape]
            if raised:
                t = min(ceil, max(base, max(raised)))
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
