"""Not-a-face learner (sections 10 and 18): what the owner removes as "Not a face" teaches detection.

Every removal is kept in not_faces: the box's embedding, detector score, size, clarity and sharpness
(record). Detection then drops a new face (Filter.keep, called by the pipeline before anything else
uses the faces) when
- it looks like a marked one: similarity >= SIM to any marking. Measured on the owner's photos,
  things that are not faces look alike to the face model (0.52 median to the nearest other one, up to
  0.73), while clear faces stay at 0.34 or below against them; or
- the learned rule says it is no face (train): a logistic regression on the detector score, clarity,
  sharpness, size and the similarity to the nearest marking, fitted on the markings against the faces
  the owner assigned, clear or not, so the helmets and masks the owner tags keep being found. It is
  used once MIN_EACH of both exist, only at P(not a face) >= DROP_P, and only while learning is on;
and never when the face is clear (face weight >= app.analysis.faces.CLEAR_WEIGHT) or is where a face
the owner assigned already is (a re-analysis keeps the owner's faces).
The markings themselves are used whether learning is on or off, like rejected guesses.
When the owner marks a face, the unassigned automatic faces that look like it in the same way go
with it (followers); they are not markings themselves.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.db import tx
from app.learning import store
from app.util import now_iso

log = logging.getLogger(__name__)

MODEL = "not_faces"
SIM = 0.5
DROP_P = 0.9
MIN_EACH = 10
PROTECT_IOU = 0.5
PROTECT_SIM = 0.6
FEATURES = ("det_score", "clarity", "sharpness", "size", "nearest")


def _unit(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def record(conn: sqlite3.Connection, face: sqlite3.Row) -> None:
    """Keep a face the owner removes as "Not a face" (call inside the removal's transaction)."""
    with tx(conn):
        conn.execute(
            "INSERT INTO not_faces(item_id, det_score, size_px, clarity, sharpness, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (face["item_id"], float(face["det_score"]), int(face["size_px"]), face["clarity"], face["sharpness"],
             face["embedding"], now_iso()),
        )


def followers(conn: sqlite3.Connection, face: sqlite3.Row) -> list[int]:
    """The unassigned automatic faces (not the owner's, no person, no open guess) that look like `face`
    the way Filter drops them: similarity >= SIM and face weight below CLEAR_WEIGHT."""
    from app.analysis.faces import CLEAR_WEIGHT
    from app.analysis.index import get_index

    emb = _unit(face["embedding"])
    if emb is None:
        return []
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        ids = [i for i in idx.unassigned_lookalikes(emb, SIM, CLEAR_WEIGHT) if i != int(face["id"])]
    if not ids:
        return []
    out: list[int] = []
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" * len(chunk))
        out.extend(int(r[0]) for r in conn.execute(
            f"SELECT id FROM faces WHERE id IN ({marks}) AND person_id IS NULL AND assigned_by = 'auto'", chunk))
    return sorted(out)


def _features(det_score: float, size_px: float, clarity: float, sharpness: float, nearest: float) -> list[float]:
    return [float(det_score), float(clarity) / 30.0, math.log1p(max(float(sharpness), 0.0)) / 5.0,
            math.log(max(float(size_px), 1.0)) / 6.0, float(nearest)]


@dataclass
class Filter:
    """What detection drops (see the module text). protect: (box, embedding) of the faces the owner
    assigned on the item being analysed."""

    marks: np.ndarray  # (n, 512) unit embeddings of the markings
    model: dict[str, Any] | None = None
    protect: list[tuple[tuple[float, float, float, float], np.ndarray]] = field(default_factory=list)

    def nearest(self, emb: np.ndarray) -> float:
        if not self.marks.size:
            return -1.0
        return float(np.max(self.marks @ np.asarray(emb, dtype=np.float32)))

    def probability(self, det_score: float, size_px: int, clarity: float | None, sharpness: float | None,
                    nearest: float) -> float | None:
        """P(not a face) from the learned rule, or None without a model or without the measures."""
        if not self.model or clarity is None or sharpness is None:
            return None
        w = np.asarray([float(self.model["w"].get(k, 0.0)) for k in FEATURES], dtype=np.float64)
        z = float(np.asarray(_features(det_score, size_px, clarity, sharpness, nearest)) @ w + float(self.model["b"]))
        return 1.0 / (1.0 + math.exp(-z))

    def drops(self, rec: Any) -> bool:
        """rec: a FaceRecord (x, y, w, h, det_score, size_px, clarity, sharpness, embedding)."""
        from app.analysis.cluster import iou
        from app.analysis.faces import CLEAR_WEIGHT, face_weight

        if face_weight(rec.det_score, rec.size_px, rec.clarity, rec.sharpness) >= CLEAR_WEIGHT:
            return False
        emb = np.asarray(rec.embedding, dtype=np.float32)
        box = (rec.x, rec.y, rec.w, rec.h)
        for pbox, pemb in self.protect:
            if iou(box, pbox) >= PROTECT_IOU or float(pemb @ emb) >= PROTECT_SIM:
                return False
        near = self.nearest(emb)
        if near >= SIM:
            return True
        p = self.probability(rec.det_score, rec.size_px, rec.clarity, rec.sharpness, near)
        return p is not None and p >= DROP_P

    def keep(self, records: Sequence[Any]) -> list[Any]:
        kept = [r for r in records if not self.drops(r)]
        if len(kept) != len(records):
            log.info("not-a-face: dropped %d of %d detected faces", len(records) - len(kept), len(records))
        return kept


def load(conn: sqlite3.Connection, item_id: int | None = None) -> Filter:
    """The filter for analysing an item: every marking, the learned rule while learning is on, and the
    faces the owner assigned on that item (protected)."""
    vs = [v for v in (_unit(r[0]) for r in conn.execute("SELECT embedding FROM not_faces ORDER BY id")) if v is not None]
    marks = np.stack(vs) if vs else np.zeros((0, 512), dtype=np.float32)
    model = store.load(MODEL) if store.enabled(conn) else None
    rule = (model or {}).get("rule")
    protect: list[tuple[tuple[float, float, float, float], np.ndarray]] = []
    if item_id is not None:
        for r in conn.execute("SELECT x, y, w, h, embedding FROM faces WHERE item_id = ? AND assigned_by = 'user' "
                              "AND person_id IS NOT NULL", (int(item_id),)):
            v = _unit(r["embedding"])
            if v is not None:
                protect.append(((r["x"], r["y"], r["w"], r["h"]), v))
    return Filter(marks=marks, model=rule if isinstance(rule, dict) else None, protect=protect)


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    """Fit the rule on the measured markings (label 1) against the measured faces the owner assigned
    (label 0); the nearest-marking feature of a marking leaves the marking itself out."""
    from app.learning.people import fit_logistic

    neg = conn.execute("SELECT det_score, size_px, clarity, sharpness, embedding FROM not_faces "
                       "WHERE clarity IS NOT NULL AND sharpness IS NOT NULL ORDER BY id").fetchall()
    pos = conn.execute("SELECT det_score, size_px, clarity, sharpness, embedding FROM faces WHERE assigned_by = 'user' "
                       "AND person_id IS NOT NULL AND clarity IS NOT NULL AND sharpness IS NOT NULL").fetchall()
    all_marks = [v for v in (_unit(r[0]) for r in conn.execute("SELECT embedding FROM not_faces ORDER BY id"))
                 if v is not None]
    total = conn.execute("SELECT COUNT(*) FROM not_faces").fetchone()[0]
    rule = None
    if len(neg) >= MIN_EACH and len(pos) >= MIN_EACH and all_marks:
        M = np.stack(all_marks)
        xs: list[list[float]] = []
        ys: list[float] = []
        for label, rows in ((1.0, neg), (0.0, pos)):
            for r in rows:
                e = _unit(r["embedding"])
                if e is None:
                    continue
                sims = M @ e
                if label == 1.0:
                    sims = sims[sims < 0.9999]  # the marking itself
                near = float(sims.max()) if sims.size else -1.0
                xs.append(_features(r["det_score"], r["size_px"], r["clarity"], r["sharpness"], near))
                ys.append(label)
        if len(set(ys)) == 2:
            w, b = fit_logistic(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
            rule = {"w": dict(zip(FEATURES, w)), "b": b, "markings": len(neg), "faces": len(pos)}
    store.save(MODEL, {"rule": rule, "markings": int(total), "trained_at": now_iso()})
    return status()


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    rule = model.get("rule") or None
    return {"markings": int(model.get("markings", 0)), "rule": bool(rule),
            "min_each": MIN_EACH, "trained_at": model.get("trained_at")}
