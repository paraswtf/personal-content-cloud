"""Face ages (section 10): what the FairFace age groups say about a face, and keeping them measured.

CONTRACT.
- faces.age_groups: BLOB float32[9], the age-group probabilities of the face's crop (app.ml.ages.GROUPS, youngest
  first, summing to 1); NULL = not measured (the model is missing, or the background pass has not reached it).
- decode(blob) -> (9,) float32 | None; youth(groups) -> float in [0, 1]: P(under 10) = P(0-2) + P(3-9), the signal
  matching reads. Measured on the owner's library (2026-11-09): adults and teens about 0 (young adults of 18 to
  22 may read 10-19 but not under 10), babies and toddlers 0.7 to 0.9, children of about 10 to 12 0.3 to 0.45.
- expected_age(groups) -> years (group midpoints weighted by probability); band(groups) -> 'baby' | 'child' |
  'teen' | 'adult' (the likeliest of 0-2, 3-9, 10-19, the rest), for words shown to the owner.
- measure(crops) -> list[bytes | None]: the age groups of face crops (PIL RGB, or BGR arrays) as BLOBs; all None
  without the model. Never raises: a failure leaves the faces unmeasured.
- measure_stored_faces(conn, budget_s=None) -> {"faces", "complete"}: faces without age_groups get them from
  their stored crop (the same 192 px crop analysis keeps), in id order, BATCH at a time, one short transaction
  each. Not an owner action (no undo). When no face is left it sets kv READY_KEY and moves faces_version once,
  so the child band switches on for the whole library at once. Nothing while the model is missing or models
  are off; a face without a crop file is skipped (and tried again on the next start); a model failure stops
  the pass until the next start.
- The child band: is_child_band(u10, weight) (vectorised) / in_child_band(groups, weight): P(under 10) >=
  CHILD_U10 on a face of weight >= CHILD_MIN_WEIGHT; ready(conn): kv READY_KEY is set. Matching and learning
  read the band only when ready (app.analysis.index, cluster, app.learning.people).
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

N_GROUPS = 9
# The child band (section 10): a face the model is fairly sure is a young child, clear enough to judge. Babies and
# toddlers read P(under 10) 0.7 to 0.9. Children of about 10 to 14 read 0.3 to 0.5 and stay out on purpose: ArcFace
# still links them to their adult selves (0.49 to 0.68 on the owner's library), so they keep the usual matching.
# Blurred crops are left out by the weight.
CHILD_U10 = 0.50
CHILD_MIN_WEIGHT = 0.35
READY_KEY = "face_ages_ready"  # kv: MODEL_REV once every stored face with a crop has its age (the band switches on)
MODEL_REV = "fd04cdd"  # the age model's pinned commit (app.ml.models 'ages'); a new model means a new value
MIDPOINTS = np.array([1.0, 6.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0], dtype=np.float32)
BATCH = 32


def decode(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    return v if v.shape[0] == N_GROUPS else None


def encode(groups: np.ndarray) -> bytes:
    return np.asarray(groups, dtype=np.float32).reshape(N_GROUPS).tobytes()


def youth(groups: np.ndarray | bytes | None) -> float | None:
    """P(under 10), or None when unknown."""
    g = decode(groups) if isinstance(groups, (bytes, bytearray, memoryview)) else groups
    if g is None:
        return None
    return float(np.clip(float(g[0]) + float(g[1]), 0.0, 1.0))


def is_child_band(u10: Any, weight: Any) -> np.ndarray:
    """Vectorised: P(under 10) >= CHILD_U10 on a face of weight >= CHILD_MIN_WEIGHT. Unknown ages (NaN) are not;
    nothing is while settings.face_age_matching is off. Callers also check ready()."""
    from app.config import settings

    u10 = np.asarray(u10, dtype=np.float32)
    if not settings.face_age_matching:
        return np.zeros(u10.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        return (u10 >= CHILD_U10) & (np.asarray(weight, dtype=np.float32) >= CHILD_MIN_WEIGHT)


def in_child_band(groups: np.ndarray | bytes | None, weight: float) -> bool:
    """is_child_band for one face from its age groups (a BLOB or an array) and its face weight."""
    u10 = youth(groups)
    return u10 is not None and bool(is_child_band(np.float32(u10), np.float32(weight)))


def ready(conn: sqlite3.Connection) -> bool:
    """The ages of the stored faces are all read (kv READY_KEY): only then does the child band count, so it
    switches on for the whole library at once and not face by face while the background pass runs."""
    from app.db import kv_get

    return kv_get(conn, READY_KEY, "") == MODEL_REV


def mark_ready(conn: sqlite3.Connection) -> bool:
    """Set the readiness flag (and move faces_version once, so the index reads the band) when it is not set yet.
    Call inside a tx. True when it changed."""
    from app.db import bump_faces_version, kv_set

    if ready(conn):
        return False
    kv_set(conn, READY_KEY, MODEL_REV)
    bump_faces_version(conn)
    return True


def expected_age(groups: np.ndarray | bytes | None) -> float | None:
    g = decode(groups) if isinstance(groups, (bytes, bytearray, memoryview)) else groups
    if g is None:
        return None
    return float(np.dot(np.asarray(g, dtype=np.float32), MIDPOINTS))


def band(groups: np.ndarray | bytes | None) -> str | None:
    """'baby' (0-2), 'child' (3-9), 'teen' (10-19) or 'adult': the likeliest, for words shown to the owner."""
    g = decode(groups) if isinstance(groups, (bytes, bytearray, memoryview)) else groups
    if g is None:
        return None
    g = np.asarray(g, dtype=np.float32)
    shares = (float(g[0]), float(g[1]), float(g[2]), float(g[3:].sum()))
    return ("baby", "child", "teen", "adult")[int(np.argmax(shares))]


def _engine() -> Any | None:
    from app.analysis.pipeline import _model

    return _model("ages")


def available() -> bool:
    return _engine() is not None


def measure(crops: list[Any]) -> list[bytes | None]:
    """The age groups of each crop as a BLOB, or None for all of them without the model (or on a failure)."""
    if not crops:
        return []
    mod = _engine()
    if mod is None:
        return [None] * len(crops)
    try:
        return [encode(p) for p in mod.predict(list(crops))]
    except Exception:  # noqa: BLE001 - ages are extra: the faces are stored without them
        log.exception("ages: the age model failed on %d crops", len(crops))
    if len(crops) == 1:
        return [None]
    # one crop the model cannot take must not cost the others their ages: one at a time
    out: list[bytes | None] = []
    for c in crops:
        try:
            out.append(encode(mod.predict([c])[0]))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def _model_works() -> bool:
    """The model reads a plain grey crop (tells a broken model from crops it cannot take)."""
    from PIL import Image

    return measure([Image.new("RGB", (192, 192), (128, 128, 128))])[0] is not None


def _open_crop(path) -> Any | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:  # noqa: BLE001 - a missing or broken crop: skipped
        return None


def measure_stored_faces(conn: sqlite3.Connection, budget_s: float | None = None) -> dict[str, Any]:
    from app import storage
    from app.db import bump_faces_version, tx

    if _engine() is None:
        return {"faces": 0, "complete": False}
    started = time.monotonic()
    cursor = 0
    done = 0
    while True:
        rows = conn.execute(
            "SELECT id FROM faces WHERE id > ? AND age_groups IS NULL ORDER BY id LIMIT ?", (cursor, BATCH)
        ).fetchall()
        if not rows:
            with tx(conn):
                # the index reads the child band of every face at once; a later pass (faces stored while the model
                # was missing) moves faces_version too, or the index would keep them out of the band
                if not mark_ready(conn) and done:
                    bump_faces_version(conn)
            return {"faces": done, "complete": True}
        ids = [int(r[0]) for r in rows]
        cursor = ids[-1]
        crops = [_open_crop(storage.face_crop_path(fid)) for fid in ids]
        have = [i for i, c in enumerate(crops) if c is not None]
        if not have:
            continue
        blobs = measure([crops[i] for i in have])
        if all(b is None for b in blobs) and not _model_works():
            return {"faces": done, "complete": False}  # the model itself fails: the next start tries again
        with tx(conn):
            for i, blob in zip(have, blobs):
                if blob is not None:
                    done += conn.execute("UPDATE faces SET age_groups = ? WHERE id = ? AND age_groups IS NULL",
                                         (blob, ids[i])).rowcount
        if budget_s is not None and time.monotonic() - started >= budget_s:
            return {"faces": done, "complete": False}
