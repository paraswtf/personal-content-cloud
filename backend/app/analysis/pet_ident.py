"""Pet identity vectors (section 22): the DINOv2 CLS of each stored pet crop, for the pet naming learner
(app.learning.pets), which tells two cats of one home apart where the SigLIP look embedding alone does not.

CONTRACT.
- available() -> bool: pets on, models allowed (PCC_NO_MODELS unset) and app.ml.dino present.
- ensure(conn, animal_ids) -> int: fills animals.ident for those crops that have none (from their crop files);
  returns how many it filled. Does nothing when not available. Call inside a tx or not (one UPDATE per crop).
- backfill(conn, budget_s) -> [animal ids]: fills missing vectors, newest crops first, a batch per transaction, and
  lets the trained model name each batch (app.analysis.pets.rename_crops).
- housekeeping(conn) -> dict: starts backfill on a background thread (worker.run_in_background, nice 19) when crops
  lack their vector and the workers are on; then, when the pet model is newer than the last pass (kv
  'pets_named_model'), the model names every automatic crop again.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any

import numpy as np

from app import storage
from app.config import settings

log = logging.getLogger(__name__)

BATCH = 32
KV_NAMED = "pets_named_model"
KV_FILLED = "pets_ident_filled"
RUNNER_KEY = "pet_ident"
RUNNER_BUDGET_S = 6 * 3600.0


def available() -> bool:
    if not settings.pets_enabled or os.environ.get("PCC_NO_MODELS") == "1":
        return False
    try:
        from app.ml import dino

        return dino.available()
    except Exception:  # noqa: BLE001 - a broken model install means no vectors, never a failed analysis
        log.debug("dino unavailable for pet identities", exc_info=True)
        return False


def _crop(animal_id: int) -> np.ndarray | None:
    path = storage.animal_crop_path(animal_id)
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except (OSError, ValueError):
        return None


def vectors(animal_ids: list[int]) -> dict[int, bytes]:
    """{animal id: float32[384] bytes} for the crops whose file can be read."""
    from app.ml import dino

    out: dict[int, bytes] = {}
    ids: list[int] = []
    crops: list[np.ndarray] = []
    for aid in animal_ids:
        c = _crop(int(aid))
        if c is not None and c.size:
            ids.append(int(aid))
            crops.append(c)
    for start in range(0, len(crops), BATCH):
        v = np.asarray(dino.embed(crops[start:start + BATCH]), dtype=np.float32)
        for aid, row in zip(ids[start:start + BATCH], v):
            out[aid] = row.astype(np.float32).tobytes()
    return out


def vector_of_rgb(rgb: np.ndarray) -> bytes | None:
    """The float32[384] bytes of one RGB crop, or None when the model fails."""
    try:
        from app.ml import dino

        return np.asarray(dino.embed([rgb]), dtype=np.float32)[0].tobytes()
    except Exception:  # noqa: BLE001 - the background fill takes it from the crop file later
        log.exception("pets: identity vector failed")
        return None


def ensure(conn: sqlite3.Connection, animal_ids: list[int]) -> int:
    if not animal_ids or not available():
        return 0
    marks = ",".join("?" * len(animal_ids))
    missing = [int(r[0]) for r in conn.execute(
        f"SELECT id FROM animals WHERE id IN ({marks}) AND ident IS NULL", [int(a) for a in animal_ids])]
    if not missing:
        return 0
    try:
        got = vectors(missing)
    except Exception:  # noqa: BLE001 - naming falls back to the look embedding
        log.exception("pets: identity vectors failed")
        return 0
    for aid, blob in got.items():
        conn.execute("UPDATE animals SET ident = ? WHERE id = ? AND ident IS NULL", (blob, aid))
    return len(got)


def backfill(conn: sqlite3.Connection, budget_s: float) -> list[int]:
    """Fills missing vectors, newest crops first, for at most budget_s seconds; each batch commits and the model
    names its crops (rename_crops). Returns the crops that got a vector."""
    from app.analysis import pets
    from app.db import tx

    filled: list[int] = []
    if not available():
        return filled
    end = time.monotonic() + budget_s
    # First the owner's crops and every crop the owner took away from a pet (the training rows and the veto),
    # then the automatic ones, which the model names batch by batch once the veto is complete.
    for owners in (1, 0):
        below = 1 << 62  # a crop without a readable file is passed over, not tried again in this pass
        while time.monotonic() < end:
            ids = [int(r[0]) for r in conn.execute(
                "SELECT id FROM animals WHERE ident IS NULL AND embedding IS NOT NULL AND id < ? AND "
                "(assigned_by = 'user' OR id IN (SELECT ref_id FROM guess_rejections WHERE kind = 'animal')) = ? "
                "ORDER BY id DESC LIMIT ?", (below, owners, BATCH))]
            if not ids:
                break
            below = min(ids)
            got = vectors(ids)
            with tx(conn):
                for aid, blob in got.items():
                    conn.execute("UPDATE animals SET ident = ? WHERE id = ? AND ident IS NULL", (blob, aid))
                if got and not owners:
                    pets.rename_crops(conn, list(got))
            filled.extend(got)
    return filled


def _missing(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM animals WHERE ident IS NULL AND embedding IS NOT NULL LIMIT 1").fetchone() is not None


def _matters(conn: sqlite3.Connection, filled: list[int]) -> bool:
    """Whether the filled crops change what the pet model trains on: the owner's crops, crops the owner took away,
    or crops of a species that has a model (its far negatives). Crops of other species do not retrain anything."""
    from app.learning import pets as learned

    species = list((learned.current() or {}).keys())
    for start in range(0, len(filled), 500):
        chunk = filled[start:start + 500]
        marks = ",".join("?" * len(chunk))
        sp_marks = ",".join("?" * len(species)) or "NULL"
        if conn.execute(
            f"SELECT 1 FROM animals WHERE id IN ({marks}) AND (assigned_by = 'user' OR species IN ({sp_marks}) "
            f"OR id IN (SELECT ref_id FROM guess_rejections WHERE kind = 'animal')) LIMIT 1", [*chunk, *species]
        ).fetchone():
            return True
    return False


def _runner() -> None:
    """The background thread: its own connection, nice 19, until every readable crop has its vector."""
    from app import worker
    from app.db import connect

    worker._lower_priority()
    conn = connect()
    try:
        filled = backfill(conn, RUNNER_BUDGET_S)
        n = len(filled)
        if n:
            log.info("pets: %d crops got their identity vector", n)
        if n and _matters(conn, filled):
            # the learning signature holds this stamp, so the pet model trains again with every vector filled
            from app.db import kv_set, tx
            from app.util import now_iso

            with tx(conn):
                kv_set(conn, KV_FILLED, now_iso())
    finally:
        conn.close()


def housekeeping(conn: sqlite3.Connection) -> dict[str, int]:
    """Starts the background vector fill when crops lack their vector (with the workers on), then, when the pet
    model is newer than the last pass, lets it name the automatic crops again."""
    from app.analysis import pets
    from app.db import kv_get, kv_set, tx
    from app.learning import pets as learned

    out = {"pets_renamed": 0}
    if settings.workers_enabled and available() and _missing(conn):
        from app import worker

        if not worker.background_running(RUNNER_KEY):
            worker.run_in_background(RUNNER_KEY, _runner)
    stamp = learned.trained_at()
    if learned.current() is None or stamp is None or (kv_get(conn, KV_NAMED, "") or "") == stamp:
        return out
    with tx(conn):
        moved, _items = pets.rename_crops(conn)
        kv_set(conn, KV_NAMED, stamp)
    out["pets_renamed"] = moved
    if moved:
        log.info("pets: the trained model named %d automatic crops", moved)
    return out
