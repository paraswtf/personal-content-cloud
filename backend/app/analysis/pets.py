"""Pet identities (section 22): animal boxes of one species are matched to the pets of that species
through the centroid of each pet's assigned crops, and unassigned boxes that look alike become a new
unnamed pet. The owner's assignments are locked ('user') and teach each pet its own threshold
(learning.people, section 18).

CONTRACT.
- assign_new_animals(conn, item_id) -> int: assigns the item's unassigned automatic boxes (with an
  embedding); returns the number assigned (to an existing or a new pet). Call inside a tx.
- carry_over(conn, old_rows, new_ids) -> int: user-assigned boxes of a re-analysed item keep their
  pet on the best-overlapping new box (IoU >= CARRY_IOU). Call inside a tx.
- pet_centroids(conn, species) -> {person_id: unit vector} over the assigned crops of that species.
- new_pet(conn, species, name=None) -> person id (persons.kind 'pet').
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np

from app.analysis import cluster
from app.config import settings
from app.util import now_iso

log = logging.getLogger(__name__)

CARRY_IOU = 0.5
POOL_LIMIT = 2000   # unassigned boxes of one species considered for a new pet


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def new_pet(conn: sqlite3.Connection, species: str, name: str | None = None) -> int:
    return cluster.new_person(conn, name, kind="pet", species=species)


def pet_centroids(conn: sqlite3.Connection, species: str) -> dict[int, np.ndarray]:
    sums: dict[int, np.ndarray] = {}
    for r in conn.execute(
        "SELECT a.person_id, a.embedding FROM animals a JOIN persons p ON p.id = a.person_id "
        "WHERE p.kind = 'pet' AND a.species = ? AND a.embedding IS NOT NULL", (species,)
    ):
        v = _vec(r["embedding"])
        if v is None:
            continue
        pid = int(r["person_id"])
        sums[pid] = sums[pid] + v if pid in sums else v.copy()
    out: dict[int, np.ndarray] = {}
    for pid, total in sums.items():
        n = float(np.linalg.norm(total))
        if n > 0:
            out[pid] = (total / n).astype(np.float32)
    return out


def match_threshold(person_id: int) -> float:
    return cluster.learned_threshold(person_id, float(settings.pet_match_threshold))


def assign_new_animals(conn: sqlite3.Connection, item_id: int) -> int:
    rows = conn.execute(
        "SELECT id, species, embedding FROM animals WHERE item_id = ? AND person_id IS NULL "
        "AND assigned_by = 'auto' AND embedding IS NOT NULL ORDER BY det_score DESC, id", (item_id,)
    ).fetchall()
    if not rows:
        return 0
    assigned = 0
    now = now_iso()
    by_species: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_species.setdefault(str(r["species"]), []).append(r)
    for species, boxes in by_species.items():
        centroids = pet_centroids(conn, species)
        pids = list(centroids)
        matrix = np.stack([centroids[p] for p in pids]) if pids else None
        leftovers: list[tuple[int, np.ndarray]] = []
        for r in boxes:
            v = _vec(r["embedding"])
            if v is None:
                continue
            if matrix is not None:
                sims = matrix @ v
                order = np.argsort(-sims)
                hit = next((pids[i] for i in order if float(sims[i]) >= match_threshold(pids[i])), None)
                if hit is not None:
                    conn.execute("UPDATE animals SET person_id = ?, assigned_by = 'auto' WHERE id = ?", (hit, int(r["id"])))
                    conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, hit))
                    # the pet's centroid moves with its new crop
                    c = centroids[hit] + v
                    n = float(np.linalg.norm(c))
                    if n > 0:
                        centroids[hit] = (c / n).astype(np.float32)
                        matrix = np.stack([centroids[p] for p in pids])
                    assigned += 1
                    continue
            leftovers.append((int(r["id"]), v))
        if leftovers:
            assigned += _group_new(conn, species, leftovers, now)
    return assigned


def _group_new(conn: sqlite3.Connection, species: str, boxes: list[tuple[int, np.ndarray]], now: str) -> int:
    """Unassigned boxes of one species that match another unassigned box (of any item) at
    >= pet_new_threshold form a new unnamed pet. Boxes of one item never pair with each other
    (two dogs in one photo are two dogs)."""
    ids = [i for i, _ in boxes]
    marks = ",".join("?" * len(ids))
    pool = conn.execute(
        f"SELECT id, item_id, embedding FROM animals WHERE species = ? AND person_id IS NULL AND assigned_by = 'auto' "
        f"AND embedding IS NOT NULL AND id NOT IN ({marks}) ORDER BY id DESC LIMIT ?", [species, *ids, POOL_LIMIT]
    ).fetchall()
    pool_vecs = [(int(r["id"]), int(r["item_id"]), _vec(r["embedding"])) for r in pool]
    pool_vecs = [(i, it, v) for i, it, v in pool_vecs if v is not None]
    threshold = float(settings.pet_new_threshold)
    assigned = 0
    taken: set[int] = set()
    for aid, v in boxes:
        if aid in taken:
            continue
        best: tuple[float, int] | None = None
        for pid_, _item, pv in pool_vecs:
            if pid_ in taken:
                continue
            s = float(pv @ v)
            if s >= threshold and (best is None or s > best[0]):
                best = (s, pid_)
        if best is None:
            continue
        pet = new_pet(conn, species)
        conn.execute(f"UPDATE animals SET person_id = ?, assigned_by = 'auto' WHERE id IN (?, ?)", (pet, aid, best[1]))
        taken.update({aid, best[1]})
        assigned += 2
        # other pool members close to the new pet's centroid join it too
        c = v + next(pv for pid_, _i, pv in pool_vecs if pid_ == best[1])
        c = c / float(np.linalg.norm(c))
        for pid_, _item, pv in pool_vecs:
            if pid_ in taken:
                continue
            if float(pv @ c) >= threshold:
                conn.execute("UPDATE animals SET person_id = ?, assigned_by = 'auto' WHERE id = ?", (pet, pid_))
                taken.add(pid_)
                assigned += 1
        log.info("pets: new %s identity %d from %d boxes", species, pet, len([t for t in taken]))
    return assigned


def carry_over(conn: sqlite3.Connection, old_rows: list[sqlite3.Row], new_ids: list[int]) -> int:
    """User-assigned animal boxes of the previous analysis keep their pet on the best-overlapping new
    box of the same species (IoU >= CARRY_IOU)."""
    olds = [r for r in old_rows if r["assigned_by"] == "user" and r["person_id"] is not None]
    if not olds or not new_ids:
        return 0
    marks = ",".join("?" * len(new_ids))
    news = conn.execute(f"SELECT id, species, x, y, w, h FROM animals WHERE id IN ({marks}) AND person_id IS NULL",
                        new_ids).fetchall()
    used: set[int] = set()
    moved = 0
    for o in olds:
        best: tuple[float, int] | None = None
        for n in news:
            if int(n["id"]) in used or n["species"] != o["species"]:
                continue
            s = cluster.iou((o["x"], o["y"], o["w"], o["h"]),
                            (n["x"], n["y"], n["w"], n["h"]))
            if s >= CARRY_IOU and (best is None or s > best[0]):
                best = (s, int(n["id"]))
        if best is not None:
            conn.execute("UPDATE animals SET person_id = ?, assigned_by = 'user' WHERE id = ?",
                         (int(o["person_id"]), best[1]))
            used.add(best[1])
            moved += 1
    return moved
