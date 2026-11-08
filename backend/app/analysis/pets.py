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
- set_species(conn, animal_ids, species) -> SpeciesChange: the owner's species for these boxes (section 22,
  species corrections): species_source 'user', a 'species' learning event per box, a box in a pet of
  another species leaves it and is matched to the pets of the new species, and the automatic look-alikes
  of the corrected crops (cosine >= SPECIES_FOLLOW_SIM, same old species) follow. Call inside a tx.
- set_pet_species(conn, person_id, species) -> SpeciesChange: the same for every crop of a pet, and
  persons.species; a pet of that name and species already there takes the crops (cluster.merge_persons).
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np

from dataclasses import dataclass, field

from app.analysis import cluster
from app.config import settings
from app.util import now_iso

log = logging.getLogger(__name__)

CARRY_IOU = 0.5
POOL_LIMIT = 2000   # unassigned boxes of one species considered for a new pet
SPECIES_FOLLOW_SIM = 0.85  # an automatic crop this close to a corrected one takes its species too
FOLLOW_POOL = 5000  # automatic crops of the old species compared with the corrected ones


@dataclass
class SpeciesChange:
    updated: int = 0          # boxes the owner's choice applied to
    followed: int = 0         # look-alike crops that took the species too
    person_id: int | None = None  # the pet the change ends on (set_pet_species)
    items: set[int] = field(default_factory=set)


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


def _user_species(row: sqlite3.Row) -> bool:
    try:
        return row["species_source"] == "user"
    except (IndexError, KeyError):
        return False


def carry_over(conn: sqlite3.Connection, old_rows: list[sqlite3.Row], new_ids: list[int]) -> int:
    """Boxes of the previous analysis the owner touched keep what they said on the best-overlapping new
    box (IoU >= CARRY_IOU): a user assignment keeps its pet (the new box must be of the same species
    unless the owner also set the species), and a user species (section 22, species corrections)
    goes to the new box whatever the detector said this time."""
    olds = [r for r in old_rows if (r["assigned_by"] == "user" and r["person_id"] is not None) or _user_species(r)]
    if not olds or not new_ids:
        return 0
    marks = ",".join("?" * len(new_ids))
    news = conn.execute(f"SELECT id, species, x, y, w, h FROM animals WHERE id IN ({marks}) AND person_id IS NULL",
                        new_ids).fetchall()
    used: set[int] = set()
    moved = 0
    for o in olds:
        user_species = _user_species(o)
        best: tuple[float, int] | None = None
        for n in news:
            if int(n["id"]) in used or (not user_species and n["species"] != o["species"]):
                continue
            s = cluster.iou((o["x"], o["y"], o["w"], o["h"]),
                            (n["x"], n["y"], n["w"], n["h"]))
            if s >= CARRY_IOU and (best is None or s > best[0]):
                best = (s, int(n["id"]))
        if best is None:
            continue
        if user_species:
            conn.execute("UPDATE animals SET species = ?, species_source = 'user' WHERE id = ?", (o["species"], best[1]))
        if o["assigned_by"] == "user" and o["person_id"] is not None:
            conn.execute("UPDATE animals SET person_id = ?, assigned_by = 'user' WHERE id = ?",
                         (int(o["person_id"]), best[1]))
        used.add(best[1])
        moved += 1
    return moved


# ------------------------------------------------------------------ species corrections (section 22)
def _record_species(conn: sqlite3.Connection, row: sqlite3.Row, species: str, *, followed: bool) -> None:
    try:
        from app.learning import events

        events.record(conn, "species", None, {
            "from": str(row["species"]), "to": species, "score": float(row["det_score"] or 0.0), "followed": followed,
        }, item_id=int(row["item_id"]), person_id=int(row["person_id"]) if row["person_id"] is not None else None)
    except Exception:  # noqa: BLE001 - learning never blocks the action
        log.exception("could not record the species change")


def _leave_other_species(conn: sqlite3.Connection, row: sqlite3.Row, species: str, left: set[int]) -> None:
    """A box in a pet of another species leaves it (the owner's lock on the assignment goes with it: the
    pet was wrong, so the match is decided again)."""
    if row["person_id"] is None:
        return
    pet = conn.execute("SELECT species FROM persons WHERE id = ?", (row["person_id"],)).fetchone()
    if pet is not None and pet["species"] == species:
        return
    conn.execute("UPDATE animals SET person_id = NULL, assigned_by = 'auto' WHERE id = ?", (int(row["id"]),))
    left.add(int(row["person_id"]))


def _followers(conn: sqlite3.Connection, corrected: list[sqlite3.Row], species: str) -> list[sqlite3.Row]:
    """Automatic crops with the old species of a corrected crop within SPECIES_FOLLOW_SIM of it."""
    seeds: dict[str, list[np.ndarray]] = {}
    for r in corrected:
        v = _vec(r["embedding"])
        if v is not None and r["species"] != species:
            seeds.setdefault(str(r["species"]), []).append(v)
    if not seeds:
        return []
    done = {int(r["id"]) for r in corrected}
    out: list[sqlite3.Row] = []
    for old, vecs in seeds.items():
        matrix = np.stack(vecs)
        for r in conn.execute(
            "SELECT id, item_id, person_id, assigned_by, species, species_source, det_score, embedding FROM animals "
            "WHERE species = ? AND species_source = 'auto' AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?",
            (old, FOLLOW_POOL),
        ):
            if int(r["id"]) in done:
                continue
            v = _vec(r["embedding"])
            if v is not None and float(np.max(matrix @ v)) >= SPECIES_FOLLOW_SIM:
                out.append(r)
                done.add(int(r["id"]))
    return out


def set_species(conn: sqlite3.Connection, animal_ids: list[int], species: str) -> SpeciesChange:
    ids = sorted({int(i) for i in animal_ids})
    if not ids:
        return SpeciesChange()
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, item_id, person_id, assigned_by, species, species_source, det_score, embedding FROM animals "
        f"WHERE id IN ({marks}) ORDER BY id", ids,
    ).fetchall()
    change = SpeciesChange()
    left: set[int] = set()
    for r in rows:
        _record_species(conn, r, species, followed=False)
        conn.execute("UPDATE animals SET species = ?, species_source = 'user' WHERE id = ?", (species, int(r["id"])))
        _leave_other_species(conn, r, species, left)
        change.updated += 1
        change.items.add(int(r["item_id"]))
    for r in _followers(conn, rows, species):
        _record_species(conn, r, species, followed=True)
        conn.execute("UPDATE animals SET species = ? WHERE id = ?", (species, int(r["id"])))
        _leave_other_species(conn, r, species, left)
        change.followed += 1
        change.items.add(int(r["item_id"]))
    now = now_iso()
    for item_id in sorted(change.items):
        assign_new_animals(conn, item_id)
    if left:
        conn.execute(f"UPDATE persons SET updated_at = ? WHERE id IN ({','.join('?' * len(left))})", [now, *sorted(left)])
        cluster.prune_empty_persons(conn, left)
    _settle_pet_species(conn, change.items)
    return change


def _settle_pet_species(conn: sqlite3.Connection, item_ids: set[int]) -> None:
    """A pet whose crops all carry one species that differs from persons.species follows them."""
    if not item_ids:
        return
    marks = ",".join("?" * len(item_ids))
    for r in conn.execute(
        f"SELECT p.id, p.species, MIN(a.species) AS lo, MAX(a.species) AS hi FROM persons p JOIN animals a ON a.person_id = p.id "
        f"WHERE p.kind = 'pet' AND p.id IN (SELECT person_id FROM animals WHERE item_id IN ({marks}) AND person_id IS NOT NULL) "
        "GROUP BY p.id", sorted(item_ids),
    ).fetchall():
        if r["lo"] == r["hi"] and r["lo"] != r["species"]:
            conn.execute("UPDATE persons SET species = ?, updated_at = ? WHERE id = ?", (r["lo"], now_iso(), int(r["id"])))


def set_pet_species(conn: sqlite3.Connection, person_id: int, species: str) -> SpeciesChange:
    pet = conn.execute("SELECT id, name, name_key, kind, species FROM persons WHERE id = ?", (person_id,)).fetchone()
    if pet is None or pet["kind"] != "pet":
        raise ValueError("Not a pet")
    rows = conn.execute(
        "SELECT id, item_id, person_id, assigned_by, species, species_source, det_score, embedding FROM animals "
        "WHERE person_id = ? ORDER BY id", (person_id,),
    ).fetchall()
    change = SpeciesChange(person_id=int(person_id))
    for r in rows:
        if r["species"] != species or r["species_source"] != "user":
            _record_species(conn, r, species, followed=False)
        conn.execute("UPDATE animals SET species = ?, species_source = 'user' WHERE id = ?", (species, int(r["id"])))
        change.updated += 1
        change.items.add(int(r["item_id"]))
    for r in _followers(conn, rows, species):
        _record_species(conn, r, species, followed=True)
        conn.execute("UPDATE animals SET species = ? WHERE id = ?", (species, int(r["id"])))
        change.followed += 1
        change.items.add(int(r["item_id"]))
        left: set[int] = set()
        _leave_other_species(conn, r, species, left)
        if left:
            cluster.prune_empty_persons(conn, left)
        assign_new_animals(conn, int(r["item_id"]))
    now = now_iso()
    conn.execute("UPDATE persons SET species = ?, updated_at = ? WHERE id = ?", (species, now, int(person_id)))
    if pet["name"]:
        other = cluster.find_person_by_name(conn, pet["name"], exclude_id=int(person_id), kind="pet")
        if other is not None:
            row = conn.execute("SELECT species FROM persons WHERE id = ?", (other,)).fetchone()
            if row is not None and (row["species"] in (None, species)):
                cluster.merge_persons(conn, int(other), [int(person_id)])
                change.person_id = int(other)
    return change
