"""Grouping faces into people, and person maintenance (merge, suggestions, pruning).

Adapted from the hub's faces/cluster.py (docs/ARCHITECTURE.md section 10).

CONTRACT.
- assign_new_faces(conn, face_ids, exclude_face_ids=None) -> {face_id: person_id | None}
- carry_over(conn, old_faces, new_face_ids) -> int (reprocess keeps names, user locks and rejected
  guesses, pairing old and new faces with the pure plan_carry; rejected_ids / move_rejections /
  drop_rejections keep guess_rejections on live boxes)
- recluster_unassigned(conn) -> int (never creates people; refreshes the FTS rows it changed)
- merge_persons(conn, target_id, source_ids): moves faces, bodies, item_tags, guesses,
  guess_rejections, merge_dismissals, related_dismissals and the cover; deletes the sources. ValueError on bad ids.
- merge_suggestions(conn, limit) -> [(a, b, similarity)] with a < b
- new_person(conn, name) / find_person_by_name(conn, name) / prune_empty_persons(conn, ids) /
  clear_guesses_for(conn, person_ids): person helpers for the pipeline and the people API.

Every function that changes faces, bodies or persons runs inside `with tx(conn):` (nesting is
fine) and calls app.db.bump_faces_version(conn) when it changed anything.

User decisions are locked: faces with assigned_by='user' (including user-unassigned faces,
person_id NULL) are never touched by automatic assignment or reclustering. 'guess' faces have no
person yet and stay eligible for automatic assignment.
"""
from __future__ import annotations

import heapq
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.db import bump_faces_version, tx
from app.analysis.faces import face_weight, is_clear_face
from app.analysis.index import FaceIndex, get_index
from app.util import clean_name, now_iso

log = logging.getLogger(__name__)

NAME_MAX = 120
CARRY_IOU = 0.3
CARRY_IOU_MIN_SIM = 0.3  # an IoU match must also look alike (same person ~0.7+, others <= ~0.3)
CARRY_SIM = 0.6
CARRY_USER_SIM = 0.4  # looser last pass, only for user-locked old faces
CARRY_USER_IOU = 0.1
_IN_CHUNK = 500
_RECLUSTER_CHUNK = 200  # faces scored per recluster step (whole items, so a photo is never split)
_SUGGEST_CHUNK = 1024  # rows of the centroid similarity matrix computed at a time
AUTO_STATES = ("auto", "guess")  # assigned_by values automatic assignment may change


# ------------------------------------------------------------------ person helpers
def clean_person_name(value: str | None) -> str | None:
    if value is None:
        return None
    name = clean_name(value, max_len=NAME_MAX)
    return name or None


def new_person(conn: sqlite3.Connection, name: str | None = None, *, kind: str = "person",
               species: str | None = None) -> int:
    """Insert a person (unnamed when name is None), or a pet identity (kind 'pet' with its species,
    section 22). Call inside the caller's transaction."""
    now = now_iso()
    name = clean_person_name(name)
    return int(conn.execute(
        "INSERT INTO persons(name, name_key, created_at, updated_at, kind, species) VALUES (?, ?, ?, ?, ?, ?)",
        (name, name.casefold() if name else None, now, now, kind, species if kind == "pet" else None),
    ).lastrowid)


def find_person_by_name(conn: sqlite3.Connection, name: str | None, exclude_id: int | None = None,
                        kind: str | None = "person") -> int | None:
    """The person (of that kind; None = any) whose name matches (case-insensitive), most faces or
    crops first, then lowest id."""
    clean = clean_person_name(name)
    if not clean:
        return None
    row = conn.execute(
        "SELECT p.id FROM persons p WHERE p.name_key = ? AND p.id IS NOT ? AND (? IS NULL OR p.kind = ?) "
        "ORDER BY (SELECT COUNT(*) FROM faces f WHERE f.person_id = p.id) "
        "+ (SELECT COUNT(*) FROM animals a WHERE a.person_id = p.id) DESC, p.id LIMIT 1",
        (clean.casefold(), exclude_id, kind, kind),
    ).fetchone()
    return int(row[0]) if row else None


def find_or_create_person(conn: sqlite3.Connection, name: str | None, *, kind: str = "person",
                          species: str | None = None) -> int | None:
    clean = clean_person_name(name)
    if not clean:
        return None
    found = find_person_by_name(conn, clean, kind=kind)
    return found if found is not None else new_person(conn, clean, kind=kind, species=species)


def clear_guesses_for(conn: sqlite3.Connection, person_ids: list[int] | set[int]) -> int:
    """Drop open guesses that point at these people (call before deleting them). Returns rows changed."""
    ids = sorted({int(p) for p in person_ids})
    n = 0
    for chunk in _chunks(ids):
        marks = ",".join("?" * len(chunk))
        for table in ("faces", "bodies"):
            n += conn.execute(
                f"UPDATE {table} SET guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL, "
                f"assigned_by = CASE WHEN assigned_by = 'guess' THEN 'auto' ELSE assigned_by END "
                f"WHERE guess_person_id IN ({marks})",
                chunk,
            ).rowcount
    return n


def prune_empty_persons(conn: sqlite3.Connection, person_ids: set[int] | list[int]) -> int:
    """Delete unnamed people (and pets) left with no faces, bodies, animal boxes or tags (call inside a tx)."""
    ids = sorted({int(p) for p in person_ids if p is not None})
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    empty = [int(r[0]) for r in conn.execute(
        f"SELECT id FROM persons WHERE id IN ({marks}) AND name IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM faces f WHERE f.person_id = persons.id) "
        "AND NOT EXISTS (SELECT 1 FROM bodies b WHERE b.person_id = persons.id) "
        "AND NOT EXISTS (SELECT 1 FROM animals a WHERE a.person_id = persons.id) "
        "AND NOT EXISTS (SELECT 1 FROM item_tags t WHERE t.person_id = persons.id)",
        ids,
    )]
    if not empty:
        return 0
    clear_guesses_for(conn, empty)
    emarks = ",".join("?" * len(empty))
    conn.execute(f"DELETE FROM persons WHERE id IN ({emarks})", empty)
    bump_faces_version(conn)
    return len(empty)


# ------------------------------------------------------------------ guess rejections
def rejected_ids(conn: sqlite3.Connection, kind: str, ids: list[int]) -> set[int]:
    """The box ids (kind 'face', 'body' or 'animal') that have guess_rejections rows."""
    out: set[int] = set()
    for chunk in _chunks(sorted({int(i) for i in ids})):
        marks = ",".join("?" * len(chunk))
        out.update(int(r[0]) for r in conn.execute(
            f"SELECT DISTINCT ref_id FROM guess_rejections WHERE kind = ? AND ref_id IN ({marks})", [kind, *chunk]))
    return out


def move_rejections(conn: sqlite3.Connection, kind: str, old_id: int, new_id: int) -> None:
    """A re-detected box takes over the rejected guesses of its predecessor."""
    conn.execute("UPDATE OR IGNORE guess_rejections SET ref_id = ? WHERE kind = ? AND ref_id = ?",
                 (int(new_id), kind, int(old_id)))
    conn.execute("DELETE FROM guess_rejections WHERE kind = ? AND ref_id = ?", (kind, int(old_id)))


def remove_faces(conn: sqlite3.Connection, face_ids: list[int]) -> None:
    """Delete face rows the way DELETE /api/faces/{id} does: the rows, their rejections, and the covers
    they were (a body's face_id clears through the schema). Counts, pruning, versions and crop files are
    the caller's. Call inside a tx."""
    ids = sorted({int(f) for f in face_ids})
    if not ids:
        return
    for chunk in _chunks(ids):
        marks = ",".join("?" * len(chunk))
        conn.execute(f"UPDATE persons SET cover_face_id = NULL WHERE cover_face_id IN ({marks})", chunk)
        conn.execute(f"DELETE FROM faces WHERE id IN ({marks})", chunk)
    drop_rejections(conn, "face", ids)


def drop_rejections(conn: sqlite3.Connection, kind: str, ids: list[int]) -> None:
    """Forget the rejections of deleted boxes (guess_rejections has no foreign key to them)."""
    for chunk in _chunks(sorted({int(i) for i in ids})):
        marks = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM guess_rejections WHERE kind = ? AND ref_id IN ({marks})", [kind, *chunk])


# ------------------------------------------------------------------ assignment
def is_good_face(det_score: float, size_px: int) -> bool:
    """Good enough to start a new person: size >= settings.face_good_min_size_px and
    det_score >= settings.face_good_det_score."""
    return int(size_px) >= settings.face_good_min_size_px and float(det_score) >= settings.face_good_det_score


def _chunks(ids: list[int], n: int = _IN_CHUNK):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def _face_rows(conn: sqlite3.Connection, face_ids: list[int]) -> list[sqlite3.Row]:
    rows: list[sqlite3.Row] = []
    for chunk in _chunks(sorted(set(int(f) for f in face_ids))):
        marks = ",".join("?" * len(chunk))
        rows.extend(conn.execute(
            "SELECT f.id, f.item_id, f.person_id, f.assigned_by, f.det_score, f.size_px, f.quality, f.clarity, "
            f"f.sharpness, i.kind FROM faces f JOIN items i ON i.id = f.item_id WHERE f.id IN ({marks})",
            chunk,
        ).fetchall())
    return rows


def _used_people(conn: sqlite3.Connection, item_ids: list[int], skip_face_ids: set[int]) -> dict[int, set[int]]:
    """People already present (via a face) in each of the given items."""
    used: dict[int, set[int]] = defaultdict(set)
    for chunk in _chunks(item_ids):
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT id, item_id, person_id FROM faces WHERE item_id IN ({marks}) AND person_id IS NOT NULL",
            chunk,
        ):
            if r["id"] not in skip_face_ids:
                used[r["item_id"]].add(r["person_id"])
    return used


def _set_person(conn: sqlite3.Connection, idx: FaceIndex | None, face_id: int, person_id: int) -> bool:
    cur = conn.execute(
        "UPDATE faces SET person_id = ?, assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
        "guess_evidence = NULL WHERE id = ? AND person_id IS NULL AND assigned_by IN ('auto', 'guess')",
        (person_id, face_id),
    )
    if cur.rowcount:
        if idx is not None:
            idx.set_person(face_id, person_id)
        return True
    return False


def _clear(row: sqlite3.Row) -> bool:
    return is_clear_face(row["det_score"], row["size_px"], row["clarity"], row["sharpness"])


def _threshold(row: sqlite3.Row) -> float:
    """The cut a face must reach to join a person: a good face (is_good_face) from
    cluster_match_threshold at face weight 1 rising linearly to the stricter
    cluster_match_threshold_low as the weight falls to 0 (a blurred, masked or turned-away face needs
    stronger evidence); any other face cluster_match_threshold_low, as before the weight existed."""
    good, low = float(settings.cluster_match_threshold), float(settings.cluster_match_threshold_low)
    if not is_good_face(row["det_score"], row["size_px"]):
        return low
    w = face_weight(row["det_score"], row["size_px"], row["clarity"], row["sharpness"])
    return max(good, low - (low - good) * w)


def _score(idx: FaceIndex, faces: list[sqlite3.Row], exclude: frozenset[int],
           conn: sqlite3.Connection | None = None, threshold: float | None = None) -> list[list[tuple[int, float]]]:
    """For each face: the people it may join, best first (score >= the face's cut, or `threshold`
    for every face when given). The cut is the default for the face's quality, moved by what the
    owner taught about the person (section 18): a learned threshold replaces the good-face default
    (lower after confirmations and merges, higher after rejections) and a weaker face keeps its
    extra margin on top. A person whose rejected examples (faces the owner took away from them)
    look more like the face than the person's own faces do is left out.
    A face in the child band (section 10, app.analysis.ages: a baby or young child) is scored against
    each person's child-band faces only, its cut is settings.cluster_child_threshold plus the same
    low-quality extra (or `threshold` when that is higher), a learned threshold only ever raises it
    (it was learned from adults), and it needs a lead of CHILD_LEAD over the next person (young
    children, siblings above all, look alike to the face model; settings.child_lead). Call with idx.lock held,
    after idx.sync."""
    if not faces:
        return []
    ids = [r["id"] for r in faces]
    embs = idx.embeddings_of(ids)
    child = idx.child_flags(ids)
    out: list[list[tuple[int, float]]] = []
    scored: list[list[tuple[int, float]]] = []
    good = float(settings.cluster_match_threshold)
    for r, scores, young in zip(faces, idx.person_scores_many(embs, settings.cluster_topk, exclude, child=child),
                                child):
        thr = _threshold(r)
        margin = thr - good  # 0 for a good face, the low-quality extra otherwise
        kept: list[tuple[int, float]] = []
        child_cut = float(settings.cluster_child_threshold) + margin
        for pid, score in scores:
            learned = _learned(pid) if threshold is None else None
            if young:
                cut = child_cut if threshold is None else max(float(threshold), child_cut)
                if learned is not None:
                    cut = max(cut, learned + margin)
            elif threshold is not None:
                cut = float(threshold)
            else:
                cut = thr if learned is None else learned + margin
            if score >= cut:
                kept.append((pid, score))
        out.append(kept)
        scored.append(scores)
    if conn is not None:
        out = _drop_rejected_lookalikes(conn, embs, out)
    for i, young in enumerate(child):
        if young and out[i]:
            # One option only, with the lead over every other person scored (below the cut too): a best that is
            # taken in the photo, or dropped as a rejected look-alike, never falls through to an ambiguous second.
            best_pid, best = out[i][0]
            runner = max((sc for pid, sc in scored[i] if pid != best_pid), default=0.0)
            out[i] = out[i][:1] if best - runner >= float(settings.child_lead) else []
    return out


def rejected_examples(conn: sqlite3.Connection, person_ids: list[int] | set[int]) -> dict[int, np.ndarray]:
    """{person_id: (n, 512) embeddings of the faces the owner rejected for that person} (a face that is the
    person's again, through a merge, is left out)."""
    ids = sorted({int(p) for p in person_ids})
    out: dict[int, list[np.ndarray]] = {}
    for chunk in _chunks(ids):
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT g.person_id, f.embedding FROM guess_rejections g JOIN faces f ON f.id = g.ref_id "
            f"WHERE g.kind = 'face' AND g.person_id IN ({marks}) AND f.person_id IS NOT g.person_id", chunk,
        ):
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape[0]:
                out.setdefault(int(r["person_id"]), []).append(v)
    return {pid: np.stack(vs) for pid, vs in out.items()}


def _drop_rejected_lookalikes(conn: sqlite3.Connection, embs: np.ndarray,
                              options: list[list[tuple[int, float]]]) -> list[list[tuple[int, float]]]:
    pids = {pid for opts in options for pid, _ in opts}
    if not pids:
        return options
    examples = rejected_examples(conn, pids)
    if not examples:
        return options
    out: list[list[tuple[int, float]]] = []
    for emb, opts in zip(embs, options):
        kept = []
        for pid, score in opts:
            ex = examples.get(pid)
            if ex is not None and float(np.max(ex @ emb)) >= score:
                continue  # looks more like someone the owner said is not this person
            kept.append((pid, score))
        out.append(kept)
    return out


def followers(conn: sqlite3.Connection, seed_ids: list[int], person_id: int) -> list[int]:
    """The automatically assigned faces of `person_id` that look like the seed faces the owner just
    took away from that person: closer to a seed (cosine >= FOLLOW_SIM) than to the person's other
    faces (top-k mean, the seeds left out). The owner's own assignments never follow."""
    seeds = [int(s) for s in seed_ids]
    if not seeds:
        return []
    marks = ",".join("?" * len(seeds))
    seed_rows = conn.execute(f"SELECT id, embedding FROM faces WHERE id IN ({marks})", seeds).fetchall()
    if not seed_rows:
        return []
    s = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in seed_rows])
    rows = conn.execute(
        f"SELECT id, assigned_by, embedding FROM faces WHERE person_id = ? AND id NOT IN ({marks}) ORDER BY id",
        [int(person_id), *seeds],
    ).fetchall()
    if not rows:
        return []
    e = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    to_seed = (e @ s.T).max(axis=1)
    among = e @ e.T
    np.fill_diagonal(among, -1.0)
    k = max(1, int(settings.cluster_topk))
    if among.shape[1] > 1:
        top = -np.sort(-among, axis=1)[:, :k]
        rest = np.where(top > -1.0, top, np.nan)
        rest_score = np.nan_to_num(np.nanmean(rest, axis=1), nan=-1.0)
    else:
        rest_score = np.full(e.shape[0], -1.0)
    out = []
    for r, near_seed, own in zip(rows, to_seed, rest_score):
        if r["assigned_by"] == "user":
            continue
        if float(near_seed) >= FOLLOW_SIM and float(near_seed) > float(own):
            out.append(int(r["id"]))
    return out


def record_rejections(conn: sqlite3.Connection, face_ids: list[int], person_id: int) -> None:
    """Remember that these faces are not `person_id` (guess_rejections): guesses skip them and
    matching keeps their look-alikes away from that person. Call inside a tx."""
    record_box_rejections(conn, "face", face_ids, person_id)


def record_box_rejections(conn: sqlite3.Connection, kind: str, box_ids: list[int], person_id: int) -> None:
    """Remember that these boxes (kind 'face', 'body' or 'animal') are not `person_id`: the owner took
    them away from that person or pet (section 10, section 22). Call inside a tx."""
    now = now_iso()
    for bid in sorted({int(b) for b in box_ids}):
        conn.execute("INSERT OR IGNORE INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES (?, ?, ?, ?)",
                     (kind, bid, int(person_id), now))


FOLLOW_SIM = 0.55  # a face this close to a face the owner took away looks like the same wrong person


def _learned(person_id: int) -> float | None:
    """The person's own match threshold from the owner's confirmations, merges and rejections
    (section 18), or None when nothing was learned for them."""
    try:
        from app.learning import people as learned_people

        t = learned_people.person_threshold(int(person_id))
    except Exception:  # noqa: BLE001 - learning never blocks clustering
        return None
    return float(t) if t is not None else None


def learned_threshold(person_id: int, default: float) -> float:
    """_learned, else the default."""
    t = _learned(person_id)
    return float(t) if t is not None else float(default)


def _group_by_item(rows: list[sqlite3.Row]) -> tuple[dict[int, list[sqlite3.Row]], list[int]]:
    """Faces per item (best quality first) and the items, best face first."""
    by_item: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    for faces in by_item.values():
        faces.sort(key=lambda r: (-r["quality"], r["id"]))
    item_order = sorted(by_item, key=lambda m: (-by_item[m][0]["quality"], m))
    return by_item, item_order


def _eligible(rows: list[sqlite3.Row], exclude: frozenset[int]) -> list[sqlite3.Row]:
    return [r for r in rows
            if r["person_id"] is None and r["assigned_by"] in AUTO_STATES and r["id"] not in exclude]


def _apply_item(
    conn: sqlite3.Connection, idx: FaceIndex | None, faces: list[sqlite3.Row],
    options: list[list[tuple[int, float]]], taken: set[int], create_people: bool, result: dict[int, int | None],
) -> int:
    """Assign the faces of one item from their options. Returns how many got a person."""
    changed = 0
    if faces[0]["kind"] != "video":
        # One person per photo: best-scoring (face, person) pairs win, the rest take their next option.
        cands: list[tuple[float, int, int]] = []
        for i, opts in enumerate(options):
            for pid, score in opts:
                if pid not in taken:
                    cands.append((score, i, pid))
        cands.sort(key=lambda c: (-c[0], faces[c[1]]["id"], c[2]))
        done: set[int] = set()
        for score, i, pid in cands:
            if i in done or pid in taken:
                continue
            if _set_person(conn, idx, faces[i]["id"], pid):
                result[faces[i]["id"]] = pid
                taken.add(pid)
                changed += 1
            done.add(i)
        if create_people:
            for i, r in enumerate(faces):
                if i in done or not _clear(r):
                    continue
                pid = new_person(conn)
                if _set_person(conn, idx, r["id"], pid):
                    result[r["id"]] = pid
                    taken.add(pid)
                    changed += 1
    else:
        # Videos may show one person several times.
        for r, opts in zip(faces, options):
            pid: int | None = opts[0][0] if opts else None
            if pid is None:
                if not (create_people and _clear(r)):
                    continue
                pid = new_person(conn)
            if _set_person(conn, idx, r["id"], pid):
                result[r["id"]] = pid
                changed += 1
    return changed


def _auto_assign(
    conn: sqlite3.Connection, face_ids: list[int], exclude_face_ids: set[int] | None, create_people: bool
) -> dict[int, int | None]:
    exclude = frozenset(int(f) for f in (exclude_face_ids or ()))
    idx = get_index()
    result: dict[int, int | None] = {}
    try:
        with tx(conn), idx.lock:
            idx.sync(conn)
            rows = _face_rows(conn, face_ids)
            for r in rows:
                result[r["id"]] = r["person_id"]
            eligible = _eligible(rows, exclude)
            if not eligible:
                return result
            eligible_ids = {r["id"] for r in eligible}
            by_item, item_order = _group_by_item(eligible)
            used = _used_people(conn, item_order, eligible_ids | exclude)
            changed = 0
            for item_id in item_order:
                # Scored item by item: faces assigned (or people started) for earlier items count.
                faces = by_item[item_id]
                options = _score(idx, faces, exclude, conn)
                changed += _apply_item(conn, idx, faces, options, used[item_id], create_people, result)
            if changed:
                bump_faces_version(conn)
            log.debug("auto-assigned %d of %d faces", changed, len(eligible))
    except BaseException:
        # The in-memory person ids (and anything synced inside this transaction) were rolled back.
        idx.invalidate()
        raise
    return result


def assign_new_faces(conn: sqlite3.Connection, face_ids: list[int],
                     exclude_face_ids: set[int] | None = None) -> dict[int, int | None]:
    """Assign freshly inserted auto faces (person_id NULL, assigned_by 'auto') to people.

    For each face (best quality first): score = FaceIndex.person_scores(emb, settings.cluster_topk,
    exclude_face_ids); join the best person if score >= the face's cut (_threshold: from
    settings.cluster_match_threshold for a clear face to settings.cluster_match_threshold_low as its
    face weight falls). In a photo one person appears at most once. Otherwise a clear face
    (faces.is_clear_face) starts a new unnamed person; other faces stay unassigned (a blurred, masked
    or turned-away face never starts a person).
    Returns {face_id: person_id | None}. The faces must be in the database already (the index syncs
    inside the same transaction)."""
    return _auto_assign(conn, face_ids, exclude_face_ids, create_people=True)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two normalised (x, y, w, h) boxes."""
    return _iou(a, b)


def plan_carry(old_boxes: list[tuple[float, float, float, float]], old_emb: np.ndarray, user: list[bool] | np.ndarray,
               new_boxes: list[tuple[float, float, float, float]], new_emb: np.ndarray,
               ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Which new face takes over each old face's decisions when an item is analysed again (pure;
    indexes into the inputs).

    pairs, one-to-one, best pairs first within each pass: (1) box IoU >= CARRY_IOU with similarity
    >= CARRY_IOU_MIN_SIM, (2) similarity >= CARRY_SIM, (3) user-locked old faces only: similarity >=
    CARRY_USER_SIM, or IoU >= CARRY_USER_IOU with similarity >= CARRY_IOU_MIN_SIM. The user-locked old
    faces go through all three passes before any other old face takes a new one, so an owner's choice
    never loses to an automatic one.
    follows, (old, new): an old face left without a pair whose best new face passes pass 1 or 2. A
    video tracked again has fewer, longer tracks, so several old tracks map onto one new track; the
    followers pass on their rejected guesses only."""
    if not len(old_boxes) or not len(new_boxes):
        return [], []
    ious = np.array([[_iou(a, b) for b in new_boxes] for a in old_boxes], dtype=np.float64)
    sim = np.asarray(old_emb, dtype=np.float64) @ np.asarray(new_emb, dtype=np.float64).T
    locked = np.asarray(user, dtype=bool)
    iou_gated = np.where(sim >= CARRY_IOU_MIN_SIM, ious, 0.0)
    strict = ((iou_gated, CARRY_IOU), (sim, CARRY_SIM))
    loose = (sim >= CARRY_USER_SIM) | ((ious >= CARRY_USER_IOU) & (sim >= CARRY_IOU_MIN_SIM))
    pairs: list[tuple[int, int]] = []
    used_old: set[int] = set()
    used_new: set[int] = set()

    def take(matrix: np.ndarray, thr: float, rows: np.ndarray) -> None:
        cand = np.argwhere((matrix >= thr) & rows[:, None])
        if not len(cand):
            return
        for k in np.argsort(-matrix[cand[:, 0], cand[:, 1]], kind="stable"):
            i, j = int(cand[k][0]), int(cand[k][1])
            if i in used_old or j in used_new:
                continue
            used_old.add(i)
            used_new.add(j)
            pairs.append((i, j))

    for matrix, thr in strict:
        take(matrix, thr, locked)
    take(np.where(loose, sim, -1.0), CARRY_IOU_MIN_SIM, locked)
    for matrix, thr in strict:
        take(matrix, thr, ~locked)
    follows: list[tuple[int, int]] = []
    for i in range(len(old_boxes)):
        if i in used_old:
            continue
        ok = (iou_gated[i] >= CARRY_IOU) | (sim[i] >= CARRY_SIM)
        if ok.any():
            follows.append((i, int(np.argmax(np.where(ok, sim[i], -np.inf)))))
    return pairs, follows


def carry_over(conn: sqlite3.Connection, old_faces: list[sqlite3.Row], new_face_ids: list[int]) -> int:
    """After re-detecting faces on a reanalysed item, the old faces' decisions move to the matching
    new faces (same item). plan_carry pairs them; every old face that pairs with or follows a new face
    is one of its claimants, and the new face takes the person and assigned_by of its best claimant:
    one the owner locked first, then one with a named person, then the most alike (a video tracked
    again has fewer, longer tracks, and the owner may have named only one of the old pieces). The
    rejected guesses of every claimant move to it. An automatic person carried onto a face that now
    holds a rejection of that person is dropped, so the face is assigned afresh. Only old faces that
    had a person, were user-locked or had rejected guesses take part. Returns the number of faces
    whose person was carried."""
    rejected = rejected_ids(conn, "face", [int(o["id"]) for o in old_faces])
    olds = [o for o in old_faces
            if o["person_id"] is not None or o["assigned_by"] == "user" or int(o["id"]) in rejected]
    if not olds or not new_face_ids:
        return 0
    news: list[sqlite3.Row] = []
    for chunk in _chunks([int(f) for f in new_face_ids]):
        marks = ",".join("?" * len(chunk))
        news.extend(conn.execute(
            f"SELECT id, x, y, w, h, embedding FROM faces WHERE id IN ({marks}) ORDER BY id", chunk
        ).fetchall())
    if not news:
        return 0
    pairs, follows = plan_carry(
        [(o["x"], o["y"], o["w"], o["h"]) for o in olds],
        np.stack([np.frombuffer(o["embedding"], dtype=np.float32) for o in olds]),
        [o["assigned_by"] == "user" for o in olds],
        [(n["x"], n["y"], n["w"], n["h"]) for n in news],
        np.stack([np.frombuffer(n["embedding"], dtype=np.float32) for n in news]),
    )
    if not pairs and not follows:
        return 0
    claims: dict[int, list[int]] = {}
    for i, j in [*pairs, *follows]:
        claims.setdefault(j, []).append(i)
    pids = sorted({int(o["person_id"]) for o in olds if o["person_id"] is not None})
    named: set[int] = set()
    for chunk in _chunks(pids):
        marks = ",".join("?" * len(chunk))
        named |= {int(r[0]) for r in conn.execute(
            f"SELECT id FROM persons WHERE id IN ({marks}) AND name IS NOT NULL AND TRIM(name) != ''", chunk)}
    old_e = np.stack([np.frombuffer(o["embedding"], dtype=np.float32) for o in olds])
    new_e = np.stack([np.frombuffer(n["embedding"], dtype=np.float32) for n in news])
    carried = 0
    took_rejections: set[int] = set()
    with tx(conn):
        for j, members in claims.items():
            deciding = [i for i in members if olds[i]["person_id"] is not None or olds[i]["assigned_by"] == "user"]
            if not deciding:
                continue
            o = olds[max(deciding, key=lambda i: (olds[i]["assigned_by"] == "user",
                                                    olds[i]["person_id"] is not None and int(olds[i]["person_id"]) in named,
                                                    float(old_e[i] @ new_e[j])))]
            conn.execute(
                "UPDATE faces SET person_id = ?, assigned_by = ?, by_guess = ? WHERE id = ?",
                (o["person_id"], o["assigned_by"], int(o["by_guess"]) if "by_guess" in o.keys() else 0, news[j]["id"]),
            )
            carried += 1
        for i, j in [*pairs, *follows]:
            if int(olds[i]["id"]) in rejected:
                move_rejections(conn, "face", int(olds[i]["id"]), int(news[j]["id"]))
                took_rejections.add(int(news[j]["id"]))
        for chunk in _chunks(sorted(took_rejections)):
            marks = ",".join("?" * len(chunk))
            carried -= conn.execute(
                f"UPDATE faces SET person_id = NULL, assigned_by = 'auto' WHERE id IN ({marks}) "
                "AND assigned_by IN ('auto', 'guess') AND person_id IS NOT NULL AND EXISTS ("
                "SELECT 1 FROM guess_rejections g WHERE g.kind = 'face' AND g.ref_id = faces.id "
                "AND g.person_id = faces.person_id)", chunk,
            ).rowcount
        if claims:
            bump_faces_version(conn)
    return carried


def recluster_unassigned(conn: sqlite3.Connection) -> int:
    """Retry automatic assignment for unassigned auto faces (thresholds as in assign_new_faces,
    never creating people). Returns how many faces got a person.

    Works in steps of about _RECLUSTER_CHUNK faces (whole items). Each step scores its faces with no
    write transaction open (holding only the index lock), then applies the result in one short
    transaction that skips faces no longer unassigned and people that no longer exist."""
    rows = conn.execute(
        "SELECT id, item_id, quality FROM faces WHERE person_id IS NULL AND assigned_by IN ('auto', 'guess')"
    ).fetchall()
    if not rows:
        return 0
    from app.analysis.guess import refresh_fts

    by_item, item_order = _group_by_item(rows)
    total = 0
    chunk: list[int] = []
    for i, item_id in enumerate(item_order):
        chunk.extend(r["id"] for r in by_item[item_id])
        if len(chunk) >= _RECLUSTER_CHUNK or i == len(item_order) - 1:
            changed, items = _recluster_chunk(conn, chunk)
            total += changed
            # items_fts.people lists the names on each item: the new assignments must reach text search.
            refresh_fts(conn, items)
            chunk = []
    log.info("recluster: %d of %d unassigned faces got a person", total, len(rows))
    return total


def _recluster_chunk(conn: sqlite3.Connection, face_ids: list[int]) -> tuple[int, set[int]]:
    """(faces that got a person, the items they are in)."""
    idx = get_index()
    empty: frozenset[int] = frozenset()
    touched: set[int] = set()
    with idx.lock:
        idx.sync(conn)
        eligible = _eligible(_face_rows(conn, face_ids), empty)
        if not eligible:
            return 0, touched
        options = dict(zip((r["id"] for r in eligible), _score(idx, eligible, empty, conn)))
    pids = sorted({pid for opts in options.values() for pid, _ in opts})
    if not pids:
        return 0, touched
    result: dict[int, int | None] = {}
    with tx(conn):
        alive: set[int] = set()
        for part in _chunks(pids):
            marks = ",".join("?" * len(part))
            alive.update(r[0] for r in conn.execute(f"SELECT id FROM persons WHERE id IN ({marks})", part))
        still = _eligible(_face_rows(conn, list(options)), empty)
        if not still:
            return 0, touched
        by_item, item_order = _group_by_item(still)
        used = _used_people(conn, item_order, {r["id"] for r in still})
        changed = 0
        for item_id in item_order:
            faces = by_item[item_id]
            opts = [[(pid, sc) for pid, sc in options[r["id"]] if pid in alive] for r in faces]
            # The index is not touched here: the next sync reloads it from the bumped version.
            n = _apply_item(conn, None, faces, opts, used[item_id], False, result)
            if n:
                touched.add(int(item_id))
            changed += n
        if changed:
            sync_bodies_to_faces(conn, list(by_item))
            bump_faces_version(conn)
    return changed, touched


def sync_bodies_to_faces(conn: sqlite3.Connection, item_ids: list[int]) -> int:
    """Bodies that contain a face follow that face's person (unless the body is user-locked)."""
    n = 0
    for chunk in _chunks(sorted({int(i) for i in item_ids})):
        marks = ",".join("?" * len(chunk))
        n += conn.execute(
            "UPDATE bodies SET person_id = (SELECT f.person_id FROM faces f WHERE f.id = bodies.face_id), "
            "assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL "
            f"WHERE item_id IN ({marks}) AND face_id IS NOT NULL AND assigned_by != 'user' "
            "AND person_id IS NOT (SELECT f.person_id FROM faces f WHERE f.id = bodies.face_id)",
            chunk,
        ).rowcount
    return n


# ------------------------------------------------------------------ merge
def merge_persons(conn: sqlite3.Connection, target_id: int, source_ids: list[int]) -> None:
    """Move all faces, bodies, tags, open guesses and rejections of source people to target, keep each
    face's assigned_by, give target the first source name (and cover) if it has none, delete the
    sources. Raises ValueError if target is in sources or any id does not exist."""
    sources = list(dict.fromkeys(int(s) for s in source_ids))
    target_id = int(target_id)
    if not sources:
        raise ValueError("No people to merge")
    if target_id in sources:
        raise ValueError("A person cannot be merged into themselves")
    with tx(conn):
        marks = ",".join("?" * (len(sources) + 1))
        found = {r["id"]: r for r in conn.execute(
            f"SELECT id, name, name_key, cover_face_id, kind, species FROM persons WHERE id IN ({marks})",
            [target_id, *sources]
        )}
        missing = [i for i in [target_id, *sources] if i not in found]
        if missing:
            raise ValueError(f"Unknown person id(s): {', '.join(map(str, missing))}")
        kinds = {found[i]["kind"] for i in [target_id, *sources]}
        if len(kinds) > 1:
            raise ValueError("A pet and a person cannot be merged")
        if kinds == {"pet"} and len({found[i]["species"] for i in [target_id, *sources]}) > 1:
            raise ValueError("Pets of different species cannot be merged")
        s = ",".join("?" * len(sources))
        conn.execute(f"UPDATE faces SET person_id = ? WHERE person_id IN ({s})", [target_id, *sources])
        conn.execute(f"UPDATE bodies SET person_id = ? WHERE person_id IN ({s})", [target_id, *sources])
        conn.execute(f"UPDATE animals SET person_id = ? WHERE person_id IN ({s})", [target_id, *sources])
        for table in ("faces", "bodies"):
            conn.execute(f"UPDATE {table} SET guess_person_id = ? WHERE guess_person_id IN ({s})",
                         [target_id, *sources])
            # A guess for someone who now already is the box's person is moot.
            conn.execute(
                f"UPDATE {table} SET guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL, "
                "assigned_by = CASE WHEN assigned_by = 'guess' THEN 'auto' ELSE assigned_by END "
                "WHERE guess_person_id = ? AND person_id IS NOT NULL",
                (target_id,),
            )
        conn.execute(
            "INSERT OR IGNORE INTO item_tags(item_id, person_id, source, relation, created_at) "
            f"SELECT item_id, ?, MIN(source), MIN(relation), MIN(created_at) FROM item_tags WHERE person_id IN ({s}) "
            "GROUP BY item_id",
            [target_id, *sources],
        )
        conn.execute(
            "INSERT OR IGNORE INTO guess_rejections(kind, ref_id, person_id, created_at) "
            f"SELECT kind, ref_id, ?, MIN(created_at) FROM guess_rejections WHERE person_id IN ({s}) "
            "GROUP BY kind, ref_id",
            [target_id, *sources],
        )
        # A guess re-pointed to the target on a box that already turned the target down is dropped
        # (after the copy above, so rejections of a source count as rejections of the target).
        for kind, table in (("face", "faces"), ("body", "bodies")):
            conn.execute(
                f"UPDATE {table} SET guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL, "
                "assigned_by = CASE WHEN assigned_by = 'guess' THEN 'auto' ELSE assigned_by END "
                "WHERE guess_person_id = ? AND id IN (SELECT ref_id FROM guess_rejections WHERE kind = ? AND person_id = ?)",
                (target_id, kind, target_id),
            )
        conn.execute(
            "INSERT OR IGNORE INTO merge_dismissals(person_a, person_b, created_at) "
            "SELECT MIN(?, other), MAX(?, other), MIN(created_at) FROM ("
            "  SELECT CASE WHEN person_a IN ({s}) THEN person_b ELSE person_a END AS other, created_at "
            "  FROM merge_dismissals WHERE person_a IN ({s}) OR person_b IN ({s})"
            ") WHERE other != ? AND other NOT IN ({s}) GROUP BY other".format(s=s),
            [target_id, target_id, *sources, *sources, *sources, target_id, *sources],
        )
        conn.execute(f"UPDATE learning_events SET person_id = ? WHERE person_id IN ({s})", [target_id, *sources])
        # "Not related to NAME" (section 10) carries over, as rejected guesses do
        conn.execute(
            "INSERT OR IGNORE INTO related_dismissals(item_id, person_id, created_at) "
            f"SELECT item_id, ?, MIN(created_at) FROM related_dismissals WHERE person_id IN ({s}) GROUP BY item_id",
            [target_id, *sources],
        )
        target = found[target_id]
        now = now_iso()
        if target["name"] is None:
            named = next((found[x] for x in sources if found[x]["name"] is not None), None)
            if named is not None:
                conn.execute(
                    "UPDATE persons SET name = ?, name_key = ? WHERE id = ?",
                    (named["name"], named["name_key"], target_id),
                )
        if target["cover_face_id"] is None:
            cover = next((found[x]["cover_face_id"] for x in sources if found[x]["cover_face_id"] is not None), None)
            if cover is not None:
                conn.execute("UPDATE persons SET cover_face_id = ? WHERE id = ?", (cover, target_id))
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, target_id))
        conn.execute(f"DELETE FROM persons WHERE id IN ({s})", sources)
        bump_faces_version(conn)
    log.info("merged people %s into %d", sources, target_id)


# ------------------------------------------------------------------ suggestions
PULL_FLOOR = 0.30   # a look-alike is never pulled along below this similarity, whatever the owner accepted
PULL_MARGIN = 0.05  # the merged person must lead the runner-up by this much for a look-alike to be pulled along


@dataclass
class Accepted:
    """What a merge or a move teaches (section 18, section 10 for the child band)."""

    accepted: float | None = None  # how alike the adult faces are: may loosen the person and pulls look-alikes
    child_accepted: float | None = None  # how alike the child-band faces are: pulls childhood groups only
    cross_age: bool = False  # child-band faces met only adult faces, or adult faces only child-band ones
    below_floor: float | None = None  # an adult similarity under PULL_FLOOR, kept for the record only
    young: bool = False  # child-band faces were among those added (the person's childhood photos grew)

    def features(self) -> dict:
        """The learning event's fields (app.learning.people reads `accepted`)."""
        return {
            "accepted": None if self.accepted is None else round(float(self.accepted), 4),
            "below_floor": None if self.below_floor is None else round(float(self.below_floor), 4),
            "child_accepted": None if self.child_accepted is None else round(float(self.child_accepted), 4),
            "cross_age": self.cross_age or None,
        }


def _person_face_ids(conn: sqlite3.Connection, person_id: int) -> list[int]:
    return [int(r[0]) for r in conn.execute(
        "SELECT id FROM faces WHERE person_id = ? AND embedding IS NOT NULL", (int(person_id),))]


def _alike(idx: FaceIndex, target_id: int, ids: list[int], exclude: frozenset[int]) -> float:
    """The mean over these faces of their top-k score for the target (0 below MIN_SIM). Child-band faces meet the
    target's child-band faces only (FaceIndex.person_scores_many). Call with idx.lock held, after sync."""
    sims = []
    for scores in idx.person_scores_many(idx.embeddings_of(ids), settings.cluster_topk, exclude,
                                         child=idx.child_flags(ids)):
        sims.append(next((sc for pid, sc in scores if pid == int(target_id)), 0.0))
    return float(np.mean(sims)) if sims else 0.0


def learn_accepted(conn: sqlite3.Connection, target_id: int, groups: list[list[int]],
                   left_out: set[int] | frozenset[int] = frozenset()) -> Accepted:
    """How alike each group of faces is to the target's other faces, band by band (section 10): adult faces
    against the target's adult faces (its child-band faces left out of the reference), child-band faces against
    its child-band faces; per band, the least over the groups. A band with no reference on the target side is a
    cross-age link: the owner's say, nothing to learn. An adult value under PULL_FLOOR is kept as below_floor and
    teaches nothing either (a face the matching sees as unlike the person would loosen them for good). `left_out`:
    faces that are neither group nor reference (look-alikes that came along). Call inside a tx."""
    idx = get_index()
    out = Accepted()
    with idx.lock:
        idx.sync(conn)
        moving = {f for g in groups for f in g} | {int(f) for f in left_out}
        out.young = any(bool(idx.child_flags([int(f) for f in g]).any()) for g in groups if g)
        target_faces = [f for f in _person_face_ids(conn, target_id) if f not in moving]
        if not target_faces:
            return out
        t_child = idx.child_flags(target_faces)
        t_child_ids = frozenset(f for f, c in zip(target_faces, t_child) if c)
        has_adult_ref = len(t_child_ids) < len(target_faces)
        exclude = frozenset(moving)
        for ids in groups:
            ids = [int(f) for f in ids]
            if not ids:
                continue
            flags = idx.child_flags(ids)
            adult = [f for f, c in zip(ids, flags) if not c]
            young = [f for f, c in zip(ids, flags) if c]
            if adult:
                if has_adult_ref:
                    a = _alike(idx, target_id, adult, exclude | t_child_ids)
                    out.accepted = a if out.accepted is None else min(out.accepted, a)
                else:
                    out.cross_age = True
            if young:
                if t_child_ids:
                    c = _alike(idx, target_id, young, exclude)
                    out.child_accepted = c if out.child_accepted is None else min(out.child_accepted, c)
                else:
                    out.cross_age = True
    if out.accepted is not None and out.accepted < PULL_FLOOR:
        out.below_floor, out.accepted = out.accepted, None
    return out


def accept_faces(conn: sqlite3.Connection, target_id: int, face_ids: list[int],
                 came_along: list[int] | tuple[int, ...] = ()) -> tuple[float | None, list[int]]:
    """The owner moved these faces to `target_id` (a bulk move on a selection, section 10): learned like a
    merge (learn_accepted, band by band: the moved faces against the faces the target had before, the
    look-alikes that `came_along` left out). A 'merge' learning event for the target records the
    Accepted fields with {faces, followed}; with an adult `accepted` the unnamed look-alike groups
    follow (follow_merge), and with childhood photos added the childhood groups do (follow_childhood).
    An adult value below PULL_FLOOR, and a cross-age link, teach nothing and pull nothing. The faces
    must already be the target's. Returns (accepted, the pulled person ids). Call inside a tx."""
    ids = sorted({int(f) for f in face_ids})
    target_id = int(target_id)
    if not ids:
        return None, []
    learned = learn_accepted(conn, target_id, [ids], left_out={int(f) for f in came_along})
    if learned.below_floor is not None:
        log.info("bulk move to person %s: accepted %.2f is below %.2f, not learned as a merge", target_id,
                 learned.below_floor, PULL_FLOOR)
    pulled = _follow_learned(conn, target_id, learned)
    try:
        from app.learning import events

        events.record(conn, "merge", 1, {**learned.features(), "faces": len(ids), "followed": len(pulled)},
                      person_id=target_id)
    except Exception:  # noqa: BLE001 - learning never blocks the action
        log.exception("could not record the bulk move for learning")
    return learned.accepted, pulled


def _follow_learned(conn: sqlite3.Connection, target_id: int, learned: Accepted) -> list[int]:
    """The look-alike groups a merge or a move justifies: adult ones at the adult value it taught, then childhood
    ones when childhood photos were added (their own cut, section 10)."""
    pulled = follow_merge(conn, target_id, learned.accepted) if learned.accepted is not None else []
    if learned.young:
        pulled += follow_childhood(conn, target_id, learned.child_accepted)
    return pulled


def follow_childhood(conn: sqlite3.Connection, target_id: int, child_accepted: float | None) -> list[int]:
    """follow_merge for childhood groups (section 10): unnamed groups whose faces are all in the child band,
    pulled when every face matches the target's childhood photos first at >= max(child_accepted,
    settings.cluster_child_threshold), ahead of the next person by PULL_MARGIN (and settings.child_lead, _score)."""
    cut = max(float(child_accepted or 0.0), float(settings.cluster_child_threshold))
    return follow_merge(conn, target_id, cut, childhood=True)


def follow_merge(conn: sqlite3.Connection, target_id: int, accepted: float, *, childhood: bool = False) -> list[int]:
    """The unnamed, automatically grouped people who look like `target_id` at least as much as the
    people the owner just merged into them (section 10): every face of such a person matches the
    target first, at >= max(accepted, PULL_FLOOR, the target's learned threshold), ahead of any other
    person by PULL_MARGIN, with the rejected look-alikes rule applied. Never a named person, one with a
    face, body or tag the owner assigned, one dismissed against the target, or one seen in the same
    photo as the target. With `childhood`, only groups whose faces are all in the child band, and the
    target's learned threshold (taught by adults) is not applied. Merges them into the target; returns
    their ids."""
    target_id = int(target_id)
    cut = max(float(accepted), PULL_FLOOR)
    learned = None if childhood else _learned(target_id)
    if learned is not None:
        cut = max(cut, float(learned))
    dismissed = {int(r[0]) for r in conn.execute(
        "SELECT CASE WHEN person_a = ? THEN person_b ELSE person_a END FROM merge_dismissals "
        "WHERE person_a = ? OR person_b = ?", (target_id, target_id, target_id))}
    together = {int(r[0]) for r in conn.execute(
        "WITH present AS ("
        "  SELECT item_id, person_id FROM faces WHERE person_id IS NOT NULL UNION "
        "  SELECT item_id, person_id FROM bodies WHERE person_id IS NOT NULL UNION "
        "  SELECT item_id, person_id FROM item_tags WHERE relation = 'in'), "
        "mine AS (SELECT p.item_id FROM present p JOIN items i ON i.id = p.item_id AND i.kind != 'video' "
        "         WHERE p.person_id = ?) "
        "SELECT DISTINCT person_id FROM present WHERE item_id IN (SELECT item_id FROM mine) AND person_id != ?",
        (target_id, target_id))}
    candidates = [int(r["id"]) for r in conn.execute(
        "SELECT p.id FROM persons p WHERE p.kind = 'person' AND p.name IS NULL AND p.id != ? "
        "AND NOT EXISTS (SELECT 1 FROM faces f WHERE f.person_id = p.id AND f.assigned_by = 'user') "
        "AND NOT EXISTS (SELECT 1 FROM bodies b WHERE b.person_id = p.id AND b.assigned_by = 'user') "
        "AND NOT EXISTS (SELECT 1 FROM item_tags t WHERE t.person_id = p.id) "
        "AND EXISTS (SELECT 1 FROM faces f WHERE f.person_id = p.id AND f.embedding IS NOT NULL) ORDER BY p.id",
        (target_id,))]
    followers: list[int] = []
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        pool = [c for c in candidates if c not in dismissed and c not in together]
        if childhood:
            pool = [c for c in pool if _all_child(conn, idx, c)]
        # Quick pass, exact: a person follows only when every face of theirs scores the target at >= cut, and no
        # score exceeds the face's best similarity to a face of the target. One matrix product for all of them
        # (scoring each of a thousand unnamed groups took seconds); the full check runs for the rest.
        least = idx.least_best_match(target_id, pool)
        for cid in pool:
            if least.get(cid, -1.0) < cut:
                continue
            ids = [int(r["id"]) for r in conn.execute(
                "SELECT id FROM faces WHERE person_id = ? AND embedding IS NOT NULL", (cid,))]
            rows = _face_rows(conn, ids)
            if not rows:
                continue
            ok = True
            for opts in _score(idx, rows, frozenset(ids), conn, threshold=cut):
                if not opts or opts[0][0] != target_id or (len(opts) > 1 and opts[0][1] - opts[1][1] < PULL_MARGIN):
                    ok = False
                    break
            if ok:
                followers.append(cid)
    if followers:
        merge_persons(conn, target_id, followers)
        log.info("merge into person %s pulled %d look-alike(s) along at >= %.2f", target_id, len(followers), cut)
    return followers


def merge_and_follow(conn: sqlite3.Connection, target_id: int, source_ids: list[int]) -> tuple[float | None, list[int]]:
    """The owner's merge (merge_persons), then the look-alikes it justifies (_follow_learned), and a
    'merge' learning event for the target with what it taught (learn_accepted, section 18: the
    person's threshold loosens to the adult similarity the owner accepted; under PULL_FLOOR, or across
    the child band, nothing is learned and nothing pulled). Returns (accepted similarity, follower ids)."""
    sources = list(dict.fromkeys(int(s) for s in source_ids))
    learned = (learn_accepted(conn, int(target_id), [_person_face_ids(conn, s) for s in sources])
               if sources else Accepted())
    merge_persons(conn, target_id, sources)
    followers = _follow_learned(conn, int(target_id), learned)
    try:
        from app.learning import events

        events.record(conn, "merge", 1, {**learned.features(), "sources": len(sources), "followed": len(followers)},
                      person_id=int(target_id))
    except Exception:  # noqa: BLE001 - learning never blocks the action
        log.exception("could not record the merge for learning")
    return learned.accepted, followers


def _all_child(conn: sqlite3.Connection, idx: FaceIndex, person_id: int) -> bool:
    """Every face of the person is in the child band. Call with idx.lock held, after sync."""
    ids = _person_face_ids(conn, person_id)
    return bool(ids) and bool(idx.child_flags(ids).all())


def _co_occurring(conn: sqlite3.Connection) -> set[tuple[int, int]]:
    """Pairs (a < b) of people who are both in the same photo or document (faces, bodies or tags)."""
    return {
        (r[0], r[1])
        for r in conn.execute(
            "WITH present AS ("
            "  SELECT item_id, person_id FROM faces WHERE person_id IS NOT NULL UNION "
            "  SELECT item_id, person_id FROM bodies WHERE person_id IS NOT NULL UNION "
            "  SELECT item_id, person_id FROM item_tags WHERE relation = 'in') "
            "SELECT DISTINCT a.person_id, b.person_id FROM present a "
            "JOIN present b ON b.item_id = a.item_id AND b.person_id > a.person_id "
            "JOIN items i ON i.id = a.item_id AND i.kind != 'video'"
        )
    }


def merge_suggestions(conn: sqlite3.Connection, limit: int = 30) -> list[tuple[int, int, float]]:
    """Pairs (a_id, b_id, similarity) with a_id < b_id, sorted by centroid cosine similarity desc,
    similarity >= settings.suggest_threshold. Excludes dismissed pairs (merge_dismissals) and pairs
    that appear together in one photo (they are different people)."""
    if limit <= 0:
        return []
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        generation = idx.generation
        cached = idx.cache_get("suggestions")
        if cached is None:
            pids, C = idx.centroids()
    if cached is None:
        cached = (pids, C, _co_occurring(conn))
        idx.cache_put("suggestions", generation, cached)
    pids, C, co_occurring = cached  # type: ignore[misc]
    if pids.size < 2:
        return []
    order = np.argsort(pids)
    pids, C = pids[order], C[order]
    excluded = {(r[0], r[1]) for r in conn.execute("SELECT person_a, person_b FROM merge_dismissals")}
    excluded |= co_occurring
    thr = float(settings.suggest_threshold)
    heap: list[tuple[float, int, int]] = []  # min-heap of the best `limit` pairs
    n = pids.size
    for start in range(0, n, _SUGGEST_CHUNK):
        block = C[start:start + _SUGGEST_CHUNK] @ C.T  # (b, n)
        rows, cols = np.nonzero(block >= thr)
        abs_rows = rows + start
        upper = cols > abs_rows  # pids are sorted, so this gives a_id < b_id
        rows, cols, abs_rows = rows[upper], cols[upper], abs_rows[upper]
        if rows.size == 0:
            continue
        sims = block[rows, cols]
        for k in np.argsort(-sims, kind="stable"):
            sv = float(sims[k])
            if len(heap) >= limit and sv <= heap[0][0]:
                break  # sorted desc: nothing later in this block can enter the heap
            a, b = int(pids[abs_rows[k]]), int(pids[cols[k]])
            if (a, b) in excluded:
                continue
            item = (sv, -a, -b)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)
    best = sorted(heap, key=lambda t: (-t[0], -t[1], -t[2]))
    return [(-a, -b, float(sv)) for sv, a, b in best]


def childhood_suggestions(conn: sqlite3.Connection, limit: int = 30) -> list[tuple[int, int, float]]:
    """Pairs (a_id, b_id, similarity), a_id < b_id, of people whose childhood photos (faces in the child band,
    section 10) look alike: the cosine of their child-band centroids >= settings.child_suggest_threshold (0.40,
    below the child cut as the adult suggestions sit below theirs). Leaves out dismissed pairs and pairs seen in
    one photo. Best first. (The suggestions route shows such a pair as a childhood one even when
    merge_suggestions has it too.)"""
    if limit <= 0:
        return []
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        pids, C = idx.centroids(child=True)
    if pids.size < 2:
        return []
    order = np.argsort(pids)
    pids, C = pids[order], C[order]
    excluded = {(r[0], r[1]) for r in conn.execute("SELECT person_a, person_b FROM merge_dismissals")}
    excluded |= _co_occurring(conn)
    kinds = {int(r[0]): r[1] for r in conn.execute(
        f"SELECT id, kind FROM persons WHERE id IN ({','.join('?' * pids.size)})", pids.tolist())}
    sims = C @ C.T
    rows, cols = np.nonzero(np.triu(sims >= float(settings.child_suggest_threshold), k=1))
    pairs = [(int(pids[i]), int(pids[j]), float(sims[i, j])) for i, j in zip(rows.tolist(), cols.tolist())]
    pairs = [t for t in pairs if (t[0], t[1]) not in excluded and kinds.get(t[0]) == "person" and kinds.get(t[1]) == "person"]
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    return pairs[:limit]


CHILDHOOD_SHARE = 0.5  # a group is a childhood group when this much of its face weight is in the child band


def childhood_candidates(conn: sqlite3.Connection, person_id: int, limit: int = 3) -> list[tuple[int, float]]:
    """For an unnamed group of childhood photos (CHILDHOOD_SHARE of its face weight in the child band, section
    10): the named people it may be as a child, best first. First those whose own childhood photos look like the
    group's (child-band centroids at >= settings.child_suggest_threshold), by that; then the rest by the cosine of
    the group's child-band centroid and the person's adult centroid. No bar: the face model links a young child to
    their adult self only weakly (about 0.2 to 0.35), so this ranks for the owner, who decides. Leaves out people
    seen in one photo with the group and pairs the owner dismissed. [] for a named person or a group that is not
    mostly childhood photos."""
    person_id = int(person_id)
    row = conn.execute("SELECT name, kind FROM persons WHERE id = ?", (person_id,)).fetchone()
    if row is None or row["name"] or row["kind"] != "person" or limit <= 0:
        return []
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        ids = _person_face_ids(conn, person_id)
        if not ids:
            return []
        flags = idx.child_flags(ids)
        weights = idx.weights_of(ids)
        total = float(weights.sum())
        if total <= 0 or float(weights[flags].sum()) / total < CHILDHOOD_SHARE:
            return []
        gp, gc = idx.centroids([person_id], child=True)
        ap, ac = idx.centroids(child=False)
        cp, cc = idx.centroids(child=True)
    if gp.size == 0 or (ap.size == 0 and cp.size == 0):
        return []
    named = {int(r[0]) for r in conn.execute("SELECT id FROM persons WHERE kind = 'person' AND name IS NOT NULL")}
    dismissed = {int(r[0]) for r in conn.execute(
        "SELECT CASE WHEN person_a = ? THEN person_b ELSE person_a END FROM merge_dismissals "
        "WHERE person_a = ? OR person_b = ?", (person_id, person_id, person_id))}
    together = {b if a == person_id else a for a, b in _co_occurring(conn) if person_id in (a, b)}
    def fits(pid: int) -> bool:
        return pid in named and pid not in dismissed and pid not in together and pid != person_id

    young = {int(p): float(sv) for p, sv in zip(cp.tolist(), (cc @ gc[0]).tolist())
             if fits(int(p)) and float(sv) >= float(settings.child_suggest_threshold)}
    grown = {int(p): float(sv) for p, sv in zip(ap.tolist(), (ac @ gc[0]).tolist()) if fits(int(p))}
    ranked = [(1, sv, pid) for pid, sv in young.items()]
    ranked += [(0, sv, pid) for pid, sv in grown.items() if pid not in young]
    ranked.sort(reverse=True)
    return [(pid, sv) for _, sv, pid in ranked[:limit]]
