"""Grouping faces into people, and person maintenance (merge, suggestions, pruning).

Adapted from the hub's faces/cluster.py (docs/ARCHITECTURE.md section 10).

CONTRACT.
- assign_new_faces(conn, face_ids, exclude_face_ids=None) -> {face_id: person_id | None}
- carry_over(conn, old_faces, new_face_ids) -> int (reprocess keeps names, user locks and rejected
  guesses; rejected_ids / move_rejections / drop_rejections keep guess_rejections on live boxes)
- recluster_unassigned(conn) -> int (never creates people; refreshes the FTS rows it changed)
- merge_persons(conn, target_id, source_ids): moves faces, bodies, item_tags, guesses,
  guess_rejections, merge_dismissals and the cover; deletes the sources. ValueError on bad ids.
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

import numpy as np

from app.config import settings
from app.db import bump_faces_version, tx
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
    """The box ids (kind 'face' or 'body') that have guess_rejections rows."""
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
            "SELECT f.id, f.item_id, f.person_id, f.assigned_by, f.det_score, f.size_px, f.quality, i.kind "
            f"FROM faces f JOIN items i ON i.id = f.item_id WHERE f.id IN ({marks})",
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


def _threshold(row: sqlite3.Row) -> float:
    if is_good_face(row["det_score"], row["size_px"]):
        return settings.cluster_match_threshold
    return settings.cluster_match_threshold_low


def _score(idx: FaceIndex, faces: list[sqlite3.Row], exclude: frozenset[int],
           conn: sqlite3.Connection | None = None) -> list[list[tuple[int, float]]]:
    """For each face: the people it may join, best first (score >= the face's threshold). A person
    whose rejected examples (faces the owner took away from them, section 18) look more like the
    face than the person's own faces do is left out. Call with idx.lock held, after idx.sync."""
    if not faces:
        return []
    embs = idx.embeddings_of([r["id"] for r in faces])
    out: list[list[tuple[int, float]]] = []
    for r, scores in zip(faces, idx.person_scores_many(embs, settings.cluster_topk, exclude)):
        thr = _threshold(r)
        out.append([(pid, score) for pid, score in scores if score >= max(thr, learned_threshold(pid, thr))])
    if conn is not None:
        out = _drop_rejected_lookalikes(conn, embs, out)
    return out


def rejected_examples(conn: sqlite3.Connection, person_ids: list[int] | set[int]) -> dict[int, np.ndarray]:
    """{person_id: (n, 512) embeddings of the faces the owner rejected for that person}."""
    ids = sorted({int(p) for p in person_ids})
    out: dict[int, list[np.ndarray]] = {}
    for chunk in _chunks(ids):
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT g.person_id, f.embedding FROM guess_rejections g JOIN faces f ON f.id = g.ref_id "
            f"WHERE g.kind = 'face' AND g.person_id IN ({marks})", chunk,
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
    now = now_iso()
    for fid in sorted({int(f) for f in face_ids}):
        conn.execute("INSERT OR IGNORE INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (fid, int(person_id), now))


FOLLOW_SIM = 0.55  # a face this close to a face the owner took away looks like the same wrong person


def learned_threshold(person_id: int, default: float) -> float:
    """The person's own match threshold from the owner's confirmations and rejections (section 18),
    else the default."""
    try:
        from app.learning import people as learned_people

        t = learned_people.person_threshold(int(person_id))
    except Exception:  # noqa: BLE001 - learning never blocks clustering
        t = None
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
                if i in done or not is_good_face(r["det_score"], r["size_px"]):
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
                if not (create_people and is_good_face(r["det_score"], r["size_px"])):
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
    exclude_face_ids); join the best person if score >= settings.cluster_match_threshold (good face)
    or settings.cluster_match_threshold_low (other faces). In a photo one person appears at most once.
    Otherwise a good face starts a new unnamed person; other faces stay unassigned.
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


def carry_over(conn: sqlite3.Connection, old_faces: list[sqlite3.Row], new_face_ids: list[int]) -> int:
    """After re-detecting faces on a reprocessed item, copy person_id + assigned_by from old faces to
    the matching new faces (same item): match by normalised-box IoU >= 0.3 with embedding similarity
    >= 0.3, then by similarity >= 0.6, then (user-locked old faces only) by similarity >= 0.4 or
    IoU >= 0.1 with similarity >= 0.3. Each old/new face is used once, best pairs first within each
    pass. Only old faces that had a person or were user-locked are carried; old faces with rejected
    guesses take part too, and only their guess_rejections move to the new face. Returns the number
    of faces whose person was carried."""
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
    old_boxes = [(o["x"], o["y"], o["w"], o["h"]) for o in olds]
    new_boxes = [(n["x"], n["y"], n["w"], n["h"]) for n in news]
    ious = np.array([[_iou(a, b) for b in new_boxes] for a in old_boxes], dtype=np.float64)
    old_e = np.stack([np.frombuffer(o["embedding"], dtype=np.float32) for o in olds])
    new_e = np.stack([np.frombuffer(n["embedding"], dtype=np.float32) for n in news])
    sim = old_e @ new_e.T

    user = np.array([o["assigned_by"] == "user" for o in olds], dtype=bool)[:, None]
    iou_gated = np.where(sim >= CARRY_IOU_MIN_SIM, ious, 0.0)
    loose = (sim >= CARRY_USER_SIM) | ((ious >= CARRY_USER_IOU) & (sim >= CARRY_IOU_MIN_SIM))
    user_pass = np.where(user & loose, sim, -1.0)
    pairs: list[tuple[int, int]] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for matrix, thr in ((iou_gated, CARRY_IOU), (sim, CARRY_SIM), (user_pass, CARRY_IOU_MIN_SIM)):
        cand = np.argwhere(matrix >= thr)
        order = np.argsort(-matrix[cand[:, 0], cand[:, 1]], kind="stable") if len(cand) else []
        for k in order:
            i, j = int(cand[k][0]), int(cand[k][1])
            if i in used_old or j in used_new:
                continue
            used_old.add(i)
            used_new.add(j)
            pairs.append((i, j))
    if not pairs:
        return 0
    carried = 0
    with tx(conn):
        for i, j in pairs:
            o = olds[i]
            if o["person_id"] is not None or o["assigned_by"] == "user":
                conn.execute(
                    "UPDATE faces SET person_id = ?, assigned_by = ? WHERE id = ?",
                    (o["person_id"], o["assigned_by"], news[j]["id"]),
                )
                carried += 1
            if int(o["id"]) in rejected:
                move_rejections(conn, "face", int(o["id"]), int(news[j]["id"]))
        if carried:
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
            "INSERT OR IGNORE INTO item_tags(item_id, person_id, source, created_at) "
            f"SELECT item_id, ?, MIN(source), MIN(created_at) FROM item_tags WHERE person_id IN ({s}) GROUP BY item_id",
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
def _co_occurring(conn: sqlite3.Connection) -> set[tuple[int, int]]:
    """Pairs (a < b) of people who are both in the same photo or document (faces, bodies or tags)."""
    return {
        (r[0], r[1])
        for r in conn.execute(
            "WITH present AS ("
            "  SELECT item_id, person_id FROM faces WHERE person_id IS NOT NULL UNION "
            "  SELECT item_id, person_id FROM bodies WHERE person_id IS NOT NULL UNION "
            "  SELECT item_id, person_id FROM item_tags) "
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
