"""Grouping faces into people, and person maintenance (merge, suggestions).

CONTRACT (implemented by the faces agent). Rules are in docs/SPEC.md section "Clustering".
Every function that changes faces/persons runs inside `with tx(conn):` (nesting is fine) and
calls app.db.bump_faces_version(conn) when it changed anything.

Admin decisions are locked: faces with assigned_by='admin' (including admin-unassigned faces,
person_id NULL) are never touched by automatic assignment or reclustering.
"""
from __future__ import annotations

import heapq
import logging
import sqlite3
from collections import defaultdict

import numpy as np

from app.changelog.capture import claim
from app.config import settings
from app.db import bump_faces_version, tx
from app.faces.index import FaceIndex, get_index
from app.util import now_iso

log = logging.getLogger(__name__)

CARRY_IOU = 0.3
CARRY_IOU_MIN_SIM = 0.3  # an IoU match must also look alike (same person ~0.7+, others <= ~0.3)
CARRY_SIM = 0.6
CARRY_ADMIN_SIM = 0.4  # looser last pass, only for admin-locked old faces
CARRY_ADMIN_IOU = 0.1
_IN_CHUNK = 500
_RECLUSTER_CHUNK = 200  # faces scored per recluster step (whole items, so a photo is never split)
_SUGGEST_CHUNK = 1024  # rows of the centroid similarity matrix computed at a time


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
            "SELECT f.id, f.media_id, f.person_id, f.assigned_by, f.det_score, f.size_px, f.quality, m.kind "
            f"FROM faces f JOIN media m ON m.id = f.media_id WHERE f.id IN ({marks})",
            chunk,
        ).fetchall())
    return rows


def _used_people(conn: sqlite3.Connection, media_ids: list[int], skip_face_ids: set[int]) -> dict[int, set[int]]:
    """People already present (via a face) in each of the given media."""
    used: dict[int, set[int]] = defaultdict(set)
    for chunk in _chunks(media_ids):
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT id, media_id, person_id FROM faces WHERE media_id IN ({marks}) AND person_id IS NOT NULL",
            chunk,
        ):
            if r["id"] not in skip_face_ids:
                used[r["media_id"]].add(r["person_id"])
    return used


def _new_person(conn: sqlite3.Connection) -> int:
    now = now_iso()
    return int(conn.execute(
        "INSERT INTO persons(name, name_key, hidden, created_at, updated_at) VALUES (NULL, NULL, 0, ?, ?)",
        (now, now),
    ).lastrowid)


def _set_person(conn: sqlite3.Connection, idx: FaceIndex | None, face_id: int, person_id: int) -> bool:
    cur = conn.execute(
        "UPDATE faces SET person_id = ? WHERE id = ? AND person_id IS NULL AND assigned_by = 'auto'",
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


def _score(idx: FaceIndex, faces: list[sqlite3.Row], exclude: frozenset[int]) -> list[list[tuple[int, float]]]:
    """For each face: the people it may join, best first (score >= the face's threshold).
    Call with idx.lock held, after idx.sync."""
    if not faces:
        return []
    embs = idx.embeddings_of([r["id"] for r in faces])
    out: list[list[tuple[int, float]]] = []
    for r, scores in zip(faces, idx.person_scores_many(embs, settings.cluster_topk, exclude)):
        thr = _threshold(r)
        out.append([(pid, score) for pid, score in scores if score >= thr])
    return out


def _group_by_media(rows: list[sqlite3.Row]) -> tuple[dict[int, list[sqlite3.Row]], list[int]]:
    """Faces per item (best quality first) and the items, best face first."""
    by_media: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_media[r["media_id"]].append(r)
    for faces in by_media.values():
        faces.sort(key=lambda r: (-r["quality"], r["id"]))
    media_order = sorted(by_media, key=lambda m: (-by_media[m][0]["quality"], m))
    return by_media, media_order


def _eligible(rows: list[sqlite3.Row], exclude: frozenset[int]) -> list[sqlite3.Row]:
    return [r for r in rows if r["person_id"] is None and r["assigned_by"] == "auto" and r["id"] not in exclude]


def _apply_media(
    conn: sqlite3.Connection, idx: FaceIndex | None, faces: list[sqlite3.Row],
    options: list[list[tuple[int, float]]], taken: set[int], create_people: bool, result: dict[int, int | None],
) -> int:
    """Assign the faces of one item from their options. Returns how many got a person."""
    changed = 0
    if faces[0]["kind"] == "image":
        # One person per photo: best-scoring (face, person) pairs win, the rest take their
        # next option.
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
                pid = _new_person(conn)
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
                pid = _new_person(conn)
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
            by_media, media_order = _group_by_media(eligible)
            used = _used_people(conn, media_order, eligible_ids | exclude)
            changed = 0
            for media_id in media_order:
                # Scored item by item: faces assigned (or people started) for earlier items count.
                faces = by_media[media_id]
                options = _score(idx, faces, exclude)
                changed += _apply_media(conn, idx, faces, options, used[media_id], create_people, result)
            if changed:
                bump_faces_version(conn)
            log.debug("auto-assigned %d of %d faces", changed, len(eligible))
    except BaseException:
        # The in-memory person ids (and anything synced inside this transaction) were rolled back.
        idx.invalidate()
        raise
    return result


def assign_new_faces(conn: sqlite3.Connection, face_ids: list[int], exclude_face_ids: set[int] | None = None) -> dict[int, int | None]:
    """Assign freshly inserted auto faces (person_id NULL, assigned_by 'auto') to people.

    For each face (best quality first): score = FaceIndex.person_scores(emb, settings.cluster_topk,
    exclude_face_ids); join the best person if score >= settings.cluster_match_threshold (good face)
    or settings.cluster_match_threshold_low (other faces). In a PHOTO one person appears at most once:
    skip people already used by another face of the same media (take the next best). Otherwise a good
    face starts a new unnamed person; other faces stay unassigned. Returns {face_id: person_id|None}.
    The new faces must be visible to the index (call index.sync(conn) inside the same transaction after
    inserting them, or pass embeddings directly).
    """
    return _auto_assign(conn, face_ids, exclude_face_ids, create_people=True)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def carry_over(conn: sqlite3.Connection, old_faces: list[sqlite3.Row], new_face_ids: list[int]) -> int:
    """After re-detecting faces on a replaced/reprocessed item, copy person_id + assigned_by from old
    faces to the matching new faces (same item): match by normalised-box IoU >= 0.3 with embedding
    similarity >= 0.3, then by embedding similarity >= 0.6, then (admin-locked old faces only) by
    similarity >= 0.4 or IoU >= 0.1 with similarity >= 0.3. Each old/new face is used once, best
    pairs first within each pass. Only old faces that had a person or were admin-locked are carried.
    Returns the number carried."""
    olds = [o for o in old_faces if o["person_id"] is not None or o["assigned_by"] == "admin"]
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
    iou = np.array([[_iou(a, b) for b in new_boxes] for a in old_boxes], dtype=np.float64)
    old_e = np.stack([np.frombuffer(o["embedding"], dtype=np.float32) for o in olds])
    new_e = np.stack([np.frombuffer(n["embedding"], dtype=np.float32) for n in news])
    sim = old_e @ new_e.T

    # 1. Box overlap, but only between faces that also look alike: boxes of overlapping people, or
    #    of a video track whose best frame moved, must not hand a name or a lock to someone else.
    # 2. Embedding alone. 3. Admin-locked old faces only, looser, so an admin decision is not lost
    #    and the face auto-assigned against it.
    admin = np.array([o["assigned_by"] == "admin" for o in olds], dtype=bool)[:, None]
    iou_gated = np.where(sim >= CARRY_IOU_MIN_SIM, iou, 0.0)
    loose = (sim >= CARRY_ADMIN_SIM) | ((iou >= CARRY_ADMIN_IOU) & (sim >= CARRY_IOU_MIN_SIM))
    admin_pass = np.where(admin & loose, sim, -1.0)
    pairs: list[tuple[int, int]] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for matrix, thr in ((iou_gated, CARRY_IOU), (sim, CARRY_SIM), (admin_pass, CARRY_IOU_MIN_SIM)):
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
    with tx(conn):
        for i, j in pairs:
            conn.execute(
                "UPDATE faces SET person_id = ?, assigned_by = ? WHERE id = ?",
                (olds[i]["person_id"], olds[i]["assigned_by"], news[j]["id"]),
            )
        bump_faces_version(conn)
    return len(pairs)


def recluster_unassigned(conn: sqlite3.Connection) -> int:
    """Retry automatic assignment for unassigned auto faces (thresholds as in assign_new_faces,
    never creating people). Returns how many faces got a person.

    Works in steps of about _RECLUSTER_CHUNK faces (whole items). Each step scores its faces with no
    write transaction open (holding only the index lock), then applies the result in one short
    transaction that skips faces no longer unassigned/auto and people that no longer exist, so
    uploads and admin actions are never blocked for long."""
    rows = conn.execute(
        "SELECT id, media_id, quality FROM faces WHERE person_id IS NULL AND assigned_by = 'auto'"
    ).fetchall()
    if not rows:
        return 0
    by_media, media_order = _group_by_media(rows)
    total = 0
    chunk: list[int] = []
    for i, media_id in enumerate(media_order):
        chunk.extend(r["id"] for r in by_media[media_id])
        if len(chunk) >= _RECLUSTER_CHUNK or i == len(media_order) - 1:
            total += _recluster_chunk(conn, chunk)
            chunk = []
    log.info("recluster: %d of %d unassigned faces got a person", total, len(rows))
    return total


def _recluster_chunk(conn: sqlite3.Connection, face_ids: list[int]) -> int:
    idx = get_index()
    empty: frozenset[int] = frozenset()
    # 1. Score, outside any write transaction.
    with idx.lock:
        idx.sync(conn)
        eligible = _eligible(_face_rows(conn, face_ids), empty)
        if not eligible:
            return 0
        options = dict(zip((r["id"] for r in eligible), _score(idx, eligible, empty)))
    pids = sorted({pid for opts in options.values() for pid, _ in opts})
    if not pids:
        return 0
    # 2. Apply in a short transaction, re-checking what may have changed meanwhile.
    result: dict[int, int | None] = {}
    with tx(conn):
        alive: set[int] = set()
        for part in _chunks(pids):
            marks = ",".join("?" * len(part))
            alive.update(r[0] for r in conn.execute(f"SELECT id FROM persons WHERE id IN ({marks})", part))
        still = _eligible(_face_rows(conn, list(options)), empty)
        if not still:
            return 0
        by_media, media_order = _group_by_media(still)
        used = _used_people(conn, media_order, {r["id"] for r in still})
        changed = 0
        for media_id in media_order:
            faces = by_media[media_id]
            opts = [[(pid, sc) for pid, sc in options[r["id"]] if pid in alive] for r in faces]
            # The index is not touched here: the next sync reloads it from the bumped version.
            changed += _apply_media(conn, None, faces, opts, used[media_id], False, result)
        if changed:
            bump_faces_version(conn)
    return changed


def merge_persons(conn: sqlite3.Connection, target_id: int, source_ids: list[int]) -> None:
    """Move all faces and manual tags of source people to target, keep each face's assigned_by,
    give target the first source name if target has none, delete the sources.
    Linked profiles (person_links) move to the target after its own, in source order; a link the
    target already has is skipped. No per-person limit here: a merge never drops links.
    Raises ValueError if target is in sources or any id does not exist."""
    sources = list(dict.fromkeys(int(s) for s in source_ids))
    target_id = int(target_id)
    if not sources:
        raise ValueError("No people to merge")
    if target_id in sources:
        raise ValueError("A person cannot be merged into themselves")
    with tx(conn):
        marks = ",".join("?" * (len(sources) + 1))
        found = {r["id"]: r for r in conn.execute(
            f"SELECT id, name, name_key FROM persons WHERE id IN ({marks})", [target_id, *sources]
        )}
        missing = [i for i in [target_id, *sources] if i not in found]
        if missing:
            raise ValueError(f"Unknown person id(s): {', '.join(map(str, missing))}")
        smarks = ",".join("?" * len(sources))
        conn.execute(f"UPDATE faces SET person_id = ? WHERE person_id IN ({smarks})", [target_id, *sources])
        # Tags and "different people" decisions the target already has are claimed by the merge, so
        # undoing the target's own tag or decision later keeps what the merge relies on.
        for r in conn.execute(
            f"SELECT media_id FROM media_tags WHERE person_id = ? AND media_id IN "
            f"(SELECT media_id FROM media_tags WHERE person_id IN ({smarks}))", [target_id, *sources],
        ).fetchall():
            claim(conn, "media_tags", {"media_id": int(r[0]), "person_id": target_id}, int(r[0]))
        for r in conn.execute(
            "SELECT DISTINCT CASE WHEN person_a IN ({s}) THEN person_b ELSE person_a END AS other "
            "FROM merge_dismissals WHERE person_a IN ({s}) OR person_b IN ({s})".format(s=smarks),
            [*sources, *sources, *sources],
        ).fetchall():
            other = int(r[0])
            if other == target_id or other in sources:
                continue
            lo, hi = min(target_id, other), max(target_id, other)
            if conn.execute("SELECT 1 FROM merge_dismissals WHERE person_a = ? AND person_b = ?", (lo, hi)).fetchone():
                claim(conn, "merge_dismissals", {"person_a": lo, "person_b": hi})
        conn.execute(
            "INSERT OR IGNORE INTO media_tags(media_id, person_id, created_at) "
            f"SELECT media_id, ?, MIN(created_at) FROM media_tags WHERE person_id IN ({smarks}) GROUP BY media_id",
            [target_id, *sources],
        )
        # "Different people" decisions against a source now apply to the merged person.
        conn.execute(
            "INSERT OR IGNORE INTO merge_dismissals(person_a, person_b, created_at) "
            "SELECT MIN(?, other), MAX(?, other), MIN(created_at) FROM ("
            "  SELECT CASE WHEN person_a IN ({s}) THEN person_b ELSE person_a END AS other, created_at "
            "  FROM merge_dismissals WHERE person_a IN ({s}) OR person_b IN ({s})"
            ") WHERE other != ? AND other NOT IN ({s}) GROUP BY other".format(s=smarks),
            [target_id, target_id, *sources, *sources, *sources, target_id, *sources],
        )
        target = found[target_id]
        now = now_iso()
        if target["name"] is None:
            named = next((found[s] for s in sources if found[s]["name"] is not None), None)
            if named is not None:
                conn.execute(
                    "UPDATE persons SET name = ?, name_key = ?, updated_at = ? WHERE id = ?",
                    (named["name"], named["name_key"], now, target_id),
                )
        _move_links(conn, target_id, sources, now)
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, target_id))
        conn.execute(f"DELETE FROM persons WHERE id IN ({smarks})", sources)
        bump_faces_version(conn)
    log.info("merged people %s into %d", sources, target_id)


def _move_links(conn: sqlite3.Connection, target_id: int, sources: list[int], now: str) -> None:
    """Copy the sources' linked profiles to target (call inside a tx, before the sources are
    deleted; their own rows go with them by ON DELETE CASCADE)."""
    pos = int(conn.execute(
        "SELECT COALESCE(MAX(position), -1) FROM person_links WHERE person_id = ?", (target_id,)
    ).fetchone()[0])
    for sid in sources:
        for r in conn.execute(
            "SELECT kind, value, label, created_at FROM person_links WHERE person_id = ? ORDER BY position, id",
            (sid,),
        ).fetchall():
            cur = conn.execute(
                "INSERT OR IGNORE INTO person_links(person_id, kind, value, label, position, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (target_id, r["kind"], r["value"], r["label"], pos + 1, r["created_at"], now),
            )
            if cur.rowcount:
                pos += 1
            else:
                have = conn.execute(
                    "SELECT id FROM person_links WHERE person_id = ? AND kind = ? AND value = ?",
                    (target_id, r["kind"], r["value"]),
                ).fetchone()
                if have is not None:
                    claim(conn, "person_links", {"id": int(have[0])})


def _co_occurring(conn: sqlite3.Connection) -> set[tuple[int, int]]:
    """Pairs (a < b) of people who both have a face in the same photo."""
    return {
        (r[0], r[1])
        for r in conn.execute(
            "SELECT DISTINCT a.person_id, b.person_id FROM faces a "
            "JOIN faces b ON b.media_id = a.media_id AND b.person_id > a.person_id "
            "JOIN media m ON m.id = a.media_id AND m.kind = 'image' "
            "WHERE a.person_id IS NOT NULL"
        )
    }


def merge_suggestions(conn: sqlite3.Connection, limit: int = 30) -> list[tuple[int, int, float]]:
    """Pairs (a_id, b_id, similarity) with a_id < b_id, sorted by centroid cosine similarity desc,
    similarity >= settings.suggest_threshold. Excludes hidden people, dismissed pairs
    (merge_dismissals), and pairs that appear together in the same photo (they are different people)."""
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
        # Centroids and co-occurring pairs only change with faces/persons, so they are kept until
        # the index changes (a new faces_version or an assignment).
        cached = (pids, C, _co_occurring(conn))
        idx.cache_put("suggestions", generation, cached)
    pids, C, co_occurring = cached  # type: ignore[misc]
    if pids.size < 2:
        return []
    hidden = {r[0] for r in conn.execute("SELECT id FROM persons WHERE hidden != 0")}
    if hidden:
        keep = ~np.isin(pids, np.fromiter(hidden, dtype=np.int64))
        pids, C = pids[keep], C[keep]
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
            s = float(sims[k])
            if len(heap) >= limit and s <= heap[0][0]:
                break  # sorted desc: nothing later in this block can enter the heap
            a, b = int(pids[abs_rows[k]]), int(pids[cols[k]])
            if (a, b) in excluded:
                continue
            item = (s, -a, -b)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)
    best = sorted(heap, key=lambda t: (-t[0], -t[1], -t[2]))
    return [(-a, -b, float(s)) for s, a, b in best]
