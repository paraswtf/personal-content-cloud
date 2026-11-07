"""Person guessing from same day, same scene and same clothes (docs/ARCHITECTURE.md section 10).

CONTRACT.
- guess_item(conn, item_id) -> int: (re)computes guesses for the item's open boxes: faces with no
  person that are not user-locked, and bodies without a face and without a person. Returns how many
  boxes changed. Writes assigned_by='guess', guess_person_id, guess_score, guess_evidence; clears a
  stale guess when nothing qualifies any more.
- propagate(conn, person_id, item_id) -> int: after the user names, confirms or tags someone, runs
  guess_item on every item in the time window of item_id (or of every item of the person when
  item_id is None) that does not already show the person.
- confirm(conn, face_ids, body_ids) -> int and reject(conn, face_ids, body_ids) -> int: the review
  actions (confirm: person_id set, assigned_by 'user', item_tags row for a faceless body, a 'guess'
  job per (person, item); reject: guess_rejections row and the guess cleared).
- enqueue_guess(conn, person_id, item_id) and refresh_fts(conn, item_ids): small helpers shared with
  the people API (lazy imports of app.jobs and app.documents.extract).

Scoring for an open box D in item I against a reference box F (face or body of a person with
assigned_by user or auto) in item J:
- J within settings.guess_window_h hours of I, or the same local calendar date (settings.local_timezone);
- scene: cos(frame-0 embeddings of I and J) >= settings.guess_scene_sim, or both have GPS within
  settings.guess_gps_m metres;
- clothes: cos(torso(D), torso(F)) >= settings.guess_clothes_sim;
- face: when D is a face, it must not match any other person at >= FACE_CONTRADICTION; a match with
  the candidate >= FACE_WEAK raises the face term above the neutral 0.5;
- score = 0.5 * clothes + 0.3 * scene + 0.2 * face_term; the best person must lead the runner-up by
  >= MARGIN. Persons already present in I and rejected pairs are never proposed.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from app.config import settings
from app.db import bump_faces_version, tx
from app.util import json_dumps, local_date, now_iso, parse_iso

log = logging.getLogger(__name__)

FACE_CONTRADICTION = 0.40
FACE_WEAK = 0.20
FACE_STRONG = 0.45
MARGIN = 0.05
W_CLOTHES, W_SCENE, W_FACE = 0.5, 0.3, 0.2
MAX_WINDOW_ITEMS = 2000
_IN_CHUNK = 500


@dataclass
class Box:
    kind: str  # 'face' | 'body'
    id: int
    item_id: int
    person_id: int | None
    torso: np.ndarray | None
    face: np.ndarray | None = None  # ArcFace embedding (faces only)
    face_id: int | None = None  # bodies: the face inside the box


@dataclass
class Candidate:
    person_id: int
    score: float
    evidence: dict[str, Any]


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def _chunks(ids: list[int], n: int = _IN_CHUNK):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def face_term(face_sim: float | None) -> float:
    """0.5 when there is no face evidence; rises to 1.0 as the candidate match goes from FACE_WEAK
    to FACE_STRONG."""
    if face_sim is None or face_sim < FACE_WEAK:
        return 0.5
    return 0.5 + 0.5 * min(1.0, (face_sim - FACE_WEAK) / (FACE_STRONG - FACE_WEAK))


def guess_score(clothes: float, scene: float, face_sim: float | None) -> float:
    return W_CLOTHES * clothes + W_SCENE * scene + W_FACE * face_term(face_sim)


# ------------------------------------------------------------------ reads
def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def window_items(conn: sqlite3.Connection, item: sqlite3.Row) -> list[sqlite3.Row]:
    """Live items (other than `item`) taken within the guess window or on the same local date."""
    t0 = parse_iso(item["taken_at"])
    if t0 is None:
        return []
    hours = float(settings.guess_window_h)
    lo = t0 - timedelta(hours=max(hours, 24.0)) - timedelta(seconds=1)
    hi = t0 + timedelta(hours=max(hours, 24.0)) + timedelta(seconds=1)
    rows = conn.execute(
        "SELECT id, uid, taken_at, gps_lat, gps_lon FROM items WHERE id != ? AND deleted_at IS NULL "
        "AND taken_at IS NOT NULL AND taken_at >= ? AND taken_at <= ? ORDER BY taken_at LIMIT ?",
        (item["id"], _iso(lo)[:19], _iso(hi), MAX_WINDOW_ITEMS * 2),
    ).fetchall()
    day0 = local_date(item["taken_at"], settings.local_timezone)
    out = []
    for r in rows:
        t = parse_iso(r["taken_at"])
        if t is None:
            continue
        close = abs((t - t0).total_seconds()) <= hours * 3600.0
        same_day = local_date(r["taken_at"], settings.local_timezone) == day0
        if close or same_day:
            out.append(r)
    return out[:MAX_WINDOW_ITEMS]


def _scene_embeddings(conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for chunk in _chunks(sorted(set(item_ids))):
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT item_id, model, embedding FROM item_embeddings WHERE frame_index = 0 AND item_id IN ({marks}) "
            "ORDER BY item_id, model",
            chunk,
        ):
            if int(r["item_id"]) not in out:
                v = _vec(r["embedding"])
                if v is not None:
                    out[int(r["item_id"])] = v
    return out


def _present_people(conn: sqlite3.Connection, item_id: int) -> set[int]:
    return {int(r[0]) for r in conn.execute(
        "SELECT person_id FROM faces WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM bodies WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM item_tags WHERE item_id = ?",
        (item_id, item_id, item_id),
    )}


def open_boxes(conn: sqlite3.Connection, item_id: int) -> list[Box]:
    """Boxes of the item that may receive a guess."""
    boxes: list[Box] = []
    for r in conn.execute(
        "SELECT id, item_id, torso_embedding, embedding FROM faces WHERE item_id = ? AND person_id IS NULL "
        "AND assigned_by IN ('auto', 'guess') ORDER BY id",
        (item_id,),
    ):
        boxes.append(Box("face", int(r["id"]), int(r["item_id"]), None, _vec(r["torso_embedding"]), _vec(r["embedding"])))
    for r in conn.execute(
        "SELECT id, item_id, torso_embedding FROM bodies WHERE item_id = ? AND person_id IS NULL "
        "AND face_id IS NULL AND assigned_by IN ('auto', 'guess') ORDER BY id",
        (item_id,),
    ):
        boxes.append(Box("body", int(r["id"]), int(r["item_id"]), None, _vec(r["torso_embedding"])))
    return boxes


def _reference_boxes(conn: sqlite3.Connection, item_ids: list[int], exclude_people: set[int]) -> list[Box]:
    refs: list[Box] = []
    for chunk in _chunks(sorted(set(item_ids))):
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT id, item_id, person_id, torso_embedding FROM faces WHERE item_id IN ({marks}) "
            "AND person_id IS NOT NULL AND assigned_by IN ('user', 'auto') AND torso_embedding IS NOT NULL",
            chunk,
        ):
            if int(r["person_id"]) in exclude_people:
                continue
            v = _vec(r["torso_embedding"])
            if v is not None:
                refs.append(Box("face", int(r["id"]), int(r["item_id"]), int(r["person_id"]), v, face_id=int(r["id"])))
        for r in conn.execute(
            f"SELECT id, item_id, person_id, face_id, torso_embedding FROM bodies WHERE item_id IN ({marks}) "
            "AND person_id IS NOT NULL AND assigned_by IN ('user', 'auto') AND torso_embedding IS NOT NULL",
            chunk,
        ):
            if int(r["person_id"]) in exclude_people:
                continue
            v = _vec(r["torso_embedding"])
            if v is not None:
                refs.append(Box("body", int(r["id"]), int(r["item_id"]), int(r["person_id"]), v,
                                face_id=r["face_id"]))
    return refs


def _rejections(conn: sqlite3.Connection, boxes: list[Box]) -> dict[tuple[str, int], set[int]]:
    out: dict[tuple[str, int], set[int]] = {}
    for kind in ("face", "body"):
        ids = [b.id for b in boxes if b.kind == kind]
        for chunk in _chunks(ids):
            marks = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT ref_id, person_id FROM guess_rejections WHERE kind = ? AND ref_id IN ({marks})",
                [kind, *chunk],
            ):
                out.setdefault((kind, int(r["ref_id"])), set()).add(int(r["person_id"]))
    return out


def _face_scores(conn: sqlite3.Connection, boxes: list[Box]) -> dict[int, dict[int, float]]:
    """{face box id: {person_id: top-k mean similarity}} from the face index."""
    faces = [b for b in boxes if b.kind == "face" and b.face is not None]
    if not faces:
        return {}
    from app.analysis.index import get_index

    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        scores = idx.person_scores_many(np.stack([b.face for b in faces]), settings.cluster_topk,
                                        frozenset(b.id for b in faces))
    return {b.id: dict(s) for b, s in zip(faces, scores)}


# ------------------------------------------------------------------ scoring
def score_box(box: Box, refs: list[Box], scene: dict[int, tuple[float, float | None, float | None, bool]],
              face_scores: dict[int, float] | None, rejected: set[int],
              item_uids: dict[int, str]) -> tuple[Candidate | None, list[Candidate]]:
    """Best qualifying candidate (or None) and the per-person ranking.
    scene: {item_id: (scene_term, scene_cos, gps_m, same_day)} for the items that passed the scene test."""
    if box.torso is None:
        return None, []
    best_by_person: dict[int, Candidate] = {}
    for ref in refs:
        pid = ref.person_id
        if pid is None or pid in rejected or ref.item_id not in scene or ref.torso is None:
            continue
        if ref.torso.shape != box.torso.shape:
            continue
        clothes = float(np.dot(box.torso, ref.torso))
        if clothes < settings.guess_clothes_sim:
            continue
        face_sim: float | None = None
        if face_scores is not None:
            if any(other != pid and s >= FACE_CONTRADICTION for other, s in face_scores.items()):
                continue
            face_sim = face_scores.get(pid)
        scene_term, scene_cos, gps_m, same_day = scene[ref.item_id]
        score = guess_score(clothes, scene_term, face_sim)
        prev = best_by_person.get(pid)
        if prev is not None and prev.score >= score:
            continue
        best_by_person[pid] = Candidate(pid, score, {
            "clothes": round(clothes, 4),
            "scene": round(scene_cos, 4) if scene_cos is not None else None,
            "face": round(face_sim, 4) if face_sim is not None else None,
            "ref_face_id": ref.face_id,
            "ref_body_id": ref.id if ref.kind == "body" else None,
            "ref_item_uid": item_uids.get(ref.item_id),
            "same_day": same_day,
            "gps_m": round(gps_m, 1) if gps_m is not None else None,
        })
    ranked = sorted(best_by_person.values(), key=lambda c: (-c.score, c.person_id))
    if not ranked:
        return None, ranked
    runner = ranked[1].score if len(ranked) > 1 else 0.0
    if ranked[0].score - runner < MARGIN:
        return None, ranked
    winner = ranked[0]
    winner.evidence["score"] = round(winner.score, 4)
    return winner, ranked


def _compute(conn: sqlite3.Connection, item: sqlite3.Row, boxes: list[Box]) -> dict[tuple[str, int], Candidate | None]:
    results: dict[tuple[str, int], Candidate | None] = {(b.kind, b.id): None for b in boxes}
    if not boxes or not item["taken_at"]:
        return results
    window = window_items(conn, item)
    if not window:
        return results
    present = _present_people(conn, int(item["id"]))
    embs = _scene_embeddings(conn, [int(item["id"]), *[int(r["id"]) for r in window]])
    e0 = embs.get(int(item["id"]))
    day0 = local_date(item["taken_at"], settings.local_timezone)
    scene: dict[int, tuple[float, float | None, float | None, bool]] = {}
    for r in window:
        jid = int(r["id"])
        cos = float(np.dot(e0, embs[jid])) if e0 is not None and jid in embs and embs[jid].shape == e0.shape else None
        gps_m = None
        if None not in (item["gps_lat"], item["gps_lon"], r["gps_lat"], r["gps_lon"]):
            gps_m = haversine_m(float(item["gps_lat"]), float(item["gps_lon"]), float(r["gps_lat"]), float(r["gps_lon"]))
        scene_ok = cos is not None and cos >= settings.guess_scene_sim
        gps_ok = gps_m is not None and gps_m <= settings.guess_gps_m
        if not (scene_ok or gps_ok):
            continue
        term = cos if scene_ok else max(float(cos or 0.0), float(settings.guess_scene_sim))
        same_day = local_date(r["taken_at"], settings.local_timezone) == day0
        scene[jid] = (float(term), cos, gps_m, bool(same_day))
    if not scene:
        return results
    refs = _reference_boxes(conn, list(scene), present)
    if not refs:
        return results
    uids = {int(r["id"]): r["uid"] for r in window}
    rejected = _rejections(conn, boxes)
    fscores = _face_scores(conn, boxes)
    for b in boxes:
        winner, _ = score_box(b, refs, scene, fscores.get(b.id) if b.kind == "face" else None,
                              rejected.get((b.kind, b.id), set()), uids)
        results[(b.kind, b.id)] = winner
    return results


def guess_item(conn: sqlite3.Connection, item_id: int) -> int:
    """(Re)compute guesses for the open boxes of one item. Returns how many boxes changed."""
    item = conn.execute(
        "SELECT id, uid, taken_at, gps_lat, gps_lon, deleted_at FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    if item is None or item["deleted_at"] is not None:
        return 0
    boxes = open_boxes(conn, int(item_id))
    if not boxes:
        return 0
    results = _compute(conn, item, boxes)
    changed = 0
    with tx(conn):
        for (kind, box_id), cand in results.items():
            table = "faces" if kind == "face" else "bodies"
            extra = "" if kind == "face" else " AND face_id IS NULL"
            if cand is None:
                cur = conn.execute(
                    f"UPDATE {table} SET assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
                    f"guess_evidence = NULL WHERE id = ? AND person_id IS NULL AND assigned_by = 'guess'{extra}",
                    (box_id,),
                )
            else:
                alive = conn.execute("SELECT 1 FROM persons WHERE id = ?", (cand.person_id,)).fetchone()
                if alive is None:
                    continue
                cur = conn.execute(
                    f"UPDATE {table} SET assigned_by = 'guess', guess_person_id = ?, guess_score = ?, guess_evidence = ? "
                    f"WHERE id = ? AND person_id IS NULL AND assigned_by IN ('auto', 'guess'){extra} "
                    "AND (guess_person_id IS NOT ? OR guess_score IS NOT ? OR guess_evidence IS NOT ?)",
                    (cand.person_id, round(cand.score, 4), json_dumps(cand.evidence), box_id,
                     cand.person_id, round(cand.score, 4), json_dumps(cand.evidence)),
                )
            changed += cur.rowcount
        if changed:
            bump_faces_version(conn)
    if changed:
        log.info("guess: item %s, %d box(es) changed", item["uid"], changed)
    return changed


def propagate(conn: sqlite3.Connection, person_id: int, item_id: int | None) -> int:
    """Re-run guessing over the time window around item_id (or around every item of the person)."""
    if conn.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
        return 0
    if item_id is not None:
        anchors = conn.execute(
            "SELECT id, uid, taken_at FROM items WHERE id = ? AND deleted_at IS NULL", (item_id,)
        ).fetchall()
    else:
        anchors = conn.execute(
            "SELECT id, uid, taken_at FROM items WHERE deleted_at IS NULL AND taken_at IS NOT NULL AND id IN ("
            "SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
            "UNION SELECT item_id FROM item_tags WHERE person_id = ?) ORDER BY taken_at LIMIT ?",
            (person_id, person_id, person_id, MAX_WINDOW_ITEMS),
        ).fetchall()
    targets: set[int] = set()
    for a in anchors:
        for r in window_items(conn, a):
            targets.add(int(r["id"]))
        if len(targets) >= MAX_WINDOW_ITEMS:
            break
    total = 0
    for jid in sorted(targets):
        if not conn.execute(
            "SELECT 1 FROM faces WHERE item_id = ? AND person_id IS NULL AND assigned_by IN ('auto', 'guess') "
            "UNION ALL SELECT 1 FROM bodies WHERE item_id = ? AND person_id IS NULL AND face_id IS NULL "
            "AND assigned_by IN ('auto', 'guess') LIMIT 1",
            (jid, jid),
        ).fetchone():
            continue
        total += guess_item(conn, jid)
    log.info("guess propagation for person %s: %d box(es) changed over %d item(s)", person_id, total, len(targets))
    return total


# ------------------------------------------------------------------ review actions
def enqueue_guess(conn: sqlite3.Connection, person_id: int, item_id: int | None) -> None:
    """Queue a 'guess' job. Payload keys always in this order so unique=True dedups."""
    try:
        from app import jobs
    except ImportError:
        return
    jobs.enqueue(conn, "guess", {"person_id": int(person_id), "item_id": int(item_id) if item_id is not None else None},
                 unique=True)


def refresh_fts(conn: sqlite3.Connection, item_ids: set[int] | list[int]) -> None:
    """Rewrite the FTS rows of these items (people names changed). Missing documents area: no-op."""
    try:
        from app.documents.extract import refresh_fts as _refresh
    except ImportError:
        return
    for iid in sorted({int(i) for i in item_ids}):
        try:
            _refresh(conn, iid)
        except Exception as e:  # search text is best effort; never fail the user's action
            log.warning("fts refresh failed for item %s: %s", iid, e)


def confirm(conn: sqlite3.Connection, face_ids: list[int], body_ids: list[int]) -> int:
    """Accept open guesses: person_id = guess_person_id, assigned_by 'user'. A body without a face
    also gets an item_tags row. Queues a 'guess' job per (person, item). Returns boxes confirmed."""
    pairs: set[tuple[int, int]] = set()
    updated = 0
    now = now_iso()
    with tx(conn):
        for chunk in _chunks(sorted({int(f) for f in face_ids})):
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT id, item_id, guess_person_id FROM faces WHERE id IN ({marks}) AND person_id IS NULL "
                "AND guess_person_id IS NOT NULL",
                chunk,
            ).fetchall()
            for r in rows:
                pid = int(r["guess_person_id"])
                conn.execute(
                    "UPDATE faces SET person_id = ?, assigned_by = 'user', guess_person_id = NULL, guess_score = NULL, "
                    "guess_evidence = NULL WHERE id = ?",
                    (pid, r["id"]),
                )
                conn.execute(
                    "UPDATE bodies SET person_id = ?, assigned_by = 'user', guess_person_id = NULL, guess_score = NULL, "
                    "guess_evidence = NULL WHERE face_id = ?",
                    (pid, r["id"]),
                )
                pairs.add((pid, int(r["item_id"])))
                updated += 1
        for chunk in _chunks(sorted({int(b) for b in body_ids})):
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT id, item_id, face_id, guess_person_id FROM bodies WHERE id IN ({marks}) AND person_id IS NULL "
                "AND guess_person_id IS NOT NULL",
                chunk,
            ).fetchall()
            for r in rows:
                pid = int(r["guess_person_id"])
                conn.execute(
                    "UPDATE bodies SET person_id = ?, assigned_by = 'user', guess_person_id = NULL, guess_score = NULL, "
                    "guess_evidence = NULL WHERE id = ?",
                    (pid, r["id"]),
                )
                if r["face_id"] is None:
                    conn.execute(
                        "INSERT OR IGNORE INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                        (r["item_id"], pid, now),
                    )
                pairs.add((pid, int(r["item_id"])))
                updated += 1
        if updated:
            for pid in {p for p, _ in pairs}:
                conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, pid))
            bump_faces_version(conn)
    for pid, iid in sorted(pairs):
        enqueue_guess(conn, pid, iid)
    refresh_fts(conn, {i for _, i in pairs})
    return updated


def reject(conn: sqlite3.Connection, face_ids: list[int], body_ids: list[int]) -> int:
    """Turn down open guesses: remember the pair in guess_rejections and clear the guess."""
    updated = 0
    now = now_iso()
    with tx(conn):
        for kind, table, ids in (("face", "faces", face_ids), ("body", "bodies", body_ids)):
            for chunk in _chunks(sorted({int(i) for i in ids})):
                marks = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT id, guess_person_id FROM {table} WHERE id IN ({marks}) AND guess_person_id IS NOT NULL "
                    "AND person_id IS NULL",
                    chunk,
                ).fetchall()
                for r in rows:
                    conn.execute(
                        "INSERT OR IGNORE INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES (?, ?, ?, ?)",
                        (kind, r["id"], r["guess_person_id"], now),
                    )
                    conn.execute(
                        f"UPDATE {table} SET assigned_by = CASE WHEN assigned_by = 'guess' THEN 'auto' ELSE assigned_by END, "
                        "guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL WHERE id = ?",
                        (r["id"],),
                    )
                    updated += 1
        if updated:
            bump_faces_version(conn)
    return updated
