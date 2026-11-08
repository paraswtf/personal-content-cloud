"""Person guessing from the same hour, same scene and same clothes (docs/ARCHITECTURE.md section 10).

CONTRACT.
- guess_item(conn, item_id) -> int: (re)computes guesses for the item's open boxes: faces with no
  person that are not user-locked, and bodies without a face and without a person. Returns how many
  boxes changed. Writes assigned_by='guess', guess_person_id, guess_score, guess_evidence; clears a
  stale guess when nothing qualifies any more.
- propagate(conn, person_id, item_id) -> int: after the user names, confirms or tags someone, runs
  guess_item on every item in the time window of item_id (or of every item of the person when
  item_id is None) that does not already show the person.
- confirm(conn, face_ids, body_ids) -> int and reject(conn, face_ids, body_ids) -> int: the review
  actions (confirm: person_id set, assigned_by 'user', item_tags row with source 'guess' for a faceless
  body, a 'guess' job per (person, item); a box whose guessed person is already in the item is not
  confirmed and its guess is cleared; reject: guess_rejections row and the guess cleared).
- enqueue_guess(conn, person_id, item_id) and refresh_fts(conn, item_ids): small helpers shared with
  the people API (lazy imports of app.jobs and app.documents.extract).
- apply_window_change(conn) -> int: at startup (ingest.recover_on_startup). When settings.guess_window_h
  differs from the value the open guesses were made with (kv guess_window_h), every guess whose reference
  photo now lies outside the window is withdrawn and its item queued for guessing again.

Scoring for an open box D in item I against a reference box F (face or body of a person with
assigned_by user or auto) in item J:
- J taken within settings.guess_window_h hours (1 h) before or after I, never further, whatever the
  calendar day: clothes are evidence only close in time;
- scene: cos(frame-0 embeddings of I and J) >= settings.guess_scene_sim, or both have GPS within
  settings.guess_gps_m metres;
- clothes: cos(torso(D), torso(F)) >= settings.guess_clothes_sim;
- face: when D is a face, it must not match any other person at >= FACE_CONTRADICTION; a match with
  the candidate >= FACE_WEAK raises the face term above the neutral 0.5;
- score = 0.5 * clothes + 0.3 * scene + 0.2 * face_term; the best person must lead the runner-up by
  >= MARGIN. Persons already present in I and rejected pairs are never proposed. In a photo one
  person is proposed for at most one box (best score first; videos may show a person several times).
- Vault (NSFW) items take no part: they are never references and never get guesses.
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
from app.db import bump_faces_version, faces_version, kv_get, kv_set, tx
from app.util import json_dumps, json_loads, local_date, now_iso, parse_iso

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


def guess_features(clothes: float, scene: float, face_sim: float | None, *, same_day: bool = False,
                   has_gps: bool = False) -> dict[str, Any]:
    return {"clothes": round(float(clothes), 4), "scene_term": round(float(scene), 4),
            "face_term": round(face_term(face_sim), 4), "same_day": bool(same_day), "has_gps": bool(has_gps)}


def guess_score(clothes: float, scene: float, face_sim: float | None, *, same_day: bool = False,
                has_gps: bool = False) -> float:
    """The fixed mix, or the probability of the weights learned from the owner's confirmations and
    rejections once enough of them exist (section 18)."""
    try:
        from app.learning import people as learned_people

        p = learned_people.guess_probability(guess_features(clothes, scene, face_sim, same_day=same_day, has_gps=has_gps))
    except Exception:  # noqa: BLE001 - learning never blocks guessing
        p = None
    if p is not None:
        return float(p)
    return W_CLOTHES * clothes + W_SCENE * scene + W_FACE * face_term(face_sim)


# ------------------------------------------------------------------ reads
def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# The unary '+' keeps SQLite on the items_taken range index. Without ANALYZE statistics it would
# otherwise pick the deleted_at / nsfw equality index and scan every live item.
WINDOW_BEFORE_SQL = (
    "SELECT id, uid, taken_at, gps_lat, gps_lon FROM items WHERE taken_at >= ? AND taken_at < ? AND id != ? "
    "AND +deleted_at IS NULL AND +nsfw = 0 ORDER BY taken_at DESC LIMIT ?"
)
WINDOW_AFTER_SQL = (
    "SELECT id, uid, taken_at, gps_lat, gps_lon FROM items WHERE taken_at >= ? AND taken_at <= ? AND id != ? "
    "AND +deleted_at IS NULL AND +nsfw = 0 ORDER BY taken_at ASC LIMIT ?"
)


def window_items(conn: sqlite3.Connection, item: sqlite3.Row) -> list[sqlite3.Row]:
    """Live library items (other than `item`, never vault items) taken within settings.guess_window_h
    hours before or after it, never further whatever the calendar day: the MAX_WINDOW_ITEMS closest in
    time, ordered by taken_at."""
    t0 = parse_iso(item["taken_at"])
    if t0 is None:
        return []
    hours = float(settings.guess_window_h)
    lo = t0 - timedelta(hours=hours) - timedelta(seconds=1)
    hi = t0 + timedelta(hours=hours) + timedelta(seconds=1)
    limit = MAX_WINDOW_ITEMS
    # Before and from t0 separately, each nearest first, so a dense hour cannot crowd out the
    # neighbours on one side.
    rows = [
        *conn.execute(WINDOW_BEFORE_SQL, (_iso(lo)[:19], item["taken_at"], item["id"], limit)).fetchall(),
        *conn.execute(WINDOW_AFTER_SQL, (item["taken_at"], _iso(hi), item["id"], limit)).fetchall(),
    ]
    out: list[tuple[float, sqlite3.Row]] = []
    for r in rows:
        t = parse_iso(r["taken_at"])
        if t is None:
            continue
        dt = abs((t - t0).total_seconds())
        if dt <= hours * 3600.0:
            out.append((dt, r))
    out.sort(key=lambda x: (x[0], int(x[1]["id"])))
    nearest = [r for _, r in out[:limit]]
    nearest.sort(key=lambda r: (r["taken_at"], int(r["id"])))
    return nearest


def _scene_embeddings(conn: sqlite3.Connection, item_ids: list[int],
                      cache: dict[str, Any] | None = None) -> dict[int, np.ndarray]:
    """{item_id: frame-0 embedding}. `cache` (one propagate run) keeps what was loaded already."""
    if cache is not None:
        known: dict[int, np.ndarray | None] = cache.setdefault("emb", {})
        missing = sorted({int(i) for i in item_ids} - known.keys())
        if missing:
            found = _scene_embeddings(conn, missing)
            for i in missing:
                known[i] = found.get(i)
        return {int(i): known[int(i)] for i in item_ids if known.get(int(i)) is not None}  # type: ignore[misc]
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


def _reference_boxes(conn: sqlite3.Connection, item_ids: list[int], exclude_people: set[int],
                     cache: dict[str, Any] | None = None) -> list[Box]:
    if cache is not None:
        # Guessing never changes who a box belongs to, so one propagate run can reuse the references.
        known: dict[int, list[Box]] = cache.setdefault("refs", {})
        missing = sorted({int(i) for i in item_ids} - known.keys())
        if missing:
            for i in missing:
                known[i] = []
            for ref in _reference_boxes(conn, missing, set()):
                known[ref.item_id].append(ref)
        return [ref for i in sorted({int(i) for i in item_ids}) for ref in known[i]
                if ref.person_id not in exclude_people]
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
def score_box(box: Box, refs: list[Box], scene: dict[int, tuple[float, float | None, float | None, bool, int | None]],
              face_scores: dict[int, float] | None, rejected: set[int],
              item_uids: dict[int, str]) -> tuple[Candidate | None, list[Candidate]]:
    """Best qualifying candidate (or None) and the per-person ranking.
    scene: {item_id: (scene_term, scene_cos, gps_m, same_day, minutes_apart)} for the items that passed the
    scene test."""
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
        scene_term, scene_cos, gps_m, same_day, minutes_apart = scene[ref.item_id]
        score = guess_score(clothes, scene_term, face_sim, same_day=bool(same_day), has_gps=gps_m is not None)
        prev = best_by_person.get(pid)
        if prev is not None and prev.score >= score:
            continue
        best_by_person[pid] = Candidate(pid, score, {
            "clothes": round(clothes, 4),
            "scene_term": round(float(scene_term), 4),
            "scene": round(scene_cos, 4) if scene_cos is not None else None,
            "face": round(face_sim, 4) if face_sim is not None else None,
            "ref_face_id": ref.face_id,
            "ref_body_id": ref.id if ref.kind == "body" else None,
            "ref_item_uid": item_uids.get(ref.item_id),
            "same_day": same_day,
            "minutes_apart": minutes_apart,
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


def _compute(conn: sqlite3.Connection, item: sqlite3.Row, boxes: list[Box],
             cache: dict[str, Any] | None = None) -> dict[tuple[str, int], Candidate | None]:
    results: dict[tuple[str, int], Candidate | None] = {(b.kind, b.id): None for b in boxes}
    if not boxes or not item["taken_at"]:
        return results
    window = window_items(conn, item)
    if not window:
        return results
    present = _present_people(conn, int(item["id"]))
    embs = _scene_embeddings(conn, [int(item["id"]), *[int(r["id"]) for r in window]], cache)
    e0 = embs.get(int(item["id"]))
    day0 = local_date(item["taken_at"], settings.local_timezone)
    t0 = parse_iso(item["taken_at"])
    scene: dict[int, tuple[float, float | None, float | None, bool, int | None]] = {}
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
        t = parse_iso(r["taken_at"])
        minutes = int(round(abs((t - t0).total_seconds()) / 60.0)) if t is not None and t0 is not None else None
        scene[jid] = (float(term), cos, gps_m, bool(same_day), minutes)
    if not scene:
        return results
    refs = _reference_boxes(conn, list(scene), present, cache)
    if not refs:
        return results
    uids = {int(r["id"]): r["uid"] for r in window}
    rejected = _rejections(conn, boxes)
    fscores = _face_scores(conn, boxes)

    def score(b: Box, taken: set[int]) -> Candidate | None:
        return score_box(b, refs, scene, fscores.get(b.id) if b.kind == "face" else None,
                         rejected.get((b.kind, b.id), set()) | taken, uids)[0]

    if item["kind"] == "video":
        for b in boxes:
            results[(b.kind, b.id)] = score(b, set())
        return results
    # One person per photo: the best-scoring (box, person) wins, the other boxes are scored again
    # without that person (the margin then applies among the people left).
    pending = list(boxes)
    taken: set[int] = set()
    while pending:
        scored = [(b, score(b, taken)) for b in pending]
        winners = [(b, c) for b, c in scored if c is not None]
        if not winners:
            break
        b, c = min(winners, key=lambda t: (-t[1].score, t[0].kind, t[0].id))
        results[(b.kind, b.id)] = c
        taken.add(c.person_id)
        pending.remove(b)
    return results


def _advance_index(conn: sqlite3.Connection, nested: bool, base: int | None, new: int | None) -> None:
    """Guess writes touch only guess fields, which the face index does not hold: an index that was
    current at `base` is current at `new` too, so the next sync need not re-read every face."""
    if nested or base is None or new is None:
        return
    from app.analysis.index import get_index

    get_index().advance(base, new)


def guess_item(conn: sqlite3.Connection, item_id: int, cache: dict[str, Any] | None = None) -> int:
    """(Re)compute guesses for the open boxes of one item. Returns how many boxes changed.
    `cache` lets propagate() share scene embeddings and reference boxes between items."""
    item = conn.execute(
        "SELECT id, uid, kind, taken_at, gps_lat, gps_lon, nsfw, deleted_at FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    if item is None or item["deleted_at"] is not None:
        return 0
    boxes = open_boxes(conn, int(item_id))
    if not boxes:
        return 0
    if item["nsfw"]:
        results: dict[tuple[str, int], Candidate | None] = {(b.kind, b.id): None for b in boxes}  # clears stale ones
    else:
        results = _compute(conn, item, boxes, cache)
    changed = 0
    nested = conn.in_transaction
    base = new = None
    with tx(conn):
        base = faces_version(conn)
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
            new = faces_version(conn)
    if changed:
        _advance_index(conn, nested, base, new)
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
        # The anchor too: a box there may still carry a guess for the person who is now present.
        targets.add(int(a["id"]))
        for r in window_items(conn, a):
            targets.add(int(r["id"]))
        if len(targets) >= MAX_WINDOW_ITEMS:
            break
    total = 0
    cache: dict[str, Any] = {}
    for jid in sorted(targets):
        if not conn.execute(
            "SELECT 1 FROM faces WHERE item_id = ? AND person_id IS NULL AND assigned_by IN ('auto', 'guess') "
            "UNION ALL SELECT 1 FROM bodies WHERE item_id = ? AND person_id IS NULL AND face_id IS NULL "
            "AND assigned_by IN ('auto', 'guess') LIMIT 1",
            (jid, jid),
        ).fetchone():
            continue
        total += guess_item(conn, jid, cache)
    log.info("guess propagation for person %s: %d box(es) changed over %d item(s)", person_id, total, len(targets))
    return total


WINDOW_KEY = "guess_window_h"  # kv: the window the open guesses were made with


def apply_window_change(conn: sqlite3.Connection) -> int:
    """Startup check (ingest.recover_on_startup). When settings.guess_window_h differs from the value
    recorded in kv, every open guess whose reference photo now lies outside the window is withdrawn (the
    box goes back to 'auto' with no guess) and a 'guess' job is queued for its item, so the worker
    proposes a nearer reference when one exists. Records the value in force. Returns how many guesses
    were withdrawn."""
    current = f"{float(settings.guess_window_h):g}"
    if kv_get(conn, WINDOW_KEY) == current:
        return 0
    limit_s = float(settings.guess_window_h) * 3600.0
    withdrawn = 0
    with tx(conn):
        for table in ("faces", "bodies"):
            rows = conn.execute(
                f"SELECT b.id, b.item_id, b.guess_person_id, b.guess_evidence, i.taken_at FROM {table} b "
                "JOIN items i ON i.id = b.item_id WHERE b.assigned_by = 'guess' AND b.guess_person_id IS NOT NULL "
                "AND b.person_id IS NULL"
            ).fetchall()
            for r in rows:
                try:
                    ev = json_loads(r["guess_evidence"]) or {}
                except Exception:
                    ev = {}
                ref_uid = ev.get("ref_item_uid") if isinstance(ev, dict) else None
                ref = conn.execute("SELECT taken_at FROM items WHERE uid = ?", (ref_uid,)).fetchone() if ref_uid else None
                t0 = parse_iso(r["taken_at"])
                t1 = parse_iso(ref["taken_at"]) if ref is not None else None
                if t0 is not None and t1 is not None and abs((t1 - t0).total_seconds()) <= limit_s:
                    continue
                conn.execute(
                    f"UPDATE {table} SET assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
                    "guess_evidence = NULL WHERE id = ? AND person_id IS NULL AND assigned_by = 'guess'",
                    (r["id"],),
                )
                enqueue_guess(conn, int(r["guess_person_id"]), int(r["item_id"]))
                withdrawn += 1
        if withdrawn:
            bump_faces_version(conn)
        kv_set(conn, WINDOW_KEY, current)
    if withdrawn:
        log.info("guess window is now %s h: withdrew %d guess(es) made against a reference further away", current, withdrawn)
    return withdrawn


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


def record_outcome(conn: sqlite3.Connection, row: sqlite3.Row, person_id: int, label: int) -> None:
    """Keep the evidence terms of a confirmed (1) or rejected (0) guess for the people learner
    (section 18); the guess fields are cleared right after."""
    try:
        from app.learning import events

        ev = json_loads(row["guess_evidence"], {}) or {}
        scene_term = ev.get("scene_term")
        if scene_term is None:
            scene_term = ev.get("scene") if ev.get("scene") is not None else float(settings.guess_scene_sim)
        features = guess_features(float(ev.get("clothes") or 0.0), float(scene_term), ev.get("face"),
                                  same_day=bool(ev.get("same_day")), has_gps=ev.get("gps_m") is not None)
        features["score"] = ev.get("score")
        events.record(conn, "guess", label, features, item_id=int(row["item_id"]), person_id=int(person_id))
    except Exception:  # noqa: BLE001 - learning never blocks a confirmation
        log.exception("learning: could not record a guess outcome")


def confirm(conn: sqlite3.Connection, face_ids: list[int], body_ids: list[int]) -> int:
    """Accept open guesses: person_id = guess_person_id, assigned_by 'user'. A body without a face
    also gets an item_tags row (source 'guess', removed again when that body is reassigned or deleted).
    A box whose person is already in the item (also through a box confirmed earlier in this call)
    is not confirmed: its guess is cleared. Queues a 'guess' job per (person, item). Returns boxes
    confirmed."""
    pairs: set[tuple[int, int]] = set()
    updated = 0
    dropped = 0
    now = now_iso()
    present: dict[int, set[int]] = {}

    def already(table: str, r: sqlite3.Row, pid: int) -> bool:
        """True (and the stale guess cleared) when pid is already in the box's item."""
        nonlocal dropped
        iid = int(r["item_id"])
        if iid not in present:
            present[iid] = _present_people(conn, iid)
        if pid not in present[iid]:
            present[iid].add(pid)
            return False
        conn.execute(
            f"UPDATE {table} SET assigned_by = CASE WHEN assigned_by = 'guess' THEN 'auto' ELSE assigned_by END, "
            "guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL WHERE id = ?",
            (r["id"],),
        )
        dropped += 1
        return True

    with tx(conn):
        for chunk in _chunks(sorted({int(f) for f in face_ids})):
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT id, item_id, guess_person_id, guess_evidence FROM faces WHERE id IN ({marks}) AND person_id IS NULL "
                "AND guess_person_id IS NOT NULL",
                chunk,
            ).fetchall()
            for r in rows:
                pid = int(r["guess_person_id"])
                if already("faces", r, pid):
                    continue
                record_outcome(conn, r, pid, 1)
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
                f"SELECT id, item_id, face_id, guess_person_id, guess_evidence FROM bodies WHERE id IN ({marks}) "
                "AND person_id IS NULL AND guess_person_id IS NOT NULL",
                chunk,
            ).fetchall()
            for r in rows:
                pid = int(r["guess_person_id"])
                if already("bodies", r, pid):
                    continue
                record_outcome(conn, r, pid, 1)
                conn.execute(
                    "UPDATE bodies SET person_id = ?, assigned_by = 'user', guess_person_id = NULL, guess_score = NULL, "
                    "guess_evidence = NULL WHERE id = ?",
                    (pid, r["id"]),
                )
                if r["face_id"] is None:
                    conn.execute(
                        "INSERT OR IGNORE INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'guess', ?)",
                        (r["item_id"], pid, now),
                    )
                pairs.add((pid, int(r["item_id"])))
                updated += 1
        if updated:
            for pid in {p for p, _ in pairs}:
                conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, pid))
        if updated or dropped:
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
                    f"SELECT id, item_id, guess_person_id, guess_evidence FROM {table} WHERE id IN ({marks}) "
                    "AND guess_person_id IS NOT NULL AND person_id IS NULL",
                    chunk,
                ).fetchall()
                for r in rows:
                    record_outcome(conn, r, int(r["guess_person_id"]), 0)
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
