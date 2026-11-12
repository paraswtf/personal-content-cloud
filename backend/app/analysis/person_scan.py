"""Look again with recognition trained on one person (docs/ARCHITECTURE.md section 10).

app.learning.person_model holds the model; this module applies it to faces that are not on the person:

- scan(conn, person_id): every face that is not theirs, not the owner's own say, in a live item outside the vault,
  not turned down for them before, in an item they are not in yet. A loose face or one in an unnamed group that the
  model gives GUESS_P or more becomes a GUESS (basis 'face'), one per photo. guess_item writes the guess (the scan
  runs it for every photo it picked); the scan itself only takes a face out of its unnamed group first, and only
  when the guess will stand (the group goes when it is left empty; the evidence remembers it, was_person_id). Faces
  the app put on other NAMED people are never touched: the sure ones are listed in the result ("elsewhere").
  Nothing is ever tagged here.
- merge_item(conn, item, boxes, results): the same judgement for the open faces of one item, merged into what the
  clothes and trait paths found (guess.guess_item calls it), so a model guess is recomputed on every run instead of
  being erased, and a new photo of the person gets one when it is analysed.
- regroup(conn, faces): a face taken out of a group whose guess was turned down or taken back goes back into that
  group (or, the group gone, a clear face starts an unnamed person of its own). No matching step, never a named
  person: exactly what the scan undid.
- withdraw(conn, person_id): takes the model's open guesses back and pauses it (no new guesses until the next scan).
- start / state: one background thread (worker.run_in_background) for training and scanning, with its progress.

A candidate is dropped when a face the owner took off the person is nearer to it than the person's own faces are
(cosine >= max(top-3, VETO_SIM)), and when another named person scores it at least as high as the person does,
unless that person's own model says it is not them.
"""
from __future__ import annotations

import contextvars
import logging
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.db import bump_faces_version, tx
from app.learning import person_model
from app.util import json_dumps, json_loads, now_iso

log = logging.getLogger(__name__)

RUNNER_KEY = "person-model"
BASIS = "face"
VETO_SIM = 0.6
SCORE_CHUNK = 2048
WRITE_CHUNK = 200
ELSEWHERE_KEPT = 200
_IN_CHUNK = 500
# CASE keeps json_extract away from malformed JSON (AND does not promise an order in SQLite)
BASIS_SQL = "(CASE WHEN json_valid({col}) THEN json_extract({col}, '$.basis') END)"

Progress = Callable[[int, int, str], None]


def evidence(model: person_model.Model, p: float, top3: float, *, was_person_id: int | None = None,
             also: str | None = None) -> dict[str, Any]:
    """The stored evidence of a model guess: byte-stable (rounded values, one key order)."""
    ev: dict[str, Any] = {"basis": BASIS, "score": round(float(p), 4), "face": round(float(top3), 4),
                          "p": round(float(p), 4),
                          "model": {"rev": person_model.MODEL_REV, "trained_at": model.trained_at}}
    if was_person_id is not None:
        ev["was_person_id"] = int(was_person_id)
    if also:
        ev["also"] = also
    return ev


def _chunks(seq: list[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _removed(conn: sqlite3.Connection, person_id: int) -> np.ndarray | None:
    """Vectors of the faces the owner took off the person (and that are not theirs again)."""
    from app.analysis import cluster

    ex = cluster.rejected_examples(conn, [int(person_id)]).get(int(person_id))
    if ex is None or not len(ex):
        return None
    ex = np.asarray(ex, dtype=np.float32)
    return ex / np.maximum(np.linalg.norm(ex, axis=1, keepdims=True), 1e-9)


def _named_people(conn: sqlite3.Connection) -> set[int]:
    return {int(r[0]) for r in conn.execute(
        "SELECT id FROM persons WHERE name IS NOT NULL AND name != '' AND kind = 'person'")}


class _Judge:
    """One model's verdict on faces: probability, top-3 and the veto of the removed look-alikes."""

    def __init__(self, conn: sqlite3.Connection, model: person_model.Model) -> None:
        self.model = model
        self.removed = _removed(conn, model.person_id)

    def score(self, embs: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(p, top-3, vetoed) per row. The face index must be synced."""
        embs = np.asarray(embs, dtype=np.float32).reshape(-1, person_model.DIM)
        top3 = self.model.top3(embs)
        p = self.model.probability(embs, weights, top3=top3)
        veto = np.zeros(embs.shape[0], dtype=bool)
        if self.removed is not None and embs.shape[0]:
            near = (embs @ self.removed.T).max(axis=1)
            veto = near >= np.maximum(top3, VETO_SIM)
        return p, top3, veto


def _better_elsewhere(idx: Any, person_id: int, face_ids: list[int], embs: np.ndarray, weights: np.ndarray,
                      top3: np.ndarray, named: set[int]) -> np.ndarray:
    """Per face: another named person scores it at least as high as its top-3 for this person, and that person's
    own model (when they have one) does not say otherwise. Call with idx.lock held, after sync."""
    out = np.zeros(len(face_ids), dtype=bool)
    if not face_ids:
        return out
    scores = idx.person_scores_many(embs, settings.cluster_topk, frozenset(int(f) for f in face_ids),
                                    child=idx.child_flags(face_ids))
    for i, ranked in enumerate(scores):
        for other, sc in ranked:
            if other == person_id or other not in named or sc < float(top3[i]):
                continue
            theirs = person_model.load(int(other))
            if theirs is not None and not theirs.gate(embs[i], float(weights[i]), float(sc)):
                continue
            out[i] = True
            break
    return out


# ------------------------------------------------------------------ the scan
@dataclass
class _Pick:
    face_id: int
    item_id: int
    person_id: int | None  # where the face sat when it was scored (None, or an unnamed person)
    video: bool
    p: float
    top3: float


def candidates(conn: sqlite3.Connection, person_id: int) -> list[sqlite3.Row]:
    """Faces a scan looks at: (id, item_id, person_id, kind, named), named = the face sits on a named person."""
    pid = int(person_id)
    present = {int(r[0]) for r in conn.execute(
        "SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? UNION "
        "SELECT item_id FROM item_tags WHERE person_id = ? AND relation = 'in'", (pid, pid, pid))}
    rows = conn.execute(
        "SELECT f.id, f.item_id, f.person_id, i.kind, "
        "(p.id IS NOT NULL AND p.name IS NOT NULL AND p.name != '') AS named "
        "FROM faces f JOIN items i ON i.id = f.item_id LEFT JOIN persons p ON p.id = f.person_id "
        "WHERE i.deleted_at IS NULL AND i.nsfw = 0 AND f.assigned_by IN ('auto', 'guess') AND f.person_id IS NOT ? "
        "AND NOT EXISTS (SELECT 1 FROM guess_rejections g WHERE g.kind = 'face' AND g.ref_id = f.id AND g.person_id = ?) "
        "ORDER BY f.id", (pid, pid),
    ).fetchall()
    # the person is in the item already (a face, a person box or a tag): confirming would be refused there
    return [r for r in rows if int(r["item_id"]) not in present]


def scan(conn: sqlite3.Connection, person_id: int, progress: Progress | None = None) -> dict[str, Any]:
    """Score every candidate with no write transaction open, then write the guesses in short transactions that
    re-read each row. Returns (and stores under the model's `scan`) {at, looked, guesses, sure, detached,
    elsewhere: [{face_id, p, person_id}]}."""
    from app.analysis.index import get_index

    pid = int(person_id)
    if person_model.load(pid) is None:
        raise person_model.NotEnough("Recognition is not trained on this person yet.")
    if conn.execute("SELECT 1 FROM persons WHERE id = ? AND kind = 'person'", (pid,)).fetchone() is None:
        raise person_model.NotEnough("This person is not here any more.")
    person_model.update(pid, withdrawn_at=None)  # looking again un-pauses the model for guess_item
    model = person_model.load(pid)
    if model is None:
        raise person_model.NotEnough("Recognition is not trained on this person yet.")
    idx = get_index()
    judge = _Judge(conn, model)
    named = _named_people(conn)
    rows = candidates(conn, pid)
    total = len(rows)
    looked = 0
    picks: list[_Pick] = []
    elsewhere: list[dict[str, Any]] = []
    done = 0
    for chunk in _chunks(rows, SCORE_CHUNK):
        if progress is not None:
            progress(done, total, "scan")
        done += len(chunk)
        ids = [int(r["id"]) for r in chunk]
        with idx.lock:
            idx.sync(conn)
            weights = idx.weights_of(ids)
            child = idx.child_flags(ids)
            keep = [k for k in range(len(ids)) if weights[k] > 0 and not child[k]]
            if not keep:
                continue
            kept = [chunk[k] for k in keep]
            kept_ids = [ids[k] for k in keep]
            embs = idx.embeddings_of(kept_ids)
        w = weights[keep]
        looked += len(kept)
        p, top3, veto = judge.score(embs, w)
        is_named = np.fromiter((bool(r["named"]) for r in kept), dtype=bool, count=len(kept))
        ok = (p >= model.guess_p) & ~veto
        hot = [k for k in range(len(kept)) if ok[k] and not is_named[k]]
        if hot:
            with idx.lock:
                idx.sync(conn)
                worse = _better_elsewhere(idx, pid, [kept_ids[k] for k in hot], embs[hot], w[hot], top3[hot], named)
            for k, bad in zip(hot, worse):
                if bad:
                    continue
                r = kept[k]
                picks.append(_Pick(int(r["id"]), int(r["item_id"]), None if r["person_id"] is None else int(r["person_id"]),
                                   r["kind"] == "video", float(p[k]), float(top3[k])))
        for k in range(len(kept)):
            if is_named[k] and not veto[k] and p[k] >= model.sure_p:
                elsewhere.append({"face_id": int(kept[k]["id"]), "p": round(float(p[k]), 4),
                                  "person_id": int(kept[k]["person_id"])})
    if progress is not None:
        progress(total, total, "write")
    # one per photo: the best (videos may show a person several times)
    best: dict[int, _Pick] = {}
    chosen: list[_Pick] = []
    for pk in picks:
        if pk.video:
            chosen.append(pk)
        elif pk.item_id not in best or (pk.p, -pk.face_id) > (best[pk.item_id].p, -best[pk.item_id].face_id):
            best[pk.item_id] = pk
    chosen.extend(best.values())
    chosen.sort(key=lambda pk: pk.face_id)
    keep_ids = {pk.face_id for pk in picks}  # every face the model still names; guess_item picks one per photo
    # guess_item is what writes a guess on an open face, and it runs for every photo picked (below). The scan itself
    # only takes faces out of unnamed groups, and only those whose guess guess_item will keep: a face stays in
    # its group when clothes or traits name someone more strongly there, or another box of the photo keeps the person.
    grouped = [pk for pk in chosen if pk.person_id is not None and _would_stand(conn, pid, pk)]
    detached = 0
    touched: set[int] = {pk.item_id for pk in chosen}
    for part in _chunks(grouped, WRITE_CHUNK):
        n, items = _write(conn, model, part)
        detached += n
        touched |= items
    # a guess of an earlier scan that the model does not make any more goes (and its face back to a group)
    stale = [r for r in _standing(conn, pid) if int(r["id"]) not in keep_ids]
    if stale:
        touched |= {int(r["item_id"]) for r in stale}
        _clear(conn, pid, stale)
    _settle(conn, touched)
    # the safety net of _would_stand: a face taken out whose guess did not stand after all goes back
    undone = _restore_unsettled(conn, pid, grouped)
    detached = max(0, detached - undone)
    standing = _standing(conn, pid)
    elsewhere.sort(key=lambda e: (-e["p"], e["face_id"]))
    result = {
        "at": now_iso(), "looked": int(looked), "guesses": len(standing),
        "sure": sum(1 for r in standing if float(r["guess_score"] or 0.0) >= model.sure_p),
        "detached": int(detached), "elsewhere": elsewhere[:ELSEWHERE_KEPT],
    }
    person_model.update(pid, scan=result)
    log.info("person scan: person %d, %d faces looked at, %d guesses standing (%d taken out of unnamed groups), "
             "%d on other people look like them", pid, looked, len(standing), detached, len(elsewhere))
    return result


def _would_stand(conn: sqlite3.Connection, person_id: int, pk: _Pick) -> bool:
    """Whether guess_item would keep the model's guess on this face once it is an open box: the same computation,
    with the face added to the item's open boxes, and nothing written."""
    from app.analysis import guess

    item = conn.execute(
        "SELECT id, uid, kind, taken_at, gps_lat, gps_lon, nsfw, deleted_at FROM items WHERE id = ?", (pk.item_id,)).fetchone()
    face = conn.execute("SELECT id, item_id, torso_embedding, embedding FROM faces WHERE id = ?", (pk.face_id,)).fetchone()
    if item is None or face is None or item["deleted_at"] is not None or item["nsfw"]:
        return False
    try:
        boxes = guess.open_boxes(conn, pk.item_id)
        if not any(b.kind == "face" and b.id == pk.face_id for b in boxes):
            boxes.append(guess.Box("face", int(face["id"]), int(face["item_id"]), None,
                                   guess._vec(face["torso_embedding"]), guess._vec(face["embedding"])))
            boxes.sort(key=lambda b: (b.kind != "face", b.id))  # open_boxes' order: faces by id, then bodies by id
        got = merge_item(conn, item, boxes, guess._compute(conn, item, boxes, None)).get(("face", pk.face_id))
    except Exception:  # noqa: BLE001 - when in doubt the face stays in its group
        log.exception("person scan: could not try the guess of face %s", pk.face_id)
        return False
    return got is not None and got.person_id == int(person_id) and got.evidence.get("basis") == BASIS


def _settle(conn: sqlite3.Connection, item_ids: set[int]) -> None:
    """What guess_item computes for these items is what stays (clothes agreeing, one person per photo, and the
    guesses of other evidence back where a model guess went)."""
    from app.analysis import guess

    for item_id in sorted(item_ids):
        try:
            guess.guess_item(conn, item_id)
        except Exception:  # noqa: BLE001 - the write before it stands
            log.exception("person scan: could not settle the guesses of item %s", item_id)


def _restore_unsettled(conn: sqlite3.Connection, person_id: int, picks: list[_Pick]) -> int:
    back: list[tuple[int, int]] = []
    for pk in picks:
        r = conn.execute("SELECT person_id, assigned_by, guess_person_id, guess_evidence FROM faces WHERE id = ?",
                         (pk.face_id,)).fetchone()
        if r is None or r["person_id"] is not None or r["assigned_by"] == "user" or pk.person_id is None:
            continue
        ev = json_loads(r["guess_evidence"], {}) or {}
        if (r["guess_person_id"] is not None and int(r["guess_person_id"]) == int(person_id)
                and isinstance(ev, dict) and ev.get("basis") == BASIS):
            continue  # the model's guess stands
        back.append((pk.face_id, pk.person_id))
    return regroup(conn, back, clear_guess=True)


def _standing(conn: sqlite3.Connection, person_id: int) -> list[sqlite3.Row]:
    """The model's open guesses for the person in live items outside the vault."""
    return conn.execute(
        "SELECT f.id, f.item_id, f.guess_score, f.guess_evidence FROM faces f JOIN items i ON i.id = f.item_id "
        "WHERE f.guess_person_id = ? AND f.person_id IS NULL AND f.assigned_by = 'guess' "
        f"AND i.deleted_at IS NULL AND i.nsfw = 0 AND {BASIS_SQL.format(col='f.guess_evidence')} = ? ORDER BY f.id",
        (int(person_id), BASIS),
    ).fetchall()


def standing_count(conn: sqlite3.Connection, person_id: int) -> int:
    return len(_standing(conn, person_id))


def _was(evidence_json: str | None) -> int | None:
    ev = json_loads(evidence_json, {}) or {}
    v = ev.get("was_person_id") if isinstance(ev, dict) else None
    return int(v) if isinstance(v, int) else None


def _write(conn: sqlite3.Connection, model: person_model.Model, picks: list[_Pick]) -> tuple[int, set[int]]:
    """Write these guesses in one short transaction, each row read again first. Returns (faces taken out of an
    unnamed group, items changed)."""
    from app.analysis import cluster, guess

    pid = model.person_id
    detached = 0
    detached_from: set[int] = set()
    detached_items: set[int] = set()
    changed_items: set[int] = set()
    present: dict[int, set[int]] = {}
    now = now_iso()
    with tx(conn):
        if conn.execute("SELECT 1 FROM persons WHERE id = ?", (pid,)).fetchone() is None:
            return 0, set()
        for pk in picks:
            row = conn.execute(
                "SELECT f.person_id, f.assigned_by, f.guess_person_id, f.guess_score, f.guess_evidence, i.deleted_at, i.nsfw "
                "FROM faces f JOIN items i ON i.id = f.item_id WHERE f.id = ?", (pk.face_id,)).fetchone()
            if row is None or row["assigned_by"] not in ("auto", "guess") or row["deleted_at"] is not None or row["nsfw"]:
                continue
            cur = None if row["person_id"] is None else int(row["person_id"])
            if cur != pk.person_id:
                continue  # it moved since it was scored
            if cur is not None and conn.execute(
                    "SELECT 1 FROM persons WHERE id = ? AND (name IS NULL OR name = '')", (cur,)).fetchone() is None:
                continue  # its group got a name meanwhile
            if conn.execute("SELECT 1 FROM guess_rejections WHERE kind = 'face' AND ref_id = ? AND person_id = ?",
                            (pk.face_id, pid)).fetchone():
                continue
            if pk.item_id not in present:
                present[pk.item_id] = guess._present_people(conn, pk.item_id)
            if pid in present[pk.item_id]:
                continue
            held = row["guess_person_id"]
            old = json_loads(row["guess_evidence"], {}) or {}
            old = old if isinstance(old, dict) else {}
            mine = held is not None and int(held) == pid and old.get("basis") == BASIS
            if held is not None and not mine and float(row["guess_score"] or 0.0) >= pk.p:
                continue  # a guess at least as strong stays (someone else's, or this person's by clothes or traits)
            if (mine and cur is None and old.get("p") == round(pk.p, 4)
                    and (old.get("model") or {}).get("trained_at") == model.trained_at):
                continue  # the same guess already stands (guess_item may have added what else agrees)
            was = cur if cur is not None else (_was(row["guess_evidence"]) if mine else None)
            ev = evidence(model, pk.p, pk.top3, was_person_id=was)
            conn.execute(
                "UPDATE faces SET person_id = NULL, assigned_by = 'guess', guess_person_id = ?, guess_score = ?, "
                "guess_evidence = ? WHERE id = ? AND assigned_by IN ('auto', 'guess')",
                (pid, round(pk.p, 4), json_dumps(ev), pk.face_id))
            changed_items.add(pk.item_id)
            if cur is not None:
                detached += 1
                detached_from.add(cur)
                detached_items.add(pk.item_id)
                conn.execute("UPDATE persons SET cover_face_id = NULL, updated_at = ? WHERE cover_face_id = ?",
                             (now, pk.face_id))
        if detached_items:
            cluster.sync_bodies_to_faces(conn, sorted(detached_items))  # the person box around follows its face
            cluster.prune_empty_persons(conn, detached_from)
        if changed_items:
            bump_faces_version(conn)
    if detached_items:
        guess.refresh_fts(conn, detached_items)
    return detached, changed_items


def _clear(conn: sqlite3.Connection, person_id: int, rows: list[sqlite3.Row]) -> int:
    """Clear these open guesses of the person, then put the faces that were taken out of a group back."""
    ids = [int(r["id"]) for r in rows]
    back = [(int(r["id"]), was) for r in rows if (was := _was(r["guess_evidence"])) is not None]
    n = 0
    with tx(conn):
        for part in _chunks(ids, _IN_CHUNK):
            marks = ",".join("?" * len(part))
            n += conn.execute(
                "UPDATE faces SET assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL "
                f"WHERE id IN ({marks}) AND person_id IS NULL AND assigned_by = 'guess' AND guess_person_id = ?",
                [*part, int(person_id)]).rowcount
        if n:
            bump_faces_version(conn)
    regroup(conn, back)
    return n


def _items_of(rows: list[sqlite3.Row]) -> set[int]:
    return {int(r["item_id"]) for r in rows}


def regroup(conn: sqlite3.Connection, faces: list[tuple[int, int]], *, clear_guess: bool = False) -> int:
    """(face id, the unnamed person a scan took it out of) for faces that are loose again, the guess turned down or
    taken back: each goes back to that group when it still exists and still has no name, else a clear face starts
    an unnamed person of its own (as it had). Exactly what the scan undid, with no new matching: a face is never
    put on a named person here. With clear_guess a face that holds another open guess goes back too, without it.
    Returns how many found a group."""
    from app.analysis import cluster, guess
    from app.analysis.faces import is_clear_face

    if not faces:
        return 0
    done = 0
    items: set[int] = set()
    try:
        with tx(conn):
            for face_id, was in faces:
                states = "('auto', 'guess')" if clear_guess else "('auto')"
                r = conn.execute(
                    "SELECT item_id, det_score, size_px, clarity, sharpness FROM faces WHERE id = ? AND person_id IS NULL "
                    f"AND assigned_by IN {states}", (int(face_id),)).fetchone()
                if r is None:
                    continue
                target: int | None = None
                if conn.execute("SELECT 1 FROM persons WHERE id = ? AND (name IS NULL OR name = '') AND kind = 'person'",
                                (int(was),)).fetchone() is not None:
                    target = int(was)
                elif is_clear_face(r["det_score"], r["size_px"], r["clarity"], r["sharpness"]):
                    target = cluster.new_person(conn)
                if target is None:
                    continue
                conn.execute(
                    "UPDATE faces SET person_id = ?, assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
                    f"guess_evidence = NULL WHERE id = ? AND person_id IS NULL AND assigned_by IN {states}",
                    (target, int(face_id)))
                items.add(int(r["item_id"]))
                done += 1
            if done:
                cluster.sync_bodies_to_faces(conn, sorted(items))
                bump_faces_version(conn)
    except Exception:  # noqa: BLE001 - the owner's answer already happened; the face stays loose
        log.exception("person scan: could not put %d faces back into their groups", len(faces))
        return 0
    if items:
        guess.refresh_fts(conn, items)
    return done


def withdraw(conn: sqlite3.Connection, person_id: int) -> int:
    """Take the model's open guesses for the person back. The model is paused first, so guess_item does not make
    them again; the next scan un-pauses it. Returns how many guesses went."""
    pid = int(person_id)
    person_model.update(pid, withdrawn_at=now_iso())
    rows = _standing(conn, pid)
    n = _clear(conn, pid, rows)
    _settle(conn, _items_of(rows))  # a guess of other evidence that the model's had replaced comes back
    return n


# ------------------------------------------------------------------ guess_item's part
def merge_item(conn: sqlite3.Connection, item: sqlite3.Row, boxes: list[Any],
               results: dict[tuple[str, int], Any]) -> dict[tuple[str, int], Any]:
    """Add the trained models' guesses for the item's open faces to `results` (guess._compute's). The same object
    comes back untouched when no model has a say, so every other path stays byte for byte what it was.

    Against the clothes or trait candidate of a face: the same person keeps the higher score and notes the other
    evidence (`also`, or `p` on the other's evidence); a different person: the higher score wins, except that a
    candidate for an unnamed person never displaces the model's (the face may come from that very group, and a
    named guess is the one the owner can answer; turned down, the other shows next time). Then one person per photo."""
    from app.analysis import guess
    from app.analysis.index import get_index

    faces = [b for b in boxes if b.kind == "face" and b.face is not None]
    if not faces:
        return results
    trained = person_model.trained_ids()
    stored = _stored(conn, [b.id for b in faces])
    held = {f: g for f, g in stored.items() if g[2].get("basis") == BASIS}
    # a face a scan took out of an unnamed group keeps that memory whatever guess it ends with
    was_of = {f: int(g[2]["was_person_id"]) for f, g in stored.items() if isinstance(g[2].get("was_person_id"), int)}
    if not trained and not held and not was_of:
        return results
    item_id = int(item["id"])
    present = guess._present_people(conn, item_id)
    rejected = guess._rejections(conn, faces)
    active: dict[int, person_model.Model] = {}
    for pid in trained:
        m = person_model.load(pid)
        if m is not None and not m.withdrawn:
            active[pid] = m
    cands: dict[int, guess.Candidate] = {}
    named: set[int] | None = None
    live = {pid: m for pid, m in active.items() if pid not in present}
    if live:
        alive = {int(r[0]) for r in conn.execute(
            f"SELECT id FROM persons WHERE id IN ({','.join('?' * len(live))})", sorted(live))}
        named = _named_people(conn)
        idx = get_index()
        ids = [b.id for b in faces]
        embs = np.stack([b.face for b in faces]).astype(np.float32)
        with idx.lock:
            idx.sync(conn)
            weights = idx.weights_of(ids)
            child = idx.child_flags(ids)
            for pid in sorted(live):
                if pid not in alive:
                    continue
                model = live[pid]
                p, top3, veto = _Judge(conn, model).score(embs, weights)
                hot = [k for k in range(len(faces))
                       if p[k] >= model.guess_p and not veto[k] and weights[k] > 0 and not child[k]
                       and pid not in rejected.get(("face", faces[k].id), set())]
                if not hot:
                    continue
                worse = _better_elsewhere(idx, pid, [ids[k] for k in hot], embs[hot], weights[hot], top3[hot], named)
                for k, bad in zip(hot, worse):
                    if bad:
                        continue
                    prev = cands.get(ids[k])
                    if prev is not None and (prev.score, -prev.person_id) >= (float(p[k]), -pid):
                        continue
                    cands[ids[k]] = guess.Candidate(pid, float(p[k]), evidence(
                        model, float(p[k]), float(top3[k]), was_person_id=was_of.get(ids[k])))
    # A guess whose model has no say any more (paused, forgotten, learning off) stays as it is until the owner
    # answers it: an undo may have brought it back, and only a model that is at work may withdraw its own guess.
    for fid, (qid, score, ev) in held.items():
        if fid in cands or qid in active or qid in present or qid in rejected.get(("face", fid), set()):
            continue
        if conn.execute("SELECT 1 FROM persons WHERE id = ?", (qid,)).fetchone() is None:
            continue
        cands[fid] = guess.Candidate(qid, float(score), ev)
    if not cands and not was_of:
        return results
    out = dict(results)
    for fid, m in cands.items():
        key = ("face", fid)
        c = out.get(key)
        if c is None:
            out[key] = m
        elif c.person_id == m.person_id:
            if m.score >= c.score:
                out[key] = guess.Candidate(m.person_id, m.score, {**m.evidence, "also": c.evidence.get("basis") or "clothes"})
            else:
                out[key] = guess.Candidate(c.person_id, c.score, {**c.evidence, "p": m.evidence.get("p")})
        else:
            if named is None:
                named = _named_people(conn)
            if m.score > c.score or c.person_id not in named:
                out[key] = m
    if cands and item["kind"] != "video":
        # one person per photo: the better box keeps the person, the other falls back to what it had
        for _ in range(len(out) + 1):
            by_person: dict[int, list[tuple[str, int]]] = {}
            for key, c in out.items():
                if c is not None:
                    by_person.setdefault(c.person_id, []).append(key)
            twice = {pid: keys for pid, keys in by_person.items() if len(keys) > 1}
            if not twice:
                break
            for pid, keys in twice.items():
                keys.sort(key=lambda k: (-out[k].score, k[0], k[1]))
                for key in keys[1:]:
                    before = results.get(key)
                    out[key] = before if before is not None and before.person_id != pid else None
    for fid, was in was_of.items():
        c = out.get(("face", fid))
        if c is not None and c.evidence.get("was_person_id") != was:
            out[("face", fid)] = guess.Candidate(c.person_id, c.score, {**c.evidence, "was_person_id": was})
    return out


def _stored(conn: sqlite3.Connection, face_ids: list[int]) -> dict[int, tuple[int, float, dict[str, Any]]]:
    """{face id: (person, score, evidence)} for those of the faces that hold an open guess."""
    out: dict[int, tuple[int, float, dict[str, Any]]] = {}
    for part in _chunks([int(f) for f in face_ids], _IN_CHUNK):
        marks = ",".join("?" * len(part))
        for r in conn.execute(
            f"SELECT id, guess_person_id, guess_score, guess_evidence FROM faces WHERE id IN ({marks}) "
            "AND person_id IS NULL AND assigned_by = 'guess' AND guess_person_id IS NOT NULL", part,
        ):
            ev = json_loads(r["guess_evidence"], {}) or {}
            if isinstance(ev, dict):
                out[int(r["id"])] = (int(r["guess_person_id"]), float(r["guess_score"] or 0.0), ev)
    return out


# ------------------------------------------------------------------ the background run and its state
_state_lock = threading.Lock()
_state: dict[int, dict[str, Any]] = {}
_running: int | None = None


def state(person_id: int) -> dict[str, Any]:
    """{running: 'train' | 'scan' | None, progress: {done, total, step} | None, error: str | None}."""
    with _state_lock:
        s = _state.get(int(person_id)) or {}
        return {"running": s.get("running"), "progress": dict(s["progress"]) if s.get("progress") else None,
                "error": s.get("error")}


def busy_with() -> int | None:
    """The person a run is working on now, or None."""
    with _state_lock:
        return _running


def _set(person_id: int, **fields: Any) -> None:
    with _state_lock:
        _state.setdefault(int(person_id), {}).update(fields)


def start(person_id: int, *, train: bool = True, look: bool = True) -> bool:
    """Train and/or scan on the background thread (inline with settings.workers_enabled off, as the tests run).
    False when a run is already going."""
    global _running
    from app import worker

    pid = int(person_id)
    with _state_lock:
        if _running is not None:
            return False
        _running = pid
        _state[pid] = {"running": "train" if train else "scan", "progress": None, "error": None}
    if not settings.workers_enabled:
        contextvars.Context().run(_run, pid, train, look)  # no owner action: as on the thread, nothing is journaled
        return True
    if worker.run_in_background(RUNNER_KEY, _run, pid, train, look) is None:
        with _state_lock:
            _running = None
            _state[pid]["running"] = None
        return False
    return True


def _run(person_id: int, train: bool, look: bool) -> None:
    global _running
    from app import worker
    from app.db import connect

    if threading.current_thread().name.startswith("pcc-bg-"):
        worker._lower_priority()
    conn = connect()
    try:
        if train:
            _set(person_id, running="train", progress=None)
            person_model.train(conn, person_id, lambda d, t, s: _set(person_id, progress={"done": d, "total": t, "step": s}))
        if look:
            _set(person_id, running="scan", progress=None)
            scan(conn, person_id, lambda d, t, s: _set(person_id, progress={"done": d, "total": t, "step": s}))
    except person_model.NotEnough as e:
        _set(person_id, error=str(e))
    except Exception:  # noqa: BLE001 - reported through the status
        log.exception("person model: the run for person %s failed", person_id)
        _set(person_id, error="It stopped on an error. Try again.")
    finally:
        conn.close()
        with _state_lock:
            _running = None
            _state.setdefault(int(person_id), {}).update(running=None, progress=None)
