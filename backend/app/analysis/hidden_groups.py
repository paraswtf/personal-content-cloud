"""Unnamed groups stay in the background (docs/ARCHITECTURE.md section 10, "Unnamed groups stay in the background").

The matching still gathers faces nobody is named for into unnamed persons (app.analysis.cluster), and recognition,
merges and guessing use those groups as before. The API never shows such a group (api.common.shown,
api.people._hidden_person): its faces read as faces nobody is named for and are listed with the faces to name, and
a guess for an unnamed person is not shown. This module holds what happens when the owner names one of those faces.

CONTRACT.
- BASIS = 'group': the evidence basis of a guess made when the owner named a face of an unnamed group.
- rest_to_guesses(conn, leaving, target) -> (face ids, item ids). The owner just named faces of unnamed groups
  (`leaving`: {group id: the faces of it that were just moved}) as `target`, or set them to nobody (target None).
  Only those faces take the owner's choice. The faces that left are remembered as not that group
  (cluster.record_rejections; a group the move left empty is gone and skipped). With a named target, every other automatic face of each group becomes an open guess
  for the target, for the owner to confirm or turn down: it leaves the group (person_id NULL, assigned_by 'guess'),
  with evidence {basis 'group', score, face, ref_face_id, ref_item_uid, was_person_id} (the reference is the best
  face just named outside the vault and the trash; none when every named face is in one), and the person box around
  it follows (cluster.sync_bodies_to_faces). The score is the cosine of the face to the mean of the faces just
  named (0 to 1). A face stays in its group when its photo is trashed or in the vault, the target is in its photo
  already, the owner said it is not the target (guess_rejections), the owner said the whole group is not the
  target (merge_dismissals), or another face of the group in the same photo is more alike (one person per photo;
  videos may show a person several times). A group left empty goes (cluster.prune_empty_persons). Turning a guess
  down puts its face back into its group, and a group that went comes back then (person_scan.regroup, through
  guess.reject). Call inside the owner's transaction.
- hold(conn, item, boxes, results) -> results. guess_item's pass for these guesses (from guess._merge_models): a
  'group' guess stays until the owner answers it, whatever another path found for that face, while it still
  stands: its person exists and has a name, is not in the photo, and was not turned down for the face (the vault
  is guess_item's own rule: a vault photo gets no guesses at all). In a photo, a box for which another path found
  the same person loses that candidate. `results` itself comes back when the item holds no such guess.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

from app.db import bump_faces_version
from app.util import json_dumps, json_loads, now_iso

log = logging.getLogger(__name__)

BASIS = "group"
_IN_CHUNK = 500
# CASE keeps json_extract away from malformed JSON (AND does not promise an order in SQLite)
BASIS_SQL = "(CASE WHEN json_valid({col}) THEN json_extract({col}, '$.basis') END)"


def _chunks(ids: list[int], n: int = _IN_CHUNK):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def _unit(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def _named(conn: sqlite3.Connection, person_id: int) -> bool:
    return conn.execute("SELECT 1 FROM persons WHERE id = ? AND kind = 'person' AND name IS NOT NULL AND name != ''",
                        (int(person_id),)).fetchone() is not None


def _dismissed(conn: sqlite3.Connection, person_id: int) -> set[int]:
    """The people the owner said are not this person (a merge suggestion turned down)."""
    pid = int(person_id)
    return {int(r[0]) for r in conn.execute(
        "SELECT CASE WHEN person_a = ? THEN person_b ELSE person_a END FROM merge_dismissals "
        "WHERE person_a = ? OR person_b = ?", (pid, pid, pid))}


def rest_to_guesses(conn: sqlite3.Connection, leaving: dict[int, list[int]],
                    target: int | None) -> tuple[list[int], set[int]]:
    from app.analysis import cluster, guess

    guessed: list[int] = []
    items: set[int] = set()
    # a group the move left empty is gone already (cluster.prune_empty_persons): no memory, nothing left to guess
    leaving = {gid: seeds for gid, seeds in leaving.items()
               if conn.execute("SELECT 1 FROM persons WHERE id = ?", (int(gid),)).fetchone() is not None}
    if not leaving:
        return guessed, items
    for gid, seeds in sorted(leaving.items()):
        cluster.record_rejections(conn, seeds, gid)
    if target is None or not _named(conn, int(target)):
        return guessed, items
    target = int(target)
    dismissed = _dismissed(conn, target)
    rejected = {int(r[0]) for r in conn.execute(
        "SELECT ref_id FROM guess_rejections WHERE kind = 'face' AND person_id = ?", (target,))}
    present: dict[int, set[int]] = {}
    emptied: set[int] = set()
    now = now_iso()
    for gid, seeds in sorted(leaving.items()):
        if gid in dismissed:
            continue
        seed_rows = conn.execute(
            f"SELECT f.id, f.embedding, f.quality, i.uid, (i.deleted_at IS NULL AND i.nsfw = 0) AS shown "
            f"FROM faces f JOIN items i ON i.id = f.item_id "
            f"WHERE f.id IN ({','.join('?' * len(seeds))}) ORDER BY f.quality DESC, f.id", sorted(seeds)).fetchall()
        vecs = [v for v in (_unit(r["embedding"]) for r in seed_rows) if v is not None]
        if not vecs:
            continue
        ref = np.mean(np.stack(vecs), axis=0)
        ref /= max(float(np.linalg.norm(ref)), 1e-9)
        # The guesses' reference: the best named face in the library. The guesses sit outside the vault and the
        # trash, where a locked session reviews them, so a face named in a vault (or trashed) photo is never named
        # as their reference; with only such faces, the evidence has none.
        best_seed = next((r for r in seed_rows if r["shown"]), None)
        picks: dict[Any, tuple[float, sqlite3.Row]] = {}
        for r in conn.execute(
            "SELECT f.id, f.item_id, f.embedding, i.kind FROM faces f JOIN items i ON i.id = f.item_id "
            "WHERE f.person_id = ? AND f.assigned_by = 'auto' AND i.deleted_at IS NULL AND i.nsfw = 0 ORDER BY f.id",
            (gid,),
        ):
            fid, iid = int(r["id"]), int(r["item_id"])
            if fid in rejected:
                continue
            if iid not in present:
                present[iid] = guess._present_people(conn, iid)
            if target in present[iid]:
                continue
            v = _unit(r["embedding"])
            score = min(1.0, max(0.0, float(v @ ref))) if v is not None else 0.0
            key: Any = (iid, fid) if r["kind"] == "video" else iid
            held = picks.get(key)
            if held is not None and held[0] >= score:
                continue
            picks[key] = (score, r)
        for score, r in sorted(picks.values(), key=lambda t: int(t[1]["id"])):
            ev = {"basis": BASIS, "score": round(score, 4), "face": round(score, 4), "was_person_id": int(gid)}
            if best_seed is not None:
                ev.update(ref_face_id=int(best_seed["id"]), ref_item_uid=str(best_seed["uid"]))
            cur = conn.execute(
                "UPDATE faces SET person_id = NULL, assigned_by = 'guess', guess_person_id = ?, guess_score = ?, "
                "guess_evidence = ? WHERE id = ? AND person_id = ? AND assigned_by = 'auto'",
                (target, round(score, 4), json_dumps(ev), int(r["id"]), int(gid)))
            if cur.rowcount:
                guessed.append(int(r["id"]))
                items.add(int(r["item_id"]))
                emptied.add(int(gid))
                conn.execute("UPDATE persons SET cover_face_id = NULL, updated_at = ? WHERE cover_face_id = ?",
                             (now, int(r["id"])))
    if items:
        cluster.sync_bodies_to_faces(conn, sorted(items))  # the person box around follows its face
        cluster.prune_empty_persons(conn, emptied)
        bump_faces_version(conn)
        log.info("%d face(s) of %d unnamed group(s) became guesses for person %d", len(guessed), len(emptied), target)
    return guessed, items


def hold(conn: sqlite3.Connection, item: sqlite3.Row, boxes: list[Any],
         results: dict[tuple[str, int], Any]) -> dict[tuple[str, int], Any]:
    from app.analysis import guess

    if item["nsfw"]:
        return results
    faces = [int(b.id) for b in boxes if b.kind == "face"]
    held: dict[int, tuple[int, float, dict[str, Any]]] = {}
    for chunk in _chunks(faces):
        for r in conn.execute(
            f"SELECT id, guess_person_id, guess_score, guess_evidence FROM faces WHERE id IN ({','.join('?' * len(chunk))}) "
            "AND person_id IS NULL AND assigned_by = 'guess' AND guess_person_id IS NOT NULL "
            f"AND {BASIS_SQL.format(col='guess_evidence')} IS '{BASIS}'", chunk,
        ):
            ev = json_loads(r["guess_evidence"], {}) or {}
            if isinstance(ev, dict):
                held[int(r["id"])] = (int(r["guess_person_id"]), float(r["guess_score"] or 0.0), ev)
    if not held:
        return results
    present = guess._present_people(conn, int(item["id"]))
    rejected = guess._rejections(conn, [b for b in boxes if b.kind == "face" and int(b.id) in held])
    out = dict(results)
    kept: dict[int, int] = {}  # person -> the face holding their 'group' guess here
    for fid, (pid, score, ev) in sorted(held.items()):
        if pid in present or pid in rejected.get(("face", fid), set()) or not _named(conn, pid):
            continue  # it no longer stands: guess_item clears it, and the face goes back into its group
        if item["kind"] != "video" and pid in kept:
            continue
        out[("face", fid)] = guess.Candidate(pid, score, ev)
        kept.setdefault(pid, fid)
    if item["kind"] != "video" and kept:
        for key, cand in list(out.items()):
            if cand is not None and cand.person_id in kept and key != ("face", kept[cand.person_id]):
                out[key] = None  # one person per photo: the guess the owner's naming made keeps them
    return out
