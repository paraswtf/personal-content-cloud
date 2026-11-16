"""Who a face, a box or a person looks most like: the order of the name picker (docs/ARCHITECTURE.md section 10,
"Naming by likeness").

CONTRACT.
- Source: what is being named. face_ids, body_ids, animal_ids (boxes); person_id with item_ids (that person's
  boxes on those items: the selection's "Change to..." and the viewer's person editor); person_id alone (the whole
  person: "Merge into"); around_item_id (a box being drawn, or a tag on a whole item: who was around then).
- rank(conn, source) -> list[Option], best first: every person the evidence names, named or not, of any kind (the
  route keeps the named and visible people of the kinds the source can be, app.api.similar). Read only: it never
  writes, queues or learns, and holds the face index's lock only for its own matrix products.
- Option: person_id, score (0 to 1, higher is more alike: the order), likely (the matching or the guess path would
  take them: shown, it does not change the order), basis (what gave the score or the likely: 'guess', 'model',
  'face', 'clothes', 'look', 'person' or 'hour') and first (the box's own open guess, listed before everyone else).

How a source is scored, with the matching's own parts:
- A face: the face index's top-k mean against each person's faces (index.person_scores_many; a face in the child
  band meets childhood photos only) is the score, so the order is the plain similarity's. On a snapshot of the
  owner's library that order put the right person first for 99.5% of the clear faces the owner named and 85% of
  the blurred or turned ones; moving people by their learned cut, their rejected faces or a trained model put
  fewer first, so those only decide "likely": the top-k reaches the cut the matching uses for that face and that
  person (cluster.option_cut, with the person's learned threshold; a child: only the best, with the child lead
  over the next), and not when the faces the owner took off the person look more like it than their own faces do
  (cluster.rejected_examples, read for those people only). A model trained on one person (section 10,
  app.learning.person_model; on, not paused) makes its person likely from its guess level (basis 'model') and
  not likely under its gate, as the matching does.
- A person box without a face: the clothes path of the guess (guess.score_box over the photos of the hour that show
  the same scene or place): every candidate at its score, the winner (it leads by guess.MARGIN) likely, the others
  at CLOTHES_RUNNER of their score. A face scored alone is given the clothes path too (a face that matches someone
  at guess.FACE_CONTRADICTION or more keeps other people out of it). Traits and position are not run here (their
  verification takes seconds); they count through the box's open guess.
- A box's open guess (guess_person_id): that person first. Left out for a box: the people the owner said it is not
  (guess_rejections) and, in a photo, the people already on its other boxes or tagged in it. Left out everywhere:
  the people the boxes belong to now (and the person whose boxes are being moved).
- A pet box: the pets of its species by the cosine to each pet's centroid (pets.pet_centroids), likely from the
  pet's own threshold (pets.match_threshold) unless the crop looks like one the owner took off that pet
  (pets.rejected_crops).
- A single box, and around_item_id, also count who is in the photos of the same hour (guess.window_items: never a
  vault or trashed item) and not in this one, weakly (HOUR_MAX at most, by how many of those photos show them), so
  a box nothing else speaks for still lists the people who were there first.
- Several boxes: each box's score per person, weighed by the box (a face by its face weight, at least
  MIN_FACE_WEIGHT; a person box BODY_WEIGHT; a pet box 1), averaged over the boxes that name anyone (one that names
  others but not this person gives them 0 there; one that names nobody, such as a face past MAX_FACES or a person
  box with no guess, takes no part); likely and first when that holds for at least half of that weight. At most
  MAX_FACES faces are scored (the best weighted); the clothes path and the hour only for a single box.
- The whole person: the cosine of their centroid to every other person's (index.centroids, as merge suggestions),
  likely from settings.suggest_threshold; people seen with them in one photo (not a video) and pairs the owner
  dismissed are left out. A pet: its crops' centroid against the other pets of its species, likely from their
  thresholds.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

HOUR_MAX = 0.12        # the most "in photos of the same hour" gives: under a face that matches (0.4 or more)
CLOTHES_RUNNER = 0.75  # a clothes candidate that does not win the box counts this much of its score
MIN_FACE_WEIGHT = 0.15  # a blurred or turned-away face still counts a little among several boxes
BODY_WEIGHT = 0.5
ANIMAL_WEIGHT = 1.0
MAX_FACES = 12          # faces scored for several boxes (the best weighted): about 3 ms each on 38,000 faces
CENTROIDS_KEY = "similar_centroids"  # the face index's derived-data cache (dropped on every change)
_IN_CHUNK = 500


@dataclass
class Source:
    face_ids: list[int] = field(default_factory=list)
    body_ids: list[int] = field(default_factory=list)
    animal_ids: list[int] = field(default_factory=list)
    person_id: int | None = None
    item_ids: list[int] = field(default_factory=list)
    around_item_id: int | None = None


@dataclass
class Option:
    person_id: int
    score: float
    likely: bool
    basis: str
    first: bool = False


@dataclass
class _Ev:
    score: float
    likely: bool
    basis: str
    first: bool = False


@dataclass
class _Box:
    kind: str  # 'face' | 'body' | 'animal' | 'item'
    id: int
    item_id: int
    person_id: int | None
    weight: float
    evs: dict[int, _Ev] = field(default_factory=dict)
    row: Any = None
    face_scores: dict[int, float] | None = None


def _chunks(ids: list[Any], n: int = _IN_CHUNK):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def _add(evs: dict[int, _Ev], pid: int, score: float, likely: bool, basis: str, first: bool = False) -> None:
    """Evidence for a person on one box: the strongest score names the basis; likely and first hold once any says so."""
    score = float(min(1.0, max(0.0, score)))
    cur = evs.get(int(pid))
    if cur is None:
        evs[int(pid)] = _Ev(score, bool(likely), basis, bool(first))
        return
    if score > cur.score:
        cur.score, cur.basis = score, basis
    cur.likely = cur.likely or bool(likely)
    cur.first = cur.first or bool(first)


def _unique(ids: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(int(i))
    return out


def _rows(conn: sqlite3.Connection, sql: str, ids: list[int]) -> list[sqlite3.Row]:
    """Rows of `sql` (ending in "id IN ({marks})") for these ids, in the order given; unknown ids are skipped."""
    found: dict[int, sqlite3.Row] = {}
    for chunk in _chunks(ids):
        for r in conn.execute(sql.format(marks=",".join("?" * len(chunk))), chunk):
            found[int(r["id"])] = r
    return [found[i] for i in ids if i in found]


def _rejected(conn: sqlite3.Connection, kind: str, ids: list[int]) -> dict[int, set[int]]:
    """{box id: people the owner said it is not} (guess_rejections)."""
    out: dict[int, set[int]] = {}
    for chunk in _chunks(ids):
        for r in conn.execute(
            f"SELECT ref_id, person_id FROM guess_rejections WHERE kind = ? AND ref_id IN ({','.join('?' * len(chunk))})",
            [kind, *chunk],
        ):
            out.setdefault(int(r[0]), set()).add(int(r[1]))
    return out


def _guess_score(row: sqlite3.Row) -> float:
    try:
        return float(row["guess_score"] or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _open_guess(box: _Box) -> None:
    """The box's open guess: its person first (a guess the owner has not answered, on a box with no person)."""
    r = box.row
    if r is None or r["guess_person_id"] is None or r["person_id"] is not None:
        return
    _add(box.evs, int(r["guess_person_id"]), max(_guess_score(r), 0.5), True, "guess", first=True)


# ------------------------------------------------------------------ faces
FACE_SQL = (
    "SELECT id, item_id, person_id, det_score, size_px, clarity, sharpness, guess_person_id, guess_score, "
    "torso_embedding FROM faces WHERE id IN ({marks})"
)


def _face_boxes(conn: sqlite3.Connection, face_ids: list[int]) -> list[_Box]:
    rows = _rows(conn, FACE_SQL, face_ids)
    if not rows:
        return []
    from app.analysis import cluster
    from app.analysis.index import get_index

    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        held, embs, weights, _user, _items = idx.examples_of([int(r["id"]) for r in rows])
        pos = {int(f): i for i, f in enumerate(held.tolist())}
        # the best weighted faces when there are many (ties: the order given)
        order = sorted((i for i, r in enumerate(rows) if int(r["id"]) in pos),
                       key=lambda i: (-float(weights[pos[int(rows[i]["id"])]]), i))[:MAX_FACES]
        scored = [rows[i] for i in sorted(order)]
        sel = [pos[int(r["id"])] for r in scored]
        child = idx.child_flags([int(r["id"]) for r in scored]) if scored else np.zeros(0, dtype=bool)
        exclude = frozenset(int(r["id"]) for r in rows)  # the boxes being named never vote for themselves
        lists = (idx.person_scores_many(embs[sel], settings.cluster_topk, exclude, child=child)
                 if scored else [])
    boxes: list[_Box] = []
    by_id: dict[int, _Box] = {}
    for r in rows:
        w = float(weights[pos[int(r["id"])]]) if int(r["id"]) in pos else 0.0
        b = _Box("face", int(r["id"]), int(r["item_id"]), r["person_id"], max(w, MIN_FACE_WEIGHT), row=r)
        boxes.append(b)
        by_id[b.id] = b
    if scored:
        learned: dict[int, float | None] = {}

        def learned_of(pid: int) -> float | None:
            if pid not in learned:
                learned[pid] = cluster._learned(pid)
            return learned[pid]

        overs: list[list[tuple[int, float]]] = []
        for k, (r, opts) in enumerate(zip(scored, lists)):
            young = bool(child[k])
            thr = cluster._threshold(r)
            best_pid = opts[0][0] if opts else None
            lead_ok = bool(opts) and (opts[0][1] - (opts[1][1] if len(opts) > 1 else 0.0)) >= float(settings.child_lead)
            overs.append([(pid, s) for pid, s in opts
                          if s >= cluster.option_cut(thr, young, learned_of(pid)) and (not young or (pid == best_pid and lead_ok))])
        wanted = {pid for over in overs for pid, _ in over}
        examples = cluster.rejected_examples(conn, wanted) if wanted else {}  # once, for the people over a cut
        for k, (r, opts) in enumerate(zip(scored, lists)):
            b = by_id[int(r["id"])]
            b.face_scores = dict(opts)
            emb = embs[sel[k]]
            likely = set()
            for pid, s in overs[k]:
                ex = examples.get(pid)
                if ex is None or not ex.shape[0] or float(np.max(ex @ emb)) < s:
                    likely.add(pid)  # not when it looks more like a face the owner took off them
            for pid, s in opts:
                _add(b.evs, pid, float(s), pid in likely, "face")
        _trained_models(conn, [by_id[int(r["id"])] for r in scored], embs[sel],
                        np.asarray([float(weights[i]) for i in sel], dtype=np.float32), child, exclude)
    for b in boxes:
        _open_guess(b)
    return boxes


def _trained_models(conn: sqlite3.Connection, boxes: list[_Box], embs: np.ndarray, weights: np.ndarray,
                    child: np.ndarray, exclude: frozenset[int]) -> None:
    """Recognition trained on one person (section 10): at its guess level or more the model makes its person likely
    (basis 'model'); under its gate the person is not likely. The score stays the plain similarity's. Faces in the
    child band, faces that weigh nothing and paused models are left out, as in the matching."""
    try:
        from app.analysis.index import get_index
        from app.learning import person_model

        trained = person_model.trained_ids()
        rows = [i for i in range(len(boxes)) if not child[i] and weights[i] > 0]
        if not trained or not rows:
            return
        at_work = [int(r[0]) for r in conn.execute(
            f"SELECT id FROM persons WHERE id IN ({','.join('?' * len(trained))}) AND kind = 'person' "
            "AND recognition_paused_at IS NULL ORDER BY id", trained)]
        idx = get_index()
        for pid in at_work:
            model = person_model.load(pid)
            if model is None:
                continue
            with idx.lock:
                p = np.asarray(model.probability(embs[rows], weights[rows], exclude_ids=exclude, conn=conn),
                               dtype=np.float64).reshape(-1)
            for k, i in enumerate(rows):
                ev = boxes[i].evs.get(pid)
                if ev is None:
                    continue  # the plain similarity does not name them: the model does not add them
                if p[k] >= model.guess_p:
                    ev.likely, ev.basis = True, "model"
                elif p[k] < person_model.GATE_P:
                    ev.likely = False
    except Exception:  # noqa: BLE001 - a trained model never blocks the picker
        log.exception("similar: the trained recognition failed; ranking without it")


# ------------------------------------------------------------------ person boxes and the clothes path
BODY_SQL = (
    "SELECT id, item_id, person_id, face_id, guess_person_id, guess_score, torso_embedding FROM bodies "
    "WHERE id IN ({marks})"
)


def _body_boxes(rows: list[sqlite3.Row]) -> list[_Box]:
    boxes = []
    for r in rows:
        b = _Box("body", int(r["id"]), int(r["item_id"]), r["person_id"], BODY_WEIGHT, row=r)
        _open_guess(b)
        boxes.append(b)
    return boxes


def _item(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT id, uid, kind, taken_at, gps_lat, gps_lon FROM items WHERE id = ?",
                        (int(item_id),)).fetchone()


def _clothes(conn: sqlite3.Connection, box: _Box, rejected: set[int]) -> None:
    """The guess path's clothes ranking for one face or person box (guess.score_box with its own inputs)."""
    from app.analysis import guess

    torso = guess._vec(box.row["torso_embedding"]) if box.row is not None else None
    if torso is None:
        return
    item = _item(conn, box.item_id)
    if item is None or not item["taken_at"]:
        return
    inputs = guess._clothes_inputs(conn, item, None)
    if inputs is None:
        return
    refs, scene, uids = inputs
    gbox = guess.Box(box.kind, box.id, box.item_id, None, torso)
    winner, ranked = guess.score_box(gbox, refs, scene, box.face_scores if box.kind == "face" else None,
                                     set(rejected), uids)
    for c in ranked:
        won = winner is not None and c.person_id == winner.person_id
        _add(box.evs, c.person_id, c.score if won else c.score * CLOTHES_RUNNER, won, "clothes")


def _hour(conn: sqlite3.Connection, item_id: int, evs: dict[int, _Ev]) -> None:
    """Who is in the photos of the same hour (guess.window_items), weakly: HOUR_MAX for the one most of them show."""
    from app.analysis import guess

    item = _item(conn, item_id)
    if item is None or not item["taken_at"]:
        return
    ids = [int(r["id"]) for r in guess.window_items(conn, item)]
    counts: dict[int, int] = {}
    for chunk in _chunks(ids):
        m = ",".join("?" * len(chunk))
        for pid, n in conn.execute(
            "SELECT person_id, COUNT(DISTINCT item_id) FROM ("
            f"SELECT person_id, item_id FROM faces WHERE item_id IN ({m}) AND person_id IS NOT NULL UNION "
            f"SELECT person_id, item_id FROM bodies WHERE item_id IN ({m}) AND person_id IS NOT NULL UNION "
            f"SELECT person_id, item_id FROM animals WHERE item_id IN ({m}) AND person_id IS NOT NULL UNION "
            f"SELECT person_id, item_id FROM item_tags WHERE item_id IN ({m}) AND relation = 'in') GROUP BY person_id",
            [*chunk, *chunk, *chunk, *chunk],
        ):
            counts[int(pid)] = counts.get(int(pid), 0) + int(n)
    if not counts:
        return
    top = max(counts.values())
    for pid, n in counts.items():
        _add(evs, pid, HOUR_MAX * (0.5 + 0.5 * n / top), False, "hour")


def _present(conn: sqlite3.Connection, box: _Box) -> set[int]:
    """The people already in the box's photo other than through the box itself (its other boxes, its 'in' tags);
    empty for a video, where one person shows in many frames."""
    item = conn.execute("SELECT kind FROM items WHERE id = ?", (box.item_id,)).fetchone()
    if item is None or item["kind"] == "video":
        return set()
    out: set[int] = set()
    own_face = box.id if box.kind == "face" else (box.row["face_id"] if box.kind == "body" and box.row is not None else None)
    for r in conn.execute("SELECT id, person_id FROM faces WHERE item_id = ? AND person_id IS NOT NULL", (box.item_id,)):
        if own_face is None or int(r["id"]) != int(own_face):
            out.add(int(r["person_id"]))
    for r in conn.execute("SELECT id, face_id, person_id FROM bodies WHERE item_id = ? AND person_id IS NOT NULL",
                          (box.item_id,)):
        if box.kind == "body" and int(r["id"]) == box.id:
            continue
        if own_face is not None and r["face_id"] is not None and int(r["face_id"]) == int(own_face):
            continue
        out.add(int(r["person_id"]))
    for r in conn.execute("SELECT id, person_id FROM animals WHERE item_id = ? AND person_id IS NOT NULL", (box.item_id,)):
        if not (box.kind == "animal" and int(r["id"]) == box.id):
            out.add(int(r["person_id"]))
    out.update(int(r[0]) for r in conn.execute(
        "SELECT person_id FROM item_tags WHERE item_id = ? AND relation = 'in'", (box.item_id,)))
    return out


# ------------------------------------------------------------------ pets
ANIMAL_SQL = "SELECT id, item_id, person_id, species, embedding FROM animals WHERE id IN ({marks})"


def _animal_boxes(conn: sqlite3.Connection, animal_ids: list[int]) -> list[_Box]:
    from app.analysis import pets

    rows = _rows(conn, ANIMAL_SQL, animal_ids)
    centroids: dict[str, tuple[list[int], np.ndarray | None, dict[int, np.ndarray]]] = {}
    boxes = []
    for r in rows:
        b = _Box("animal", int(r["id"]), int(r["item_id"]), r["person_id"], ANIMAL_WEIGHT, row=r)
        boxes.append(b)
        v = pets._vec(r["embedding"])
        if v is None:
            continue
        species = str(r["species"])
        if species not in centroids:
            cents = pets.pet_centroids(conn, species)
            pids = sorted(cents)
            matrix = np.stack([cents[p] for p in pids]) if pids else None
            centroids[species] = (pids, matrix, pets.rejected_crops(conn, pids) if pids else {})
        pids, matrix, rejected = centroids[species]
        if matrix is None or matrix.shape[1] != v.shape[0]:
            continue
        for pid, s in zip(pids, (matrix @ v).tolist()):
            cut = pets.match_threshold(pid)
            likely = s >= cut and not pets._looks_rejected(rejected.get(pid), v)
            _add(b.evs, pid, float(s), likely, "look")
    return boxes


# ------------------------------------------------------------------ the whole person (Merge into)
def _seen_with(conn: sqlite3.Connection, person_id: int) -> set[int]:
    """People in one photo (not a video) with the person: different people (merge suggestions leave them out too)."""
    p = int(person_id)
    return {int(r[0]) for r in conn.execute(
        "WITH mine AS (SELECT item_id FROM faces WHERE person_id = :p UNION SELECT item_id FROM bodies WHERE "
        "person_id = :p UNION SELECT item_id FROM animals WHERE person_id = :p UNION SELECT item_id FROM item_tags "
        "WHERE person_id = :p AND relation = 'in'), "
        "photos AS (SELECT m.item_id FROM mine m JOIN items i ON i.id = m.item_id WHERE i.kind != 'video') "
        "SELECT person_id FROM faces WHERE item_id IN photos AND person_id IS NOT NULL AND person_id != :p UNION "
        "SELECT person_id FROM bodies WHERE item_id IN photos AND person_id IS NOT NULL AND person_id != :p UNION "
        "SELECT person_id FROM animals WHERE item_id IN photos AND person_id IS NOT NULL AND person_id != :p UNION "
        "SELECT person_id FROM item_tags WHERE item_id IN photos AND relation = 'in' AND person_id != :p",
        {"p": p})}


def _dismissed(conn: sqlite3.Connection, person_id: int) -> set[int]:
    p = int(person_id)
    return {int(r[0]) for r in conn.execute(
        "SELECT person_b FROM merge_dismissals WHERE person_a = ? UNION SELECT person_a FROM merge_dismissals "
        "WHERE person_b = ?", (p, p))}


def _centroids(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    """(person ids, unit centroids) of the face index, kept in its derived-data cache until the faces change (merge
    suggestions keep the same centroids under their own key, used when present)."""
    from app.analysis.index import get_index

    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        generation = idx.generation
        hit = idx.cache_get("suggestions")
        if hit is not None:
            return hit[0], hit[1]  # type: ignore[index]
        hit = idx.cache_get(CENTROIDS_KEY)
        if hit is not None:
            return hit  # type: ignore[return-value]
        pids, cents = idx.centroids()
    idx.cache_put(CENTROIDS_KEY, generation, (pids, cents))
    return pids, cents


def _whole_person(conn: sqlite3.Connection, person_id: int) -> list[Option]:
    row = conn.execute("SELECT id, kind, species FROM persons WHERE id = ?", (int(person_id),)).fetchone()
    if row is None:
        return []
    leave_out = _seen_with(conn, person_id) | _dismissed(conn, person_id) | {int(person_id)}
    out: list[Option] = []
    if row["kind"] == "pet":
        from app.analysis import pets

        cents = pets.pet_centroids(conn, str(row["species"] or ""))
        mine = cents.get(int(person_id))
        if mine is None:
            return []
        for pid, v in cents.items():
            if pid in leave_out or v.shape != mine.shape:
                continue
            s, cut = float(v @ mine), pets.match_threshold(pid)
            out.append(Option(int(pid), s, s >= cut, "look"))
    else:
        pids, cents = _centroids(conn)
        where = np.nonzero(pids == int(person_id))[0]
        if where.size == 0:
            return []
        sims = cents @ cents[int(where[0])]
        thr = float(settings.suggest_threshold)
        for pid, s in zip(pids.tolist(), sims.tolist()):
            if int(pid) in leave_out:
                continue
            out.append(Option(int(pid), float(s), float(s) >= thr, "person"))
    out.sort(key=lambda o: (-o.score, o.person_id))
    return out


# ------------------------------------------------------------------ the ranking
def _boxes_of(conn: sqlite3.Connection, person_id: int, item_ids: list[int]) -> tuple[list[int], list[int], list[int]]:
    """The person's faces, person boxes without a face and pet boxes on these items."""
    faces: list[int] = []
    bodies: list[int] = []
    animals: list[int] = []
    for chunk in _chunks(sorted({int(i) for i in item_ids})):
        m = ",".join("?" * len(chunk))
        faces += [int(r[0]) for r in conn.execute(
            f"SELECT id FROM faces WHERE person_id = ? AND item_id IN ({m}) ORDER BY id", [int(person_id), *chunk])]
        bodies += [int(r[0]) for r in conn.execute(
            f"SELECT id FROM bodies WHERE person_id = ? AND face_id IS NULL AND item_id IN ({m}) ORDER BY id",
            [int(person_id), *chunk])]
        animals += [int(r[0]) for r in conn.execute(
            f"SELECT id FROM animals WHERE person_id = ? AND item_id IN ({m}) ORDER BY id", [int(person_id), *chunk])]
    return faces, bodies, animals


def rank(conn: sqlite3.Connection, src: Source) -> list[Option]:
    if src.person_id is not None and not src.item_ids:
        return _whole_person(conn, int(src.person_id))
    face_ids, body_ids, animal_ids = list(src.face_ids), list(src.body_ids), list(src.animal_ids)
    leave_out: set[int] = set()
    if src.person_id is not None:
        f, b, a = _boxes_of(conn, int(src.person_id), src.item_ids)
        face_ids, body_ids, animal_ids = face_ids + f, body_ids + b, animal_ids + a
        leave_out.add(int(src.person_id))
    body_rows = _rows(conn, BODY_SQL, _unique(body_ids))
    # a person box with a face inside is named through that face
    face_ids += [int(r["face_id"]) for r in body_rows if r["face_id"] is not None]
    boxes = (_face_boxes(conn, _unique(face_ids))
             + _body_boxes([r for r in body_rows if r["face_id"] is None])
             + _animal_boxes(conn, _unique(animal_ids)))
    leave_out |= {int(b.person_id) for b in boxes if b.person_id is not None}
    rejected = {kind: _rejected(conn, kind, [b.id for b in boxes if b.kind == kind]) for kind in ("face", "body", "animal")}
    if len(boxes) == 1:
        box = boxes[0]
        if box.kind in ("face", "body"):
            _clothes(conn, box, rejected[box.kind].get(box.id, set()))
        _hour(conn, box.item_id, box.evs)
    if src.around_item_id is not None:
        around = _Box("item", 0, int(src.around_item_id), None, 1.0)
        _hour(conn, around.item_id, around.evs)
        for pid in _present(conn, around):
            around.evs.pop(pid, None)
        boxes.append(around)
    for b in boxes:
        if b.kind == "item" or not b.evs:
            continue
        drop = rejected[b.kind].get(b.id, set()) | _present(conn, b)
        for pid in drop:
            b.evs.pop(pid, None)
    # a box that names nobody (not scored, no vector, everyone it named left out) says nothing: it takes no part in
    # the average, so it does not water the others down
    return _combine([b for b in boxes if b.evs], leave_out)


def _combine(boxes: list[_Box], leave_out: set[int]) -> list[Option]:
    total = sum(b.weight for b in boxes)
    if total <= 0:
        return []
    acc: dict[int, list[Any]] = {}  # person -> [weighted score, likely weight, first weight, best part, basis]
    for b in boxes:
        for pid, ev in b.evs.items():
            if pid in leave_out:
                continue
            a = acc.setdefault(pid, [0.0, 0.0, 0.0, -1.0, ev.basis])
            part = b.weight * ev.score
            a[0] += part
            if ev.likely:
                a[1] += b.weight
            if ev.first:
                a[2] += b.weight
            if part > a[3]:
                a[3], a[4] = part, ev.basis
    half = 0.5 * total - 1e-9
    out = [Option(pid, float(a[0] / total), a[1] >= half, str(a[4]), a[2] >= half) for pid, a in acc.items()]
    out.sort(key=lambda o: (not o.first, -o.score, o.person_id))
    return out
