"""In-memory face embedding index, kept in sync with the DB (adapted from the hub's faces/index.py).

CONTRACT. A process-wide singleton (get_index()) shared by the worker and API threads, guarded by
an RLock. It caches (face id, person id, user-locked flag, face weight, item id, embedding) for every
face and refreshes when kv 'faces_version' changes. A person's faces are their examples in
person_scores and centroids, each counted by its face weight (app.analysis.faces.face_weight): a
blurred, masked or turned-away face the owner or the clustering put on someone stays theirs but
teaches little or nothing about what they look like.
  sync(conn): if the stored version differs, re-read (id, person_id, assigned_by, item_id) for all
  faces, drop rows that disappeared, load embeddings only for ids it has not seen, update the
  rest. That keeps refreshes cheap (no full embedding reload) at 100k+ faces.

State read inside an open write transaction may be rolled back later (and face ids reused, since
a rolled back AUTOINCREMENT insert gives its id back). Such a sync is remembered as tentative:
the next sync refreshes again and reloads the embeddings of the ids that sync added. A later
transaction on the same connection is told apart by conn.total_changes. Code that syncs inside a
write transaction calls invalidate() when that transaction rolls back.
A sync outside a transaction reads the version, the rows and the embeddings in one read
transaction, so they come from one snapshot.
Callers that sync and then query (cluster.py) hold `index.lock` across both, so another
thread cannot swap the state in between.
settle(conn, version) after a commit and advance(old, new) after guess-only writes keep the index
current without the full row re-read a tentative or outdated state would otherwise force.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

from app.analysis.faces import face_weights
from app.db import faces_version

log = logging.getLogger(__name__)

DIM = 512
MIN_SIM = 0.2  # person_scores only scores people with at least one face this similar
_IN_CHUNK = 500  # ids per "IN (...)" query (SQLite variable limit is far above, keeps SQL small)
_SCORE_BATCH = 64  # query embeddings per matrix product in person_scores_many


class FaceIndex:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._version: int | None = None
        self._ids = np.zeros(0, dtype=np.int64)
        self._person = np.zeros(0, dtype=np.int64)  # -1 = unassigned
        self._item = np.zeros(0, dtype=np.int64)
        self._user = np.zeros(0, dtype=bool)
        self._weight = np.zeros(0, dtype=np.float32)  # face weight: how much each face counts as an example
        self._emb = np.zeros((0, DIM), dtype=np.float32)
        self._pos: dict[int, int] = {}
        self._tentative_conn: int | None = None  # id(conn) of a sync made inside a transaction
        self._tentative_changes = 0  # conn.total_changes right after that sync
        self._tentative_ids: set[int] = set()
        self._cache: dict[str, object] = {}  # derived data (cluster.py), dropped on every change
        self._generation = 0  # bumped on every change of the in-memory state

    # -------------------------------------------------------------- derived-data cache
    @property
    def generation(self) -> int:
        return self._generation

    def cache_get(self, key: str) -> object | None:
        with self.lock:
            return self._cache.get(key)

    def cache_put(self, key: str, generation: int, value: object) -> None:
        """Store data derived from the state at `generation`. Ignored if the state changed since,
        or if it may still be rolled back (read inside a write transaction)."""
        with self.lock:
            if generation == self._generation and self._tentative_conn is None and self._version is not None:
                self._cache[key] = value

    def _changed(self) -> None:
        self._cache = {}
        self._generation += 1

    # -------------------------------------------------------------- sync
    def sync(self, conn: sqlite3.Connection, force: bool = False) -> None:
        with self.lock:
            in_tx = conn.in_transaction
            if not in_tx:
                # One read transaction: version, rows and embeddings from the same snapshot.
                conn.execute("BEGIN")
                try:
                    self._sync(conn, force, in_tx=False)
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                conn.execute("COMMIT")
            else:
                self._sync(conn, force, in_tx=True)

    def _sync(self, conn: sqlite3.Connection, force: bool, in_tx: bool) -> None:
        version = faces_version(conn)
        if self._tentative_conn is not None:
            same_tx = (in_tx and self._tentative_conn == id(conn)
                       and self._tentative_changes == conn.total_changes)
            if not (same_tx and version == self._version):
                force = True
        if not force and version == self._version:
            return
        reload_ids = self._tentative_ids
        rows = conn.execute(
            "SELECT id, person_id, assigned_by, item_id, det_score, size_px, clarity, sharpness FROM faces ORDER BY id"
        ).fetchall()
        n = len(rows)
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=n)
        person = np.fromiter((r[1] if r[1] is not None else -1 for r in rows), dtype=np.int64, count=n)
        user = np.fromiter((r[2] == "user" for r in rows), dtype=bool, count=n)
        item = np.fromiter((r[3] for r in rows), dtype=np.int64, count=n)
        weight = _weight_of(rows)
        emb = np.zeros((n, DIM), dtype=np.float32)
        # Carry cached embeddings over (both id arrays are sorted).
        hit = np.zeros(n, dtype=bool)
        if self._ids.size and n:
            pos = np.searchsorted(self._ids, ids)
            pos_c = np.minimum(pos, self._ids.size - 1)
            hit = self._ids[pos_c] == ids
            if reload_ids:
                hit &= ~np.isin(ids, np.fromiter(reload_ids, dtype=np.int64, count=len(reload_ids)))
            emb[hit] = self._emb[pos_c[hit]]
        missing = np.nonzero(~hit)[0].tolist()
        if missing:
            self._load_embeddings(conn, ids, missing, emb)
        self._set_state(ids, person, user, item, emb, version, weight=weight)
        self._mark(conn, in_tx, {int(ids[i]) for i in missing})
        log.debug("face index synced: %d faces (%d loaded), version %d", n, len(missing), version)

    def _set_state(self, ids, person, user, item, emb, version: int | None,
                   pos: dict[int, int] | None = None, *, weight: np.ndarray) -> None:
        self._ids, self._person, self._user, self._item, self._emb = ids, person, user, item, emb
        self._weight = weight
        self._pos = pos if pos is not None else {fid: i for i, fid in enumerate(ids.tolist())}
        self._version = version
        self._changed()

    def settle(self, conn: sqlite3.Connection, version: int) -> None:
        """Call right after the write transaction that last synced this index (sync_item, sync and
        set_person inside it) committed, with `version` the faces_version that transaction ended with.
        The in-memory state then is the committed state, so it stops being tentative and the next
        sync does not re-read every face. A no-op if anything else happened since."""
        with self.lock:
            if (self._tentative_conn != id(conn) or conn.in_transaction
                    or faces_version(conn) != int(version)):
                return
            self._version = int(version)
            self._tentative_conn = None
            self._tentative_ids = set()

    def advance(self, old_version: int, new_version: int) -> None:
        """A committed write moved faces_version from old_version to new_version without changing
        anything the index holds (guess fields only). Current at old means current at new."""
        with self.lock:
            if self._tentative_conn is None and self._version is not None and self._version == int(old_version):
                self._version = int(new_version)

    def _mark(self, conn: sqlite3.Connection, in_tx: bool, loaded_ids: set[int]) -> None:
        if in_tx:
            # Keep ids loaded by an earlier tentative sync of this transaction: they may be rolled
            # back too.
            prev = self._tentative_ids if self._tentative_conn == id(conn) else set()
            self._tentative_conn = id(conn)
            self._tentative_changes = conn.total_changes
            self._tentative_ids = prev | loaded_ids
        else:
            self._tentative_conn = None
            self._tentative_ids = set()

    def sync_item(self, conn: sqlite3.Connection, item_id: int, base_version: int) -> None:
        """Inside a write transaction that changed only the faces of `item_id`: refresh just that
        item's rows when the index already holds the committed state at `base_version` (the
        version read at the start of this transaction). Otherwise a full sync."""
        with self.lock:
            if (not conn.in_transaction or self._tentative_conn is not None
                    or self._version is None or self._version != int(base_version)):
                self._sync(conn, False, in_tx=conn.in_transaction)
                return
            rows = conn.execute(
                "SELECT id, person_id, assigned_by, item_id, det_score, size_px, clarity, sharpness FROM faces "
                "WHERE item_id = ? ORDER BY id",
                (int(item_id),),
            ).fetchall()
            n = len(rows)
            r_ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=n)
            r_person = np.fromiter((r[1] if r[1] is not None else -1 for r in rows), dtype=np.int64, count=n)
            r_user = np.fromiter((r[2] == "user" for r in rows), dtype=bool, count=n)
            r_item = np.full(n, int(item_id), dtype=np.int64)
            r_weight = _weight_of(rows)
            r_emb = np.zeros((n, DIM), dtype=np.float32)
            if n:
                self._load_by_id(conn, r_ids, list(range(n)), r_emb)
            keep = self._item != int(item_id)
            kept_all = bool(keep.all())
            parts = [self._ids, self._person, self._user, self._item, self._emb, self._weight]
            if not kept_all:
                parts = [a[keep] for a in parts]
            if n:
                parts = [np.concatenate([a, b]) for a, b in zip(parts, [r_ids, r_person, r_user, r_item, r_emb, r_weight])]
            pos = None
            if n and parts[0].size > n and r_ids[0] < parts[0][parts[0].size - n - 1]:
                # Some new id sorts before a kept one (rare: ids grow): restore the id order.
                order = np.argsort(parts[0], kind="stable")
                parts = [a[order] for a in parts]
            elif kept_all:
                # New faces appended after every kept row (the usual case): extend the position map.
                pos = self._pos
                base = parts[0].size - n
                for k, fid in enumerate(r_ids.tolist()):
                    pos[int(fid)] = base + k
            *state, weight = parts
            self._set_state(*state, faces_version(conn), pos=pos, weight=weight)
            self._mark(conn, True, {int(f) for f in r_ids.tolist()})
            log.debug("face index: item %s refreshed (%d faces), version %d", item_id, n, self._version)

    @staticmethod
    def _load_embeddings(conn: sqlite3.Connection, ids: np.ndarray, rows: list[int], out: np.ndarray) -> None:
        if len(rows) > len(ids) // 2:
            # Most rows are new (first load): one scan is cheaper than many IN queries.
            pos = {int(ids[i]): i for i in rows}
            for fid, blob in conn.execute("SELECT id, embedding FROM faces"):
                i = pos.get(fid)
                if i is not None:
                    out[i] = _decode(blob)
            return
        FaceIndex._load_by_id(conn, ids, rows, out)

    @staticmethod
    def _load_by_id(conn: sqlite3.Connection, ids: np.ndarray, rows: list[int], out: np.ndarray) -> None:
        for start in range(0, len(rows), _IN_CHUNK):
            chunk = rows[start:start + _IN_CHUNK]
            pos = {int(ids[i]): i for i in chunk}
            marks = ",".join("?" * len(pos))
            for fid, blob in conn.execute(f"SELECT id, embedding FROM faces WHERE id IN ({marks})", list(pos)):
                out[pos[fid]] = _decode(blob)

    def sync_media(self, conn: sqlite3.Connection, item_id: int, base_version: int) -> None:
        """Old name of sync_item (hub)."""
        self.sync_item(conn, item_id, base_version)

    def invalidate(self) -> None:
        """Force a full refresh (embeddings included) on the next sync. Call it when a transaction
        that synced or called set_person rolls back."""
        with self.lock:
            self._set_state(
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.int64), np.zeros((0, DIM), dtype=np.float32), None,
                weight=np.zeros(0, dtype=np.float32),
            )
            self._tentative_conn = None
            self._tentative_ids = set()

    def reload_faces(self, conn: sqlite3.Connection, face_ids: list[int]) -> None:
        """After a commit that changed these faces' vectors or weights (a moved box, or the undo of one): take
        them again from the database. A plain sync would not, as it only loads the vectors of faces it has not
        seen. Faces the index does not hold yet come with the next sync."""
        ids = sorted({int(f) for f in face_ids})
        if not ids:
            return
        with self.lock:
            changed = False
            for start in range(0, len(ids), _IN_CHUNK):
                chunk = ids[start:start + _IN_CHUNK]
                marks = ",".join("?" * len(chunk))
                for r in conn.execute(
                    "SELECT id, person_id, assigned_by, item_id, det_score, size_px, clarity, sharpness, embedding "
                    f"FROM faces WHERE id IN ({marks})", chunk,
                ):
                    pos = self._pos.get(int(r[0]))
                    if pos is None:
                        continue
                    self._emb[pos] = _decode(r[8])
                    self._weight[pos] = _weight_of([r])[0]
                    changed = True
            if changed:
                self._changed()

    def set_person(self, face_id: int, person_id: int | None) -> None:
        """Reflect an assignment made in the current transaction without a full refresh."""
        with self.lock:
            p = self._pos[face_id]
            self._person[p] = -1 if person_id is None else person_id
            self._changed()
            if self._tentative_conn is None:
                # The next sync refreshes the rows even if this transaction rolls back (version
                # unchanged). 0 is never an id(conn).
                self._tentative_conn = 0

    # -------------------------------------------------------------- queries
    def size(self) -> int:
        with self.lock:
            return int(self._ids.shape[0])

    def person_scores(
        self, emb: np.ndarray, topk: int, exclude_face_ids: set[int] | frozenset[int] = frozenset()
    ) -> list[tuple[int, float]]:
        """For each person with faces of weight > 0 in the index: the weighted mean of the top-k
        cosine similarities between emb and those faces (fewer if the person has fewer), each face
        counted by its face weight (and USER_WEIGHT when the owner assigned it). Sorted desc. Only
        people with at least one such similarity >= MIN_SIM are returned; their score uses all of
        those faces, weak matches included. A person whose faces all weigh 0 is never scored."""
        return self.person_scores_many(np.asarray(emb, dtype=np.float32)[None, :], topk, exclude_face_ids)[0]

    def person_scores_many(
        self, embs: np.ndarray, topk: int, exclude_face_ids: set[int] | frozenset[int] = frozenset()
    ) -> list[list[tuple[int, float]]]:
        """person_scores for each row of embs (B, 512), with one matrix product for the batch."""
        embs = np.asarray(embs, dtype=np.float32).reshape(-1, DIM)
        with self.lock:
            if self._ids.shape[0] == 0 or embs.shape[0] == 0:
                return [[] for _ in range(embs.shape[0])]
            base = (self._person >= 0) & (self._weight > 0)
            if exclude_face_ids:
                base &= ~np.isin(self._ids, np.fromiter(exclude_face_ids, dtype=np.int64))
            out: list[list[tuple[int, float]]] = []
            for start in range(0, embs.shape[0], _SCORE_BATCH):
                sims = self._emb @ embs[start:start + _SCORE_BATCH].T  # (N, b)
                out.extend(_topk_means(self._person, base, sims[:, b], topk, self._user, self._weight)
                           for b in range(sims.shape[1]))
            return out

    def centroids(self, person_ids: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(person_ids array, L2-normalised mean embeddings (P, 512)) for people with faces of weight
        > 0, each face counted by its face weight."""
        with self.lock:
            mask = (self._person >= 0) & (self._weight > 0)
            if person_ids is not None:
                mask &= np.isin(self._person, np.asarray(list(person_ids), dtype=np.int64))
            sel = np.nonzero(mask)[0]
            p, e = self._person[sel], self._emb[sel] * self._weight[sel][:, None]
        if p.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, DIM), dtype=np.float32)
        order = np.argsort(p, kind="stable")
        p, e = p[order], e[order]
        pids, starts = np.unique(p, return_index=True)
        sums = np.add.reduceat(e, starts, axis=0)
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        return pids.astype(np.int64), (sums / np.maximum(norms, 1e-12)).astype(np.float32)

    def unassigned_lookalikes(self, emb: np.ndarray, min_sim: float, max_weight: float) -> list[int]:
        """Ids of faces with no person and not locked by the owner that look like emb (cosine >=
        min_sim) and weigh less than max_weight (section 10: look-alikes of a face marked "Not a face")."""
        with self.lock:
            if self._ids.size == 0:
                return []
            sims = self._emb @ np.asarray(emb, dtype=np.float32)
            mask = (self._person < 0) & ~self._user & (self._weight < max_weight) & (sims >= min_sim)
            return [int(i) for i in self._ids[mask].tolist()]

    def embeddings_of(self, face_ids: list[int]) -> np.ndarray:
        """(n, 512) embeddings for the given face ids, in order (KeyError if unknown)."""
        with self.lock:
            rows = [self._pos[int(f)] for f in face_ids]
            return self._emb[rows].copy() if rows else np.zeros((0, DIM), dtype=np.float32)


USER_WEIGHT = 1.5  # a face the owner assigned counts this much in the top-k mean (an automatic one 1.0)


def _topk_means(person: np.ndarray, base: np.ndarray, sims: np.ndarray, topk: int,
                user: np.ndarray | None = None, weight: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Per candidate person: the weighted mean of the top-k similarities to that person's faces.
    The reference for a person is every face they have (in base), so it sharpens as faces are added;
    each counts by its face weight, times USER_WEIGHT for the owner's own assignments."""
    cand = np.unique(person[base & (sims >= MIN_SIM)])
    if cand.size == 0:
        return []
    # Every face of a candidate counts: dropping the weak ones would shrink the divisor and turn
    # the top-k mean into "best single face".
    sel = np.nonzero(base & np.isin(person, cand))[0]
    p, s = person[sel], sims[sel]
    w = np.where(user[sel], USER_WEIGHT, 1.0) if user is not None and user.shape[0] == person.shape[0] else np.ones(sel.size)
    if weight is not None and weight.shape[0] == person.shape[0]:
        w = w * weight[sel]
    order = np.lexsort((-s, p))
    p, s, w = p[order], s[order], w[order]
    starts = np.r_[0, np.nonzero(np.diff(p))[0] + 1]
    group = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, p.size]))
    rank = np.arange(p.size) - starts[group]
    keep = rank < max(int(topk), 1)
    totals = np.bincount(group[keep], weights=(s * w)[keep], minlength=starts.size)
    counts = np.bincount(group[keep], weights=w[keep], minlength=starts.size)
    means = totals / np.maximum(counts, 1e-9)
    pids = p[starts]
    out = sorted(zip(pids.tolist(), means.tolist()), key=lambda t: (-t[1], t[0]))
    return [(int(a), float(b)) for a, b in out]


def _weight_of(rows: list[sqlite3.Row] | list[tuple]) -> np.ndarray:
    """The face weight of face rows (id, person_id, assigned_by, item_id, det_score, size_px, clarity,
    sharpness)."""
    n = len(rows)
    if not n:
        return np.zeros(0, dtype=np.float32)
    nan = float("nan")
    det = np.fromiter((r[4] for r in rows), dtype=np.float64, count=n)
    size = np.fromiter((r[5] for r in rows), dtype=np.float64, count=n)
    clarity = np.fromiter((nan if r[6] is None else r[6] for r in rows), dtype=np.float64, count=n)
    sharp = np.fromiter((nan if r[7] is None else r[7] for r in rows), dtype=np.float64, count=n)
    return face_weights(det, size, clarity, sharp).astype(np.float32)


def _decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=DIM)


_index: FaceIndex | None = None
_index_db: Path | None = None
_index_lock = threading.Lock()


def get_index() -> FaceIndex:
    """Process-wide singleton. It must also fully reload when settings.db_path changes
    (tests point settings at a fresh database per test)."""
    global _index, _index_db
    from app.config import settings

    db_path = settings.db_path
    with _index_lock:
        if _index is None or _index_db != db_path:
            _index = FaceIndex()
            _index_db = db_path
        return _index


def reset_index() -> None:
    """Drop the singleton (tests)."""
    global _index, _index_db
    with _index_lock:
        _index = None
        _index_db = None
