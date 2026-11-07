"""Semantic search over the stored image embeddings (table media_embeddings).

- Engine access goes through get_engine() / engine_available() / current_model(), which import
  app.search.engine lazily: a missing or broken model never breaks app import or startup, and
  tests replace these three functions with a fake engine.
- EmbeddingMatrix keeps every embedding of the current model in memory (one row per photo, one
  per sampled video frame, and one per person crop of a photo face). It refreshes when kv
  'search_version' (bumped by the indexer in the same transaction as its writes) or a row count
  (cascade deletes) changes, and then loads only the rows it has not seen. Thread-safe: sync
  swaps whole arrays under a lock, so a search that took references to the arrays keeps a
  consistent snapshot.
- search(): cosine similarity of the text embedding against every row, best frame per item,
  engine.relevance() -> 0..1, then two gates, then the visibility / kind / people filters in SQL
  and paging by offset (score desc, id asc):
  1. relevance >= max(settings.search_min_relevance, settings.search_contrast (20) x the query's
     median relevance over the whole index). A real query lifts a few items far above its median
     (on the event photos "group photo" tops out ~5700x its median); gibberish reads to the model
     like an unknown name and lifts many items a little (median ~50x), so most of it drops out
     (34 made-up strings: 1658 results -> 124). A larger factor would also drop sparse real
     queries ("table" tops out ~100x).
  2. A query that names a colour must match that colour better than the same query with any
     other colour ("a man in a green shirt" vs "... orange shirt"): the model otherwise matches
     the object ("shirt") and returns people in any colour of shirt. Checked per row (photo or
     video frame), so a video passes on the frames that show the colour.
- Person crops: one person in a busy scene barely registers in the whole photo (a pale green shirt
  at a restaurant table scores near 0 for "person with green shirt"), so every photo face also
  has a head-and-torso crop. Crop rows add photos on their own: best crop per photo, the same
  colour check per row, and a relevance of at least max(the threshold of gate 1,
  settings.search_person_min_relevance (0.4)). They never feed the median of gate 1, so every
  whole-photo result stays; a photo found both ways is ranked by the higher of its two scores.
  The floor sits above every crop score of 40 made-up strings (highest 0.363) and 15 absent
  objects (0.355, "a football" on a TV showing cricket); crops lift name-like gibberish because
  they look like portraits, which is why a fixed floor is used and no contrast factor (gibberish
  often scores one crop high and the rest near 0, as sharply as a real query).
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
from dataclasses import dataclass, field

import numpy as np

from app.api.common import MEDIA_ITEM_COLUMNS, people_filter_sql
from app.config import settings
from app.db import kv_get

log = logging.getLogger(__name__)

FALLBACK_MODEL_ID = "ViT-B-16-SigLIP__webli"  # app.search.engine.MODEL_ID, used when it cannot be imported
SEARCH_VERSION_KEY = "search_version"
_IN_CHUNK = 500  # ids per "IN (...)" query

# Colours the colour check compares; synonyms map onto one of them.
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey")
_COLOR_SYNONYMS = {"gray": "grey", "violet": "purple", "maroon": "red", "navy": "blue"}
_WORD = re.compile(r"[A-Za-z]+")


def query_color(q: str) -> tuple[str, str] | None:
    """(word as typed, canonical colour) for the first colour word in the query, else None."""
    for m in _WORD.finditer(q):
        word = m.group(0).lower()
        color = word if word in COLORS else _COLOR_SYNONYMS.get(word)
        if color:
            return m.group(0), color
    return None


def color_variants(q: str, word: str) -> list[str]:
    """The query once per colour in COLORS, with the colour word replaced (whole word only)."""
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return [pattern.sub(color, q, count=1) for color in COLORS]


class SearchUnavailable(RuntimeError):
    """The search model is missing, disabled or failed to run."""


# --------------------------------------------------------------------- engine access
def engine_available() -> bool:
    """True when search is enabled and the model files are present."""
    if not settings.search_enabled:
        return False
    try:
        from app.search.engine import search_available
    except Exception:
        log.exception("search engine module could not be imported")
        return False
    try:
        return bool(search_available())
    except Exception:
        log.exception("could not check the search model files")
        return False


def get_engine():
    """The process-wide search engine (loads the model on first use)."""
    from app.search.engine import get_search_engine

    return get_search_engine()


def current_model() -> str:
    """Model id the embeddings are stored under."""
    try:
        from app.search.engine import MODEL_ID
    except Exception:
        return FALLBACK_MODEL_ID
    return str(MODEL_ID)


def bump_search_version(conn: sqlite3.Connection) -> None:
    """Call inside the transaction that changes media_embeddings."""
    conn.execute(
        "INSERT INTO kv(key, value) VALUES (?, '1') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
        (SEARCH_VERSION_KEY,),
    )


# --------------------------------------------------------------------- in-memory matrix
@dataclass
class _Snapshot:
    media: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))  # NaN for photos
    emb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    person_media: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # photo of each crop
    person_emb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


class EmbeddingMatrix:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._model: str | None = None
        self._sig: tuple[str | None, int, int] | None = None
        # (media_id, frame_index) -> (version, frame_time_s) of the row loaded at that position
        self._keys: list[tuple[int, int]] = []
        self._meta: dict[tuple[int, int], tuple[int, float | None]] = {}
        self._person_keys: list[int] = []  # face id of each person-crop row
        self._snap = _Snapshot()

    def reset(self) -> None:
        with self.lock:
            self._model = None
            self._sig = None
            self._keys = []
            self._meta = {}
            self._person_keys = []
            self._snap = _Snapshot()

    def snapshot(self) -> _Snapshot:
        with self.lock:
            return self._snap

    @property
    def size(self) -> int:
        return int(self.snapshot().media.shape[0])

    def sync(self, conn: sqlite3.Connection, model: str) -> None:
        with self.lock:
            own_tx = not conn.in_transaction
            if own_tx:
                conn.execute("BEGIN")  # one read snapshot for the signature, keys and blobs
            try:
                self._sync_locked(conn, model)
            finally:
                if own_tx:
                    conn.execute("COMMIT")

    def _sync_locked(self, conn: sqlite3.Connection, model: str) -> None:
        sig = (
            kv_get(conn, SEARCH_VERSION_KEY),
            int(conn.execute("SELECT COUNT(*) FROM media_embeddings WHERE model = ?", (model,)).fetchone()[0]),
            int(conn.execute(
                "SELECT COUNT(*) FROM person_embeddings WHERE model = ? AND embedding IS NOT NULL", (model,)
            ).fetchone()[0]),
        )
        if model == self._model and sig == self._sig:
            return
        if model != self._model:
            self._keys, self._meta, self._person_keys, self._snap = [], {}, [], _Snapshot()
        rows = conn.execute(
            "SELECT media_id, frame_index, version, frame_time_s FROM media_embeddings WHERE model = ? "
            "ORDER BY media_id, frame_index",
            (model,),
        ).fetchall()
        old_pos = {k: i for i, k in enumerate(self._keys)}
        keys: list[tuple[int, int]] = []
        meta: dict[tuple[int, int], tuple[int, float | None]] = {}
        keep_src: list[int] = []  # position in the old arrays, or -1 for a row to load
        missing: dict[int, set[int]] = {}
        for r in rows:
            key = (int(r["media_id"]), int(r["frame_index"]))
            m = (int(r["version"]), r["frame_time_s"])
            keys.append(key)
            meta[key] = m
            pos = old_pos.get(key)
            if pos is not None and self._meta.get(key) == m:
                keep_src.append(pos)
            else:
                keep_src.append(-1)
                missing.setdefault(key[0], set()).add(key[1])

        loaded: dict[tuple[int, int], np.ndarray] = {}
        ids = sorted(missing)
        for i in range(0, len(ids), _IN_CHUNK):
            chunk = ids[i:i + _IN_CHUNK]
            ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT media_id, frame_index, embedding FROM media_embeddings "
                f"WHERE model = ? AND media_id IN ({ph})",
                [model, *chunk],
            ):
                key = (int(r["media_id"]), int(r["frame_index"]))
                if key[1] in missing.get(key[0], ()):
                    loaded[key] = np.frombuffer(r["embedding"], dtype=np.float32)

        old = self._snap
        dims = {v.shape[0] for v in loaded.values()}
        if old.emb.shape[0]:
            dims.add(int(old.emb.shape[1]))
        if len(dims) > 1:
            log.warning("search embeddings of model %s have mixed sizes %s; using the most common", model, dims)
        dim = _common_dim(loaded, old) if dims else 0

        n = len(keys)
        emb = np.zeros((n, dim), dtype=np.float32)
        valid = np.ones(n, dtype=bool)
        for i, (key, src) in enumerate(zip(keys, keep_src)):
            if src >= 0 and old.emb.shape[1] == dim:
                emb[i] = old.emb[src]
                continue
            v = loaded.get(key)
            if v is None or v.shape[0] != dim:
                valid[i] = False  # deleted between the two reads, or a bad blob
                continue
            emb[i] = v
        media = np.fromiter((k[0] for k in keys), dtype=np.int64, count=n)
        times = np.fromiter(
            ((meta[k][1] if meta[k][1] is not None else math.nan) for k in keys), dtype=np.float64, count=n
        )
        if not valid.all():
            emb, media, times = emb[valid], media[valid], times[valid]
            keys = [k for k, ok in zip(keys, valid) if ok]
            meta = {k: meta[k] for k in keys}
        person_media, person_emb, person_keys, person_loaded = self._sync_people(conn, model, dim, old)
        self._snap = _Snapshot(media=media, times=times, emb=emb, person_media=person_media, person_emb=person_emb)
        self._keys, self._meta, self._person_keys = keys, meta, person_keys
        self._model, self._sig = model, sig
        log.debug("search matrix: %d rows (%d loaded), %d person crops (%d loaded)",
                  len(keys), len(loaded), len(person_keys), person_loaded)

    def _sync_people(self, conn: sqlite3.Connection, model: str, dim: int,
                     old: _Snapshot) -> tuple[np.ndarray, np.ndarray, list[int], int]:
        """Person-crop rows in face id order: (photo ids, embeddings, face ids, rows loaded).
        Rows already in `old` are reused; a face never changes photo or crop under one model."""
        rows = conn.execute(
            "SELECT p.face_id, f.media_id FROM person_embeddings p JOIN faces f ON f.id = p.face_id "
            "WHERE p.model = ? AND p.embedding IS NOT NULL ORDER BY p.face_id",
            (model,),
        ).fetchall()
        old_pos = {fid: i for i, fid in enumerate(self._person_keys)}
        reuse = old.person_emb.shape[0] > 0 and (not dim or old.person_emb.shape[1] == dim)
        missing = [int(r["face_id"]) for r in rows if not (reuse and int(r["face_id"]) in old_pos)]
        loaded: dict[int, np.ndarray] = {}
        for i in range(0, len(missing), _IN_CHUNK):
            chunk = missing[i:i + _IN_CHUNK]
            ph = ",".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT face_id, embedding FROM person_embeddings WHERE model = ? AND face_id IN ({ph}) "
                "AND embedding IS NOT NULL",
                [model, *chunk],
            ):
                loaded[int(r["face_id"])] = np.frombuffer(r["embedding"], dtype=np.float32)
        if not dim:  # no photo or frame rows yet: the crops set the size
            dims = [v.shape[0] for v in loaded.values()] + ([int(old.person_emb.shape[1])] if reuse else [])
            dim = max(set(dims), key=dims.count) if dims else 0
        keys: list[int] = []
        media: list[int] = []
        vecs: list[np.ndarray] = []
        for r in rows:
            fid = int(r["face_id"])
            pos = old_pos.get(fid) if reuse else None
            v = old.person_emb[pos] if pos is not None else loaded.get(fid)
            if v is None or v.shape[0] != dim:
                continue  # deleted between the two reads, or a bad blob
            keys.append(fid)
            media.append(int(r["media_id"]))
            vecs.append(v)
        emb = np.stack(vecs).astype(np.float32, copy=False) if vecs else np.zeros((0, dim), dtype=np.float32)
        return np.asarray(media, dtype=np.int64), emb, keys, len(loaded)


def _common_dim(loaded: dict[tuple[int, int], np.ndarray], old: _Snapshot) -> int:
    counts: dict[int, int] = {}
    for v in loaded.values():
        counts[v.shape[0]] = counts.get(v.shape[0], 0) + 1
    if old.emb.shape[0]:
        counts[int(old.emb.shape[1])] = counts.get(int(old.emb.shape[1]), 0) + int(old.emb.shape[0])
    return max(counts, key=lambda d: (counts[d], d))


_matrix = EmbeddingMatrix()


def get_matrix() -> EmbeddingMatrix:
    return _matrix


def reset_matrix() -> None:
    """Drop the in-memory matrix (tests: each test has a fresh database)."""
    _matrix.reset()


def _best_rows(media: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Index of the highest-scoring row of each item, items in id order."""
    order = np.lexsort((-cos, media))  # by media id, best row first within each item
    media_sorted = media[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = media_sorted[1:] != media_sorted[:-1]
    return order[first]


# --------------------------------------------------------------------- search
@dataclass
class SearchHitRow:
    row: sqlite3.Row  # MEDIA_ITEM_COLUMNS
    score: float
    match_time_s: float | None


@dataclass
class SearchResult:
    hits: list[SearchHitRow]
    total: int
    next_offset: int | None


def search(
    conn: sqlite3.Connection,
    q: str,
    person_ids: list[int] | None = None,
    kind: str | None = None,
    limit: int = 60,
    offset: int = 0,
    include_hidden: bool = False,
) -> SearchResult:
    """Items matching the text query, best first. Raises SearchUnavailable when the model is
    missing or fails."""
    empty = SearchResult(hits=[], total=0, next_offset=None)
    ids_filter = sorted({int(p) for p in person_ids or []})
    if ids_filter and not include_hidden:
        ph = ",".join("?" * len(ids_filter))
        visible = conn.execute(
            f"SELECT COUNT(*) FROM persons WHERE id IN ({ph}) AND hidden = 0", ids_filter
        ).fetchone()[0]
        if visible != len(ids_filter):
            return empty  # a hidden (or unknown) person must not reveal which items they are in
    if not engine_available():
        raise SearchUnavailable("search model is not available")
    model = current_model()
    try:
        engine = get_engine()
        query = np.asarray(engine.embed_text(q), dtype=np.float32).reshape(-1)
    except Exception as e:
        log.exception("search: could not embed the query")
        raise SearchUnavailable(str(e)) from e

    _matrix.sync(conn, model)
    snap = _matrix.snapshot()
    whole = snap.media.shape[0] > 0 and snap.emb.shape[1] == query.shape[0]
    people = (settings.search_person_crops and snap.person_media.shape[0] > 0
              and snap.person_emb.shape[1] == query.shape[0])
    if not whole and not people:
        return empty
    threshold = float(settings.search_min_relevance)
    if whole:
        cos = snap.emb @ query
        best = _best_rows(snap.media, cos)
        rel = np.asarray(engine.relevance(cos[best]), dtype=np.float64).reshape(-1)
        if settings.search_contrast > 0:
            threshold = max(threshold, float(settings.search_contrast) * float(np.median(rel)))
        if not (rel >= threshold).any():
            whole = False
    floor = max(threshold, float(settings.search_person_min_relevance))
    if people:
        pcos = snap.person_emb @ query
        if not (np.asarray(engine.relevance(pcos.max()), dtype=np.float64) >= floor).any():
            people = False
    if not whole and not people:
        return empty
    color = query_color(q) if settings.search_color_check else None
    variants = None
    if color is not None:
        try:
            variants = np.stack([np.asarray(engine.embed_text(v), dtype=np.float32).reshape(-1)
                                 for v in color_variants(q, color[0])])
        except Exception as e:
            log.exception("search: colour check failed")
            raise SearchUnavailable(str(e)) from e
    found: dict[int, tuple[float, float]] = {}  # media id -> (relevance, frame time or NaN)
    if whole:
        if variants is not None:
            # checked per row, so a video counts with the frames that show the colour (and is
            # scored, and seeks, by the best of those)
            ok = (snap.emb @ variants.T).argmax(axis=1) == COLORS.index(color[1])
            best = _best_rows(snap.media, np.where(ok, cos, -np.inf))
            best = best[ok[best]]  # items without a single row of the colour drop out
            rel = np.asarray(engine.relevance(cos[best]), dtype=np.float64).reshape(-1)
        keep = rel >= threshold
        for mid, r, t in zip(snap.media[best][keep].tolist(), rel[keep].tolist(), snap.times[best][keep].tolist()):
            found[mid] = (r, t)
    if people:
        # person crops add photos on their own floor; they never feed the median above
        if variants is not None:
            pok = (snap.person_emb @ variants.T).argmax(axis=1) == COLORS.index(color[1])
            pbest = _best_rows(snap.person_media, np.where(pok, pcos, -np.inf))
            pbest = pbest[pok[pbest]]
        else:
            pbest = _best_rows(snap.person_media, pcos)
        prel = np.asarray(engine.relevance(pcos[pbest]), dtype=np.float64).reshape(-1)
        for mid, r in zip(snap.person_media[pbest].tolist(), prel.tolist()):
            if r >= floor and r > found.get(mid, (-1.0, math.nan))[0]:
                found[mid] = (r, math.nan)
    if not found:
        return empty
    media_ids = np.fromiter(found, dtype=np.int64, count=len(found))
    rel = np.fromiter((v[0] for v in found.values()), dtype=np.float64, count=len(found))
    times = np.fromiter((v[1] for v in found.values()), dtype=np.float64, count=len(found))
    order = np.lexsort((media_ids, -rel))
    media_ids, rel, times = media_ids[order], rel[order], times[order]

    rows = _filter_rows(conn, [int(x) for x in media_ids], ids_filter, kind, include_hidden)
    hits: list[SearchHitRow] = []
    for mid, score, t in zip(media_ids.tolist(), rel.tolist(), times.tolist()):
        row = rows.get(mid)
        if row is None:
            continue
        match = None if row["kind"] != "video" or math.isnan(t) else round(float(t), 6)
        hits.append(SearchHitRow(row=row, score=round(min(1.0, max(0.0, float(score))), 6), match_time_s=match))
    total = len(hits)
    page = hits[offset:offset + limit]
    next_offset = offset + limit if offset + limit < total else None
    return SearchResult(hits=page, total=total, next_offset=next_offset)


def _filter_rows(conn: sqlite3.Connection, ids: list[int], person_ids: list[int], kind: str | None,
                 include_hidden: bool) -> dict[int, sqlite3.Row]:
    where: list[str] = []
    params: list = []
    if not include_hidden:
        where.append("m.hidden = 0")
    if kind is not None:
        where.append("m.kind = ?")
        params.append(kind)
    if person_ids:
        sql, p = people_filter_sql(person_ids)
        where.append(sql)
        params.extend(p)
    out: dict[int, sqlite3.Row] = {}
    for i in range(0, len(ids), _IN_CHUNK):
        chunk = ids[i:i + _IN_CHUNK]
        ph = ",".join("?" * len(chunk))
        cond = " AND ".join([f"m.id IN ({ph})", *where])
        for r in conn.execute(f"SELECT {MEDIA_ITEM_COLUMNS} FROM media m WHERE {cond}", [*chunk, *params]):
            out[int(r["id"])] = r
    return out


# --------------------------------------------------------------------- progress
def indexed_count(conn: sqlite3.Connection, model: str | None = None, include_hidden: bool = True) -> int:
    """Items with embeddings for the model and their current version."""
    model = model or current_model()
    hidden = "" if include_hidden else "m.hidden = 0 AND "
    return int(conn.execute(
        f"SELECT COUNT(*) FROM media m WHERE {hidden}EXISTS (SELECT 1 FROM media_embeddings e "
        "WHERE e.media_id = m.id AND e.model = ? AND e.version = m.version)",
        (model,),
    ).fetchone()[0])
