"""Search: SigLIP semantic search over stored embeddings plus FTS5 full-text search, merged.

Contract (docs/ARCHITECTURE.md sections 12 and 13):

- search(conn, q, *, mode, category, kind, person_ids, date_from, date_to, vault, limit, offset)
  -> SearchResult(hits, total, next_offset, indexed, indexing). hits are dicts
  {id, score, match_time_s, snippet, source}; the API turns them into SearchHit cards.
- Nothing here writes. The analysis stage writes item_embeddings and faces/bodies.torso_embedding
  and bumps kv 'search_version' (app.db.bump_search_version) in the same transaction; the
  documents stage keeps items_fts current (app.documents.extract.refresh_fts).
- The SigLIP engine is reached through app.ml.siglip (available(), get_engine(), MODEL_ID),
  imported lazily by _siglip_module() so a missing model or a broken import never breaks the
  app, and tests replace that one function with a fake. Without the engine (or with
  settings.search_enabled off) semantic search is skipped: mode auto answers with text hits only
  and reports indexed 0; mode semantic raises SearchUnavailable (the API answers 503).
  Full-text search needs no model and always runs.

Semantic engine (adapted from the hub):
- EmbeddingMatrix keeps every whole-image row of the current model in memory (frame 0 is the
  whole image, page 1 of a document or a video poster; videos add one row per sampled frame) and
  every person crop row (faces.torso_embedding, plus bodies.torso_embedding of bodies without a
  face). It refreshes when (kv search_version, count of item_embeddings rows of the model, count
  of torso embeddings) changes, loads only the rows it has not seen (IN chunks of 500) and swaps
  whole arrays under a lock, so a running search keeps a consistent snapshot. Rows of an older
  item version are ignored.
- Scoring: cosine of the text embedding against every row, best row per item,
  engine.relevance() -> 0..1, then the hub's gates:
  1. relevance >= max(settings.search_min_relevance, settings.search_contrast x the query's
     median relevance over the whole index): a real query lifts a few items far above its
     median, gibberish lifts many a little.
  2. A query naming a colour must match that colour better than the same query with any other
     colour, checked per row (a video passes on the frames that show the colour).
  Person crops add items on their own floor max(gate 1 threshold,
  settings.search_person_min_relevance) and never feed the median; an item found both ways
  keeps the higher relevance. Video hits (and crops of video faces) carry match_time_s.

Text engine:
- The query is sanitised into FTS5 terms: every whitespace-separated piece becomes one quoted
  string (so punctuation, column filters and operators lose their meaning and a piece like
  "1234-5678" becomes a phrase), a trailing * keeps prefix search, bare AND / OR / NOT / NEAR are
  dropped, at most MAX_TERMS terms.
- First the strict query (every term, implicit AND) runs; when it finds nothing and there are
  several terms, the relaxed query (any term, OR) runs and its scores are halved. Ranking is
  bm25 with column weights FTS_WEIGHTS (filename, text, doc, people, labels). Items with no
  items_fts row yet (just uploaded) are also matched on their file name with LIKE.
- Snippets: snippet(items_fts, 1, <b>, </b>, '...', 12) over the text column (the doc column
  when the text column has no match), HTML-escaped around the markers so the frontend may render
  them; None when neither column matched (a file name or label match).

Merge (mode auto): text hits go first when the strict query found an item through its file
name, text, document fields or people (a "strong" text hit, labels alone do not count),
otherwise semantic hits go first; the other list follows with the items not already listed.
Scores are normalised per list against that list's best hit (best = 1.0); an item found by both
gets source 'both', the higher of its two scores, the text snippet and the semantic match time.
mode semantic / text use one engine only.

Filters (all in SQL): app.api.common.visible_sql(vault=..., misclicks='include',
dups='collapse') (trash and NSFW excluded, or only NSFW inside the vault; mis-click shots are
searchable; duplicate copies collapse onto the best of their group), category, kind, people (all
of them, app.api.common.people_filter_sql), and the capture date COALESCE(taken_at, created_at)
from date_from up to and including date_to. Paging is by offset over the whole merged list,
recomputed on every page (capped at TEXT_LIMIT text hits).

indexed = items (not in the trash, NSFW or not as the vault flag says) with an embedding of the
current model and version, 0 when the engine is unavailable; indexing = some live item still
waits for analysis or text extraction.
"""
from __future__ import annotations

import html
import logging
import math
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Hashable

import numpy as np

from app.config import settings
from app.db import kv_get

log = logging.getLogger(__name__)

FALLBACK_MODEL_ID = "ViT-B-16-SigLIP__webli"  # app.ml.siglip.MODEL_ID when that module cannot be imported
SEARCH_VERSION_KEY = "search_version"
MODES = ("auto", "semantic", "text")
_IN_CHUNK = 500  # ids per "IN (...)" query

# Full-text search
MAX_TERMS = 16
TEXT_LIMIT = 2000  # FTS candidates per query
FTS_WEIGHTS = (2.0, 1.0, 2.0, 3.0, 0.5)  # bm25 weights: filename, text, doc, people, labels
STRONG_COLUMNS = "{filename text doc people}"  # a strict hit here puts text hits first
RELAXED_FACTOR = 0.5  # score factor for hits of the OR query
FILENAME_ONLY_SCORE = 0.25  # items without an items_fts row, matched on their file name
SNIPPET_TOKENS = 12
_OPEN, _CLOSE = "\x02", "\x03"  # snippet markers, replaced by <b> </b> after escaping
_FTS_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})
_KEEP = re.compile(r"[^\w\-./@&+']+", re.UNICODE)
_HAS_WORD = re.compile(r"[^\W_]", re.UNICODE)

# Colours the colour check compares; synonyms map onto one of them.
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey")
_COLOR_SYNONYMS = {"gray": "grey", "violet": "purple", "maroon": "red", "navy": "blue"}
_WORD = re.compile(r"[A-Za-z]+")


class SearchUnavailable(RuntimeError):
    """The requested engines cannot answer (semantic model missing in mode semantic, or both failed)."""


# --------------------------------------------------------------------- engine access
def _siglip_module():
    """app.ml.siglip, imported on first use (tests replace this function)."""
    from app.ml import siglip

    return siglip


def current_model() -> str:
    """Model id the embeddings are stored under."""
    try:
        return str(getattr(_siglip_module(), "MODEL_ID", FALLBACK_MODEL_ID))
    except Exception:
        return FALLBACK_MODEL_ID


def engine_available() -> bool:
    """True when semantic search is enabled and the SigLIP model can be used."""
    if not settings.search_enabled:
        return False
    try:
        return bool(_siglip_module().available())
    except Exception:
        log.warning("semantic search engine is not available", exc_info=True)
        return False


def get_engine():
    """The process-wide SigLIP engine (loads the model on first use)."""
    return _siglip_module().get_engine()


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


# --------------------------------------------------------------------- in-memory matrix
def _empty_emb() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float32)


@dataclass
class _Block:
    """Rows of one kind (whole images and frames, or person crops) in key order."""
    keys: list[Hashable] = field(default_factory=list)
    metas: list[Any] = field(default_factory=list)  # changes when a row is rewritten
    items: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))  # NaN when not a frame
    emb: np.ndarray = field(default_factory=_empty_emb)

    @property
    def size(self) -> int:
        return len(self.keys)

    @property
    def dim(self) -> int:
        return int(self.emb.shape[1]) if self.emb.ndim == 2 else 0


@dataclass
class _Snapshot:
    frames: _Block = field(default_factory=_Block)
    crops: _Block = field(default_factory=_Block)


def _vector(blob: bytes | None) -> np.ndarray | None:
    if not blob or len(blob) % 4:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _refresh_block(old: _Block, rows: list[tuple[Hashable, int, float | None, Any]],
                   load: Callable[[list[Hashable]], dict[Hashable, np.ndarray]]) -> tuple[_Block, int]:
    """New block for rows (key, item id, time, meta); rows whose key and meta are unchanged are
    copied from `old`, the others loaded. Keeps the most common embedding size."""
    old_pos = {k: i for i, k in enumerate(old.keys)}
    missing = []
    for key, _item, _t, meta in rows:
        pos = old_pos.get(key)
        if pos is None or old.metas[pos] != meta:
            missing.append(key)
    loaded = load(missing) if missing else {}
    counts: Counter[int] = Counter(int(v.shape[0]) for v in loaded.values())
    if old.size and old.dim:
        counts[old.dim] += old.size
    if len(counts) > 1:
        log.warning("search embeddings have mixed sizes %s; using the most common", dict(counts))
    dim = max(counts, key=lambda d: (counts[d], d)) if counts else 0

    keys: list[Hashable] = []
    metas: list[Any] = []
    items: list[int] = []
    times: list[float] = []
    vecs: list[np.ndarray] = []
    for key, item, t, meta in rows:
        v = loaded.get(key)
        if v is None:
            pos = old_pos.get(key)
            if pos is not None and old.metas[pos] == meta and old.dim == dim:
                v = old.emb[pos]
        if v is None or int(v.shape[0]) != dim:
            continue  # deleted between the two reads, or a bad blob
        keys.append(key)
        metas.append(meta)
        items.append(int(item))
        times.append(float(t) if t is not None else math.nan)
        vecs.append(v)
    emb = np.stack(vecs).astype(np.float32, copy=False) if vecs else np.zeros((0, dim), dtype=np.float32)
    block = _Block(keys=keys, metas=metas, items=np.asarray(items, dtype=np.int64),
                   times=np.asarray(times, dtype=np.float64), emb=emb)
    return block, len(loaded)


class EmbeddingMatrix:
    """Every searchable embedding of one model in memory (see the module docstring)."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._model: str | None = None
        self._sig: tuple[Any, ...] | None = None
        self._snap = _Snapshot()

    def reset(self) -> None:
        with self.lock:
            self._model = None
            self._sig = None
            self._snap = _Snapshot()

    def snapshot(self) -> _Snapshot:
        with self.lock:
            return self._snap

    @property
    def size(self) -> int:
        return self.snapshot().frames.size

    def sync(self, conn: sqlite3.Connection, model: str) -> None:
        with self.lock:
            own_tx = not conn.in_transaction
            if own_tx:
                conn.execute("BEGIN")  # one read snapshot for the signature, keys and blobs
            try:
                self._sync_locked(conn, model)
            finally:
                if own_tx and conn.in_transaction:
                    conn.execute("COMMIT")

    @staticmethod
    def signature(conn: sqlite3.Connection, model: str) -> tuple[Any, ...]:
        return (
            kv_get(conn, SEARCH_VERSION_KEY),
            int(conn.execute("SELECT COUNT(*) FROM item_embeddings WHERE model = ?", (model,)).fetchone()[0]),
            int(conn.execute(
                "SELECT (SELECT COUNT(*) FROM faces WHERE torso_embedding IS NOT NULL) + "
                "(SELECT COUNT(*) FROM bodies WHERE face_id IS NULL AND torso_embedding IS NOT NULL)"
            ).fetchone()[0]),
        )

    def _sync_locked(self, conn: sqlite3.Connection, model: str) -> None:
        sig = self.signature(conn, model)
        if model == self._model and sig == self._sig:
            return
        old = self._snap if model == self._model else _Snapshot()

        frame_rows = [
            ((int(r["item_id"]), int(r["frame_index"])), int(r["item_id"]), r["frame_time_s"],
             (int(r["version"]), r["frame_time_s"], r["created_at"]))
            for r in conn.execute(
                "SELECT e.item_id, e.frame_index, e.version, e.frame_time_s, e.created_at FROM item_embeddings e "
                "JOIN items i ON i.id = e.item_id AND i.version = e.version "
                "WHERE e.model = ? ORDER BY e.item_id, e.frame_index",
                (model,),
            )
        ]

        def load_frames(keys: list[Hashable]) -> dict[Hashable, np.ndarray]:
            wanted = set(keys)
            ids = sorted({k[0] for k in keys})  # type: ignore[index]
            out: dict[Hashable, np.ndarray] = {}
            for i in range(0, len(ids), _IN_CHUNK):
                chunk = ids[i:i + _IN_CHUNK]
                ph = ",".join("?" * len(chunk))
                for r in conn.execute(
                    f"SELECT item_id, frame_index, embedding FROM item_embeddings "
                    f"WHERE model = ? AND item_id IN ({ph})",
                    [model, *chunk],
                ):
                    key = (int(r["item_id"]), int(r["frame_index"]))
                    v = _vector(r["embedding"])
                    if key in wanted and v is not None:
                        out[key] = v
            return out

        # Person crops: faces by id, faceless bodies by -id (one key space).
        crop_rows = [
            (int(r["key"]), int(r["item_id"]), r["frame_time_s"], r["created_at"])
            for r in conn.execute(
                "SELECT id AS key, item_id, frame_time_s, created_at FROM faces WHERE torso_embedding IS NOT NULL "
                "UNION ALL "
                "SELECT -id AS key, item_id, frame_time_s, created_at FROM bodies "
                "WHERE face_id IS NULL AND torso_embedding IS NOT NULL "
                "ORDER BY 1"
            )
        ]

        def load_crops(keys: list[Hashable]) -> dict[Hashable, np.ndarray]:
            out: dict[Hashable, np.ndarray] = {}
            face_ids = sorted(int(k) for k in keys if int(k) > 0)  # type: ignore[call-overload]
            body_ids = sorted(-int(k) for k in keys if int(k) < 0)  # type: ignore[call-overload]
            for table, ids, sign in (("faces", face_ids, 1), ("bodies", body_ids, -1)):
                for i in range(0, len(ids), _IN_CHUNK):
                    chunk = ids[i:i + _IN_CHUNK]
                    ph = ",".join("?" * len(chunk))
                    for r in conn.execute(
                        f"SELECT id, torso_embedding FROM {table} WHERE id IN ({ph}) AND torso_embedding IS NOT NULL",
                        chunk,
                    ):
                        v = _vector(r["torso_embedding"])
                        if v is not None:
                            out[sign * int(r["id"])] = v
            return out

        frames, n_frames = _refresh_block(old.frames, frame_rows, load_frames)
        crops, n_crops = _refresh_block(old.crops, crop_rows, load_crops)
        self._snap = _Snapshot(frames=frames, crops=crops)
        self._model, self._sig = model, sig
        log.debug("search matrix: %d rows (%d loaded), %d person crops (%d loaded)",
                  frames.size, n_frames, crops.size, n_crops)


_matrix = EmbeddingMatrix()


def get_matrix() -> EmbeddingMatrix:
    return _matrix


def reset_matrix() -> None:
    """Drop the in-memory matrix (tests: each test has a fresh database)."""
    _matrix.reset()


def _best_rows(items: np.ndarray, cos: np.ndarray) -> np.ndarray:
    """Index of the highest-scoring row of each item, items in id order."""
    if items.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((-cos, items))  # by item id, best row first within each item
    sorted_items = items[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_items[1:] != sorted_items[:-1]
    return order[first]


# --------------------------------------------------------------------- filters
@dataclass
class _Filters:
    sql: str
    params: list[Any]


def _iso_date(value: date | str | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _filters(*, vault: bool, category: str | None, kind: str | None, person_ids: list[int] | None,
             date_from: date | str | None, date_to: date | str | None) -> _Filters:
    from app.api.common import people_filter_sql, visible_sql

    parts = [visible_sql(vault=vault, misclicks="include", dups="collapse", alias="i")]
    params: list[Any] = []
    if category:
        parts.append("i.category = ?")
        params.append(category)
    if kind:
        parts.append("i.kind = ?")
        params.append(kind)
    ids = sorted({int(p) for p in person_ids or []})
    if ids:
        sql, p = people_filter_sql(ids, alias="i")
        parts.append(sql)
        params.extend(p)
    from app.util import local_day_start_utc

    start, end = _iso_date(date_from), _iso_date(date_to)
    if start is not None:
        parts.append("COALESCE(i.taken_at, i.created_at) >= ?")
        params.append(local_day_start_utc(start.isoformat(), settings.local_timezone))
    if end is not None:
        parts.append("COALESCE(i.taken_at, i.created_at) < ?")
        params.append(local_day_start_utc(end.isoformat(), settings.local_timezone, next_day=True))
    return _Filters(sql=" AND ".join(parts), params=params)


def _filter_ids(conn: sqlite3.Connection, ids: list[int], filt: _Filters) -> set[int]:
    out: set[int] = set()
    for i in range(0, len(ids), _IN_CHUNK):
        chunk = ids[i:i + _IN_CHUNK]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT i.id FROM items i WHERE i.id IN ({ph}) AND {filt.sql}",
                              [*chunk, *filt.params]):
            out.add(int(r["id"]))
    return out


# --------------------------------------------------------------------- semantic
@dataclass
class _Hit:
    id: int
    score: float  # normalised 0..1 within its list
    match_time_s: float | None = None
    snippet: str | None = None


def _semantic_hits(conn: sqlite3.Connection, q: str, filt: _Filters) -> list[_Hit]:
    """Gated semantic hits, best first, filtered. Raises SearchUnavailable without an engine."""
    if not engine_available():
        raise SearchUnavailable("semantic search model is not available")
    model = current_model()
    try:
        engine = get_engine()
        query = np.asarray(engine.embed_text(q), dtype=np.float32).reshape(-1)
    except Exception as e:
        log.exception("search: could not embed the query")
        raise SearchUnavailable(str(e)) from e

    _matrix.sync(conn, model)
    snap = _matrix.snapshot()
    fr, cr = snap.frames, snap.crops
    whole = fr.size > 0 and fr.dim == query.shape[0]
    people = cr.size > 0 and cr.dim == query.shape[0]
    if not whole and not people:
        return []
    try:
        threshold = float(settings.search_min_relevance)
        cos = np.zeros(0, dtype=np.float32)
        best = np.zeros(0, dtype=np.int64)
        rel = np.zeros(0, dtype=np.float64)
        if whole:
            cos = fr.emb @ query
            best = _best_rows(fr.items, cos)
            rel = np.asarray(engine.relevance(cos[best]), dtype=np.float64).reshape(-1)
            if settings.search_contrast > 0:
                threshold = max(threshold, float(settings.search_contrast) * float(np.median(rel)))
            if not (rel >= threshold).any():
                whole = False
        floor = max(threshold, float(settings.search_person_min_relevance))
        pcos = np.zeros(0, dtype=np.float32)
        if people:
            pcos = cr.emb @ query
            if not (np.asarray(engine.relevance(pcos.max()), dtype=np.float64) >= floor).any():
                people = False
        if not whole and not people:
            return []
        color = query_color(q) if settings.search_color_check else None
        variants = None
        if color is not None:
            variants = np.stack([np.asarray(engine.embed_text(v), dtype=np.float32).reshape(-1)
                                 for v in color_variants(q, color[0])])
        found: dict[int, tuple[float, float]] = {}  # item id -> (relevance, frame time or NaN)
        if whole:
            if variants is not None:
                ok = (fr.emb @ variants.T).argmax(axis=1) == COLORS.index(color[1])  # type: ignore[index]
                best = _best_rows(fr.items, np.where(ok, cos, -np.inf))
                best = best[ok[best]]  # items without a single row of the colour drop out
                rel = np.asarray(engine.relevance(cos[best]), dtype=np.float64).reshape(-1)
            keep = rel >= threshold
            for iid, r, t in zip(fr.items[best][keep].tolist(), rel[keep].tolist(), fr.times[best][keep].tolist()):
                found[int(iid)] = (float(r), float(t))
        if people:
            if variants is not None:
                pok = (cr.emb @ variants.T).argmax(axis=1) == COLORS.index(color[1])  # type: ignore[index]
                pbest = _best_rows(cr.items, np.where(pok, pcos, -np.inf))
                pbest = pbest[pok[pbest]]
            else:
                pbest = _best_rows(cr.items, pcos)
            prel = np.asarray(engine.relevance(pcos[pbest]), dtype=np.float64).reshape(-1)
            for iid, r, t in zip(cr.items[pbest].tolist(), prel.tolist(), cr.times[pbest].tolist()):
                if r >= floor and r > found.get(int(iid), (-1.0, math.nan))[0]:
                    found[int(iid)] = (float(r), float(t))
    except SearchUnavailable:
        raise
    except Exception as e:
        log.exception("search: semantic scoring failed")
        raise SearchUnavailable(str(e)) from e
    if not found:
        return []

    allowed = _filter_ids(conn, list(found), filt)
    if not allowed:
        return []
    kinds = _kinds(conn, allowed)
    ordered = sorted(allowed, key=lambda i: (-found[i][0], i))
    top = max(found[i][0] for i in ordered) or 1.0
    hits: list[_Hit] = []
    for iid in ordered:
        r, t = found[iid]
        match = None if kinds.get(iid) != "video" or math.isnan(t) else round(t, 6)
        hits.append(_Hit(id=iid, score=round(min(1.0, max(0.0, r / top)), 6), match_time_s=match))
    return hits


def _kinds(conn: sqlite3.Connection, ids: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    id_list = sorted(ids)
    for i in range(0, len(id_list), _IN_CHUNK):
        chunk = id_list[i:i + _IN_CHUNK]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT id, kind FROM items WHERE id IN ({ph})", chunk):
            out[int(r["id"])] = r["kind"]
    return out


# --------------------------------------------------------------------- full text
def _clean_pieces(q: str) -> list[tuple[str, bool]]:
    """(cleaned piece, prefix) per usable whitespace-separated piece of the query."""
    out: list[tuple[str, bool]] = []
    for raw in str(q).split():
        prefix = raw.endswith("*")
        words = _KEEP.sub(" ", raw.replace('"', " ").replace("*", " ")).split()
        core = " ".join(w for w in words if w not in _FTS_OPERATORS)
        if not core or not _HAS_WORD.search(core):
            continue
        out.append((core, prefix))
        if len(out) >= MAX_TERMS:
            break
    return out


def fts_terms(q: str) -> list[str]:
    """Sanitised FTS5 terms: each piece quoted ("a b" is a phrase), a trailing * kept as prefix
    search, FTS operators and punctuation-only pieces dropped."""
    return [f'"{core}"' + ("*" if prefix else "") for core, prefix in _clean_pieces(q)]


def fts_query(q: str, *, any_term: bool = False) -> str | None:
    """MATCH expression for the query (every term, or any term), None when nothing is left."""
    terms = fts_terms(q)
    if not terms:
        return None
    return (" OR " if any_term else " ").join(terms)


def _render_snippet(raw: str | None) -> str | None:
    if not raw or _OPEN not in raw:
        return None
    text = " ".join(html.escape(raw, quote=False).split())
    return text.replace(_OPEN, "<b>").replace(_CLOSE, "</b>")


def _fts_rows(conn: sqlite3.Connection, match: str, filt: _Filters, limit: int) -> list[sqlite3.Row]:
    weights = ", ".join(str(w) for w in FTS_WEIGHTS)
    sql = (
        f"SELECT items_fts.rowid AS id, bm25(items_fts, {weights}) AS rank, "
        f"snippet(items_fts, 1, ?, ?, '...', {SNIPPET_TOKENS}) AS s_text, "
        f"snippet(items_fts, 2, ?, ?, '...', {SNIPPET_TOKENS}) AS s_doc "
        "FROM items_fts JOIN items i ON i.id = items_fts.rowid "
        f"WHERE items_fts MATCH ? AND {filt.sql} "
        "ORDER BY rank, i.id LIMIT ?"
    )
    return conn.execute(sql, [_OPEN, _CLOSE, _OPEN, _CLOSE, match, *filt.params, limit]).fetchall()


def _fts_any(conn: sqlite3.Connection, match: str, filt: _Filters) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM items_fts JOIN items i ON i.id = items_fts.rowid WHERE items_fts MATCH ? AND {filt.sql} LIMIT 1",
        [match, *filt.params],
    ).fetchone()
    return row is not None


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _filename_only_hits(conn: sqlite3.Connection, pieces: list[tuple[str, bool]], filt: _Filters,
                        limit: int, exclude: set[int]) -> list[_Hit]:
    """Items that have no items_fts row yet, matched on every piece of their file name."""
    words = [core for core, _prefix in pieces]
    if not words or limit <= 0:
        return []
    conds = " AND ".join("i.original_filename LIKE ? ESCAPE '\\'" for _ in words)
    rows = conn.execute(
        f"SELECT i.id FROM items i WHERE {filt.sql} AND {conds} "
        "AND NOT EXISTS (SELECT 1 FROM items_fts WHERE items_fts.rowid = i.id) "
        "ORDER BY i.id DESC LIMIT ?",
        [*filt.params, *(f"%{_like_escape(w)}%" for w in words), limit],
    ).fetchall()
    return [_Hit(id=int(r["id"]), score=FILENAME_ONLY_SCORE) for r in rows if int(r["id"]) not in exclude]


def _text_hits(conn: sqlite3.Connection, q: str, filt: _Filters, limit: int = TEXT_LIMIT) -> tuple[list[_Hit], bool]:
    """(hits best first, strong) for the query. Raises sqlite3.Error when FTS itself fails."""
    pieces = _clean_pieces(q)
    terms = [f'"{core}"' + ("*" if prefix else "") for core, prefix in pieces]
    if not terms:
        return [], False
    strict = " ".join(terms)
    factor = 1.0
    rows = _fts_rows(conn, strict, filt, limit)
    strong = bool(rows) and _fts_any(conn, f"{STRONG_COLUMNS} : ({strict})", filt)
    if not rows and len(terms) > 1:
        rows = _fts_rows(conn, " OR ".join(terms), filt, limit)
        factor = RELAXED_FACTOR
    hits: list[_Hit] = []
    if rows:
        goodness = [max(0.0, -float(r["rank"] or 0.0)) for r in rows]
        top = max(goodness)
        for r, g in zip(rows, goodness):
            norm = g / top if top > 0 else 1.0
            snippet = _render_snippet(r["s_text"]) or _render_snippet(r["s_doc"])
            hits.append(_Hit(id=int(r["id"]), score=round(factor * min(1.0, norm), 6), snippet=snippet))
    seen = {h.id for h in hits}
    hits.extend(_filename_only_hits(conn, pieces, filt, limit - len(hits), seen))
    return hits, strong


# --------------------------------------------------------------------- search
@dataclass
class SearchResult:
    hits: list[dict[str, Any]]  # {id, score, match_time_s, snippet, source}
    total: int
    next_offset: int | None
    indexed: int
    indexing: bool
    semantic_used: bool = False  # the semantic engine answered (False when it was skipped or missing)


def _merge(semantic: list[_Hit], text: list[_Hit], text_first: bool) -> list[dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    lists = ((text, "text"), (semantic, "semantic")) if text_first else ((semantic, "semantic"), (text, "text"))
    for hits, source in lists:
        for h in hits:
            cur = out.get(h.id)
            if cur is None:
                out[h.id] = {"id": h.id, "score": h.score, "match_time_s": h.match_time_s,
                             "snippet": h.snippet, "source": source}
                continue
            if cur["source"] == source:
                continue
            cur["source"] = "both"
            cur["score"] = max(cur["score"], h.score)
            cur["snippet"] = cur["snippet"] or h.snippet
            if cur["match_time_s"] is None:
                cur["match_time_s"] = h.match_time_s
    return list(out.values())


def search(
    conn: sqlite3.Connection,
    q: str,
    *,
    mode: str = "auto",
    category: str | None = None,
    kind: str | None = None,
    person_ids: list[int] | None = None,
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    vault: bool = False,
    limit: int = 60,
    offset: int = 0,
) -> SearchResult:
    """Items matching the query, best first (see the module docstring). Raises SearchUnavailable
    when mode is semantic and the model is unavailable, or when every engine asked failed."""
    if mode not in MODES:
        raise ValueError(f"unknown search mode {mode!r}")
    q = " ".join(str(q).split())
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    filt = _filters(vault=vault, category=category, kind=kind, person_ids=person_ids,
                    date_from=date_from, date_to=date_to)

    semantic: list[_Hit] = []
    semantic_ok = False
    semantic_error: SearchUnavailable | None = None
    if mode in ("auto", "semantic") and q:
        try:
            semantic = _semantic_hits(conn, q, filt)
            semantic_ok = True
        except SearchUnavailable as e:
            semantic_error = e
            if mode == "semantic":
                raise
            log.info("search: semantic engine skipped (%s); text search only", e)

    text: list[_Hit] = []
    strong = False
    if mode in ("auto", "text") and q:
        try:
            text, strong = _text_hits(conn, q, filt)
        except sqlite3.Error as e:
            log.exception("search: full-text query failed")
            if mode == "text" or not semantic_ok:
                raise SearchUnavailable(str(e)) from (semantic_error or e)

    if mode == "semantic":
        merged = _merge(semantic, [], text_first=False)
    elif mode == "text":
        merged = _merge([], text, text_first=True)
    else:
        merged = _merge(semantic, text, text_first=strong)

    total = len(merged)
    page = merged[offset:offset + limit]
    next_offset = offset + limit if offset + limit < total else None
    indexed = indexed_count(conn, vault=vault) if (semantic_ok or engine_available()) else 0
    return SearchResult(hits=page, total=total, next_offset=next_offset, indexed=indexed,
                        indexing=indexing_pending(conn), semantic_used=semantic_ok)


# --------------------------------------------------------------------- progress
def indexed_count(conn: sqlite3.Connection, model: str | None = None, *, vault: bool | None = None) -> int:
    """Live items with an embedding of the model for their current version. vault None counts
    NSFW and other items, True only NSFW, False only the others."""
    model = model or current_model()
    nsfw = "" if vault is None else f"AND i.nsfw = {1 if vault else 0} "
    return int(conn.execute(
        f"SELECT COUNT(*) FROM items i WHERE i.deleted_at IS NULL {nsfw}"
        "AND EXISTS (SELECT 1 FROM item_embeddings e WHERE e.item_id = i.id AND e.model = ? AND e.version = i.version)",
        (model,),
    ).fetchone()[0])


def indexing_pending(conn: sqlite3.Connection) -> bool:
    """True while a live item still waits for analysis (embeddings) or text extraction (FTS)."""
    row = conn.execute(
        "SELECT 1 FROM items WHERE deleted_at IS NULL AND (analysis_status = 'pending' OR text_status = 'pending') "
        "LIMIT 1"
    ).fetchone()
    return row is not None
