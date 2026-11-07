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
  whole arrays under a lock, so a running search keeps a consistent snapshot (a block whose rows
  did not change is kept as it is, not copied). Rows of an older item version are ignored.
- Scoring: cosine of the text embedding against every row, best row per item,
  engine.relevance() -> 0..1, then the hub's gates:
  1. relevance >= max(settings.search_min_relevance, min(CONTRAST_CAP, settings.search_contrast
     x the query's median relevance over the whole index)): a real query lifts a few items far
     above its median, gibberish lifts many a little. The cap keeps clear matches when the
     subject fills most of a (small) library.
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
  them; None when neither column matched (a file name or label match). They are made only for
  the hits of the returned page, with the MATCH expression the text hits came from.

Script (mode auto): a query whose letters are all non-Latin (Devanagari, say) skips the semantic
pass, because SigLIP only knows English text; the answer is the text hits alone and semantic_used
is False. A mixed query ("राहुल passport") runs both engines as usual.
Indic text: items_fts keeps Indic combining marks inside tokens (app.db.FTS_TOKENIZE), so the
query keeps them too (_KEEP), drops zero-width joiners, and searches a piece with a Devanagari
nukta letter in both its precomposed and its base + U+093C spelling.

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
from date_from up to and including date_to. Paging is by offset over the whole merged list
(capped at TEXT_LIMIT text hits). Page 1 (offset 0) computes it; later pages reuse that list for
PAGE_CACHE_TTL_S (an in-process cache keyed by query, mode and filters), so a list that changes
while indexing runs does not skip items; their hits are re-checked against the filters.

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
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Hashable

import numpy as np

from app.config import settings
from app.db import FTS_TOKENCHARS, kv_get

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
PAGE_CACHE_TTL_S = 600.0  # later pages reuse the merged list of page 1 for this long
PAGE_CACHE_SIZE = 8  # merged lists kept (one per query and filter set)
_OPEN, _CLOSE = "\x02", "\x03"  # snippet markers, replaced by <b> </b> after escaping
_FTS_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})
# Characters kept inside a query piece: word characters, a few joiners, and the Indic combining marks
# that items_fts treats as token characters (app.db.FTS_TOKENCHARS). Python's \w does not match
# combining marks, so without them "राहुल" would fall apart into "र ह ल".
_KEEP = re.compile(r"[^\w\-./@&+'" + re.escape(FTS_TOKENCHARS) + r"]+", re.UNICODE)
_HAS_WORD = re.compile(r"[^\W_]", re.UNICODE)
_JOINERS = re.compile("[\u200c\u200d]")  # zero-width non-joiner / joiner: removed, they only shape glyphs
_QUERY_DIGITS = str.maketrans({chr(0x0966 + i): str(i) for i in range(10)})  # the index stores ASCII digits
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_LATIN_LETTER = re.compile(r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff]")
# Devanagari letters with a nukta exist precomposed (U+0929, U+0931, U+0934, U+0958..U+095F) and as
# base letter + U+093C. OCR, PDF text layers and keyboards use either, so a piece containing one is
# searched in both spellings. Other text is left exactly as typed.
_NUKTA = "\u093c"
_NUKTA_COMPOSED = {base + _NUKTA: chr(cp) for cp, base in (
    (0x0929, "\u0928"), (0x0931, "\u0930"), (0x0934, "\u0933"), (0x0958, "\u0915"), (0x0959, "\u0916"),
    (0x095A, "\u0917"), (0x095B, "\u091c"), (0x095C, "\u0921"), (0x095D, "\u0922"), (0x095E, "\u092b"),
    (0x095F, "\u092f"))}
_NUKTA_DECOMPOSED = {v: k for k, v in _NUKTA_COMPOSED.items()}

# Contrast gate: settings.search_contrast x median relevance, capped at this relevance (about
# cosine 0.10 on SigLIP), so a subject that fills most of the library, or a library of a few
# items, still finds its clear matches. Gibberish stays far below the cap.
CONTRAST_CAP = 0.25

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
    """The query once per colour in COLORS, with the colour word replaced. The word is found with
    the same tokenisation as query_color (_WORD), so "blue_car" or "blue2" get "red_car", "red2"."""
    for m in _WORD.finditer(q):
        if m.group(0).lower() == word.lower():
            return [q[:m.start()] + color + q[m.end():] for color in COLORS]
    return [q] * len(COLORS)


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
    copied from `old`, the others loaded. Keeps the most common embedding size. When every row
    is unchanged (a visibility-only search_version bump), `old` itself is returned: no copy."""
    old_pos = {k: i for i, k in enumerate(old.keys)}
    missing = []
    for key, _item, _t, meta in rows:
        pos = old_pos.get(key)
        if pos is None or old.metas[pos] != meta:
            missing.append(key)
    if not missing and len(rows) == old.size:
        return old, 0
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
    """Drop the in-memory matrix and the page cache (tests: each test has a fresh database)."""
    _matrix.reset()
    with _page_lock:
        _page_cache.clear()


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
             date_from: date | str | None, date_to: date | str | None,
             tag_names: list[str] | None = None, live: bool | None = None) -> _Filters:
    from app.api.common import people_filter_sql, tags_filter_sql, visible_sql

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
    if tag_names:
        sql, p = tags_filter_sql(list(tag_names)[:10], alias="i")
        parts.append(sql)
        params.extend(p)
    if live is not None:
        parts.append(f"i.live = {1 if live else 0}")
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
                threshold = max(threshold, min(CONTRAST_CAP, float(settings.search_contrast) * float(np.median(rel))))
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
        texts = color_variants(q, color[0]) if color is not None else []
        if len(set(texts)) == len(COLORS):  # the colour word could be replaced: check it
            variants = np.stack([np.asarray(engine.embed_text(v), dtype=np.float32).reshape(-1)
                                 for v in texts])
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
    for raw in _JOINERS.sub("", str(q)).translate(_QUERY_DIGITS).split():
        prefix = raw.endswith("*")
        words = _KEEP.sub(" ", raw.replace('"', " ").replace("*", " ")).split()
        core = " ".join(w for w in words if w not in _FTS_OPERATORS)
        if not core or not _HAS_WORD.search(core):
            continue
        out.append((core, prefix))
        if len(out) >= MAX_TERMS:
            break
    return out


def _nukta_spellings(core: str) -> list[str]:
    """The piece as typed, plus its other nukta spelling when it has a Devanagari nukta letter."""
    if _NUKTA not in core and not any(c in _NUKTA_DECOMPOSED for c in core):
        return [core]
    composed = core
    for pair, single in _NUKTA_COMPOSED.items():
        composed = composed.replace(pair, single)
    decomposed = "".join(_NUKTA_DECOMPOSED.get(c, c) for c in composed)
    return list(dict.fromkeys((core, composed, decomposed)))


def _term(core: str, prefix: bool) -> str:
    star = "*" if prefix else ""
    forms = [f'"{form}"{star}' for form in _nukta_spellings(core)]
    return forms[0] if len(forms) == 1 else "(" + " OR ".join(forms) + ")"


def _every(terms: list[str]) -> str:
    """Terms joined so that all must match. FTS5 allows the implicit AND (a space) only between
    phrases, so an OR group needs an explicit AND."""
    return (" AND " if any(t.startswith("(") for t in terms) else " ").join(terms)


def fts_terms(q: str) -> list[str]:
    """Sanitised FTS5 terms: each piece quoted ("a b" is a phrase), a trailing * kept as prefix
    search, FTS operators and punctuation-only pieces dropped. Indic combining marks stay inside
    the word and zero-width joiners are removed; a piece with a Devanagari nukta letter becomes
    ("as typed" OR "other spelling")."""
    return [_term(core, prefix) for core, prefix in _clean_pieces(q)]


def fts_query(q: str, *, any_term: bool = False) -> str | None:
    """MATCH expression for the query (every term, or any term), None when nothing is left."""
    terms = fts_terms(q)
    if not terms:
        return None
    return " OR ".join(terms) if any_term else _every(terms)


def _render_snippet(raw: str | None) -> str | None:
    if not raw or _OPEN not in raw:
        return None
    text = " ".join(html.escape(raw, quote=False).split())
    return text.replace(_OPEN, "<b>").replace(_CLOSE, "</b>")


def _fts_rows(conn: sqlite3.Connection, match: str, filt: _Filters, limit: int) -> list[sqlite3.Row]:
    """(id, rank) of the candidates; snippets are made later, for the returned page only."""
    weights = ", ".join(str(w) for w in FTS_WEIGHTS)
    sql = (
        f"SELECT items_fts.rowid AS id, bm25(items_fts, {weights}) AS rank "
        "FROM items_fts JOIN items i ON i.id = items_fts.rowid "
        f"WHERE items_fts MATCH ? AND {filt.sql} "
        "ORDER BY rank, i.id LIMIT ?"
    )
    return conn.execute(sql, [match, *filt.params, limit]).fetchall()


def _snippets(conn: sqlite3.Connection, match: str, ids: list[int]) -> dict[int, str]:
    """Rendered snippet per id (text column, else doc column) for the query `match`."""
    out: dict[int, str] = {}
    for i in range(0, len(ids), _IN_CHUNK):
        chunk = ids[i:i + _IN_CHUNK]
        ph = ",".join("?" * len(chunk))
        # The index holds folded text (ASCII digits, no joiners); the snippet shown is mapped back onto
        # the original text so Devanagari digits and joiner-shaped conjuncts read as written.
        from app.documents import extract

        originals = extract.snippet_originals(conn, chunk)
        for r in conn.execute(
            f"SELECT rowid AS id, snippet(items_fts, 1, ?, ?, '...', {SNIPPET_TOKENS}) AS s_text, "
            f"snippet(items_fts, 2, ?, ?, '...', {SNIPPET_TOKENS}) AS s_doc "
            f"FROM items_fts WHERE items_fts MATCH ? AND rowid IN ({ph})",
            [_OPEN, _CLOSE, _OPEN, _CLOSE, match, *chunk],
        ):
            o_text, o_doc = originals.get(int(r["id"]), ("", ""))
            s_text = extract.restore_snippet(r["s_text"], o_text, _OPEN, _CLOSE)
            s_doc = extract.restore_snippet(r["s_doc"], o_doc, _OPEN, _CLOSE)
            snippet = _render_snippet(s_text) or _render_snippet(s_doc)
            if snippet:
                out[int(r["id"])] = snippet
    return out


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


def _text_hits(conn: sqlite3.Connection, q: str, filt: _Filters,
               limit: int = TEXT_LIMIT) -> tuple[list[_Hit], bool, str | None]:
    """(hits best first without snippets, strong, MATCH expression the hits came from, for
    _snippets). Raises sqlite3.Error when FTS itself fails."""
    pieces = _clean_pieces(q)
    terms = [_term(core, prefix) for core, prefix in pieces]
    if not terms:
        return [], False, None
    strict = _every(terms)
    match = strict
    factor = 1.0
    rows = _fts_rows(conn, strict, filt, limit)
    strong = bool(rows) and _fts_any(conn, f"{STRONG_COLUMNS} : ({strict})", filt)
    if not rows and len(terms) > 1:
        match = " OR ".join(terms)
        rows = _fts_rows(conn, match, filt, limit)
        factor = RELAXED_FACTOR
    hits: list[_Hit] = []
    if rows:
        goodness = [max(0.0, -float(r["rank"] or 0.0)) for r in rows]
        top = max(goodness)
        for r, g in zip(rows, goodness):
            norm = g / top if top > 0 else 1.0
            hits.append(_Hit(id=int(r["id"]), score=round(factor * min(1.0, norm), 6)))
    seen = {h.id for h in hits}
    hits.extend(_filename_only_hits(conn, pieces, filt, limit - len(hits), seen))
    return hits, strong, match


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
    tag_names: list[str] | None = None,
    live: bool | None = None,
) -> SearchResult:
    """Items matching the query, best first (see the module docstring). Raises SearchUnavailable
    when mode is semantic and the model is unavailable, or when every engine asked failed."""
    if mode not in MODES:
        raise ValueError(f"unknown search mode {mode!r}")
    q = " ".join(str(q).split())
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    filt = _filters(vault=vault, category=category, kind=kind, person_ids=person_ids,
                    date_from=date_from, date_to=date_to, tag_names=tag_names, live=live)

    key = (q, mode, filt.sql, tuple(filt.params))
    cached = _page_cache_get(key) if offset > 0 else None
    if cached is not None:
        merged, match, semantic_ok = cached
        page = [dict(h) for h in merged[offset:offset + limit]]
        visible = _filter_ids(conn, [int(h["id"]) for h in page], filt)
        page = [h for h in page if int(h["id"]) in visible]  # trashed or hidden since page 1
    else:
        merged, match, semantic_ok = _merged(conn, q, mode, filt)
        _page_cache_put(key, (merged, match, semantic_ok))
        page = [dict(h) for h in merged[offset:offset + limit]]
    if match:
        want = [int(h["id"]) for h in page if h["source"] != "semantic"]
        if want:
            try:
                found = _snippets(conn, match, want)
            except sqlite3.Error:
                log.exception("search: snippet query failed")
                found = {}
            for h in page:
                h["snippet"] = found.get(int(h["id"]), h["snippet"])

    total = len(merged)
    next_offset = offset + limit if offset + limit < total else None
    indexed = indexed_count(conn, vault=vault) if (semantic_ok or engine_available()) else 0
    return SearchResult(hits=page, total=total, next_offset=next_offset, indexed=indexed,
                        indexing=indexing_pending(conn), semantic_used=semantic_ok)


# Merged lists by (query, mode, filters): later pages of a search reuse the list page 1 built,
# so items that move while indexing runs (or a flip from semantic-first to text-first) are not
# skipped or repeated. A page-1 request (offset 0) always recomputes.
_page_lock = threading.Lock()
_page_cache: OrderedDict[tuple[Any, ...], tuple[float, tuple[list[dict[str, Any]], str | None, bool]]] = OrderedDict()


def _page_cache_get(key: tuple[Any, ...]) -> tuple[list[dict[str, Any]], str | None, bool] | None:
    with _page_lock:
        entry = _page_cache.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry[0] > PAGE_CACHE_TTL_S:
            del _page_cache[key]
            return None
        _page_cache.move_to_end(key)
        return entry[1]


def _page_cache_put(key: tuple[Any, ...], value: tuple[list[dict[str, Any]], str | None, bool]) -> None:
    with _page_lock:
        _page_cache[key] = (time.monotonic(), value)
        _page_cache.move_to_end(key)
        while len(_page_cache) > PAGE_CACHE_SIZE:
            _page_cache.popitem(last=False)


def text_only_query(q: str) -> bool:
    """True when the query has letters and none of them is Latin (a Hindi or Marathi query, say).
    SigLIP is trained on English text, so mode auto skips the semantic pass for such a query: its
    hits would be noise and its gates could only push real text hits down. Queries of digits and
    punctuation alone are not affected."""
    return bool(_LETTER.search(q)) and not _LATIN_LETTER.search(q)


def _merged(conn: sqlite3.Connection, q: str, mode: str,
            filt: _Filters) -> tuple[list[dict[str, Any]], str | None, bool]:
    """(merged hits without snippets, FTS MATCH expression for the snippets, semantic used)."""
    semantic: list[_Hit] = []
    semantic_ok = False
    semantic_error: SearchUnavailable | None = None
    text_only = mode == "auto" and text_only_query(q)
    if mode in ("auto", "semantic") and q and not text_only:
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
    match: str | None = None
    if mode in ("auto", "text") and q:
        try:
            text, strong, match = _text_hits(conn, q, filt)
        except sqlite3.Error as e:
            log.exception("search: full-text query failed")
            if mode == "text" or not semantic_ok:
                raise SearchUnavailable(str(e)) from (semantic_error or e)

    if mode == "semantic":
        merged = _merge(semantic, [], text_first=False)
    elif mode == "text":
        merged = _merge([], text, text_first=True)
    else:
        merged = _merge(semantic, text, text_first=strong or text_only)
    return merged, match, semantic_ok


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
