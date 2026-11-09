"""Related tags suggested from the owner's own (section 10).

An item the owner marked as related to someone (item_tags relation 'related', source 'user': about them, without them
in it) makes other items candidates for the same mark: items that look like it, or that look somewhat like it and were
taken at the same moment or the same place.

How alike is judged against each related item's own baseline. SigLIP whole-image similarity alone cannot tell the same
scene from a merely similar one: on the owner's library, random pairs reach 0.77 at the 99th percentile, and a dark,
red-lit video was 0.83 to 0.90 alike to unrelated dark scenes, while the same candle-and-rose decoration seen from other
angles was 0.84 to 0.90. Measured as z = (s - centre) / spread of the related item's similarity to every item (the
median and 1.4826 x the median absolute deviation, which the few real look-alikes cannot pull along), the decoration
stood far above the dark scenes (see Z_ALIKE).

CONTRACT.
- suggest(conn, person_id, *, visible, seed_visible=SEEDS_SQL, limit) -> list[Suggestion(item_id, score, reason,
  because)] for one person, from their related items within `seed_visible` (live, on one side of the vault; the
  MAX_SEEDS most recently marked):
  reason 'alike' (z >= Z_ALIKE), 'moment' (z >= Z_CONTEXT and the same moment: capture times within MOMENT_S, or, when
  either has none, saved in the same batch: created within BATCH_S and at most BATCH_IDS ids apart) or 'place' (z >=
  Z_CONTEXT and GPS within PLACE_M); score = z plus MOMENT_BONUS / PLACE_BONUS. `because` is the related item it comes
  from. Never an item already tagged with them (either relation), one they are in through a face, body or pet box, one
  the owner said is not related (related_dismissals), a related item itself, or an item outside `visible` (a SQL
  condition on items alias i: the vault, the trash, mis-clicks and duplicates are the caller's).
- suggest_for_item(conn, item_id, *, visible) -> list[(person_id, reason, because)]: the people an item may be related
  to, the same rules (each person's related items within `visible` scored against the item). Baselines are cached by
  the related item's own vectors and the library's size in BASELINE_BUCKET steps (_baseline_key), for both.
- dismiss(conn, person_id, item_ids): "Not related to NAME", remembered (call inside the owner's action).
Without the SigLIP vectors (no model, nothing analysed) nothing is suggested.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from app.util import now_iso

Z_ALIKE = 4.5  # stands out this far above the related item's baseline: looks alike, wherever it was taken
Z_CONTEXT = 3.0  # enough with the same moment or place
MOMENT_S = 3 * 3600  # capture times this close are the same moment (an evening, an outing)
BATCH_S = 120  # without capture times: saved within this many seconds ...
BATCH_IDS = 30  # ... and at most this many items apart (one save or sync batch, in camera-roll order)
PLACE_M = 100.0  # GPS this close is the same place
MOMENT_BONUS = 0.5
PLACE_BONUS = 0.3
MIN_ROWS = 50  # a baseline needs this many items to compare with


@dataclass
class Suggestion:
    item_id: int
    score: float
    reason: str  # 'alike' | 'moment' | 'place'
    because: int  # the related item it comes from


@dataclass
class _Meta:
    taken: float | None  # capture time, epoch seconds
    created: float | None
    lat: float | None
    lon: float | None


def _epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _metas(conn: sqlite3.Connection, ids: list[int]) -> dict[int, _Meta]:
    out: dict[int, _Meta] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for r in conn.execute(
            f"SELECT id, taken_at, created_at, gps_lat, gps_lon FROM items WHERE id IN ({','.join('?' * len(chunk))})",
            chunk,
        ):
            out[int(r[0])] = _Meta(_epoch(r[1]), _epoch(r[2]), r[3], r[4])
    return out


def same_moment(a: _Meta, b: _Meta, id_a: int, id_b: int) -> bool:
    if a.taken is not None and b.taken is not None:
        return abs(a.taken - b.taken) <= MOMENT_S
    if a.created is None or b.created is None:
        return False
    return abs(a.created - b.created) <= BATCH_S and abs(int(id_a) - int(id_b)) <= BATCH_IDS


def meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


def same_place(a: _Meta, b: _Meta) -> bool:
    if None in (a.lat, a.lon, b.lat, b.lon):
        return False
    return meters(float(a.lat), float(a.lon), float(b.lat), float(b.lon)) <= PLACE_M


# ------------------------------------------------------------------ vectors
BASELINE_BUCKET = 512  # a related item's baseline stays valid while the library grows by fewer rows than this
MAX_SEEDS = 200  # the person's most recently marked related items that count


def _frames(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray, str]:
    """(item id per row, L2-normalised rows, the model) from the search matrix (app.search.service)."""
    from app.search import service

    m = service.get_matrix()
    model = service.current_model()
    m.sync(conn, model)
    frames = m.snapshot().frames
    return frames.items, frames.emb, model


def _rows(items: np.ndarray, emb: np.ndarray, item_id: int) -> np.ndarray:
    if emb.ndim != 2 or not emb.shape[0]:
        return np.zeros((0, 0), dtype=np.float32)
    return emb[items == int(item_id)]


def _robust(values: np.ndarray) -> tuple[float, float]:
    """(median, 1.4826 x the median absolute deviation): the baseline's centre and spread, which the few items that
    really look alike (the ones being looked for) cannot pull along as they would a mean and a standard deviation."""
    med = float(np.median(values))
    return med, 1.4826 * float(np.median(np.abs(values - med)))


_baselines: dict[tuple[Any, ...], tuple[float, float]] = {}
_baseline_lock = threading.Lock()


def _baseline_key(model: str, seed: int, rows: np.ndarray, n_rows: int) -> tuple[Any, ...]:
    """A related item's baseline is a median over thousands of items: it holds while its own vectors stay the same
    and the library grows by less than BASELINE_BUCKET rows (re-analysing it changes the vectors, so the key)."""
    digest = hashlib.blake2b(np.ascontiguousarray(rows).tobytes(), digest_size=16).hexdigest()
    return model, int(seed), digest, int(n_rows) // BASELINE_BUCKET


def _cached(key: tuple[Any, ...]) -> tuple[float, float] | None:
    with _baseline_lock:
        return _baselines.get(key)


def _remember(key: tuple[Any, ...], value: tuple[float, float]) -> None:
    with _baseline_lock:
        if len(_baselines) > 4096:
            _baselines.clear()
        _baselines[key] = value


def _per_item(items: np.ndarray, sims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(item ids, best similarity of each item) from per-row similarities ((rows,) or (rows, k))."""
    if items.size == 0:
        return items, sims
    order = np.argsort(items, kind="stable")
    si = items[order]
    starts = np.r_[0, np.nonzero(np.diff(si))[0] + 1]
    return si[starts], np.maximum.reduceat(sims[order], starts, axis=0)


def _scores(items: np.ndarray, emb: np.ndarray, seeds: list[int], model: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """[(seed, item ids, z of each item against it)] for the related items with vectors, from one matrix product."""
    found = [(int(sd), _rows(items, emb, sd)) for sd in seeds]
    found = [(sd, r) for sd, r in found if r.shape[0]]
    if not found:
        return []
    counts = [r.shape[0] for _, r in found]
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    per_row = np.maximum.reduceat(emb @ np.vstack([r for _, r in found]).T, starts, axis=1)  # (rows, seeds)
    ids, best = _per_item(items, per_row)  # (items,), (items, seeds)
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for j, (sd, rows) in enumerate(found):
        others = ids != sd
        if int(others.sum()) < MIN_ROWS:
            continue
        key = _baseline_key(model, sd, rows, emb.shape[0])
        stats = _cached(key)
        if stats is None:
            stats = _robust(best[others, j])
            _remember(key, stats)
        if stats[1] <= 1e-6:
            continue
        out.append((sd, ids, (best[:, j] - stats[0]) / stats[1]))
    return out


def _baseline(items: np.ndarray, emb: np.ndarray, seed: int, model: str) -> tuple[np.ndarray, float, float] | None:
    """The related item's rows and its (centre, spread) over every item (cached, _baseline_key)."""
    rows = _rows(items, emb, seed)
    if rows.shape[0] == 0:
        return None
    key = _baseline_key(model, seed, rows, emb.shape[0])
    stats = _cached(key)
    if stats is None:
        ids, best = _per_item(items, (emb @ rows.T).max(axis=1))
        others = ids != seed
        if int(others.sum()) < MIN_ROWS:
            return None
        stats = _robust(best[others])
        _remember(key, stats)
    if stats[1] <= 1e-6:
        return None
    return rows, stats[0], stats[1]


# ------------------------------------------------------------------ rules
def _reason(z: float, moment: bool, place: bool) -> tuple[str, float] | None:
    if z >= Z_ALIKE:
        return "alike", z + (MOMENT_BONUS if moment else 0.0) + (PLACE_BONUS if place else 0.0)
    if z >= Z_CONTEXT and moment:
        return "moment", z + MOMENT_BONUS + (PLACE_BONUS if place else 0.0)
    if z >= Z_CONTEXT and place:
        return "place", z + PLACE_BONUS
    return None


SEEDS_SQL = "i.deleted_at IS NULL AND i.nsfw = 0"  # the related items that count: live, outside the vault


def _seeds(conn: sqlite3.Connection, person_id: int, seed_visible: str) -> list[int]:
    """The person's related items (the owner's marks), most recent first, at most MAX_SEEDS."""
    return [int(r[0]) for r in conn.execute(
        f"SELECT t.item_id FROM item_tags t JOIN items i ON i.id = t.item_id WHERE t.person_id = ? "
        f"AND t.relation = 'related' AND t.source = 'user' AND {seed_visible} ORDER BY t.created_at DESC, t.item_id DESC "
        "LIMIT ?", (int(person_id), MAX_SEEDS))]


def _taken(conn: sqlite3.Connection, person_id: int) -> set[int]:
    """Items already tagged with the person (either relation), with them in it, or dismissed for them."""
    return {int(r[0]) for r in conn.execute(
        "SELECT item_id FROM item_tags WHERE person_id = ? UNION SELECT item_id FROM faces WHERE person_id = ? "
        "UNION SELECT item_id FROM bodies WHERE person_id = ? UNION SELECT item_id FROM animals WHERE person_id = ? "
        "UNION SELECT item_id FROM related_dismissals WHERE person_id = ?", (int(person_id),) * 5)}


def suggest(conn: sqlite3.Connection, person_id: int, *, visible: str, seed_visible: str = SEEDS_SQL,
            limit: int = 24) -> list[Suggestion]:
    seeds = _seeds(conn, person_id, seed_visible)
    if not seeds or limit <= 0:
        return []
    items, emb, model = _frames(conn)
    if items.size == 0:
        return []
    best: dict[int, Suggestion] = {}
    scored = _scores(items, emb, seeds, model)
    if not scored:
        return []
    near = {int(i) for _, ids, z in scored for i in ids[z >= Z_CONTEXT].tolist()}
    skip = _taken(conn, person_id) | set(seeds)
    near -= skip
    if not near:
        return []
    allowed = set()
    cand = sorted(near)
    for start in range(0, len(cand), 500):
        chunk = cand[start:start + 500]
        allowed |= {int(r[0]) for r in conn.execute(
            f"SELECT i.id FROM items i WHERE i.id IN ({','.join('?' * len(chunk))}) AND {visible}", chunk)}
    if not allowed:
        return []
    metas = _metas(conn, sorted(allowed | set(seeds)))
    for seed, ids, z in scored:
        sm = metas.get(seed)
        keep = z >= Z_CONTEXT
        for i, zi in zip(ids[keep].tolist(), z[keep].tolist()):
            if i not in allowed:
                continue
            cm = metas.get(i)
            moment = bool(sm and cm and same_moment(cm, sm, i, seed))
            place = bool(sm and cm and same_place(cm, sm))
            got = _reason(float(zi), moment, place)
            if got is None:
                continue
            prev = best.get(i)
            if prev is None or got[1] > prev.score:
                best[i] = Suggestion(item_id=i, score=round(got[1], 3), reason=got[0], because=seed)
    out = sorted(best.values(), key=lambda s: (-s.score, s.item_id))
    return out[:limit]


def suggest_for_item(conn: sqlite3.Connection, item_id: int, *, visible: str) -> list[tuple[int, str, int]]:
    item_id = int(item_id)
    if conn.execute(f"SELECT 1 FROM items i WHERE i.id = ? AND {visible}", (item_id,)).fetchone() is None:
        return []
    seeds_by_person: dict[int, list[int]] = {}
    for r in conn.execute(
        f"SELECT t.person_id, t.item_id FROM item_tags t JOIN items i ON i.id = t.item_id "
        f"WHERE t.relation = 'related' AND t.source = 'user' AND t.item_id != ? AND {visible}", (item_id,)
    ):
        seeds_by_person.setdefault(int(r[0]), []).append(int(r[1]))
    if not seeds_by_person:
        return []
    items, emb, model = _frames(conn)
    mine = _rows(items, emb, item_id)
    if mine.shape[0] == 0:
        return []
    present = {int(r[0]) for r in conn.execute(
        "SELECT person_id FROM item_tags WHERE item_id = ? UNION SELECT person_id FROM faces WHERE item_id = ? "
        "AND person_id IS NOT NULL UNION SELECT person_id FROM bodies WHERE item_id = ? AND person_id IS NOT NULL "
        "UNION SELECT person_id FROM animals WHERE item_id = ? AND person_id IS NOT NULL "
        "UNION SELECT person_id FROM related_dismissals WHERE item_id = ?", (item_id,) * 5)}
    metas = _metas(conn, [item_id, *sorted({s for v in seeds_by_person.values() for s in v})])
    out: list[tuple[float, int, str, int]] = []
    for pid, seeds in seeds_by_person.items():
        if pid in present:
            continue
        top: tuple[float, str, int] | None = None
        for seed in seeds:
            base = _baseline(items, emb, seed, model)
            if base is None:
                continue
            rows, mu, sd = base
            z = (float((mine @ rows.T).max()) - mu) / sd
            cm, sm = metas.get(item_id), metas.get(seed)
            moment = bool(sm and cm and same_moment(cm, sm, item_id, seed))
            place = bool(sm and cm and same_place(cm, sm))
            got = _reason(z, moment, place)
            if got is not None and (top is None or got[1] > top[0]):
                top = (got[1], got[0], seed)
        if top is not None:
            out.append((top[0], pid, top[1], top[2]))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [(pid, reason, seed) for _, pid, reason, seed in out]


def dismiss(conn: sqlite3.Connection, person_id: int, item_ids: list[int]) -> int:
    """Remember that these items are not related to the person. Call inside the owner's action (a tx)."""
    now = now_iso()
    n = 0
    for iid in sorted({int(i) for i in item_ids}):
        n += conn.execute("INSERT OR IGNORE INTO related_dismissals(item_id, person_id, created_at) VALUES (?, ?, ?)",
                          (iid, int(person_id), now)).rowcount
    return n
