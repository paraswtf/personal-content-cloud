"""Photos taken together, lined up (docs/ARCHITECTURE.md section 35): the links table and the runner that fills it.

CONTRACT.
- A photo takes part when it is a live, dated image outside the vault, analysed, not a Live Photo's clip, not a
  screenshot or a document, and has at least one face or person box (ELIGIBLE_SQL). Photos are ordered by capture
  time, then the EXIF sub-second, then id.
- place_links(a_item, b_item, rev, kind, m, inliers, spread, dt_s, computed_at): one row per pair of a photo and one
  of its next AHEAD photos within MAX_GAP_S seconds, a before b in that order. kind 'still' | 'moved' | 'burst'
  (app.analysis.place_align) or 'none' (did not line up, or the two are one duplicate group or one Live pair: the
  same moment is no movement); m is the 2 x 3 matrix a -> b in normalised coordinates as JSON ('' for 'none').
- place_scans(item_id, rev, sig, scanned_at): a photo is done when its pairs exist; sig is the ids of its next
  photos ("12,15,16"), so a photo that arrives later in the middle of a sequence makes its neighbours pending again.
- pending_items(conn, limit, skip) -> [(item id, [next ids])], by capture time, newest first.
- run_pending(conn, budget_s=None, limit=None) -> {items, pairs, aligned, complete}: measures outside any
  transaction (one photo read and described at a time through place_align's cache), then one short write per photo.
  A photo whose measuring raises stays pending; after ABORTS_PER_ITEM failures its pairs are stored as 'none'.
  After a photo's usable links are written, the photos of its pairs that hold an open box are queued for a fresh
  guess on the traits runner thread (traits.requeue: one queue for every re-guess).
- kick(conn=None), start(conn): the runner thread ("place", worker.run_in_background), its own connection, nice 19,
  rounds of ROUND_BUDGET_S, the traits runner's memory stops; it ends by itself when nothing is pending. Nothing
  when the workers are off (tests call run_pending) or PLACE_ENABLED is off.
- apply_settings_change(conn) -> int, at startup: when the position rules changed (a setting, or constants of a
  later version: place_match.rules_sig, kv place_rules), the photos whose open guesses rest on position are queued
  for a fresh guess.
- links_for(conn, item_ids) -> [Link]: the usable links (current rev, kind other than 'none') touching these items.
- progress(conn) -> {scanned, eligible, ready, rev, last_error, enabled} for Settings > Learning.
Both tables are derived data: not journalled for undo, rows go with their items (ON DELETE CASCADE).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Iterable, NamedTuple

from app.analysis import place_align
from app.config import settings
from app.db import kv_get, kv_set, tx
from app.util import now_iso, parse_iso

log = logging.getLogger(__name__)

PLACE_REV = "1"
RUNNER_KEY = "place"
READY_KEY = "place_ready"         # kv: the rev the whole library was last lined up with
RULES_KEY = "place_rules"         # kv: the rules the open position guesses were made under (place_match.rules_sig)
AHEAD = 3                         # each photo is paired with this many next photos...
MAX_GAP_S = 300.0                 # ...taken at most this long after it
BATCH_ITEMS = 16
ROUND_BUDGET_S = 30.0
LOOP_SLEEP_S = 2.0
IDLE_ROUNDS = 2
RSS_GROWTH = 2 * 1024 ** 3
RSS_CAP = 7 * 1024 ** 3
ABORTS_PER_ITEM = 3               # a photo that fails this often is stored as not lined up (until the next rev)

ELIGIBLE_SQL = (
    "SELECT i.id, i.taken_at, COALESCE(i.exif_subsec, '') AS sub FROM items i "
    "WHERE i.kind = 'image' AND i.deleted_at IS NULL AND i.nsfw = 0 AND i.taken_at IS NOT NULL "
    "AND i.analysis_status = 'done' AND i.companion_of IS NULL "
    "AND COALESCE(i.category, '') NOT IN ('screenshots', 'documents') "
    "AND (EXISTS (SELECT 1 FROM faces f WHERE f.item_id = i.id) OR EXISTS (SELECT 1 FROM bodies b WHERE b.item_id = i.id))"
)
_LEADS = ", ".join(f"LEAD(id, {k}) OVER w AS n{k}, LEAD(taken_at, {k}) OVER w AS t{k}" for k in range(1, AHEAD + 1))
_SIG = " || ".join(
    f"(CASE WHEN n{k} IS NOT NULL AND (julianday(t{k}) - julianday(taken_at)) * 86400.0 <= :gap "
    f"THEN n{k} || ',' ELSE '' END)" for k in range(1, AHEAD + 1))
_SEQUENCE_SQL = (
    f"WITH e AS ({ELIGIBLE_SQL}), o AS (SELECT id, taken_at, {_LEADS} FROM e WINDOW w AS (ORDER BY taken_at, sub, id)), "
    f"s AS (SELECT id, taken_at, {_SIG} AS sig FROM o) "
)
# By capture time, newest first: a photo's next photos were then just described, so each photo costs one read.
_PENDING_SQL = _SEQUENCE_SQL + (
    "SELECT s.id, s.sig FROM s LEFT JOIN place_scans p ON p.item_id = s.id "
    "WHERE (p.item_id IS NULL OR p.rev != :rev OR p.sig != s.sig) "
    "AND s.id NOT IN (SELECT value FROM json_each(:skip)) ORDER BY s.taken_at DESC, s.id DESC LIMIT :limit"
)
_ITEM_COLUMNS = "id, uid, version, taken_at, exif_subsec, dup_group_id, content_id, live_video_item_id, deleted_at, nsfw"

_state_lock = threading.Lock()
_state: dict[str, Any] = {"last_error": None, "fails": {}}


class Link(NamedTuple):
    a_item: int
    b_item: int
    kind: str
    m: tuple[float, ...]
    inliers: int
    dt_s: float


# ------------------------------------------------------------------ readiness and reads
def enabled() -> bool:
    return bool(settings.place_enabled)


def ready_rev(conn: sqlite3.Connection) -> str | None:
    return kv_get(conn, READY_KEY) or None


def ready(conn: sqlite3.Connection) -> bool:
    return ready_rev(conn) == PLACE_REV


def _sig_ids(sig: str | None) -> list[int]:
    return [int(x) for x in (sig or "").split(",") if x]


def pending_items(conn: sqlite3.Connection, limit: int, skip: Iterable[int] = ()) -> list[tuple[int, list[int]]]:
    """Photos whose pairs with their next photos are missing, of another rev, or of another set of next photos."""
    rows = conn.execute(_PENDING_SQL, {"gap": MAX_GAP_S + 0.5, "rev": PLACE_REV, "limit": int(limit),
                                       "skip": json.dumps(sorted({int(i) for i in skip}))}).fetchall()
    return [(int(r[0]), _sig_ids(r[1])) for r in rows]


def capture_time(row: sqlite3.Row) -> float | None:
    """Seconds since the epoch, with the EXIF sub-second when the photo has one."""
    t = parse_iso(row["taken_at"])
    if t is None:
        return None
    out = t.timestamp()
    sub = row["exif_subsec"]
    if isinstance(sub, str) and sub.isdigit():
        out += float("0." + sub)
    return out


def links_for(conn: sqlite3.Connection, item_ids: Iterable[int]) -> list[Link]:
    """The usable links touching any of these photos (either side), each once."""
    ids = sorted({int(i) for i in item_ids})
    out: dict[tuple[int, int], Link] = {}
    for start in range(0, len(ids), 400):
        chunk = ids[start:start + 400]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT a_item, b_item, kind, m, inliers, dt_s FROM place_links WHERE rev = ? AND kind != 'none' "
            f"AND (a_item IN ({marks}) OR b_item IN ({marks}))", [PLACE_REV, *chunk, *chunk],
        ):
            try:
                m = tuple(float(v) for v in json.loads(r["m"]))
            except (TypeError, ValueError):
                continue
            if len(m) != 6:
                continue
            out[(int(r["a_item"]), int(r["b_item"]))] = Link(int(r["a_item"]), int(r["b_item"]), str(r["kind"]), m,
                                                           int(r["inliers"] or 0), float(r["dt_s"] or 0.0))
    return list(out.values())


# ------------------------------------------------------------------ measuring
def _boxes(conn: sqlite3.Connection, item_id: int) -> list[place_align.Box]:
    return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in conn.execute(
        "SELECT x, y, w, h FROM faces WHERE item_id = ? UNION ALL SELECT x, y, w, h FROM bodies WHERE item_id = ?",
        (item_id, item_id))]


def _features(conn: sqlite3.Connection, row: sqlite3.Row) -> place_align.Features | None:
    def loader():
        gray = place_align.load_gray(row)
        return None if gray is None else (gray, _boxes(conn, int(row["id"])))

    return place_align.features((int(row["id"]), int(row["version"])), loader)


def _same_moment(a: sqlite3.Row, b: sqlite3.Row) -> bool:
    """One duplicate group, or the two halves of one Live Photo: the same moment is no movement."""
    if a["dup_group_id"] is not None and a["dup_group_id"] == b["dup_group_id"]:
        return True
    if a["content_id"] and a["content_id"] == b["content_id"]:
        return True
    return a["live_video_item_id"] == b["id"] or b["live_video_item_id"] == a["id"]


def measure_pair(conn: sqlite3.Connection, a: sqlite3.Row, b: sqlite3.Row,
                 cache: dict[str, Any] | None = None) -> tuple[str, str, int, float, float]:
    """(kind, m as JSON, inliers, spread, dt_s) for a before b. Reads pixels; writes nothing."""
    ta, tb = capture_time(a), capture_time(b)
    dt = round(abs(tb - ta), 3) if ta is not None and tb is not None else 0.0
    if _same_moment(a, b):
        return "none", "", 0, 0.0, dt
    al = place_align.align(_features(conn, a), _features(conn, b))
    if al is None and dt <= place_align.BURST_S:
        from app.analysis.guess import _scene_embeddings

        embs = _scene_embeddings(conn, [int(a["id"]), int(b["id"])], cache)
        ea, eb = embs.get(int(a["id"])), embs.get(int(b["id"]))
        cos = float(ea @ eb) if ea is not None and eb is not None and ea.shape == eb.shape else None
        al = place_align.burst(dt, cos)
    if al is None:
        return "none", "", 0, 0.0, dt
    return al.kind, json.dumps([round(float(v), 6) for v in al.m]), int(al.inliers), float(al.spread), dt


def _item_row(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row | None:
    return conn.execute(f"SELECT {_ITEM_COLUMNS} FROM items WHERE id = ?", (int(item_id),)).fetchone()


def _open_box(conn: sqlite3.Connection, item_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM faces WHERE item_id = ? AND person_id IS NULL AND assigned_by IN ('auto', 'guess') "
        "UNION ALL SELECT 1 FROM bodies WHERE item_id = ? AND person_id IS NULL AND face_id IS NULL "
        "AND assigned_by IN ('auto', 'guess') LIMIT 1", (item_id, item_id)).fetchone() is not None


def _set_error(msg: str | None) -> None:
    with _state_lock:
        _state["last_error"] = msg


def run_pending(conn: sqlite3.Connection, *, budget_s: float | None = None, limit: int | None = None) -> dict[str, Any]:
    """Line pending photos up with their next photos, newest first, within budget_s seconds and `limit` photos."""
    out: dict[str, Any] = {"items": 0, "pairs": 0, "aligned": 0, "complete": False}
    if not enabled():
        return out
    deadline = None if budget_s is None else time.monotonic() + float(budget_s)
    tried: list[int] = []
    cache: dict[str, Any] = {}
    while True:
        batch = pending_items(conn, BATCH_ITEMS, tried)
        if not batch:
            break
        for item_id, nexts in batch:
            if (limit is not None and out["items"] >= limit) or (deadline is not None and time.monotonic() > deadline):
                return out
            tried.append(item_id)
            try:
                pairs, aligned = _measure_item(conn, item_id, nexts, cache)
            except Exception as e:  # noqa: BLE001 - one photo never stops the runner; it stays pending
                log.exception("place: lining up item %s failed", item_id)
                _set_error(f"{type(e).__name__}: {e}"[:200])
                with _state_lock:
                    fails = _state["fails"][item_id] = _state["fails"].get(item_id, 0) + 1
                if fails >= ABORTS_PER_ITEM:
                    _give_up(conn, item_id, nexts)
                continue
            out["items"] += 1
            out["pairs"] += pairs
            out["aligned"] += aligned
    out["complete"] = not pending_items(conn, 1)
    if out["complete"]:
        _set_error(None)
    return out


def _measure_item(conn: sqlite3.Connection, item_id: int, nexts: list[int], cache: dict[str, Any]) -> tuple[int, int]:
    """Measure the photo's missing pairs (no transaction), write them and its scan row, queue the re-guesses.
    Returns (pairs measured, of them lined up)."""
    a = _item_row(conn, item_id)
    if a is None:
        return 0, 0
    have = {int(r[0]) for r in conn.execute(
        "SELECT b_item FROM place_links WHERE a_item = ? AND rev = ?", (item_id, PLACE_REV))}
    rows: list[tuple[int, str, str, int, float, float]] = []
    for nid in nexts:
        if nid in have:
            continue
        b = _item_row(conn, nid)
        if b is None:
            continue
        kind, m, inliers, spread, dt = measure_pair(conn, a, b, cache)
        rows.append((nid, kind, m, inliers, spread, dt))
    now = now_iso()
    with tx(conn):
        alive = {int(r[0]) for r in conn.execute(
            f"SELECT id FROM items WHERE id IN ({','.join('?' * (len(rows) + 1))})", [item_id, *[r[0] for r in rows]])}
        if item_id not in alive:
            return 0, 0
        for nid, kind, m, inliers, spread, dt in rows:
            if nid in alive:
                conn.execute(
                    "INSERT OR REPLACE INTO place_links(a_item, b_item, rev, kind, m, inliers, spread, dt_s, computed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (item_id, nid, PLACE_REV, kind, m, inliers, spread, dt, now))
        conn.execute("INSERT OR REPLACE INTO place_scans(item_id, rev, sig, scanned_at) VALUES (?, ?, ?, ?)",
                     (item_id, PLACE_REV, "".join(f"{n}," for n in nexts), now))
    usable = [r[0] for r in rows if r[1] != "none"]
    if usable:
        _requeue(conn, [i for i in (item_id, *usable) if _open_box(conn, i)])
    return len(rows), len(usable)


def _give_up(conn: sqlite3.Connection, item_id: int, nexts: list[int]) -> None:
    """A photo that keeps failing: its missing pairs are stored as not lined up, so it is not tried at every kick."""
    now = now_iso()
    with tx(conn):
        alive = {int(r[0]) for r in conn.execute(
            f"SELECT id FROM items WHERE id IN ({','.join('?' * (len(nexts) + 1))})", [item_id, *nexts])}
        if item_id not in alive:
            return
        for nid in nexts:
            if nid in alive:
                conn.execute(
                    "INSERT OR IGNORE INTO place_links(a_item, b_item, rev, kind, m, inliers, spread, dt_s, computed_at) "
                    "VALUES (?, ?, ?, 'none', '', 0, 0, 0, ?)", (item_id, nid, PLACE_REV, now))
        conn.execute("INSERT OR REPLACE INTO place_scans(item_id, rev, sig, scanned_at) VALUES (?, ?, ?, ?)",
                     (item_id, PLACE_REV, "".join(f"{n}," for n in nexts), now))
    with _state_lock:
        _state["fails"].pop(item_id, None)
    log.warning("place: item %s failed %d times; stored as not lined up", item_id, ABORTS_PER_ITEM)


def _requeue(conn: sqlite3.Connection, item_ids: list[int]) -> None:
    if not item_ids:
        return
    try:
        from app.analysis import traits as atraits

        atraits.requeue(conn, item_ids)
        atraits.kick(conn)
    except Exception:  # noqa: BLE001 - the next guess of these photos reads the links anyway
        log.exception("place: could not queue %d photo(s) for a fresh guess", len(item_ids))


# ------------------------------------------------------------------ runner
def _rss() -> int:
    from app.analysis.traits import _rss as rss

    return rss()


def _mark_ready_if_done(conn: sqlite3.Connection) -> bool:
    if ready(conn) or pending_items(conn, 1):
        return False
    with tx(conn):
        kv_set(conn, READY_KEY, PLACE_REV)
    log.info("place: every photo is lined up with its next photos (rev %s)", PLACE_REV)
    return True


def _has_work(conn: sqlite3.Connection) -> bool:
    return enabled() and (not ready(conn) or bool(pending_items(conn, 1)))


def kick(conn: sqlite3.Connection | None = None) -> None:
    """Start the runner thread unless it runs. Nothing when the workers are off (tests call run_pending) or
    PLACE_ENABLED is off. The thread itself looks for work and ends when there is none, so the caller (the analysis
    job, housekeeping) never pays for the pending query."""
    if not settings.workers_enabled or not enabled():
        return
    from app import worker

    if not worker.background_running(RUNNER_KEY):
        worker.run_in_background(RUNNER_KEY, _loop)


def start(conn: sqlite3.Connection) -> None:
    """On startup (ingest.recover_on_startup): guesses made under other rules are queued for a fresh guess, an
    empty library is ready at once; then kick."""
    try:
        apply_settings_change(conn)
    except Exception:  # noqa: BLE001 - the check never blocks the start
        log.exception("place: the settings check failed")
    if enabled():
        _mark_ready_if_done(conn)
    kick(conn)


def apply_settings_change(conn: sqlite3.Connection) -> int:
    """When the position rules differ from the ones recorded (PLACE_ENABLED, PLACE_ONLY_GUESSES, or the constants
    of another version), the photos whose open guesses rest on position (basis 'place' or 'clothes+place') are
    queued for a fresh guess, which withdraws what the rules no longer support. Returns how many were queued."""
    from app.analysis import place_match

    sig = place_match.rules_sig()
    if kv_get(conn, RULES_KEY) == sig:
        return 0
    basis = "(CASE WHEN json_valid(guess_evidence) THEN json_extract(guess_evidence, '$.basis') END) IN ('place', 'clothes+place')"
    items = [int(r[0]) for r in conn.execute(
        f"SELECT item_id FROM faces WHERE assigned_by = 'guess' AND person_id IS NULL AND {basis} UNION "
        f"SELECT item_id FROM bodies WHERE assigned_by = 'guess' AND person_id IS NULL AND {basis}")]
    _requeue(conn, items)
    with tx(conn):
        kv_set(conn, RULES_KEY, sig)
    if items:
        log.info("place: the position rules changed; %d photo(s) queued for a fresh guess", len(items))
    return len(items)


def _loop() -> None:
    from app import worker
    from app.db import connect

    worker._lower_priority()
    conn = connect()
    base = _rss()
    idle = 0
    try:
        while True:
            r = run_pending(conn, budget_s=ROUND_BUDGET_S)
            done = _mark_ready_if_done(conn)
            now = _rss()
            if now - base > RSS_GROWTH or now > RSS_CAP:
                log.warning("place: the runner stops for now, the process holds %d MB", now // 2 ** 20)
                return
            idle = 0 if (r["items"] or done) else idle + 1
            if idle >= IDLE_ROUNDS:
                return
            if r["complete"]:
                time.sleep(LOOP_SLEEP_S)
                if not _has_work(conn):
                    return
    finally:
        conn.close()


def progress(conn: sqlite3.Connection) -> dict[str, Any]:
    """{scanned, eligible, ready, rev, last_error, enabled} for the Learning tab."""
    eligible = conn.execute(f"SELECT COUNT(*) FROM ({ELIGIBLE_SQL})").fetchone()[0]
    scanned = conn.execute(
        f"SELECT COUNT(*) FROM place_scans p JOIN ({ELIGIBLE_SQL}) e ON e.id = p.item_id WHERE p.rev = ?",
        (PLACE_REV,)).fetchone()[0]
    with _state_lock:
        err = _state["last_error"]
    return {"scanned": int(scanned), "eligible": int(eligible), "ready": ready(conn), "rev": ready_rev(conn),
            "last_error": err, "enabled": enabled()}
