"""Body traits (section 34): tattoos, mehndi and nails read from people boxes, kept as evidence for guesses.

CONTRACT.
- Rows: body_traits (one per trait region: kind in KINDS, part in PARTS, a normalised region, quality, `vec`
  float32[768] = DINOv2 CLS of the crop then of its mirror, `detail` = the native crop as WebP for
  app.analysis.trait_sift, `attrs` JSON, a 192 px WebP `crop` for the UI) and trait_scans (one per measured body:
  rev, the body's box and the item's box fingerprint, a status in STATUSES, the bare limbs). Derived, never
  journalled; a rescan deletes and inserts, so ids are never reused.
- The item is the unit: a body is pending when its scan is missing, has another rev, or its item_boxes differs
  from the item's fingerprint now (ITEM_BOXES_SQL); any box change in a photo re-measures every body in it. A
  'vault' scan is pending once the item leaves the vault, and 'doc', 'child', 'no_pixels' and 'error' scans once
  they are RECHECK_S old (pending_items, newest item first).
- wanted_rev() = "TRAITS_REV:" + the sorted kinds measured (kinds_on: mehndi only with TRAITS_MEHNDI), e.g.
  "2:nails,tattoo"; never the models that loaded.
  ready_rev(conn) is kv READY_KEY: the last rev the whole library was measured with, set when nothing is pending
  (an empty library is ready at start); ready(conn) is True once one exists (matching keeps using the rows of
  that rev during a later rescan). usable_sql() is the WHERE fragment that keeps a row only when its scan is
  'ok', of rev :ready_rev, measured on the body's current box, on a live item outside the vault and not a
  screenshot or document.
- engines() / available(): the four engines (pose, hands, dino, and the shared SigLIP engine); measuring needs
  all four and settings.traits_enabled. siglip_images(crops) turns RGB arrays into the PIL RGB images SigLIP
  takes (a raw array would be read as BGR).
- The runner (one thread, worker.run_in_background key RUNNER_KEY, nice 19, its own connection): kick() starts it
  when the workers are on, it is not running, and there is work (the requeue, the sweep, a calibration sample,
  or pending bodies with every engine present). Each round: the sweep (trait_match.sweep_open, then SWEPT_KEY)
  when a new rev became ready; drain_requeue; run_pending in ROUND_BUDGET_S steps, draining the queue between
  them, until nothing is pending, an engine fails or the process grew RSS_GROWTH (or passed RSS_CAP); ready
  marked when nothing is pending; a calibration sample (trait_match.calibration_sample) when due. It sleeps
  LOOP_SLEEP_S and asks again before it exits, because kicks are dropped while it runs.
- run_pending(conn, budget_s, limit) -> {items, bodies, complete, aborted}: measures each pending item
  (trait_regions.measure_item, outside any transaction) and writes it in one short transaction that first checks
  the item's fingerprint and bodies are unchanged (else the item stays pending). An engine failure stops the run
  with nothing written for the item and the message in progress().last_error; our own error on one body writes
  status 'error' for it. After a write: trait_match.hide_hits_for_rows for the new tattoo and mehndi rows, then,
  when ready and settings.trait_matching, guess_item on the item and trait_match.targets_for_rows of the new
  rows on named bodies into the requeue. Without every engine: {..., paused: [keys]}, nothing written or deleted.
- requeue / drain_requeue: trait_requeue (survives a restart); drain runs guess_item on each queued item, newest
  first, with one shared cache, and needs no model.
- after_undo(conn, tables, item_ids) (from undo.undo after its transaction): trait hides replayed ->
  trait_match.after_hide_change; boxes or items replayed -> kick. prune(conn) (housekeeping) drops the
  descriptors of hides gone for longer than undo.KEEP_DAYS whose removal the undo journal no longer holds.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable, NamedTuple

import numpy as np

from app.config import settings
from app.db import kv_get, kv_set, tx
from app.util import now_iso, parse_iso

log = logging.getLogger(__name__)

TRAITS_REV = "2"   # 2: the reviewed gates (tattoo without the skin/ink cuts, ownership, mehndi off)
KINDS = ("mehndi", "nails", "tattoo")
PARTS = ("forearm_l", "forearm_r", "upper_arm_l", "upper_arm_r", "calf_l", "calf_r", "hand_l", "hand_r")
FAMILIES = ("forearm", "upper_arm", "calf", "hand")
# Limb strip share of a limb crop (the SIFT mask is side * share / 1.15 wide); hands have no strip.
PART_SHARE: dict[str, float | None] = {"forearm": 0.6, "upper_arm": 0.7, "calf": 0.55, "hand": None}
STATUSES = ("ok", "small", "child", "doc", "vault", "video", "no_pixels", "error")
VERDICTS = ("not_trait", "not_theirs", "shown")
READY_KEY = "traits_ready"        # kv: the rev the whole library was last measured with
SWEPT_KEY = "traits_swept"        # kv: the ready rev the open boxes were last re-guessed for
RULES_KEY = "trait_rules"         # kv: hash of TRAITS_REV and the trait settings (trait_match.apply_settings_change)
CALIB_KEY = "trait_calib"         # kv: the last calibration sample (JSON)
CALIB_AT_KEY = "trait_calib_at"   # kv: when it was taken
VEC_DIM = 384                     # one DINOv2-small CLS; vec holds two (crop, mirror)
ROW_COLUMNS = "id, body_id, item_id, kind, part, x, y, w, h, quality, attrs, created_at"
# The fingerprints, written and compared with SQLite's own printf so both sides round the same way.
BOX_SQL = "printf('%.4f,%.4f,%.4f,%.4f', b.x, b.y, b.w, b.h)"  # alias b = bodies
ITEM_BOXES_SQL = ("SELECT group_concat(printf('%d:%.4f,%.4f,%.4f,%.4f', id, x, y, w, h), ';' ORDER BY id) "
                  "FROM bodies WHERE item_id = ?")
SIDES = {"l": "left", "r": "right"}
PART_WORDS = {"forearm": "forearm", "upper_arm": "upper arm", "calf": "calf", "hand": "hand"}


class Engines(NamedTuple):
    pose: Any     # app.ml.pose or None
    hands: Any    # app.ml.hands or None
    dino: Any     # app.ml.dino or None
    siglip: Any   # app.ml.siglip.SiglipEngine (get_engine()) or None


# ------------------------------------------------------------------ parts and rows
def family(part: str) -> str:
    """The part without its side: forearm_l -> forearm. Matching compares families, never sides."""
    return part.rsplit("_", 1)[0] if part.endswith(("_l", "_r")) else part


def share_for(part: str) -> float | None:
    return PART_SHARE.get(family(part))


def decode_vec(blob: bytes | None) -> np.ndarray | None:
    """vec as float32[768] (crop, then mirror), or None when missing or of another size."""
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    return v if v.shape[0] == 2 * VEC_DIM else None


def label(kind: str, part: str, attrs: dict[str, Any] | None = None) -> str:
    """Words for a row: "Tattoo on the left forearm", "Red nail polish, long"."""
    attrs = attrs or {}
    fam, _, side = part.rpartition("_") if part.endswith(("_l", "_r")) else (part, "", "")
    where = f"the {SIDES[side]} {PART_WORDS.get(fam, fam)}" if side else f"the {PART_WORDS.get(fam, fam)}"
    if kind == "nails":
        bits = [f"{attrs['color'].capitalize()} nail polish" if attrs.get("color") else "Nail polish"]
        if attrs.get("finish") and attrs["finish"] != "plain":
            bits[0] += f" ({attrs['finish']})"
        if attrs.get("length"):
            bits.append(str(attrs["length"]))
        return ", ".join(bits)
    return f"{'Mehndi' if kind == 'mehndi' else 'Tattoo'} on {where}"


def usable_sql() -> str:
    """WHERE fragment for usable rows, aliases t (body_traits), s (trait_scans), b (bodies), i (items). It binds
    the named parameter :ready_rev, so the whole query must use named parameters."""
    return (f"s.body_id = t.body_id AND s.status = 'ok' AND s.rev = :ready_rev AND s.box = {BOX_SQL} "
            "AND i.deleted_at IS NULL AND +i.nsfw = 0 AND COALESCE(i.category, '') NOT IN ('screenshots', 'documents')"
            + ("" if settings.traits_mehndi else " AND t.kind != 'mehndi'"))


def body_box(conn: sqlite3.Connection, body_id: int) -> str | None:
    """The body's box as trait_scans.box stores it."""
    row = conn.execute(f"SELECT {BOX_SQL} FROM bodies b WHERE b.id = ?", (int(body_id),)).fetchone()
    return row[0] if row else None


def item_fingerprint(conn: sqlite3.Connection, item_id: int) -> str | None:
    """Every body box of the item, as trait_scans.item_boxes stores it; None when the item has no body."""
    row = conn.execute(ITEM_BOXES_SQL, (int(item_id),)).fetchone()
    return row[0] if row else None


def rows_for_bodies(conn: sqlite3.Connection, body_ids: list[int]) -> dict[int, list[sqlite3.Row]]:
    """The rows to show for these bodies (ROW_COLUMNS only, never a BLOB): measured with the body's current box,
    status 'ok', of the ready rev (or of the wanted rev before the library is ready). Ordered by kind, part, id."""
    out: dict[int, list[sqlite3.Row]] = {int(b): [] for b in body_ids}
    if not out:
        return out
    rev = ready_rev(conn) or wanted_rev()
    ids = sorted(out)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ", ".join("?" * len(chunk))
        cols = ", ".join(f"t.{c.strip()}" for c in ROW_COLUMNS.split(","))
        for r in conn.execute(
                f"SELECT {cols} FROM body_traits t JOIN trait_scans s ON s.body_id = t.body_id "
                f"JOIN bodies b ON b.id = t.body_id WHERE t.body_id IN ({marks}) AND s.status = 'ok' AND s.rev = ? "
                f"AND s.box = {BOX_SQL} ORDER BY t.body_id, t.kind, t.part, t.id", (*chunk, rev)):
            out[int(r["body_id"])].append(r)
    return out


# ------------------------------------------------------------------ engines
def engines() -> Engines:
    """The four engines, None where a model is missing or models are off (PCC_NO_MODELS)."""
    from app.analysis.pipeline import _model

    sig = _model("siglip")
    siglip = None
    if sig is not None:
        try:
            siglip = sig.get_engine()
        except Exception as e:  # noqa: BLE001 - a broken SigLIP install only pauses the traits runner
            log.warning("traits: SigLIP engine unavailable: %s", e)
    return Engines(_model("pose"), _model("hands"), _model("dino"), siglip)


def missing(e: Engines | None = None) -> list[str]:
    """The engines that are missing, by key. Without `e` it only checks that each model is present (no session is
    built: SigLIP's get_engine() loads about 1 GB), so progress() stays cheap."""
    if e is not None:
        return [k for k in Engines._fields if getattr(e, k) is None]
    from app.analysis.pipeline import _model

    return [k for k in Engines._fields if _model(k) is None]


def available() -> bool:
    return bool(settings.traits_enabled) and not missing()


def siglip_images(crops_rgb: list[np.ndarray]) -> list[Any]:
    """RGB uint8 arrays as PIL RGB images, the form SigLIP reads without swapping red and blue."""
    from PIL import Image

    return [Image.fromarray(np.ascontiguousarray(c, dtype=np.uint8), "RGB") for c in crops_rgb]


# ------------------------------------------------------------------ readiness
def kinds_on() -> tuple[str, ...]:
    """The kinds measured now: mehndi only with TRAITS_MEHNDI (it read plain hands as mehndi on the owner's library)."""
    return tuple(k for k in KINDS if k != "mehndi" or settings.traits_mehndi)


def wanted_rev() -> str:
    """TRAITS_REV and the kinds measured: switching mehndi on or off rescans by itself."""
    return f"{TRAITS_REV}:{','.join(sorted(kinds_on()))}"


def ready_rev(conn: sqlite3.Connection) -> str | None:
    return kv_get(conn, READY_KEY) or None


def ready(conn: sqlite3.Connection) -> bool:
    """True once the library was measured completely with some rev (matching then uses that rev's rows)."""
    return ready_rev(conn) is not None


# ------------------------------------------------------------------ runner
RUNNER_KEY = "traits"             # worker.run_in_background key of the runner thread
BATCH_ITEMS = 16                  # pending items fetched per query
ROUND_BUDGET_S = 30.0             # run_pending and drain_requeue budgets per step of the loop
LOOP_SLEEP_S = 2.0                # the loop sleeps this long and asks again before it exits (kicks are dropped
                                  # while it runs)
IDLE_ROUNDS = 2                   # rounds in a row without progress before the loop gives up
RECHECK_S = 86400.0               # gated statuses (doc, child, no_pixels, error) are checked again after a day
RSS_GROWTH = 2 * 1024 ** 3        # the loop stops when the process grew this much since it started...
RSS_CAP = 7 * 1024 ** 3           # ...or passes this (the review server's limit is 8 GB); the next kick resumes
CALIB_EVERY_ROWS = 200            # a new calibration sample after this many new rows...
CALIB_MIN_S = 6 * 3600.0          # ...and at most this often
GATED = ("doc", "child", "no_pixels", "error")
_FP_EXPR = "group_concat(printf('%d:%.4f,%.4f,%.4f,%.4f', id, x, y, w, h), ';' ORDER BY id)"
assert ITEM_BOXES_SQL == f"SELECT {_FP_EXPR} FROM bodies WHERE item_id = ?"  # one text for both fingerprints
_PENDING_SQL = f"""
WITH fp AS (SELECT item_id, {_FP_EXPR} AS boxes FROM bodies GROUP BY item_id)
SELECT DISTINCT b.item_id FROM bodies b JOIN items i ON i.id = b.item_id JOIN fp ON fp.item_id = b.item_id
LEFT JOIN trait_scans s ON s.body_id = b.id
WHERE i.deleted_at IS NULL AND i.analysis_status = 'done'
  AND b.item_id NOT IN (SELECT value FROM json_each(:skip))
  AND (s.body_id IS NULL OR s.rev != :rev OR s.item_boxes != fp.boxes
       OR (s.status = 'vault' AND i.nsfw = 0)
       OR (s.status IN ({", ".join(f"'{g}'" for g in GATED)}) AND s.scanned_at < :day_ago))
ORDER BY b.item_id DESC LIMIT :limit"""

_state_lock = threading.Lock()
_state: dict[str, Any] = {"last_error": None, "calib_tried": None}


class _Written(NamedTuple):
    rows: list[tuple[int, int, str]]  # (row id, body id, kind) of the new rows


def pending_items(conn: sqlite3.Connection, limit: int, skip: Iterable[int] = ()) -> list[int]:
    """Items with a body to measure, newest first: no scan, another rev, a box of the item changed since, out of
    the vault now, or a gated status older than RECHECK_S."""
    return [int(r[0]) for r in conn.execute(_PENDING_SQL, {
        "skip": json.dumps(sorted({int(i) for i in skip})), "rev": wanted_rev(),
        "day_ago": now_iso(-RECHECK_S), "limit": int(limit)})]


def _set_error(msg: str | None) -> None:
    with _state_lock:
        _state["last_error"] = msg


def _rss() -> int:
    """Resident memory of this process in bytes (0 where /proc is missing)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _rss_stop(base: int) -> bool:
    now = _rss()
    return now - base > RSS_GROWTH or now > RSS_CAP


def _mark_ready_if_done(conn: sqlite3.Connection) -> bool:
    """Set READY_KEY to the wanted rev once nothing is pending (an empty library counts). Needs no model."""
    if ready_rev(conn) == wanted_rev() or pending_items(conn, 1):
        return False
    with tx(conn):
        kv_set(conn, READY_KEY, wanted_rev())
    log.info("traits: every people box is measured (rev %s)", wanted_rev())
    return True


def _sweep_due(conn: sqlite3.Connection) -> bool:
    rev = ready_rev(conn)
    return rev is not None and kv_get(conn, SWEPT_KEY) != rev


def _calib_due(conn: sqlite3.Connection) -> bool:
    """A calibration sample is due when ready and there is none yet, or CALIB_EVERY_ROWS rows were added since
    the last one (its JSON's max_id), and the last is CALIB_MIN_S old. A try that wrote nothing waits as long."""
    if not ready(conn):
        return False
    with _state_lock:
        tried = _state["calib_tried"]
    if tried is not None and time.monotonic() - tried < CALIB_MIN_S:
        return False
    top = conn.execute("SELECT MAX(id) FROM body_traits").fetchone()[0]
    if not top:
        return False
    at = parse_iso(kv_get(conn, CALIB_AT_KEY))
    if at is None:
        return True
    if (datetime.now(timezone.utc) - at).total_seconds() < CALIB_MIN_S:
        return False
    try:
        last = int(json.loads(kv_get(conn, CALIB_KEY) or "{}").get("max_id") or 0)
    except (TypeError, ValueError, AttributeError):
        last = 0
    return int(top) - last >= CALIB_EVERY_ROWS


def _has_work(conn: sqlite3.Connection) -> bool:
    """Cheap checks first: the re-guess queue, the sweep, a calibration sample, then the pending items."""
    if conn.execute("SELECT 1 FROM trait_requeue LIMIT 1").fetchone():
        return True
    if _sweep_due(conn) or _calib_due(conn):
        return True
    if settings.traits_enabled and not missing():
        return ready_rev(conn) != wanted_rev() or bool(pending_items(conn, 1))
    return False


def kick(conn: sqlite3.Connection | None = None) -> None:
    """Start the runner thread when there is work. Nothing when the workers are off (tests call run_pending and
    drain_requeue themselves) or while the runner is running (it asks again before it exits)."""
    if not settings.workers_enabled:
        return
    from app import worker

    if worker.background_running(RUNNER_KEY):
        return
    own = conn is None
    if own:
        from app.db import connect

        conn = connect()
    try:
        work = _has_work(conn)
    finally:
        if own:
            conn.close()
    if work:
        worker.run_in_background(RUNNER_KEY, _loop)


def start(conn: sqlite3.Connection) -> None:
    """On startup (ingest.recover_on_startup): mark the library ready when nothing is pending (an empty library),
    then kick when anything is pending, queued or due. The sweep runs on the runner thread, never here."""
    _mark_ready_if_done(conn)
    kick(conn)


def _loop() -> None:
    """The runner thread: its own connection, nice 19, rounds until there is no work left, the memory stop or an
    engine failure."""
    from app import worker
    from app.db import connect

    worker._lower_priority()
    conn = connect()
    base = _rss()
    idle = 0
    try:
        while True:
            progressed, stop = _round(conn, base)
            if stop:
                return
            idle = 0 if progressed else idle + 1
            if idle >= IDLE_ROUNDS:
                return
            time.sleep(LOOP_SLEEP_S)
            if not _has_work(conn):
                return
    finally:
        conn.close()


def _round(conn: sqlite3.Connection, base: int) -> tuple[bool, bool]:
    """One pass of the loop: (progress made, stop now)."""
    from app.analysis import trait_match

    progressed = False
    if _sweep_due(conn):
        rev = ready_rev(conn)
        try:
            n = trait_match.sweep_open(conn)
            if n:
                log.info("traits: %d item(s) with open boxes queued for a re-guess (rev %s)", n, rev)
        except Exception:  # noqa: BLE001 - the sweep is retried at the next ready rev only; log it
            log.exception("traits: the sweep of open boxes failed")
        with tx(conn):
            kv_set(conn, SWEPT_KEY, rev)
        progressed = True
    if drain_requeue(conn, budget_s=ROUND_BUDGET_S):
        progressed = True
    if settings.traits_enabled:
        while True:
            r = run_pending(conn, budget_s=ROUND_BUDGET_S)
            if r["items"]:
                progressed = True
            if r["aborted"]:
                return progressed, True
            if _rss_stop(base):
                log.warning("traits: the runner stops for now, the process holds %d MB", _rss() // 2 ** 20)
                return progressed, True
            if drain_requeue(conn, budget_s=ROUND_BUDGET_S):
                progressed = True
            if r["complete"] or not r["items"]:
                break
    if _mark_ready_if_done(conn):
        progressed = True
    if _calib_due(conn):
        with _state_lock:
            _state["calib_tried"] = time.monotonic()
        try:
            trait_match.calibration_sample(conn)
        except Exception:  # noqa: BLE001 - a failed sample leaves the last one in force
            log.exception("traits: the calibration sample failed")
    return progressed, False


def run_pending(conn: sqlite3.Connection, *, budget_s: float | None = None, limit: int | None = None) -> dict[str, Any]:
    """Measure pending items, newest first, each as a whole, within budget_s seconds and `limit` items.
    {items, bodies, complete (nothing is pending any more), aborted (an engine failed: nothing written for that
    item, the message in progress().last_error)}, plus `paused` (the missing engines) when it could not run."""
    out: dict[str, Any] = {"items": 0, "bodies": 0, "complete": False, "aborted": False}
    if not settings.traits_enabled:
        return out
    gone = missing()  # cheap: no session is built, so a missing model never loads SigLIP's 1 GB for nothing
    if gone:
        out["paused"] = gone
        return out
    e = engines()
    gone = missing(e)
    if gone:
        out["paused"] = gone
        return out
    from app.analysis import trait_regions

    deadline = None if budget_s is None else time.monotonic() + float(budget_s)
    tried: list[int] = []
    cache: dict[str, Any] = {}
    while True:
        ids = pending_items(conn, BATCH_ITEMS, tried)
        if not ids:
            break
        for item_id in ids:
            if (limit is not None and out["items"] >= limit) or (deadline is not None and time.monotonic() > deadline):
                return out
            tried.append(item_id)
            try:
                m = trait_regions.measure_item(conn, item_id, e)
            except trait_regions.EngineError as err:
                if _aborts_on(item_id) < ABORTS_PER_ITEM:
                    _set_error(str(err))
                    log.warning("traits: a model failed on item %s, the run stops: %s", item_id, err)
                    out["aborted"] = True
                    return out
                log.warning("traits: item %s failed %d times in a row, marked as an error: %s",
                            item_id, ABORTS_PER_ITEM, err)
                m = _error_scans(conn, item_id)
            except Exception:  # noqa: BLE001 - one odd photo must never stop the backfill
                log.exception("traits: item %s could not be measured, marked as an error", item_id)
                m = _error_scans(conn, item_id)
            if m is None:
                continue
            try:
                written = _write(conn, m)
            except Exception:  # noqa: BLE001 - the item stays pending, the others go on
                log.exception("traits: the rows of item %s could not be written", item_id)
                continue
            if written is None:
                continue
            out["items"] += 1
            out["bodies"] += len(m.scans)
            _after_write(conn, item_id, written, cache)
    out["complete"] = not pending_items(conn, 1)
    if out["complete"]:
        _set_error(None)
        _mark_ready_if_done(conn)
    return out


ABORTS_PER_ITEM = 3   # an engine failing on the same item this often in a row: that item is an error, not the engine


def _aborts_on(item_id: int) -> int:
    """Count a run stopped by an engine failure on this item; consecutive failures on the same item add up."""
    with _state_lock:
        last = _state.get("abort_item")
        n = (last[1] + 1) if last and last[0] == int(item_id) else 1
        _state["abort_item"] = (int(item_id), n)
    return n


def _error_scans(conn: sqlite3.Connection, item_id: int) -> Any:
    """An 'error' scan for every body of the item (re-checked after a day), or None when it has no bodies."""
    from app.analysis import trait_regions

    fp = item_fingerprint(conn, int(item_id))
    ids = [int(r[0]) for r in conn.execute("SELECT id FROM bodies WHERE item_id = ? ORDER BY id", (int(item_id),))]
    if fp is None or not ids:
        return None
    return trait_regions.Measured(int(item_id), fp, {bid: ("error", {}) for bid in ids})


def _write(conn: sqlite3.Connection, m: Any) -> _Written | None:
    """One short transaction per item: nothing when a box of the item changed (or a body came or went) while it
    was measured; else the item's scan rows and trait rows are replaced. None leaves the item pending."""
    now = now_iso()
    try:
        with tx(conn):
            if item_fingerprint(conn, m.item_id) != m.fingerprint:
                return None
            bodies = {int(r[0]) for r in conn.execute("SELECT id FROM bodies WHERE item_id = ?", (m.item_id,))}
            if bodies != set(m.scans):
                return None
            ids = sorted(bodies)
            marks = ", ".join("?" * len(ids))
            conn.execute(f"DELETE FROM body_traits WHERE body_id IN ({marks})", ids)
            for body_id in ids:
                status, bare = m.scans[body_id]
                if status not in STATUSES:
                    raise ValueError(f"unknown trait scan status {status!r}")
                conn.execute(
                    "INSERT INTO trait_scans(body_id, item_id, rev, box, item_boxes, status, bare, scanned_at) "
                    f"SELECT b.id, b.item_id, ?, {BOX_SQL}, ?, ?, ?, ? FROM bodies b WHERE b.id = ? "
                    "ON CONFLICT(body_id) DO UPDATE SET item_id = excluded.item_id, rev = excluded.rev, "
                    "box = excluded.box, item_boxes = excluded.item_boxes, status = excluded.status, "
                    "bare = excluded.bare, scanned_at = excluded.scanned_at",
                    (wanted_rev(), m.fingerprint, status, json.dumps(bare or {}, sort_keys=True), now, body_id))
            written: list[tuple[int, int, str]] = []
            for r in m.rows:
                if r.kind not in KINDS or r.part not in PARTS:
                    raise ValueError(f"unknown trait row {r.kind}/{r.part}")
                cur = conn.execute(
                    "INSERT INTO body_traits(body_id, item_id, kind, part, x, y, w, h, quality, vec, detail, attrs, "
                    "crop, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r.body_id, m.item_id, r.kind, r.part, r.x, r.y, r.w, r.h, round(float(r.quality), 4), r.vec,
                     r.detail, json.dumps(r.attrs, sort_keys=True), r.crop, now))
                written.append((int(cur.lastrowid), int(r.body_id), r.kind))
            return _Written(written)
    except sqlite3.IntegrityError as err:
        log.warning("traits: item %s left pending, its rows did not fit: %s", m.item_id, err)
        return None


def _after_write(conn: sqlite3.Connection, item_id: int, written: _Written, cache: dict[str, Any]) -> None:
    """Outside the write: look-alike hits of the new rows (rule c), then, once ready and matching is on, the
    item's open boxes guessed again and the photos the new named rows may now name queued for a re-guess."""
    from app.analysis import trait_match

    ink = [rid for rid, _b, kind in written.rows if kind in ("tattoo", "mehndi")]
    if ink:
        try:
            trait_match.hide_hits_for_rows(conn, ink)
        except Exception:  # noqa: BLE001 - a missing look-alike hit only shows a row the owner hid elsewhere
            log.exception("traits: the look-alike check of item %s failed", item_id)
    if not (ready(conn) and settings.trait_matching):
        return
    from app.analysis.guess import guess_item

    try:
        guess_item(conn, item_id, cache)
    except Exception:  # noqa: BLE001 - one item's guess never stops the runner
        log.exception("traits: guessing item %s again failed", item_id)
    if not ink:
        return
    marks = ", ".join("?" * len(ink))
    named = [int(r[0]) for r in conn.execute(
        f"SELECT t.id FROM body_traits t JOIN bodies b ON b.id = t.body_id WHERE t.id IN ({marks}) "
        "AND b.person_id IS NOT NULL", ink)]
    if named:
        try:
            requeue(conn, trait_match.targets_for_rows(conn, named, trait_match.REGUESS_LIMIT))
        except Exception:  # noqa: BLE001
            log.exception("traits: queueing the look-alike photos of item %s failed", item_id)


def requeue(conn: sqlite3.Connection, item_ids: Iterable[int]) -> int:
    """Queue items for a re-guess on the runner thread (INSERT OR IGNORE). Returns how many were new."""
    ids = sorted({int(i) for i in item_ids})
    if not ids:
        return 0
    now = now_iso()
    n = 0
    with tx(conn):
        for i in ids:
            n += conn.execute("INSERT OR IGNORE INTO trait_requeue(item_id, queued_at) "
                              "SELECT id, ? FROM items WHERE id = ?", (now, i)).rowcount
    return n


def drain_requeue(conn: sqlite3.Connection, *, budget_s: float | None = None) -> int:
    """guess_item on the queued items, newest first, with one shared cache; each row goes before its guess (a
    requeue during the guess queues the item again). Needs no model. Returns how many items were guessed."""
    from app.analysis.guess import guess_item

    deadline = None if budget_s is None else time.monotonic() + float(budget_s)
    cache: dict[str, Any] = {}
    n = 0
    while True:
        ids = [int(r[0]) for r in conn.execute("SELECT item_id FROM trait_requeue ORDER BY item_id DESC LIMIT 100")]
        if not ids:
            return n
        for item_id in ids:
            if deadline is not None and time.monotonic() > deadline:
                return n
            with tx(conn):
                conn.execute("DELETE FROM trait_requeue WHERE item_id = ?", (item_id,))
            try:
                if conn.execute("SELECT 1 FROM trait_hides WHERE item_id = ? LIMIT 1", (item_id,)).fetchone():
                    from app.analysis import trait_match

                    trait_match.after_hide_change(conn, [item_id])  # look-alikes of its hides, then the re-guess
                else:
                    guess_item(conn, item_id, cache)
            except Exception:  # noqa: BLE001 - one item's guess never stops the queue
                log.exception("traits: the re-guess of item %s failed", item_id)
            n += 1


def progress(conn: sqlite3.Connection) -> dict[str, Any]:
    """{scanned, eligible, ready, rev, paused, last_error, enabled} for the Learning tab."""
    rev = wanted_rev()
    eligible = conn.execute(
        "SELECT COUNT(*) FROM bodies b JOIN items i ON i.id = b.item_id "
        "WHERE i.deleted_at IS NULL AND i.analysis_status = 'done'").fetchone()[0]
    scanned = conn.execute(
        "SELECT COUNT(*) FROM trait_scans s JOIN bodies b ON b.id = s.body_id JOIN items i ON i.id = s.item_id "
        "WHERE s.rev = ? AND i.deleted_at IS NULL AND i.analysis_status = 'done'", (rev,)).fetchone()[0]
    gone = missing() if settings.traits_enabled else []
    with _state_lock:
        err = _state["last_error"]
    return {"scanned": int(scanned), "eligible": int(eligible), "ready": ready(conn), "rev": ready_rev(conn),
            "paused": gone or None, "last_error": err, "enabled": bool(settings.traits_enabled)}


def after_undo(conn: sqlite3.Connection, tables: set[str], item_ids: set[int]) -> None:
    """After undo.undo() replayed rows of these tables in these items (outside its transaction): trait hides put
    back or taken away recompute their look-alikes and re-guess (trait_match.after_hide_change); a box put back,
    moved back or removed again changes its item's fingerprint, so the runner is kicked."""
    if "trait_hides" in tables and item_ids:
        requeue(conn, sorted(item_ids))  # the runner recomputes look-alikes and re-guesses: never inside the request
        kick(conn)
    if tables & {"bodies", "items"}:
        kick(conn)


def prune(conn: sqlite3.Connection) -> int:
    """Drop trait_hide_desc rows whose hide is gone, older than undo.KEEP_DAYS, and whose hide's removal is not in
    the undo journal any more (while it is, an undo can bring the hide back and needs its descriptors).
    Returns the rows removed."""
    from app.undo import KEEP_DAYS

    cutoff = now_iso(-KEEP_DAYS * 86400.0)
    with tx(conn):
        return conn.execute(
            "DELETE FROM trait_hide_desc WHERE created_at < ? AND hide_id NOT IN (SELECT id FROM trait_hides) "
            "AND hide_id NOT IN (SELECT json_extract(old, '$.id') FROM undo_log WHERE tbl = 'trait_hides' "
            "AND op = 'delete' AND json_extract(old, '$.id') IS NOT NULL)", (cutoff,)).rowcount


def attrs_of(row: sqlite3.Row) -> dict[str, Any]:
    """A row's attrs as a dict ({} when missing or broken)."""
    try:
        v = json.loads(row["attrs"] or "{}")
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}
