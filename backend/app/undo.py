"""Undo (section 33): every change the owner makes through the API can be put back.

CONTRACT.
- An owner action is one API write request (POST, PUT, PATCH or DELETE under /api/, outside
  NOT_UNDOABLE). ActionMiddleware names it: the client's X-Action-Id header when it matches ACTION_ID_RE
  (so the several requests of one action in the app share it), else a new id. The name lives in
  app.db.current_action (a ContextVar) for the request, and the response carries X-Action-Id when the
  action changed anything.
- Journal: app.db.tx writes the undo_context row at the start of every write transaction of an action and
  removes it before COMMIT, so no other connection ever sees it. Triggers on the JOURNALED tables
  (install_triggers, from init_db) log each row change into undo_log: the row before (old) and after (new)
  as JSON, BLOB columns as hex, found again by the table's primary key. Deletes keep the whole row,
  updates only the columns that changed (plus the key and item_id), inserts all but the BLOBs.
- undo(conn, action_id) -> dict: puts back what the action changed, newest change first, in one
  transaction with foreign keys deferred. Bookkeeping columns (NOISE: updated_at and the items' face, body
  and animal counts) never block it and are not put back; the counts are counted again. It is itself an action (undo_actions.undo_of), so undoing it is a
  redo. An update puts back only the columns it changed and only where the column still holds what the
  action wrote (the analysis may have changed it since: then that column stays, and the answer says
  partial); a row the action inserted is deleted; a row it deleted is inserted again. UndoError: 404
  not_found, 410 expired (older than KEEP_DAYS), 409 already_undone, 409 later_changes (a later action, not
  undone, changed the same rows and columns; `blocking` lists them), 409 conflict (the old row no longer
  fits, for example a name taken since). Crop files the action removed come back from its stash; the crops
  of rows the undo removes go to the undo's own stash. Afterwards faces_version and search_version move and
  the caller refreshes the items' full-text rows (`items` in the answer). After the transaction,
  app.analysis.traits.after_undo(conn, tables, items) hears which tables and items were replayed (a box put
  back re-measures its photo's traits; a trait hide put back recomputes its look-alikes); a failure there is
  logged, never raised.
- item_ids(conn, action_id): the items an action touched (the vault check of the route).
- discard(path): during an action a removed crop moves to DATA_DIR/undo/<action>/ instead of being deleted.
  The same holds for the recognition trained on a person (section 10, learned/person_<id>.json): the file goes
  with its persons row (ROW_FILES), so undoing the delete or the merge of a trained person brings the model back
  with them, and a redo takes it away again.
- keep_copy(path): during an action a crop about to be replaced (a moved box, section 10) is copied there
  first. An undo that puts a box's x, y, w or h back swaps the crops: the one on disk goes to the undo's
  stash and the action's copy comes back (so a redo swaps them again), and faces whose vectors came back
  are taken again by the face index (FaceIndex.reload_faces).
- not_undoable(): a with-block whose writes stay out of the journal (the bulk purge).
- prune(conn): drops actions older than KEEP_DAYS with their stash (worker housekeeping).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from app import storage
from app.db import ActionContext, bump_faces_version, bump_search_version, current_action, tx
from app.util import now_iso

log = logging.getLogger(__name__)

KEEP_DAYS = 30
ACTION_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Writes that are not owner content, or cannot come back (a purge deletes the original files).
NOT_UNDOABLE = re.compile(
    r"^/api/(auth|uploads|system|actions|trash|search|s)(/|$)"
    r"|^/api/items/[^/]+/reprocess$|^/api/recluster$|^/api/map/refresh$|^/api/documents/[^/]+/reparse$"
)
JOURNALED: tuple[str, ...] = (
    "items", "faces", "bodies", "animals", "persons", "item_tags", "guess_rejections", "merge_dismissals",
    "learning_events", "not_faces", "albums", "album_items", "shares", "share_items", "tags", "item_user_tags",
    "tag_rejections", "documents", "related_dismissals", "trait_hides",
)
CROPS: dict[str, Callable[[int], Path]] = {
    "faces": storage.face_crop_path, "bodies": storage.body_crop_path, "animals": storage.animal_crop_path,
}
STASH_DIR = "undo"
# Bookkeeping columns: they never block an undo and are not put back; the counts are counted again.
NOISE = frozenset({"updated_at", "face_count", "body_count", "animal_count"})
BOX_COLUMNS = frozenset({"x", "y", "w", "h"})  # a box's place: its crop follows it (keep_copy)
FACE_VECTOR_COLUMNS = frozenset({"embedding", "det_score", "size_px", "clarity", "sharpness", "age_groups"})
COUNTS = {"face_count": "faces", "body_count": "bodies", "animal_count": "animals"}


def _person_model_path(person_id: int) -> Path:
    from app.learning import person_model, store

    return store.path(person_model.name_of(person_id))


# Files that go and come back with their row: a box's crop, and the recognition trained on a person (section 10).
ROW_FILES: dict[str, Callable[[int], Path]] = {**CROPS, "persons": _person_model_path}


class UndoError(Exception):
    def __init__(self, status: int, detail: str, blocking: list[str] | None = None):
        super().__init__(detail)
        self.status, self.detail, self.blocking = status, detail, blocking or []


def new_id() -> str:
    return uuid.uuid4().hex


def undoable(method: str, path: str) -> bool:
    return method in WRITE_METHODS and path.startswith("/api/") and not NOT_UNDOABLE.search(path)


@contextmanager
def not_undoable() -> Iterator[None]:
    """Writes that cannot come back (a purge deletes the original files) run outside the journal."""
    token = current_action.set(None)
    try:
        yield
    finally:
        current_action.reset(token)


# ------------------------------------------------------------------ journal
def _columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, bool, int]]:
    """(name, is_blob, pk position) per column; empty for a missing table."""
    return [(r[1], (r[2] or "").upper() == "BLOB", int(r[5])) for r in conn.execute(f"PRAGMA table_info({table})")]


# SQLite builds before 3.48 allow 127 arguments per function call, and a database whose trigger breaks
# that cannot be opened by them at all: the row JSON is built 30 columns per call.
JSON_PAIRS = 30


def _json(cols: list[tuple[str, bool, int]], alias: str, op: str) -> str:
    """The row JSON a trigger logs. A delete keeps the whole row (an undo inserts it again). An update keeps
    the key columns and item_id, and every other column only where it changed (unchanged ones are null on
    both sides, which the undo reads as unchanged). An insert keeps every column but the BLOBs (its undo
    deletes by key; the undo's own delete keeps the whole row for a redo)."""
    pairs = []
    for name, blob, pk in cols:
        value = f'hex({alias}."{name}")' if blob else f'{alias}."{name}"'
        if blob:
            value = f'CASE WHEN {alias}."{name}" IS NULL THEN NULL ELSE {value} END'
        if op == "update" and not pk and name != "item_id":
            value = f'CASE WHEN OLD."{name}" IS NOT NEW."{name}" THEN {value} END'
        elif op == "insert" and blob:
            value = "NULL"
        pairs.append((name, value))
    head, rest = pairs[:JSON_PAIRS], pairs[JSON_PAIRS:]
    expr = "json_object(" + ", ".join(f"'{n}', {v}" for n, v in head) + ")"
    for start in range(0, len(rest), JSON_PAIRS):
        chunk = rest[start:start + JSON_PAIRS]
        expr = "json_insert(" + expr + ", " + ", ".join(f"'$.\"{n}\"', {v}" for n, v in chunk) + ")"
    return expr


def install_triggers(conn: sqlite3.Connection) -> None:
    """(Re)create the journal triggers of every JOURNALED table from its current columns."""
    for table in JOURNALED:
        cols = _columns(conn, table)
        if not cols:
            continue
        for op, event in (("insert", "INSERT"), ("update", "UPDATE"), ("delete", "DELETE")):
            name = f"undo_{table}_{op}"
            old = _json(cols, "OLD", op) if op != "insert" else "NULL"
            new = _json(cols, "NEW", op) if op != "delete" else "NULL"
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            conn.execute(
                f"CREATE TRIGGER {name} AFTER {event} ON {table} WHEN EXISTS (SELECT 1 FROM undo_context) BEGIN "
                f"INSERT INTO undo_log(action_id, tbl, op, old, new) VALUES "
                f"((SELECT action_id FROM undo_context), '{table}', '{op}', {old}, {new}); END"
            )


def item_ids(conn: sqlite3.Connection, action_id: str) -> set[int]:
    out: set[int] = set()
    for r in conn.execute("SELECT tbl, old, new FROM undo_log WHERE action_id = ?", (action_id,)):
        for raw in (r["old"], r["new"]):
            row = json.loads(raw) if raw else None
            if not row:
                continue
            key = "id" if r["tbl"] == "items" else "item_id"
            if isinstance(row.get(key), int):
                out.add(int(row[key]))
    return out


# ------------------------------------------------------------------ files
def stash_dir(action_id: str) -> Path:
    return storage.root() / STASH_DIR / action_id


def _stash_path(action_id: str, path: Path) -> Path:
    return stash_dir(action_id) / Path(path).resolve().relative_to(storage.root().resolve())


def discard(path: Path | None) -> None:
    """Remove a file the current action no longer needs; inside an action it is kept for undo."""
    if path is None:
        return
    act = current_action.get()
    if act is None:
        storage.safe_unlink(path)
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        dest = _stash_path(act.id, p)
    except ValueError:  # outside the data folder: nothing to keep it beside
        storage.safe_unlink(p)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(p, dest)


def keep_copy(path: Path | None) -> None:
    """Copy a file the current action is about to replace into its stash (an undo puts it back). Outside an
    action, or when the action already kept one, nothing happens."""
    act = current_action.get()
    if path is None or act is None:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        dest = _stash_path(act.id, p)
    except ValueError:  # outside the data folder
        return
    if dest.exists():
        return  # the action's first copy is the one to put back
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)


def _move(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dest)


# ------------------------------------------------------------------ undo
def _pk(cols: list[tuple[str, bool, int]]) -> list[str]:
    return [name for name, _b, pk in sorted(cols, key=lambda c: c[2]) if pk]


def _sql_value(value: Any, blob: bool) -> Any:
    if blob and isinstance(value, str):
        return bytes.fromhex(value)
    return value


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        except (TypeError, ValueError):
            return False
    return a == b


def _current(conn: sqlite3.Connection, table: str, pk: list[str], row: dict[str, Any],
             cols: list[tuple[str, bool, int]]) -> dict[str, Any] | None:
    where = " AND ".join(f'"{k}" IS ?' for k in pk)
    blobs = {n for n, b, _p in cols if b}
    found = conn.execute(f"SELECT * FROM {table} WHERE {where}",
                         [_sql_value(row.get(k), k in blobs) for k in pk]).fetchone()
    if found is None:
        return None
    return {k: (found[k].hex().upper() if k in blobs and isinstance(found[k], bytes) else found[k]) for k in found.keys()}


def _changed(old: dict[str, Any], new: dict[str, Any], noise: bool = False) -> list[str]:
    return [k for k in new if k in old and not _same(old[k], new[k]) and (noise or k not in NOISE)]


def _later_blockers(conn: sqlite3.Connection, action_id: str, entries: list[sqlite3.Row]) -> list[str]:
    """Later owner actions, still in effect, that changed rows and columns this action changed."""
    if not entries:
        return []
    top = max(int(e["seq"]) for e in entries)
    touched: dict[tuple[str, str], set[str]] = {}
    for e in entries:
        old = json.loads(e["old"]) if e["old"] else {}
        new = json.loads(e["new"]) if e["new"] else {}
        keyrow = new or old
        cols = set(_changed(old, new)) if e["op"] == "update" else {"*"}
        touched.setdefault((e["tbl"], json.dumps(_keyof(conn, e["tbl"], keyrow))), set()).update(cols)
        if e["op"] == "update" and old:
            touched.setdefault((e["tbl"], json.dumps(_keyof(conn, e["tbl"], old))), set()).update(cols)
    blockers: list[str] = []
    for e in conn.execute(
        "SELECT l.action_id, l.tbl, l.op, l.old, l.new FROM undo_log l JOIN undo_actions a ON a.id = l.action_id "
        "WHERE l.seq > ? AND l.action_id != ? AND a.undone_by IS NULL AND a.undo_of IS NULL ORDER BY l.seq",
        (top, action_id),
    ):
        old = json.loads(e["old"]) if e["old"] else {}
        new = json.loads(e["new"]) if e["new"] else {}
        cols = set(_changed(old, new)) if e["op"] == "update" else {"*"}
        for keyrow in (old, new):
            if not keyrow:
                continue
            mine = touched.get((e["tbl"], json.dumps(_keyof(conn, e["tbl"], keyrow))))
            if mine is not None and ("*" in mine or "*" in cols or mine & cols):
                if e["action_id"] not in blockers:
                    blockers.append(e["action_id"])
                break
    return blockers


_pk_cache: dict[str, list[str]] = {}


def _keyof(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> list[Any]:
    pk = _pk_cache.get(table)
    if pk is None:
        pk = _pk_cache[table] = _pk(_columns(conn, table))
    return [row.get(k) for k in pk]


def undo(conn: sqlite3.Connection, action_id: str) -> dict[str, Any]:
    act = conn.execute("SELECT * FROM undo_actions WHERE id = ?", (action_id,)).fetchone()
    if act is None:
        raise UndoError(404, "not_found")
    if act["undone_by"]:
        raise UndoError(409, "already_undone")
    created = datetime.fromisoformat(str(act["created_at"]).replace("Z", "+00:00"))
    if created < datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS):
        raise UndoError(410, "expired")
    entries = conn.execute("SELECT * FROM undo_log WHERE action_id = ? ORDER BY seq DESC", (action_id,)).fetchall()
    blockers = _later_blockers(conn, action_id, entries)
    if blockers:
        raise UndoError(409, "later_changes", blockers)
    undo_id = new_id()
    token = current_action.set(ActionContext(undo_id, "UNDO", f"/api/actions/{action_id}/undo", undo_of=action_id))
    restored: list[tuple[str, int]] = []  # rows put back: their crops come back from this action's stash
    removed: list[tuple[str, int]] = []  # rows taken away again: their crops go to the undo's stash
    swapped: list[tuple[str, int]] = []  # boxes put back where they were: their crops swap with the stash
    vectors: list[int] = []  # faces whose vectors or weights came back: the face index takes them again
    skipped = 0
    items: set[int] = set()
    tables: set[str] = set()  # the tables whose rows were replayed (app.analysis.traits.after_undo)
    try:
        with tx(conn):
            conn.execute("PRAGMA defer_foreign_keys = ON")
            for e in entries:
                table = e["tbl"]
                cols = _columns(conn, table)
                if not cols:
                    skipped += 1
                    continue
                pk, blobs = _pk(cols), {n for n, b, _p in cols if b}
                tables.add(table)
                old = json.loads(e["old"]) if e["old"] else None
                new = json.loads(e["new"]) if e["new"] else None
                for row in (old, new):
                    if row:
                        key = "id" if table == "items" else "item_id"
                        if isinstance(row.get(key), int):
                            items.add(int(row[key]))
                if e["op"] == "insert" and new is not None:
                    where = " AND ".join(f'"{k}" IS ?' for k in pk)
                    n = conn.execute(f"DELETE FROM {table} WHERE {where}",
                                     [_sql_value(new.get(k), k in blobs) for k in pk]).rowcount
                    if n and table in ROW_FILES:
                        removed.append((table, int(new["id"])))
                    skipped += 0 if n else 1
                elif e["op"] == "delete" and old is not None:
                    if _current(conn, table, pk, old, cols) is not None:
                        skipped += 1
                        continue
                    names = [n for n, _b, _p in cols if n in old]
                    conn.execute(
                        f"INSERT INTO {table} ({', '.join(chr(34) + n + chr(34) for n in names)}) "
                        f"VALUES ({', '.join('?' * len(names))})",
                        [_sql_value(old[n], n in blobs) for n in names])
                    if table in ROW_FILES:
                        restored.append((table, int(old["id"])))
                elif e["op"] == "update" and old is not None and new is not None:
                    now = _current(conn, table, pk, new, cols)
                    if now is None:
                        skipped += 1
                        continue
                    wanted = _changed(old, new)
                    back = [c for c in wanted if _same(now.get(c), new[c])]
                    skipped += 1 if len(back) < len(wanted) else 0
                    if not back:
                        continue
                    where = " AND ".join(f'"{k}" IS ?' for k in pk)
                    conn.execute(
                        f"UPDATE {table} SET {', '.join(chr(34) + c + chr(34) + ' = ?' for c in back)} WHERE {where}",
                        [_sql_value(old[c], c in blobs) for c in back] + [_sql_value(new.get(k), k in blobs) for k in pk])
                    if table in CROPS and BOX_COLUMNS & set(back) and isinstance(new.get("id"), int):
                        swapped.append((table, int(new["id"])))
                    if table == "faces" and FACE_VECTOR_COLUMNS & set(back) and isinstance(new.get("id"), int):
                        vectors.append(int(new["id"]))
            if items:
                ids = sorted(items)
                for start in range(0, len(ids), 500):
                    chunk = ids[start:start + 500]
                    marks = ",".join("?" * len(chunk))
                    sets = ", ".join(f"{col} = (SELECT COUNT(*) FROM {tbl} WHERE item_id = items.id)" for col, tbl in COUNTS.items())
                    conn.execute(f"UPDATE items SET {sets} WHERE id IN ({marks})", chunk)
            conn.execute("UPDATE undo_actions SET undone_by = ? WHERE id = ?", (undo_id, action_id))
            bump_faces_version(conn)
            bump_search_version(conn)
    except sqlite3.IntegrityError as err:
        log.warning("undo of %s refused: %s", action_id, err)
        raise UndoError(409, "conflict") from None
    finally:
        current_action.reset(token)
    for table, row_id in restored:
        _move(_stash_path(action_id, ROW_FILES[table](row_id)), ROW_FILES[table](row_id))
    for table, row_id in removed:
        path = ROW_FILES[table](row_id)
        if path.exists():
            _move(path, _stash_path(undo_id, path))
    people = [row_id for table, row_id in [*restored, *removed] if table == "persons"]
    if people:
        from app.learning import person_model

        for row_id in people:  # a model file came back or went: what is remembered of it is stale
            person_model.file_changed(row_id)
    for table, row_id in swapped:
        path = CROPS[table](row_id)
        before = _stash_path(action_id, path)
        if before.exists():
            if path.exists():
                _move(path, _stash_path(undo_id, path))
            _move(before, path)
    if vectors:
        from app.analysis.index import get_index

        get_index().reload_faces(conn, vectors)
    try:
        from app.analysis import traits

        traits.after_undo(conn, tables, items)
    except Exception:  # noqa: BLE001 - the traits follow-up never fails an undo that already happened
        log.exception("undo: the body traits follow-up of %s failed", action_id)
    log.info("undid action %s (%d changes, %d left as they are now) as %s", action_id, len(entries), skipped, undo_id)
    return {"undo_id": undo_id, "changes": len(entries), "partial": skipped > 0, "items": sorted(items)}


# ------------------------------------------------------------------ upkeep
def prune(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Forget actions older than KEEP_DAYS, with their journal and stash. Returns how many."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=KEEP_DAYS)
    stamp = cutoff.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    old = [r[0] for r in conn.execute("SELECT id FROM undo_actions WHERE created_at < ?", (stamp,))]
    if not old:
        return 0
    for start in range(0, len(old), 500):
        chunk = old[start:start + 500]
        marks = ",".join("?" * len(chunk))
        with tx(conn):
            conn.execute(f"DELETE FROM undo_log WHERE action_id IN ({marks})", chunk)
            conn.execute(f"DELETE FROM undo_actions WHERE id IN ({marks})", chunk)
    for action_id in old:
        shutil.rmtree(stash_dir(action_id), ignore_errors=True)
    return len(old)


# ------------------------------------------------------------------ middleware
class ActionMiddleware:
    """Pure ASGI: names each undoable API write (an X-Action-Id header the client sent, else a new id),
    holds it in current_action for the request, and answers with X-Action-Id when it changed something."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not undoable(scope.get("method", ""), scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        given = ""
        for key, value in scope.get("headers") or []:
            if key == b"x-action-id":
                given = value.decode("latin-1").strip()
                break
        act = ActionContext(given if ACTION_ID_RE.fullmatch(given) else new_id(), scope["method"], scope["path"])

        async def send_with_id(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start" and act.logged:
                message["headers"] = [*message.get("headers", []), (b"x-action-id", act.id.encode("latin-1"))]
            await send(message)

        token = current_action.set(act)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            current_action.reset(token)
