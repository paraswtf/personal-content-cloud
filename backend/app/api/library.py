"""Library API: items, trash, mis-click review, duplicates, stats, client config.

CONTRACT (docs/ARCHITECTURE.md section 13). Router without a prefix, mounted at /api. Every route
requires a logged-in session. Listing visibility goes through api.common.visible_sql: NSFW items only
with vault=1 and an unlocked session (403 vault_locked otherwise), and then only NSFW items.

GET    /config                     ConfigOut
GET    /items                      ItemPage; query: category, kind, person (repeated, AND), from, to
                                   (local dates, inclusive), favorite, misclicks=exclude|include|only,
                                   dups=collapse|all, vault, sort=taken|added, cursor, limit 1..200.
                                   Keyset cursors: taken = COALESCE(taken_at, created_at) DESC, id DESC;
                                   added = created_at DESC, id DESC. Bad cursor: 400 "Invalid cursor".
GET    /items/{uid}                ItemDetail (trashed items too; NSFW needs the vault)
PATCH  /items/{uid}                ItemUpdate -> ItemDetail. User edits set *_source='user'. nsfw moves
                                   the category to 'nsfw' and back (the best non-nsfw auto score, else
                                   'misc'); misclick=false clears the reasons.
DELETE /items/{uid}                to the trash -> Ok
POST   /items/{uid}/restore        -> ItemDetail
POST   /items/bulk                 BulkIn -> BulkOut (keep = not a mis-click; purge only for trashed)
POST   /items/{uid}/reprocess      ReprocessIn -> Ok
GET    /items/{uid}/pages          [PageOut];  GET /items/{uid}/text -> ItemTextOut
GET    /trash                      ItemPage (deleted_at DESC);  POST /trash/empty -> BulkOut;
DELETE /trash/{uid}                purge one trashed item -> Ok
GET    /review/misclicks           ItemPage of flagged items (with their reasons)
GET    /duplicates                 DupGroupPage;  POST /duplicates/{uid}/best -> Ok
GET    /stats                      Stats (recent = the 12 latest cards)
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app import jobs, storage
from app.api.common import (
    ITEM_CARD_COLUMNS,
    animal_out,
    body_out,
    box_person_ids,
    document_out,
    face_out,
    item_card,
    item_urls,
    people_filter_sql,
    tags_filter_sql,
    live_video_out,
    page_sql,
    person_refs,
    url_version,
    valid_cursor_scalar,
    visible_sql,
)
from app.api.deps import get_db, require_user, want_vault
from app.auth.sessions import SessionInfo
from app.config import CATEGORIES, ITEM_KINDS, settings
from app.db import bump_search_version, tx
from app.schemas import (
    BulkIn,
    BulkOut,
    ConfigOut,
    DiskStats,
    DupGroup,
    DupGroupPage,
    GpsOut,
    ItemDetail,
    ItemPage,
    ItemTextOut,
    ItemUpdate,
    LabelOut,
    Ok,
    PageOut,
    ReprocessIn,
    Stats,
)
from app.urls import page_url
from app.util import decode_cursor, encode_cursor, json_loads, name_key, now_iso, safe_filename
from app.util import local_day_start_utc as _local_day_start_utc

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_user)])

SORT_KEYS = {"taken": "COALESCE(i.taken_at, i.created_at)", "added": "i.created_at"}
DETAIL_COLUMNS = (
    f"{ITEM_CARD_COLUMNS}, i.mime, i.size_bytes, i.fps, i.has_audio, i.camera, i.gps_lat, i.gps_lon, i.gps_alt, "
    "i.exif_datetime, i.category_source, i.category_scores, i.nsfw_score, i.nsfw_source, i.quality_scores, "
    "i.text_status, i.analysis_status, i.deleted_at, i.updated_at, i.misclick_source"
)
RECENT_LIMIT = 12


# ------------------------------------------------------------------ helpers
def _bad_cursor() -> HTTPException:
    return HTTPException(status_code=400, detail="Invalid cursor")


def _keyset(cursor: str | None, n: int = 2) -> list[Any] | None:
    if cursor is None or cursor == "":
        return None
    values = decode_cursor(cursor)
    if values is None or len(values) != n or not all(valid_cursor_scalar(v) for v in values):
        raise _bad_cursor()
    if not isinstance(values[-1], int) or isinstance(values[-1], bool):
        raise _bad_cursor()
    return values


def local_day_start_utc(value: str, *, next_day: bool = False) -> str:
    """'YYYY-MM-DD' (a local calendar date in settings.local_timezone) -> the UTC instant its day starts
    (or the next day starts), as util.local_day_start_utc. 422 when malformed."""
    try:
        return _local_day_start_utc(value, settings.local_timezone or "UTC", next_day=next_day)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date: {value!r} (use YYYY-MM-DD)")


def _page(conn: sqlite3.Connection, where: list[str], params: list[Any], sort_expr: str, cursor: str | None,
          limit: int) -> ItemPage:
    """Keyset page over items (alias i), newest first by sort_expr then id. The total is counted on
    the first page only (no cursor); later pages carry total 0 (clients read it from the first page)."""
    after = _keyset(cursor)
    total = 0
    if after is None:
        cond = " AND ".join(where) if where else "1"
        total = conn.execute(f"SELECT COUNT(*) FROM items i WHERE {cond}", params).fetchone()[0]
    sql, extra = page_sql(where, sort_expr, after)
    rows = conn.execute(sql, [*params, *extra, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = encode_cursor([rows[-1]["sort_key"], int(rows[-1]["id"])]) if more and rows else None
    return ItemPage(items=[item_card(r) for r in rows], next_cursor=next_cursor, total=int(total))


def _item_row(conn: sqlite3.Connection, uid: str) -> sqlite3.Row:
    row = conn.execute(f"SELECT {DETAIL_COLUMNS} FROM items i WHERE i.uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return row


def _guard(row: sqlite3.Row, session: SessionInfo) -> None:
    if row["nsfw"] and not session.vault_unlocked:
        raise HTTPException(status_code=403, detail="vault_locked")


def _wallet_configured() -> bool:
    try:
        from app.wallet.pkpass import configured
    except ImportError:
        return False
    try:
        return bool(configured()[0])
    except Exception:
        log.exception("wallet configuration check failed")
        return False


def _float_scores(raw: str | None) -> dict[str, float] | None:
    data = json_loads(raw)
    if not isinstance(data, dict):
        return None
    out = {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return out


def _people_ids(conn: sqlite3.Connection, item_id: int) -> list[int]:
    """Ids of the people and pets in an item (faces, bodies, animal boxes, tags), in display order."""
    rows = conn.execute(
        "SELECT p.id FROM persons p WHERE p.id IN ("
        "SELECT person_id FROM faces WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM bodies WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM animals WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM item_tags WHERE item_id = ?) "
        "ORDER BY p.kind = 'pet', p.name IS NULL, p.name_key, p.id",
        (item_id, item_id, item_id, item_id),
    ).fetchall()
    return [int(r["id"]) for r in rows]


def item_detail(conn: sqlite3.Connection, row: sqlite3.Row) -> ItemDetail:
    """ItemDetail of a row selected with DETAIL_COLUMNS."""
    item_id = int(row["id"])
    card = item_card(row)
    faces = conn.execute(
        "SELECT * FROM faces WHERE item_id = ? ORDER BY frame_time_s IS NOT NULL, frame_time_s, x, id", (item_id,)
    ).fetchall()
    bodies = conn.execute(
        "SELECT * FROM bodies WHERE item_id = ? ORDER BY frame_time_s IS NOT NULL, frame_time_s, x, id", (item_id,)
    ).fetchall()
    animals = conn.execute(
        "SELECT * FROM animals WHERE item_id = ? ORDER BY frame_time_s IS NOT NULL, frame_time_s, x, id", (item_id,)
    ).fetchall()
    tags = conn.execute(
        "SELECT p.id FROM item_tags t JOIN persons p ON p.id = t.person_id "
        "WHERE t.item_id = ? ORDER BY p.name IS NULL, p.name_key, p.id",
        (item_id,),
    ).fetchall()
    people_ids = _people_ids(conn, item_id)
    # Every person shown on the item (boxes, guesses, tags) in one lookup.
    refs = person_refs(conn, box_person_ids(faces) | box_person_ids(bodies) | {int(t["id"]) for t in tags}
                       | {int(a["person_id"]) for a in animals if a["person_id"] is not None} | set(people_ids))
    labels = conn.execute("SELECT label, score FROM item_labels WHERE item_id = ? ORDER BY score DESC, label",
                          (item_id,)).fetchall()
    doc = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()
    gps = None
    if row["gps_lat"] is not None and row["gps_lon"] is not None:
        gps = GpsOut(lat=row["gps_lat"], lon=row["gps_lon"], alt=row["gps_alt"])
    data = card.model_dump()
    data.update(
        mime=row["mime"], size_bytes=int(row["size_bytes"]), bitrate=row["bitrate"], fps=row["fps"],
        video_codec=row["video_codec"], has_audio=None if row["has_audio"] is None else bool(row["has_audio"]),
        camera=row["camera"], gps=gps, exif_datetime=row["exif_datetime"], category_source=row["category_source"],
        category_scores=_float_scores(row["category_scores"]), nsfw_score=row["nsfw_score"],
        quality_scores=json_loads(row["quality_scores"]) if row["quality_scores"] else None,
        labels=[LabelOut(label=r["label"], score=float(r["score"])) for r in labels],
        faces=[face_out(conn, f, refs) for f in faces], bodies=[body_out(conn, b, refs) for b in bodies],
        animals=[animal_out(conn, a, refs) for a in animals],
        tags=[refs[int(t["id"])] for t in tags if int(t["id"]) in refs],
        people=[refs[p] for p in people_ids if p in refs],
        document=document_out(doc, row, wallet_configured=_wallet_configured()) if doc is not None else None,
        text_status=row["text_status"], analysis_status=row["analysis_status"], preview_status=row["preview_status"],
        deleted_at=row["deleted_at"], updated_at=row["updated_at"], urls=item_urls(row, full=True),
    )
    from app.api.tags import item_tags_out

    data["user_tags"] = item_tags_out(conn, int(row["id"]))
    live_video = live_video_out(conn, row)
    data["live_video"] = live_video
    if live_video is not None:
        data["urls"].live = live_video.url
    return ItemDetail(**data)


def _refresh_fts(conn: sqlite3.Connection, item_id: int) -> None:
    try:
        from app.documents.extract import refresh_fts
    except ImportError:
        return
    try:
        refresh_fts(conn, item_id)
    except Exception:
        log.exception("could not refresh the full-text row of item %s", item_id)


def _auto_category(row: sqlite3.Row) -> str:
    """The category an item had before it was marked NSFW: its best non-nsfw auto score."""
    scores = _float_scores(row["category_scores"]) or {}
    candidates = {k: v for k, v in scores.items() if k in CATEGORIES and k != "nsfw"}
    if candidates:
        return max(candidates.items(), key=lambda kv: kv[1])[0]
    return "documents" if row["kind"] in ("pdf", "document") else "misc"


def _apply_update(conn: sqlite3.Connection, row: sqlite3.Row, body: ItemUpdate) -> bool:
    """Apply a user edit inside the caller's transaction. Returns True when the file name changed
    (the caller refreshes the full-text row after the commit)."""
    sets: dict[str, Any] = {}
    nsfw = body.nsfw
    category = body.category
    if category == "nsfw":
        nsfw, category = True, None
    if nsfw is True and not row["nsfw"]:
        sets.update(nsfw=1, nsfw_source="user", category="nsfw")
    elif nsfw is False and row["nsfw"]:
        sets.update(nsfw=0, nsfw_source="user")
        if category is None:
            sets["category"] = _auto_category(row)
    elif nsfw is not None:
        sets["nsfw_source"] = "user"  # confirms the current state; analysis must not change it
    if category is not None:
        sets.update(category=category, category_source="user")
        if row["nsfw"] and nsfw is not True:
            sets.update(nsfw=0, nsfw_source="user")
    if body.misclick is not None:
        sets.update(misclick=int(body.misclick), misclick_source="user")
        if not body.misclick:
            sets["misclick_reasons"] = None
    if body.favorite is not None:
        sets["favorite"] = int(body.favorite)
    renamed = False
    if body.filename is not None:
        name = safe_filename(body.filename, "")
        if not name:
            raise HTTPException(status_code=422, detail="The file name is empty")
        if name != row["original_filename"]:
            sets["original_filename"] = name
            renamed = True
    if not sets:
        return False
    sets["updated_at"] = now_iso()
    conn.execute(f"UPDATE items SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?",
                 [*sets.values(), int(row["id"])])
    if "nsfw" in sets:
        bump_search_version(conn)
        from app.media import live

        live.sync_companion(conn, int(row["id"]))
    _record_corrections(conn, row, sets)
    return renamed


def _record_corrections(conn: sqlite3.Connection, row: sqlite3.Row, sets: dict[str, Any]) -> None:
    """Learning events for the choices in this update (section 18): category, vault, mis-click."""
    try:
        from app.learning import events

        item_id = int(row["id"])
        if sets.get("category_source") == "user" and sets.get("category") != row["category"]:
            events.record(conn, "category", None, {"from": row["category"], "to": sets["category"],
                                                   "scores": json_loads(row["category_scores"], None)}, item_id=item_id)
        if "nsfw" in sets and int(sets["nsfw"]) != int(row["nsfw"]):
            events.record(conn, "vault", None, {"from": int(row["nsfw"]), "to": int(sets["nsfw"]),
                                                "nsfw_score": row["nsfw_score"]}, item_id=item_id)
        if "misclick" in sets and int(sets["misclick"]) != int(row["misclick"]):
            events.record(conn, "misclick", None, {"from": int(row["misclick"]), "to": int(sets["misclick"]),
                                                   "scores": json_loads(row["quality_scores"], None),
                                                   "reasons": json_loads(row["misclick_reasons"], None)}, item_id=item_id)
    except Exception:  # noqa: BLE001 - learning never blocks an edit
        log.exception("learning: could not record a correction")


# ------------------------------------------------------------------ config
@router.get("/config", response_model=ConfigOut)
def get_config() -> ConfigOut:
    from app.media.probe import ACCEPT_ATTR

    return ConfigOut(
        site_title=settings.site_title, max_upload_bytes=settings.max_upload_bytes,
        simple_upload_max_bytes=settings.simple_upload_max_bytes, chunk_size=settings.chunk_size,
        accept=ACCEPT_ATTR, categories=list(CATEGORIES), vault_unlock_min=settings.vault_unlock_min,
        trash_days=settings.trash_days,
    )


# ------------------------------------------------------------------ items
@router.get("/items", response_model=ItemPage)
def list_items(
    category: Literal[CATEGORIES] | None = None,  # type: ignore[valid-type]
    kind: Literal[ITEM_KINDS] | None = None,  # type: ignore[valid-type]
    person: list[int] = Query(default=[]),
    tag: list[str] = Query(default=[]),
    live: bool | None = None,
    date_from: str | None = Query(default=None, alias="from", max_length=10),
    date_to: str | None = Query(default=None, alias="to", max_length=10),
    favorite: bool | None = None,
    misclicks: Literal["exclude", "include", "only"] = "exclude",
    dups: Literal["collapse", "all"] = "collapse",
    sort: Literal["taken", "added"] = "taken",
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    vault: bool = Depends(want_vault),
    conn: sqlite3.Connection = Depends(get_db),
) -> ItemPage:
    where = [visible_sql(vault=vault, misclicks=misclicks, dups=dups)]
    params: list[Any] = []
    if category is not None:
        where.append("i.category = ?")
        params.append(category)
    if kind is not None:
        where.append("i.kind = ?")
        params.append(kind)
    if person:
        sql, p = people_filter_sql(person)
        where.append(sql)
        params.extend(p)
    if tag:
        sql, p = tags_filter_sql(tag[:10])
        where.append(sql)
        params.extend(p)
    if live is not None:
        where.append(f"i.live = {1 if live else 0}")
    sort_expr = SORT_KEYS[sort]
    date_expr = "COALESCE(i.taken_at, i.created_at)"
    if date_from:
        where.append(f"{date_expr} >= ?")
        params.append(local_day_start_utc(date_from))
    if date_to:
        where.append(f"{date_expr} < ?")
        params.append(local_day_start_utc(date_to, next_day=True))
    if favorite is not None:
        where.append("i.favorite = ?")
        params.append(1 if favorite else 0)
    return _page(conn, where, params, sort_expr, cursor, limit)


@router.get("/items/{uid}", response_model=ItemDetail)
def get_item(uid: str, session: SessionInfo = Depends(require_user),
             conn: sqlite3.Connection = Depends(get_db)) -> ItemDetail:
    row = _item_row(conn, uid)
    _guard(row, session)
    return item_detail(conn, row)


@router.patch("/items/{uid}", response_model=ItemDetail)
def update_item(uid: str, body: ItemUpdate, session: SessionInfo = Depends(require_user),
                conn: sqlite3.Connection = Depends(get_db)) -> ItemDetail:
    row = _item_row(conn, uid)
    _guard(row, session)
    with tx(conn):
        row = _item_row(conn, uid)
        renamed = _apply_update(conn, row, body)
    if renamed:
        _refresh_fts(conn, int(row["id"]))
    return item_detail(conn, _item_row(conn, uid))


@router.delete("/items/{uid}", response_model=Ok)
def delete_item(uid: str, session: SessionInfo = Depends(require_user),
                conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    from app import ingest

    row = _item_row(conn, uid)
    _guard(row, session)
    ingest.soft_delete(conn, int(row["id"]))
    return Ok()


@router.post("/items/{uid}/restore", response_model=ItemDetail)
def restore_item(uid: str, session: SessionInfo = Depends(require_user),
                 conn: sqlite3.Connection = Depends(get_db)) -> ItemDetail:
    from app import ingest

    row = _item_row(conn, uid)
    _guard(row, session)
    ingest.restore(conn, int(row["id"]))
    return item_detail(conn, _item_row(conn, uid))


def _bulk_value_bool(value: str | None) -> bool:
    v = (value or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise HTTPException(status_code=422, detail="value must be true or false")


@router.post("/items/bulk", response_model=BulkOut)
def bulk(body: BulkIn, session: SessionInfo = Depends(require_user),
         conn: sqlite3.Connection = Depends(get_db)) -> BulkOut:
    from app import ingest

    action = body.action
    update: ItemUpdate | None = None
    if action == "keep":
        update = ItemUpdate(misclick=False)
    elif action == "category":
        if body.value not in CATEGORIES:
            raise HTTPException(status_code=422, detail="value must be a category")
        update = ItemUpdate(category=body.value)
    elif action == "nsfw":
        update = ItemUpdate(nsfw=_bulk_value_bool(body.value))
    elif action in ("favorite", "unfavorite"):
        update = ItemUpdate(favorite=action == "favorite")
    elif action in ("tag", "untag"):
        from app.api.tags import clean_tag_name

        clean_tag_name(body.value)
    uids = list(dict.fromkeys(body.uids))
    rows: list[sqlite3.Row] = []
    for i in range(0, len(uids), 500):
        chunk = uids[i:i + 500]
        rows.extend(conn.execute(
            f"SELECT {DETAIL_COLUMNS} FROM items i WHERE i.uid IN ({','.join('?' * len(chunk))})", chunk
        ).fetchall())
    # NSFW items are only touched while the vault is open.
    rows = [r for r in rows if not r["nsfw"] or session.vault_unlocked]
    updated = 0
    renamed: list[int] = []
    if action == "delete":
        for r in rows:
            if r["deleted_at"] is None:
                ingest.soft_delete(conn, int(r["id"]))
                updated += 1
    elif action == "restore":
        for r in rows:
            if r["deleted_at"] is not None:
                ingest.restore(conn, int(r["id"]))
                updated += 1
    elif action == "purge":
        for r in rows:
            if r["deleted_at"] is not None:
                ingest.purge(conn, int(r["id"]))
                updated += 1
    elif action in ("tag", "untag"):
        from app.api import tags as tags_api

        changed: list[int] = []
        with tx(conn):
            if action == "tag":
                tag_row = tags_api.ensure_tag(conn, tags_api.clean_tag_name(body.value))
            else:
                tag_row = conn.execute("SELECT * FROM tags WHERE name_key = ?", (name_key(body.value),)).fetchone()
            for r in rows:
                if r["deleted_at"] is not None or tag_row is None:
                    continue
                if action == "tag":
                    done = tags_api.add_to_item(conn, int(r["id"]), tag_row)
                else:
                    done = tags_api.remove_from_item(conn, int(r["id"]), tag_row)
                if done:
                    changed.append(int(r["id"]))
                    updated += 1
        tags_api._refresh_fts(conn, changed)
    else:
        assert update is not None
        with tx(conn):
            for r in rows:
                fresh = conn.execute(f"SELECT {DETAIL_COLUMNS} FROM items i WHERE i.id = ?", (r["id"],)).fetchone()
                if fresh is None:
                    continue
                if _apply_update(conn, fresh, update):
                    renamed.append(int(r["id"]))
                updated += 1
    for item_id in renamed:
        _refresh_fts(conn, item_id)
    return BulkOut(updated=updated)


@router.post("/items/{uid}/reprocess", response_model=Ok)
def reprocess(uid: str, body: ReprocessIn | None = None, session: SessionInfo = Depends(require_user),
              conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    from app import ingest

    row = _item_row(conn, uid)
    _guard(row, session)
    stages = list(body.stages) if body is not None else ["analyze", "text"]
    ingest.enqueue_reprocess(conn, int(row["id"]), stages)
    return Ok()


@router.get("/items/{uid}/pages", response_model=list[PageOut])
def item_pages(uid: str, session: SessionInfo = Depends(require_user),
               conn: sqlite3.Connection = Depends(get_db)) -> list[PageOut]:
    row = _item_row(conn, uid)
    _guard(row, session)
    v = url_version(row["version"], row["nsfw"])
    stored = {int(r["page_no"]): r for r in conn.execute(
        "SELECT page_no, width, height, text, text_source FROM pages WHERE item_id = ? ORDER BY page_no",
        (row["id"],))}
    count = max([int(row["page_count"] or 0), *stored.keys()]) if row["kind"] in ("pdf", "document") else 0
    numbers = sorted(set(range(1, count + 1)) | set(stored))
    out: list[PageOut] = []
    for n in numbers:
        r = stored.get(n)
        width = r["width"] if r is not None else (row["width"] if n == 1 else None)
        height = r["height"] if r is not None else (row["height"] if n == 1 else None)
        out.append(PageOut(page_no=n, width=width, height=height, text=r["text"] if r is not None else None,
                           text_source=r["text_source"] if r is not None else "none",
                           url=page_url(uid, v, n)))  # type: ignore[arg-type]
    return out


@router.get("/items/{uid}/text", response_model=ItemTextOut)
def item_text(uid: str, session: SessionInfo = Depends(require_user),
              conn: sqlite3.Connection = Depends(get_db)) -> ItemTextOut:
    row = _item_row(conn, uid)
    _guard(row, session)
    t = conn.execute("SELECT text, source, char_count FROM item_text WHERE item_id = ?", (row["id"],)).fetchone()
    if t is None:
        raise HTTPException(status_code=404, detail="No text for this item")
    return ItemTextOut(text=t["text"], source=t["source"], char_count=int(t["char_count"]))


# ------------------------------------------------------------------ trash
@router.get("/trash", response_model=ItemPage)
def list_trash(cursor: str | None = Query(default=None, max_length=512), limit: int = Query(default=60, ge=1, le=200),
               vault: bool = Depends(want_vault), conn: sqlite3.Connection = Depends(get_db)) -> ItemPage:
    where = ["i.deleted_at IS NOT NULL"]
    if not vault:
        where.append("i.nsfw = 0")
    return _page(conn, where, [], "i.deleted_at", cursor, limit)


@router.post("/trash/empty", response_model=BulkOut)
def empty_trash(session: SessionInfo = Depends(require_user), conn: sqlite3.Connection = Depends(get_db)) -> BulkOut:
    """Purge everything in the trash (NSFW items only while the vault is open)."""
    from app import ingest

    cond = "" if session.vault_unlocked else " AND nsfw = 0"
    ids = [int(r[0]) for r in conn.execute(f"SELECT id FROM items WHERE deleted_at IS NOT NULL{cond}")]
    for item_id in ids:
        ingest.purge(conn, item_id)
    return BulkOut(updated=len(ids))


@router.delete("/trash/{uid}", response_model=Ok)
def purge_one(uid: str, session: SessionInfo = Depends(require_user),
              conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    from app import ingest

    row = _item_row(conn, uid)
    _guard(row, session)
    if row["deleted_at"] is None:
        raise HTTPException(status_code=409, detail="Only items in the trash can be deleted for good")
    ingest.purge(conn, int(row["id"]))
    return Ok()


# ------------------------------------------------------------------ review and duplicates
@router.get("/review/misclicks", response_model=ItemPage)
def review_misclicks(cursor: str | None = Query(default=None, max_length=512),
                     limit: int = Query(default=60, ge=1, le=200), vault: bool = Depends(want_vault),
                     conn: sqlite3.Connection = Depends(get_db)) -> ItemPage:
    where = [visible_sql(vault=vault, misclicks="only", dups="all")]
    return _page(conn, where, [], SORT_KEYS["taken"], cursor, limit)


@router.get("/duplicates", response_model=DupGroupPage)
def duplicates(cursor: str | None = Query(default=None, max_length=512), limit: int = Query(default=30, ge=1, le=100),
               vault: bool = Depends(want_vault), conn: sqlite3.Connection = Depends(get_db)) -> DupGroupPage:
    nsfw = 1 if vault else 0
    # Copies on the other side of the vault are not part of this view (they show as their own cards).
    base = ("i.dup_role = 'best' AND i.dup_group_id = i.id AND i.deleted_at IS NULL AND i.nsfw = ? AND EXISTS "
            "(SELECT 1 FROM items c WHERE c.dup_group_id = i.id AND c.id != i.id AND c.deleted_at IS NULL "
            "AND c.nsfw = i.nsfw)")
    total = conn.execute(f"SELECT COUNT(*) FROM items i WHERE {base}", (nsfw,)).fetchone()[0]
    params: list[Any] = [nsfw]
    cond = base
    after = _keyset(cursor, 1)
    if after is not None:
        cond += " AND i.id < ?"
        params.append(after[0])
    rows = conn.execute(f"SELECT {ITEM_CARD_COLUMNS} FROM items i WHERE {cond} ORDER BY i.id DESC LIMIT ?",
                        [*params, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    groups: list[DupGroup] = []
    for r in rows:
        copies = conn.execute(
            f"SELECT {ITEM_CARD_COLUMNS} FROM items i WHERE i.dup_group_id = ? AND i.id != ? AND i.deleted_at IS NULL "
            "AND i.nsfw = ? ORDER BY i.id",
            (r["id"], r["id"], nsfw),
        ).fetchall()
        groups.append(DupGroup(best=item_card(r), copies=[item_card(c) for c in copies]))
    next_cursor = encode_cursor([int(rows[-1]["id"])]) if more and rows else None
    return DupGroupPage(groups=groups, next_cursor=next_cursor, total=int(total))


@router.post("/duplicates/{uid}/best", response_model=Ok)
def choose_best(uid: str, session: SessionInfo = Depends(require_user),
                conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    from app.media import dedup

    row = _item_row(conn, uid)
    _guard(row, session)
    if row["deleted_at"] is not None:
        raise HTTPException(status_code=409, detail="Restore the item from the trash first")
    with tx(conn):
        if not dedup.set_best(conn, int(row["id"])):
            raise HTTPException(status_code=409, detail="This item has no duplicates")
    return Ok()


# ------------------------------------------------------------------ stats
@router.get("/stats", response_model=Stats)
def stats(conn: sqlite3.Connection = Depends(get_db)) -> Stats:
    live = "deleted_at IS NULL AND nsfw = 0 AND companion_of IS NULL"

    def n(sql: str, params: tuple = ()) -> int:
        return int(conn.execute(sql, params).fetchone()[0] or 0)

    by_kind = {k: 0 for k in ITEM_KINDS}
    for r in conn.execute(f"SELECT kind, COUNT(*) FROM items WHERE {live} GROUP BY kind"):
        by_kind[r[0]] = int(r[1])
    by_category = {c: 0 for c in CATEGORIES if c != "nsfw"}
    for r in conn.execute(f"SELECT category, COUNT(*) FROM items WHERE {live} GROUP BY category"):
        by_category[r[0]] = int(r[1])
    analysis = {"pending": 0, "done": 0, "error": 0}
    for r in conn.execute("SELECT analysis_status, COUNT(*) FROM items WHERE deleted_at IS NULL GROUP BY 1"):
        analysis[r[0]] = int(r[1])
    text = {"pending": 0, "done": 0, "error": 0}
    for r in conn.execute("SELECT text_status, COUNT(*) FROM items WHERE deleted_at IS NULL "
                          "AND text_status != 'none' GROUP BY 1"):
        text[r[0]] = int(r[1])
    free, total = storage.disk_usage()
    recent = conn.execute(
        f"SELECT {ITEM_CARD_COLUMNS} FROM items i WHERE {visible_sql()} ORDER BY i.created_at DESC, i.id DESC LIMIT ?",
        (RECENT_LIMIT,),
    ).fetchall()
    return Stats(
        items=n(f"SELECT COUNT(*) FROM items WHERE {live}"),
        by_kind=by_kind,
        by_category=by_category,
        people=n("SELECT COUNT(*) FROM persons"),
        named_people=n("SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"),
        documents=n("SELECT COUNT(*) FROM documents d JOIN items i ON i.id = d.item_id "
                    "WHERE i.deleted_at IS NULL AND i.nsfw = 0"),
        misclicks=n(f"SELECT COUNT(*) FROM items WHERE {live} AND misclick = 1"),
        guesses=n("SELECT (SELECT COUNT(*) FROM faces f JOIN items i ON i.id = f.item_id "
                  "WHERE f.person_id IS NULL AND f.guess_person_id IS NOT NULL AND i.deleted_at IS NULL AND i.nsfw = 0) + "
                  "(SELECT COUNT(*) FROM bodies b JOIN items i ON i.id = b.item_id "
                  "WHERE b.person_id IS NULL AND b.guess_person_id IS NOT NULL AND i.deleted_at IS NULL AND i.nsfw = 0)"),
        duplicates=n(f"SELECT COUNT(*) FROM items WHERE {live} AND dup_role = 'copy'"),
        trash=n("SELECT COUNT(*) FROM items WHERE deleted_at IS NOT NULL"),
        nsfw=n("SELECT COUNT(*) FROM items WHERE deleted_at IS NULL AND nsfw = 1"),
        queue=jobs.counts(conn),
        disk=DiskStats(free_bytes=free, total_bytes=total,
                       originals_bytes=n("SELECT COALESCE(SUM(size_bytes), 0) FROM items")),
        analysis=analysis,
        text=text,
        recent=[item_card(r) for r in recent],
    )
