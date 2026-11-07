"""Admin API (single hardcoded account, see app.config / app.auth).

CONTRACT (implemented by the admin agent). Router mounted at prefix /api/admin. All routes
except login/logout/session depend on app.auth.require_admin (401 / CSRF 403). Any change to
faces or persons happens in one `with tx(conn):` and calls app.db.bump_faces_version(conn).
Person names go through app.util.clean_name(..., max_len=120); empty -> NULL; name_key = casefold.

POST   /api/admin/login            LoginIn -> SessionOut   (429 after 10 failures / 10 min per IP)
POST   /api/admin/logout           -> SessionOut
GET    /api/admin/session          -> SessionOut           (never 401)
GET    /api/admin/stats            -> AdminStats   (search: {indexed, total, model} of the search index)

People
GET    /api/admin/persons?q=&filter=all|named|unnamed|hidden&sort=faces|name|recent&cursor=&limit=100 -> PersonPage
POST   /api/admin/persons          PersonCreate -> PersonOut   (optionally moves face_ids into it, locked 'admin')
GET    /api/admin/persons/{id}     -> PersonOut
PATCH  /api/admin/persons/{id}     PersonUpdate -> PersonOut   (only fields in model_fields_set;
                                   cover_face_id must be a face of this person, 400 otherwise)
DELETE /api/admin/persons/{id}     -> OkOut   faces become unassigned + locked ('admin'), tags removed
POST   /api/admin/persons/merge    MergeIn -> PersonOut   (app.faces.cluster.merge_persons)
GET    /api/admin/persons/{id}/faces?cursor=&limit=200 -> FacePage   (quality desc, id)
GET    /api/admin/persons/{id}/media?cursor=&limit=60 -> MediaPage   (includes hidden media)
       The PersonOut returned by the single-person routes above (get, create, patch, merge)
       carries `links` with every kind; the list leaves it empty.

Linked profiles (app/links.py; linkedin / instagram / website public, email / phone admin-only)
GET    /api/admin/persons/{id}/links -> list[PersonLink]   (display order)
POST   /api/admin/persons/{id}/links PersonLinkCreate -> PersonLink   (400 with the normalisation
       message; 409 if the person already has this link; 400 past MAX_LINKS_PER_PERSON)
PATCH  /api/admin/persons/{id}/links/{link_id} PersonLinkUpdate -> PersonLink   (value is
       normalised with the stored kind; 409 on a duplicate)
DELETE /api/admin/persons/{id}/links/{link_id} -> OkOut
POST   /api/admin/persons/{id}/links/reorder PersonLinkReorder -> list[PersonLink]   (ids must be
       exactly the person's link ids)
       Merging moves the sources' links to the target (duplicates skipped; the per-person limit
       applies to POST only, so a merge never drops links). Deleting a person deletes its links.

Faces
GET    /api/admin/faces/unassigned?cursor=&limit=200 -> FacePage   (person_id NULL, newest first)
POST   /api/admin/faces/assign     AssignFacesIn -> AssignOut
       person_id -> move faces there; new_person_name -> create person and move; neither -> unassign.
       All moved faces get assigned_by='admin'. Clear persons.cover_face_id that pointed at a moved
       face of another person. People left with no faces and no tags and no name are deleted.
DELETE /api/admin/faces/{id}       -> OkOut   (false detection: delete the face row and its crop file)

Merge suggestions
GET    /api/admin/suggestions?limit=30 -> SuggestionList   (app.faces.cluster.merge_suggestions,
       samples = up to 4 best face crops per person)
POST   /api/admin/suggestions/dismiss  DismissIn -> OkOut   (store (min, max) in merge_dismissals)

Media
PATCH  /api/admin/media/{uid}      MediaUpdate -> MediaItem   (hide / unhide)
DELETE /api/admin/media/{uid}      -> OkOut   (app.ingest.delete_media)
POST   /api/admin/media/{uid}/reprocess -> OkOut   (set faces_status='pending', enqueue faces job)
POST   /api/admin/media/{uid}/tags TagIn -> list[PersonRef]   (the item's manual tags afterwards;
       tagging a person who already has a face on the item is a no-op)
DELETE /api/admin/media/{uid}/tags/{person_id} -> list[PersonRef]
GET    /api/admin/media?hidden=1&cursor=&limit=60 -> MediaPage   (all media incl. hidden; hidden=1 only hidden)

Uploads / duplicates / maintenance
GET    /api/admin/uploads?status=&q=&cursor=&limit=100 -> UploadLogPage  (newest first; q matches
       uploader name or file name; can_retry = a failed upload whose file is kept in DATA_DIR/failed)
POST   /api/admin/uploads/{uid}/retry -> OkOut   (move the kept file back and enqueue ingest;
       404 unknown upload, 409 not failed or no kept file)
DELETE /api/admin/uploads/{uid}/file  -> OkOut   (delete the kept file of a failed upload; same 404/409)
GET    /api/admin/duplicates?reason=&cursor=&limit=60 -> SupersededPage   (newest first)
POST   /api/admin/duplicates/{id}/restore -> OkOut   (enqueue 'restore' job; 409 if purged)
DELETE /api/admin/duplicates/{id}  -> OkOut   (app.ingest.purge_superseded: deletes the file and
       forgets its hash, so a later re-upload of the same file is evaluated again instead of
       being reported as a duplicate)
POST   /api/admin/recluster        -> OkOut   (enqueue unique 'recluster' job; skipped while one is
       queued or running)

Notes on choices the contract leaves open:
- persons filter: all = everyone (hidden included); named / unnamed = not hidden; hidden = hidden only.
  Sort faces = face_count desc; name = name_key asc (unnamed last); recent = created_at desc. The
  list is paged with an offset cursor because face_count is computed.
- Other lists use keyset cursors on (sort key, id).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.api.deps import admin_change
from app.changelog import trash
from app.changelog.capture import claim
from app import jobs, storage
from app.api.common import (
    MEDIA_ITEM_COLUMNS,
    UPLOAD_MESSAGES,
    link_out,
    media_item,
    person_links,
    person_out,
    person_ref,
    person_stats,
    uploader_ref,
    valid_cursor_scalar,
)
from app.api.deps import get_db
from app.auth import (
    CSRF_HEADER,
    CSRF_VALUE,
    check_credentials,
    claim_login_attempt,
    clear_admin_cookie,
    client_ip,
    end_all_sessions,
    is_admin,
    require_admin,
    reset_login_failures,
    set_admin_cookie,
)
from app.config import ADMIN_USERNAME
from app.db import bump_faces_version, tx
from app.links import LABEL_MAX, MAX_LINKS_PER_PERSON, normalize
from app import persons as persons_lib
from app.search import indexer as search_indexer
from app.schemas import (
    AdminFace,
    AdminStats,
    AssignFacesIn,
    AssignOut,
    DiskStats,
    DismissIn,
    FacePage,
    LoginIn,
    MediaItem,
    MediaPage,
    MediaUpdate,
    MergeIn,
    OkOut,
    PersonCreate,
    PersonLink,
    PersonLinkCreate,
    PersonLinkReorder,
    PersonLinkUpdate,
    PersonOut,
    PersonPage,
    PersonRef,
    PersonUpdate,
    QueueStats,
    SearchStats,
    SessionOut,
    SuggestionList,
    SuggestionOut,
    SupersededItem,
    SupersededPage,
    TagIn,
    UploadLogItem,
    UploadLogPage,
)
from app.urls import face_url, superseded_file_url, superseded_thumb_url, thumb_url
from app.util import clean_name, decode_cursor, encode_cursor, now_iso

log = logging.getLogger(__name__)

def _no_store(response: Response) -> None:
    """Admin responses carry admin-only data (email and phone links, hidden items): never cache them.
    Routes that return their own Response object (file downloads) set their own headers."""
    response.headers["cache-control"] = "private, no-store"


router = APIRouter()
# admin_change ends before the response is sent ("function" scope), so History already shows the
# change's description when the next request asks.
protected = APIRouter(dependencies=[Depends(require_admin), Depends(_no_store), Depends(admin_change, scope="function")])

NAME_MAX = persons_lib.NAME_MAX


# ------------------------------------------------------------------ helpers
def _require_csrf(request: Request) -> None:
    if request.headers.get(CSRF_HEADER) != CSRF_VALUE:
        raise HTTPException(status_code=403, detail="Missing X-Requested-With header")


def _session(request: Request) -> SessionOut:
    admin = is_admin(request)
    return SessionOut(is_admin=admin, username=ADMIN_USERNAME if admin else None)


def _clean_person_name(value: str | None) -> str | None:
    """clean_name with the person limit; empty -> None."""
    return persons_lib.clean_person_name(value)


def _like(q: str) -> str:
    """LIKE pattern for a substring match (with ESCAPE '\\')."""
    esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


def _keyset(cursor: str | None, n: int) -> list[Any] | None:
    if not cursor:
        return None
    values = decode_cursor(cursor)
    if values is None or len(values) != n or not all(valid_cursor_scalar(v) for v in values):
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return values


def _desc_after(col: str, values: list[Any] | None, id_col: str) -> tuple[str, list[Any]]:
    """Keyset condition for ORDER BY col DESC, id DESC."""
    if values is None:
        return "", []
    return f"({col} < ? OR ({col} = ? AND {id_col} < ?))", [values[0], values[0], values[1]]


def _where(conds: list[str]) -> str:
    conds = [c for c in conds if c]
    return ("WHERE " + " AND ".join(conds)) if conds else ""


def _person_row(conn: sqlite3.Connection, person_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, name, name_key, hidden, cover_face_id, created_at FROM persons WHERE id = ?", (person_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return row


def _person_out(conn: sqlite3.Connection, person_id: int) -> PersonOut:
    row = _person_row(conn, person_id)
    out = person_out(row, person_stats(conn, [person_id], visible_only=False).get(person_id), admin=True)
    out.links = person_links(conn, [person_id], include_private=True).get(person_id, [])
    return out


def _media_row(conn: sqlite3.Connection, uid: str) -> sqlite3.Row:
    row = conn.execute(f"SELECT {MEDIA_ITEM_COLUMNS}, m.faces_status FROM media m WHERE m.uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Photo or video not found")
    return row


def _create_person(conn: sqlite3.Connection, name: str | None) -> int:
    return persons_lib.create_person(conn, name)


def _prune_empty_persons(conn: sqlite3.Connection, person_ids: set[int]) -> int:
    """Delete people with no name, no faces and no manual tags (call inside a tx)."""
    ids = sorted(p for p in person_ids if p is not None)
    if not ids:
        return 0
    ph = ",".join("?" * len(ids))
    cur = conn.execute(
        f"""DELETE FROM persons WHERE id IN ({ph}) AND name IS NULL
            AND NOT EXISTS (SELECT 1 FROM faces f WHERE f.person_id = persons.id)
            AND NOT EXISTS (SELECT 1 FROM media_tags t WHERE t.person_id = persons.id)
            AND NOT EXISTS (SELECT 1 FROM person_links l WHERE l.person_id = persons.id)""",
        ids,
    )
    return cur.rowcount


def _move_faces(conn: sqlite3.Connection, face_ids: list[int], target_id: int | None) -> int:
    """Move faces to target (or unassign when None), locked as 'admin'. Call inside a tx.

    Clears covers of other people that pointed at moved faces and deletes people left empty.
    Raises 404 if any face id is unknown. Returns the number of faces updated.
    """
    ids = sorted({int(f) for f in face_ids})
    if not ids:
        return 0
    ph = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT id, person_id FROM faces WHERE id IN ({ph})", ids).fetchall()
    if len(rows) != len(ids):
        raise HTTPException(status_code=404, detail="Some faces no longer exist. Reload and try again.")
    previous = {r["person_id"] for r in rows if r["person_id"] is not None}
    conn.execute(f"UPDATE faces SET person_id = ?, assigned_by = 'admin' WHERE id IN ({ph})", [target_id, *ids])
    conn.execute(
        f"UPDATE persons SET cover_face_id = NULL, updated_at = ? WHERE cover_face_id IN ({ph}) AND id IS NOT ?",
        [now_iso(), *ids, target_id],
    )
    if target_id is not None:
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now_iso(), target_id))
    previous.discard(target_id)  # type: ignore[arg-type]
    _prune_empty_persons(conn, previous)
    return len(ids)


def _admin_face(row: sqlite3.Row) -> AdminFace:
    return AdminFace(
        id=row["id"],
        media_uid=row["media_uid"],
        media_kind=row["media_kind"],
        thumb_url=face_url(row["id"]),
        det_score=row["det_score"],
        size_px=row["size_px"],
        quality=row["quality"],
        frame_time_s=row["frame_time_s"],
        person_id=row["person_id"],
        assigned_by=row["assigned_by"],
        created_at=row["created_at"],
    )


FACE_SELECT = (
    "SELECT f.id, f.person_id, f.assigned_by, f.det_score, f.size_px, f.quality, f.frame_time_s, "
    "f.created_at, m.uid AS media_uid, m.kind AS media_kind FROM faces f JOIN media m ON m.id = f.media_id"
)


def _media_tags(conn: sqlite3.Connection, media_id: int) -> list[PersonRef]:
    rows = conn.execute(
        "SELECT p.id, p.name FROM media_tags t JOIN persons p ON p.id = t.person_id WHERE t.media_id = ? "
        "ORDER BY p.name_key IS NULL, p.name_key, p.id",
        (media_id,),
    ).fetchall()
    return [person_ref(r["id"], r["name"]) for r in rows]


def _media_page(conn: sqlite3.Connection, base_conds: list[str], params: list[Any],
                cursor: str | None, limit: int) -> MediaPage:
    """Media list ordered by created_at desc, id desc (alias m)."""
    total = conn.execute(f"SELECT COUNT(*) FROM media m {_where(base_conds)}", params).fetchone()[0]
    after, after_params = _desc_after("m.created_at", _keyset(cursor, 2), "m.id")
    rows = conn.execute(
        f"SELECT {MEDIA_ITEM_COLUMNS} FROM media m {_where([*base_conds, after])} "
        "ORDER BY m.created_at DESC, m.id DESC LIMIT ?",
        [*params, *after_params, limit + 1],
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([rows[-1]["created_at"], rows[-1]["id"]]) if more and rows else None
    return MediaPage(items=[media_item(r) for r in rows], next_cursor=nxt, total=total)


# ------------------------------------------------------------------ session
@router.post("/login", response_model=SessionOut)
def login(body: LoginIn, request: Request, response: Response) -> SessionOut:
    _require_csrf(request)
    ip = client_ip(request)
    # Counted before the password is checked, under one lock, so parallel guesses cannot all
    # pass the limit before any failure is recorded. A successful login resets the count.
    if not claim_login_attempt(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in a few minutes.")
    if not check_credentials(body.username, body.password):
        log.info("admin login failed from %s", ip)
        raise HTTPException(status_code=401, detail="Wrong username or password")
    reset_login_failures(ip)
    set_admin_cookie(response, request)
    log.info("admin login from %s", ip)
    return SessionOut(is_admin=True, username=ADMIN_USERNAME)


@router.post("/logout", response_model=SessionOut)
def logout(request: Request, response: Response) -> SessionOut:
    _require_csrf(request)
    if is_admin(request):
        # Only a signed-in admin can end the sessions; otherwise anyone could log the admin out.
        end_all_sessions()
    clear_admin_cookie(response)
    return SessionOut(is_admin=False, username=None)


@router.get("/session", response_model=SessionOut)
def session(request: Request) -> SessionOut:
    return _session(request)


# ------------------------------------------------------------------ stats
@protected.get("/stats", response_model=AdminStats)
def stats(conn: sqlite3.Connection = Depends(get_db)) -> AdminStats:
    kinds = {r["kind"]: r["n"] for r in conn.execute("SELECT kind, COUNT(*) AS n FROM media GROUP BY kind")}
    photos, videos = kinds.get("image", 0), kinds.get("video", 0)

    def count(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0] or 0)

    free, total = storage.disk_usage()
    return AdminStats(
        media=photos + videos,
        photos=photos,
        videos=videos,
        hidden_media=count("SELECT COUNT(*) FROM media WHERE hidden = 1"),
        faces=count("SELECT COUNT(*) FROM faces"),
        unassigned_faces=count("SELECT COUNT(*) FROM faces WHERE person_id IS NULL"),
        persons=count("SELECT COUNT(*) FROM persons"),
        named_persons=count("SELECT COUNT(*) FROM persons WHERE name IS NOT NULL"),
        hidden_persons=count("SELECT COUNT(*) FROM persons WHERE hidden = 1"),
        uploaders=count("SELECT COUNT(*) FROM uploaders"),
        uploads_by_status={
            r["status"]: r["n"] for r in conn.execute(
                "SELECT CASE WHEN removed = 1 THEN 'removed' ELSE status END AS status, COUNT(*) AS n "
                "FROM uploads GROUP BY 1")
        },
        superseded=count("SELECT COUNT(*) FROM superseded"),
        queue=QueueStats(**jobs.counts(conn)),
        search=SearchStats(**search_indexer.progress(conn)),
        disk=DiskStats(
            free_bytes=free,
            total_bytes=total,
            originals_bytes=count("SELECT COALESCE(SUM(size_bytes), 0) FROM media"),
            superseded_bytes=count("SELECT COALESCE(SUM(size_bytes), 0) FROM superseded WHERE path IS NOT NULL"),
        ),
    )


# ------------------------------------------------------------------ people
@protected.get("/persons", response_model=PersonPage)
def list_persons(
    q: str | None = Query(default=None, max_length=200),
    filter: Literal["all", "named", "unnamed", "hidden"] = "all",
    sort: Literal["faces", "name", "recent"] = "faces",
    cursor: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_db),
) -> PersonPage:
    conds: list[str] = []
    params: list[Any] = []
    if filter == "named":
        conds.append("name IS NOT NULL AND hidden = 0")
    elif filter == "unnamed":
        conds.append("name IS NULL AND hidden = 0")
    elif filter == "hidden":
        conds.append("hidden = 1")
    needle = clean_name(q or "", 200).casefold()
    if needle:
        conds.append("(name_key LIKE ? ESCAPE '\\' OR ('unnamed #' || id) LIKE ? ESCAPE '\\')")
        params += [_like(needle), _like(needle)]
    rows = conn.execute(
        f"SELECT id, name, name_key, hidden, cover_face_id, created_at FROM persons {_where(conds)}", params
    ).fetchall()
    st = person_stats(conn, [r["id"] for r in rows], visible_only=False) if rows else {}

    def faces_of(r: sqlite3.Row) -> int:
        s = st.get(r["id"])
        return s.face_count if s else 0

    if sort == "faces":
        rows.sort(key=lambda r: (-faces_of(r), r["id"]))
    elif sort == "name":
        rows.sort(key=lambda r: (r["name_key"] is None, r["name_key"] or "", r["id"]))
    else:
        rows.sort(key=lambda r: (r["created_at"], r["id"]), reverse=True)

    values = _keyset(cursor, 1)
    offset = int(values[0]) if values and isinstance(values[0], int) and values[0] >= 0 else 0
    page = rows[offset: offset + limit]
    nxt = encode_cursor([offset + limit]) if offset + limit < len(rows) else None
    return PersonPage(
        items=[person_out(r, st.get(r["id"]), admin=True) for r in page], next_cursor=nxt, total=len(rows)
    )


@protected.post("/persons", response_model=PersonOut)
def create_person(body: PersonCreate, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    name = _clean_person_name(body.name)
    if not name:
        raise HTTPException(status_code=400, detail="Enter a name")
    with tx(conn):
        pid = _create_person(conn, name)
        if body.face_ids:
            _move_faces(conn, body.face_ids, pid)
        bump_faces_version(conn)
    return _person_out(conn, pid)


@protected.post("/persons/merge", response_model=PersonOut)
def merge(body: MergeIn, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    from app.faces import cluster

    sources = sorted({int(s) for s in body.source_ids}, key=body.source_ids.index)
    if body.target_id in sources:
        raise HTTPException(status_code=400, detail="Cannot merge a person into themselves")
    ids = [body.target_id, *sources]
    ph = ",".join("?" * len(ids))
    found = conn.execute(f"SELECT COUNT(*) FROM persons WHERE id IN ({ph})", ids).fetchone()[0]
    if found != len(ids):
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        with tx(conn):
            cluster.merge_persons(conn, body.target_id, sources)
            bump_faces_version(conn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e) or "Cannot merge these people") from e
    return _person_out(conn, body.target_id)


@protected.get("/persons/{person_id}", response_model=PersonOut)
def get_person(person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    return _person_out(conn, person_id)


@protected.patch("/persons/{person_id}", response_model=PersonOut)
def update_person(person_id: int, body: PersonUpdate, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    fields = body.model_fields_set
    with tx(conn):
        _person_row(conn, person_id)
        sets: list[str] = []
        params: list[Any] = []
        if "name" in fields:
            name = _clean_person_name(body.name)
            sets += ["name = ?", "name_key = ?"]
            params += [name, name.casefold() if name else None]
        if "hidden" in fields and body.hidden is not None:
            sets.append("hidden = ?")
            params.append(int(body.hidden))
        if "cover_face_id" in fields:
            if body.cover_face_id is not None:
                ok = conn.execute(
                    "SELECT 1 FROM faces WHERE id = ? AND person_id = ?", (body.cover_face_id, person_id)
                ).fetchone()
                if ok is None:
                    raise HTTPException(status_code=400, detail="The cover must be one of this person's faces")
            sets.append("cover_face_id = ?")
            params.append(body.cover_face_id)
        if sets:
            sets.append("updated_at = ?")
            params.append(now_iso())
            conn.execute(f"UPDATE persons SET {', '.join(sets)} WHERE id = ?", [*params, person_id])
            bump_faces_version(conn)
    return _person_out(conn, person_id)


@protected.delete("/persons/{person_id}", response_model=OkOut)
def delete_person(person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    with tx(conn):
        _person_row(conn, person_id)
        conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'admin' WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM media_tags WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM person_links WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        bump_faces_version(conn)
    return OkOut()


@protected.get("/persons/{person_id}/faces", response_model=FacePage)
def person_faces(
    person_id: int,
    cursor: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_db),
) -> FacePage:
    _person_row(conn, person_id)
    total = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id = ?", (person_id,)).fetchone()[0]
    conds = ["f.person_id = ?"]
    params: list[Any] = [person_id]
    values = _keyset(cursor, 2)
    if values is not None:  # ORDER BY quality DESC, id ASC
        conds.append("(f.quality < ? OR (f.quality = ? AND f.id > ?))")
        params += [values[0], values[0], values[1]]
    rows = conn.execute(
        f"{FACE_SELECT} {_where(conds)} ORDER BY f.quality DESC, f.id ASC LIMIT ?", [*params, limit + 1]
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([rows[-1]["quality"], rows[-1]["id"]]) if more and rows else None
    return FacePage(items=[_admin_face(r) for r in rows], next_cursor=nxt, total=total)


@protected.get("/persons/{person_id}/media", response_model=MediaPage)
def person_media(
    person_id: int,
    cursor: str | None = None,
    limit: int = Query(default=60, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> MediaPage:
    _person_row(conn, person_id)
    cond = (
        "m.id IN (SELECT media_id FROM faces WHERE person_id = ? "
        "UNION SELECT media_id FROM media_tags WHERE person_id = ?)"
    )
    return _media_page(conn, [cond], [person_id, person_id], cursor, limit)


# ------------------------------------------------------------------ person links
def _clean_label(value: str | None) -> str | None:
    if value is None:
        return None
    return clean_name(value, max_len=LABEL_MAX) or None


def _normalized(kind: str, raw: str):
    try:
        return normalize(kind, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _link_row(conn: sqlite3.Connection, person_id: int, link_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, person_id, kind, value, label, position FROM person_links WHERE id = ? AND person_id = ?",
        (link_id, person_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Link not found")
    return row


def _links_of(conn: sqlite3.Connection, person_id: int) -> list[PersonLink]:
    return person_links(conn, [person_id], include_private=True).get(person_id, [])


DUPLICATE_LINK = "This person already has that link"


@protected.get("/persons/{person_id}/links", response_model=list[PersonLink])
def list_person_links(person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> list[PersonLink]:
    _person_row(conn, person_id)
    return _links_of(conn, person_id)


@protected.post("/persons/{person_id}/links", response_model=PersonLink)
def add_person_link(person_id: int, body: PersonLinkCreate,
                    conn: sqlite3.Connection = Depends(get_db)) -> PersonLink:
    link = _normalized(body.kind, body.value)
    label = _clean_label(body.label)
    now = now_iso()
    try:
        with tx(conn):
            _person_row(conn, person_id)
            count, top = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(position), -1) FROM person_links WHERE person_id = ?", (person_id,)
            ).fetchone()
            if conn.execute(
                "SELECT 1 FROM person_links WHERE person_id = ? AND kind = ? AND value = ?",
                (person_id, body.kind, link.value),
            ).fetchone():
                raise HTTPException(status_code=409, detail=DUPLICATE_LINK)
            if count >= MAX_LINKS_PER_PERSON:
                raise HTTPException(status_code=400,
                                    detail=f"A person can have at most {MAX_LINKS_PER_PERSON} links")
            cur = conn.execute(
                "INSERT INTO person_links(person_id, kind, value, label, position, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (person_id, body.kind, link.value, label, top + 1, now, now),
            )
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=DUPLICATE_LINK) from e
    return link_out(_link_row(conn, person_id, int(cur.lastrowid)))


@protected.patch("/persons/{person_id}/links/{link_id}", response_model=PersonLink)
def update_person_link(person_id: int, link_id: int, body: PersonLinkUpdate,
                       conn: sqlite3.Connection = Depends(get_db)) -> PersonLink:
    fields = body.model_fields_set
    try:
        with tx(conn):
            row = _link_row(conn, person_id, link_id)
            sets: list[str] = []
            params: list[Any] = []
            if "value" in fields and body.value is not None:
                value = _normalized(row["kind"], body.value).value
                if value != row["value"] and conn.execute(
                    "SELECT 1 FROM person_links WHERE person_id = ? AND kind = ? AND value = ? AND id != ?",
                    (person_id, row["kind"], value, link_id),
                ).fetchone():
                    raise HTTPException(status_code=409, detail=DUPLICATE_LINK)
                sets.append("value = ?")
                params.append(value)
            if "label" in fields:
                sets.append("label = ?")
                params.append(_clean_label(body.label))
            if sets:
                sets.append("updated_at = ?")
                params.append(now_iso())
                conn.execute(f"UPDATE person_links SET {', '.join(sets)} WHERE id = ?", [*params, link_id])
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=DUPLICATE_LINK) from e
    return link_out(_link_row(conn, person_id, link_id))


@protected.delete("/persons/{person_id}/links/{link_id}", response_model=OkOut)
def delete_person_link(person_id: int, link_id: int, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    with tx(conn):
        _link_row(conn, person_id, link_id)
        conn.execute("DELETE FROM person_links WHERE id = ?", (link_id,))
    return OkOut()


@protected.post("/persons/{person_id}/links/reorder", response_model=list[PersonLink])
def reorder_person_links(person_id: int, body: PersonLinkReorder,
                         conn: sqlite3.Connection = Depends(get_db)) -> list[PersonLink]:
    with tx(conn):
        _person_row(conn, person_id)
        current = {r[0] for r in conn.execute("SELECT id FROM person_links WHERE person_id = ?", (person_id,))}
        if len(body.ids) != len(set(body.ids)) or set(body.ids) != current:
            raise HTTPException(status_code=400, detail="The links changed. Reload and try again.")
        now = now_iso()
        for pos, link_id in enumerate(body.ids):
            conn.execute("UPDATE person_links SET position = ?, updated_at = ? WHERE id = ?", (pos, now, link_id))
    return _links_of(conn, person_id)


# ------------------------------------------------------------------ faces
@protected.get("/faces/unassigned", response_model=FacePage)
def unassigned_faces(
    cursor: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_db),
) -> FacePage:
    total = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NULL").fetchone()[0]
    conds = ["f.person_id IS NULL"]
    params: list[Any] = []
    values = _keyset(cursor, 1)
    if values is not None:
        conds.append("f.id < ?")
        params.append(values[0])
    rows = conn.execute(f"{FACE_SELECT} {_where(conds)} ORDER BY f.id DESC LIMIT ?", [*params, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([rows[-1]["id"]]) if more and rows else None
    return FacePage(items=[_admin_face(r) for r in rows], next_cursor=nxt, total=total)


@protected.post("/faces/assign", response_model=AssignOut)
def assign_faces(body: AssignFacesIn, conn: sqlite3.Connection = Depends(get_db)) -> AssignOut:
    fields = body.model_fields_set
    has_name = "new_person_name" in fields and body.new_person_name is not None
    if body.person_id is not None and has_name:
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name, not both")
    new_name = _clean_person_name(body.new_person_name) if has_name else None
    if has_name and not new_name:
        raise HTTPException(status_code=400, detail="Enter a name")
    with tx(conn):
        target: int | None = None
        if body.person_id is not None:
            target = int(_person_row(conn, body.person_id)["id"])
        elif new_name:
            target = _create_person(conn, new_name)
        updated = _move_faces(conn, body.face_ids, target)
        bump_faces_version(conn)
    return AssignOut(updated=updated, person=_person_out(conn, target) if target is not None else None)


@protected.delete("/faces/{face_id}", response_model=OkOut)
def delete_face(face_id: int, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    with tx(conn):
        row = conn.execute("SELECT id, media_id, person_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Face not found")
        conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.execute(
            "UPDATE media SET face_count = (SELECT COUNT(*) FROM faces WHERE media_id = ?) WHERE id = ?",
            (row["media_id"], row["media_id"]),
        )
        if row["person_id"] is not None:
            _prune_empty_persons(conn, {row["person_id"]})
        bump_faces_version(conn)
        trash.discard(conn, storage.face_crop_path(face_id))  # kept for an undo
    return OkOut()


# ------------------------------------------------------------------ suggestions
@protected.get("/suggestions", response_model=SuggestionList)
def suggestions(
    limit: int = Query(default=30, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> SuggestionList:
    from app.faces import cluster

    pairs = cluster.merge_suggestions(conn, limit)
    if not pairs:
        return SuggestionList(items=[])
    ids = sorted({int(a) for a, _, _ in pairs} | {int(b) for _, b, _ in pairs})
    ph = ",".join("?" * len(ids))
    rows = {
        r["id"]: r
        for r in conn.execute(f"SELECT id, name, hidden, cover_face_id FROM persons WHERE id IN ({ph})", ids)
    }
    st = person_stats(conn, ids, visible_only=False)
    samples: dict[int, list[str]] = {}
    for r in conn.execute(
        f"""SELECT pid, fid FROM (
                SELECT f.person_id AS pid, f.id AS fid,
                       ROW_NUMBER() OVER (PARTITION BY f.person_id ORDER BY f.quality DESC, f.id) AS rn
                FROM faces f WHERE f.person_id IN ({ph})
            ) WHERE rn <= 4 ORDER BY pid, rn""",
        ids,
    ):
        samples.setdefault(r["pid"], []).append(face_url(r["fid"]))
    items: list[SuggestionOut] = []
    for a, b, sim in pairs:
        ra, rb = rows.get(int(a)), rows.get(int(b))
        if ra is None or rb is None:
            continue
        items.append(SuggestionOut(
            a=person_out(ra, st.get(ra["id"]), admin=True),
            b=person_out(rb, st.get(rb["id"]), admin=True),
            similarity=float(sim),
            a_samples=samples.get(ra["id"], []),
            b_samples=samples.get(rb["id"], []),
        ))
    return SuggestionList(items=items)


@protected.post("/suggestions/dismiss", response_model=OkOut)
def dismiss_suggestion(body: DismissIn, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    if body.a == body.b:
        raise HTTPException(status_code=400, detail="Pick two different people")
    lo, hi = sorted((body.a, body.b))
    found = conn.execute("SELECT COUNT(*) FROM persons WHERE id IN (?, ?)", (lo, hi)).fetchone()[0]
    if found != 2:
        raise HTTPException(status_code=404, detail="Person not found")
    with tx(conn):
        cur = conn.execute(
            "INSERT OR IGNORE INTO merge_dismissals(person_a, person_b, created_at) VALUES (?, ?, ?)",
            (lo, hi, now_iso()),
        )
        if cur.rowcount == 0:
            claim(conn, "merge_dismissals", {"person_a": lo, "person_b": hi})
    return OkOut()


# ------------------------------------------------------------------ media
@protected.get("/media", response_model=MediaPage)
def list_media(
    hidden: bool = False,
    cursor: str | None = None,
    limit: int = Query(default=60, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> MediaPage:
    return _media_page(conn, ["m.hidden = 1"] if hidden else [], [], cursor, limit)


@protected.patch("/media/{uid}", response_model=MediaItem)
def update_media(uid: str, body: MediaUpdate, conn: sqlite3.Connection = Depends(get_db)) -> MediaItem:
    with tx(conn):
        row = _media_row(conn, uid)
        conn.execute("UPDATE media SET hidden = ?, updated_at = ? WHERE id = ?", (int(body.hidden), now_iso(), row["id"]))
    return media_item(_media_row(conn, uid))


@protected.delete("/media/{uid}", response_model=OkOut)
def delete_media(uid: str, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    from app import ingest

    row = _media_row(conn, uid)
    ingest.delete_media(conn, int(row["id"]))
    log.info("admin deleted media %s", uid)
    return OkOut()


@protected.post("/media/{uid}/reprocess", response_model=OkOut)
def reprocess_media(uid: str, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    with tx(conn):
        row = _media_row(conn, uid)
        conn.execute(
            "UPDATE media SET faces_status = 'pending', updated_at = ? WHERE id = ?", (now_iso(), row["id"])
        )
        jobs.enqueue(conn, "faces", {"media_id": int(row["id"]), "version": int(row["version"])}, unique=True)
    return OkOut()


@protected.post("/media/{uid}/tags", response_model=list[PersonRef])
def add_tag(uid: str, body: TagIn, conn: sqlite3.Connection = Depends(get_db)) -> list[PersonRef]:
    has_name = body.new_person_name is not None
    if (body.person_id is None) == (not has_name):
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name")
    new_name = _clean_person_name(body.new_person_name) if has_name else None
    if has_name and not new_name:
        raise HTTPException(status_code=400, detail="Enter a name")
    with tx(conn):
        media_id = int(_media_row(conn, uid)["id"])
        if body.person_id is not None:
            pid = int(_person_row(conn, body.person_id)["id"])
        else:
            pid = _create_person(conn, new_name)
        on_item = conn.execute(
            "SELECT 1 FROM faces WHERE media_id = ? AND person_id = ? LIMIT 1", (media_id, pid)
        ).fetchone()
        if on_item is None:
            cur = conn.execute(
                "INSERT OR IGNORE INTO media_tags(media_id, person_id, created_at) VALUES (?, ?, ?)",
                (media_id, pid, now_iso()),
            )
            if cur.rowcount == 0:
                claim(conn, "media_tags", {"media_id": media_id, "person_id": pid}, media_id)
        bump_faces_version(conn)
    return _media_tags(conn, media_id)


@protected.delete("/media/{uid}/tags/{person_id}", response_model=list[PersonRef])
def remove_tag(uid: str, person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> list[PersonRef]:
    with tx(conn):
        media_id = int(_media_row(conn, uid)["id"])
        cur = conn.execute("DELETE FROM media_tags WHERE media_id = ? AND person_id = ?", (media_id, person_id))
        if cur.rowcount:
            _prune_empty_persons(conn, {person_id})
            bump_faces_version(conn)
    return _media_tags(conn, media_id)


# ------------------------------------------------------------------ uploads / duplicates
@protected.get("/uploads", response_model=UploadLogPage)
def upload_log(
    status: str | None = None,
    q: str | None = Query(default=None, max_length=200),
    cursor: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_db),
) -> UploadLogPage:
    conds: list[str] = []
    params: list[Any] = []
    if status:
        if status not in UPLOAD_MESSAGES:
            raise HTTPException(status_code=400, detail="Unknown status")
        if status == "removed":
            conds.append("u.removed = 1")
        else:
            conds.append("u.status = ? AND u.removed = 0")
            params.append(status)
    needle = clean_name(q or "", 200).casefold()
    if needle:
        conds.append("(up.name_key LIKE ? ESCAPE '\\' OR lower(u.original_filename) LIKE ? ESCAPE '\\')")
        params += [_like(needle), _like(needle)]
    base = "FROM uploads u JOIN uploaders up ON up.id = u.uploader_id LEFT JOIN media m ON m.id = u.media_id"
    total = conn.execute(f"SELECT COUNT(*) {base} {_where(conds)}", params).fetchone()[0]
    after, after_params = _desc_after("u.created_at", _keyset(cursor, 2), "u.id")
    rows = conn.execute(
        "SELECT u.id, u.uid, u.uploader_id, up.name AS uploader_name, u.original_filename, u.size_bytes, "
        "u.received_bytes, u.status, u.removed, u.message, u.remote_addr, u.created_at, u.updated_at, "
        f"m.uid AS media_uid, m.version AS media_version {base} {_where([*conds, after])} "
        "ORDER BY u.created_at DESC, u.id DESC LIMIT ?",
        [*params, *after_params, limit + 1],
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    items = [
        UploadLogItem(
            id=r["uid"],
            uploader=uploader_ref(r["uploader_id"], r["uploader_name"]),
            filename=r["original_filename"],
            size=r["size_bytes"],
            received_bytes=r["received_bytes"],
            status=r["status"],
            removed=bool(r["removed"]),
            message=UPLOAD_MESSAGES["removed"] if r["removed"] else r["message"] or UPLOAD_MESSAGES.get(r["status"]),
            media_uid=r["media_uid"],
            media_thumb_url=thumb_url(r["media_uid"], r["media_version"]) if r["media_uid"] else None,
            remote_addr=r["remote_addr"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            can_retry=r["status"] == "error" and storage.failed_path(r["uid"]).is_file(),
        )
        for r in rows
    ]
    nxt = encode_cursor([rows[-1]["created_at"], rows[-1]["id"]]) if more and rows else None
    return UploadLogPage(items=items, next_cursor=nxt, total=total)


def _upload_id(conn: sqlite3.Connection, uid: str) -> int:
    row = conn.execute("SELECT id FROM uploads WHERE uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return int(row["id"])


@protected.post("/uploads/{uid}/retry", response_model=OkOut)
def retry_upload(uid: str, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    from app import ingest

    if not ingest.requeue_failed_upload(conn, _upload_id(conn, uid)):
        raise HTTPException(status_code=409, detail="Only a failed upload whose file was kept can be retried")
    return OkOut()


@protected.delete("/uploads/{uid}/file", response_model=OkOut)
def discard_upload_file(uid: str, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    from app import ingest

    if not ingest.discard_failed_upload(conn, _upload_id(conn, uid)):
        raise HTTPException(status_code=409, detail="Only a failed upload whose file was kept can be discarded")
    return OkOut()


@protected.get("/duplicates", response_model=SupersededPage)
def list_duplicates(
    reason: Literal["replaced", "lower_quality", ""] | None = None,
    cursor: str | None = None,
    limit: int = Query(default=60, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> SupersededPage:
    conds: list[str] = []
    params: list[Any] = []
    if reason:
        conds.append("s.reason = ?")
        params.append(reason)
    total = conn.execute(f"SELECT COUNT(*) FROM superseded s {_where(conds)}", params).fetchone()[0]
    after, after_params = _desc_after("s.created_at", _keyset(cursor, 2), "s.id")
    rows = conn.execute(
        "SELECT s.*, up.name AS uploader_name FROM superseded s LEFT JOIN uploaders up ON up.id = s.uploader_id "
        f"{_where([*conds, after])} ORDER BY s.created_at DESC, s.id DESC LIMIT ?",
        [*params, *after_params, limit + 1],
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    kept_ids = sorted({r["media_id"] for r in rows if r["media_id"] is not None})
    kept: dict[int, sqlite3.Row] = {}
    if kept_ids:
        ph = ",".join("?" * len(kept_ids))
        kept = {
            r["id"]: r
            for r in conn.execute(
                f"SELECT {MEDIA_ITEM_COLUMNS}, m.size_bytes AS kept_size FROM media m WHERE m.id IN ({ph})", kept_ids
            )
        }
    items: list[SupersededItem] = []
    for r in rows:
        k = kept.get(r["media_id"]) if r["media_id"] is not None else None
        items.append(SupersededItem(
            id=r["id"],
            reason=r["reason"],
            original_filename=r["original_filename"],
            mime=r["mime"],
            size_bytes=r["size_bytes"],
            width=r["width"],
            height=r["height"],
            duration_s=r["duration_s"],
            bitrate=r["bitrate"],
            detail=r["detail"],
            uploader=uploader_ref(r["uploader_id"], r["uploader_name"]) if r["uploader_name"] is not None else None,
            created_at=r["created_at"],
            thumb_url=superseded_thumb_url(r["id"]) if r["thumb_path"] else None,
            file_url=superseded_file_url(r["id"]) if r["path"] else None,
            kept=media_item(k) if k is not None else None,
            kept_size_bytes=k["kept_size"] if k is not None else None,
        ))
    nxt = encode_cursor([rows[-1]["created_at"], rows[-1]["id"]]) if more and rows else None
    return SupersededPage(items=items, next_cursor=nxt, total=total)


def _superseded_row(conn: sqlite3.Connection, superseded_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT id, path FROM superseded WHERE id = ?", (superseded_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Duplicate not found")
    return row


@protected.post("/duplicates/{superseded_id}/restore", response_model=OkOut)
def restore_duplicate(superseded_id: int, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    row = _superseded_row(conn, superseded_id)
    if not row["path"]:
        raise HTTPException(status_code=409, detail="This file was deleted, so it cannot be restored")
    jobs.enqueue(conn, "restore", {"superseded_id": int(row["id"])}, unique=True)
    return OkOut()


@protected.delete("/duplicates/{superseded_id}", response_model=OkOut)
def purge_duplicate(superseded_id: int, conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    from app import ingest

    row = _superseded_row(conn, superseded_id)
    ingest.purge_superseded(conn, int(row["id"]))
    return OkOut()


@protected.post("/recluster", response_model=OkOut)
def recluster(conn: sqlite3.Connection = Depends(get_db)) -> OkOut:
    jobs.enqueue(conn, "recluster", {}, unique=True)
    return OkOut()


router.include_router(protected)
