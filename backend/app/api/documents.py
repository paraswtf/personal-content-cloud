"""Documents API (docs/ARCHITECTURE.md section 13): the parsed "virtual documents".

Router prefix /api/documents (mount it without another prefix). Every route needs a session
(require_user); NSFW items need the unlocked vault (403 vault_locked).
GET    /api/documents?type&group&q&cursor&limit&vault -> DocumentPage   (q = FTS match on items_fts,
       newest first, keyset cursor on the item id; near-duplicate copies are listed too)
GET    /api/documents/types?vault -> [DocumentTypeOut]  (every registered type with its visible count)
GET    /api/documents/{uid} -> DocumentOut
PATCH  /api/documents/{uid} DocumentUpdate -> DocumentOut  (user corrections: a doc_type change re-parses
       with that type and regroups; fields merge, an empty string removes a key; source becomes 'user')
POST   /api/documents/{uid}/reparse -> Ok  (detection + parsing on the stored text, no OCR, user fields kept;
       409 no_text when the item has no text)
GET    /api/documents/{uid}/wallet.pkpass -> application/vnd.apple.pkpass attachment
       (422 not_a_pass for other groups, 409 wallet_not_configured)
"""
from __future__ import annotations

import logging
import re
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.api.common import ITEM_CARD_COLUMNS, document_out, item_card, valid_cursor_scalar, visible_sql
from app.api.deps import get_db, require_user, want_vault
from app.auth.sessions import SessionInfo
from app.db import tx
from app.documents import extract
from app.documents.fields import iso_valid
from app.documents.parsers import GROUPS, REGISTRY
from app.schemas import DocumentListItem, DocumentOut, DocumentPage, DocumentTypeOut, DocumentUpdate, Ok
from app.util import clean_name, decode_cursor, encode_cursor, json_dumps, json_loads, media_file_name, now_iso

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[Depends(require_user)])

FIELD_KEY = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_FIELDS = 60


def _wallet_configured() -> bool:
    try:
        from app.wallet import pkpass

        return pkpass.configured()[0]
    except Exception:
        log.warning("wallet configuration check failed", exc_info=True)
        return False


def _item(conn: sqlite3.Connection, uid: str, session: SessionInfo) -> sqlite3.Row:
    row = conn.execute(f"SELECT {ITEM_CARD_COLUMNS}, i.nsfw AS is_nsfw FROM items i WHERE i.uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    if row["is_nsfw"] and not session.vault_unlocked:
        raise HTTPException(status_code=403, detail="vault_locked")
    return row


def _doc(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()


def _doc_or_404(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row:
    doc = _doc(conn, item_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Not found")
    return doc


@router.get("", response_model=DocumentPage)
def list_documents(
    type: str | None = Query(default=None, max_length=40),
    group: str | None = Query(default=None, max_length=40),
    q: str | None = Query(default=None, max_length=200),
    cursor: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=60, ge=1, le=200),
    vault: bool = Depends(want_vault),
    conn: sqlite3.Connection = Depends(get_db),
) -> DocumentPage:
    where = [visible_sql(vault=vault, misclicks="include", dups="all", forwards="include"),
             "EXISTS (SELECT 1 FROM documents d WHERE d.item_id = i.id"
             + (" AND d.doc_type = ?" if type else "") + (" AND d.doc_group = ?" if group else "") + ")"]
    params: list = [p for p in (type, group) if p]
    expr = extract.match_query(q) if q else None
    if expr:
        where.append("i.id IN (SELECT rowid FROM items_fts WHERE items_fts MATCH ?)")
        params.append(expr)
    base = " AND ".join(where)
    total = int(conn.execute(f"SELECT COUNT(*) FROM items i WHERE {base}", params).fetchone()[0])
    page_where, page_params = base, list(params)
    if cursor:
        cur = decode_cursor(cursor)
        if not cur or len(cur) != 1 or not isinstance(cur[0], int) or not valid_cursor_scalar(cur[0]):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        page_where += " AND i.id < ?"
        page_params.append(cur[0])
    rows = conn.execute(
        f"SELECT {ITEM_CARD_COLUMNS} FROM items i WHERE {page_where} ORDER BY i.id DESC LIMIT ?",
        (*page_params, limit + 1),
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    docs: dict[int, sqlite3.Row] = {}
    if rows:
        ids = [int(r["id"]) for r in rows]
        for d in conn.execute(f"SELECT * FROM documents WHERE item_id IN ({','.join('?' * len(ids))})", ids):
            docs[int(d["item_id"])] = d
    wallet = _wallet_configured()
    items = [DocumentListItem(item=item_card(r), document=document_out(docs[int(r["id"])], r, wallet_configured=wallet))
             for r in rows if int(r["id"]) in docs]
    next_cursor = encode_cursor([int(rows[-1]["id"])]) if more and rows else None
    return DocumentPage(items=items, next_cursor=next_cursor, total=total)


@router.get("/types", response_model=list[DocumentTypeOut])
def document_types(vault: bool = Depends(want_vault), conn: sqlite3.Connection = Depends(get_db)) -> list[DocumentTypeOut]:
    counts = {r["doc_type"]: int(r["n"]) for r in conn.execute(
        f"SELECT d.doc_type, COUNT(*) AS n FROM documents d JOIN items i ON i.id = d.item_id "
        f"WHERE {visible_sql(vault=vault, misclicks='include', dups='all', forwards='include')} GROUP BY d.doc_type")}
    out = [DocumentTypeOut(type=t, group=spec.group, label=spec.label, count=counts.get(t, 0))
           for t, spec in REGISTRY.items()]
    order = {g: i for i, g in enumerate(GROUPS)}
    out.sort(key=lambda x: order.get(x.group, len(order)))
    return out


@router.get("/{uid}", response_model=DocumentOut)
def get_document(uid: str, session: SessionInfo = Depends(require_user),
                 conn: sqlite3.Connection = Depends(get_db)) -> DocumentOut:
    item = _item(conn, uid, session)
    return document_out(_doc_or_404(conn, int(item["id"])), item, wallet_configured=_wallet_configured())


def _clean_text(value: str | None, max_len: int) -> str | None:
    if value is None:
        return None
    v = clean_name(value, max_len)
    return v or None


@router.patch("/{uid}", response_model=DocumentOut)
def update_document(uid: str, body: DocumentUpdate, session: SessionInfo = Depends(require_user),
                    conn: sqlite3.Connection = Depends(get_db)) -> DocumentOut:
    item = _item(conn, uid, session)
    item_id = int(item["id"])
    if body.doc_type is not None and body.doc_type not in REGISTRY:
        raise HTTPException(status_code=422, detail="Unknown document type")
    if body.doc_date and not iso_valid(body.doc_date):
        raise HTTPException(status_code=422, detail="doc_date must be YYYY-MM-DD")
    if body.fields:
        if len(body.fields) > MAX_FIELDS or any(not FIELD_KEY.match(k) for k in body.fields):
            raise HTTPException(status_code=422, detail="Invalid field name")
    doc = _doc(conn, item_id)
    if body.doc_type is not None and (doc is None or doc["doc_type"] != body.doc_type):
        try:
            from app.learning import events

            events.record(conn, "doc_type", None, {"from": doc["doc_type"] if doc else None, "to": body.doc_type},
                          item_id=item_id)
        except Exception:  # noqa: BLE001 - learning never blocks an edit
            log.exception("learning: could not record a document type change")
        extract.reparse(conn, item_id, forced_type=body.doc_type)
    doc = _doc_or_404(conn, item_id)
    from app.documents import parsers

    with tx(conn):
        doc = _doc_or_404(conn, item_id)
        spec = parsers.get(doc["doc_type"])
        fields = json_loads(doc["fields"], {}) or {}
        for key, value in (body.fields or {}).items():
            value = (value or "").strip()
            if not value:
                fields.pop(key, None)
                continue
            old = fields.get(key) if isinstance(fields.get(key), dict) else {}
            label = (old or {}).get("label") or spec.field_labels.get(key) or key.replace("_", " ").capitalize()
            fields[key] = {"value": value[:500], "label": label, "confidence": 1.0, "source": "user"}
        order = {k: i for i, k in enumerate(spec.field_order)}
        fields = dict(sorted(fields.items(), key=lambda kv: order.get(kv[0], len(order))))
        sets = ["fields = ?", "source = 'user'", "updated_at = ?"]
        params: list = [json_dumps(fields), now_iso()]
        if body.title is not None:
            sets.append("title = ?")
            params.append(_clean_text(body.title, 200))
        if body.doc_date is not None:
            sets.append("doc_date = ?")
            params.append(body.doc_date or None)
        if body.person_name is not None:
            sets.append("person_name = ?")
            params.append(_clean_text(body.person_name, 200))
        conn.execute(f"UPDATE documents SET {', '.join(sets)} WHERE item_id = ?", (*params, item_id))
        extract.refresh_fts(conn, item_id)
    return document_out(_doc_or_404(conn, item_id), item, wallet_configured=_wallet_configured())


@router.post("/{uid}/reparse", response_model=Ok)
def reparse_document(uid: str, session: SessionInfo = Depends(require_user),
                     conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    item = _item(conn, uid, session)
    if not extract.reparse(conn, int(item["id"])):
        raise HTTPException(status_code=409, detail="no_text")
    return Ok()


@router.get("/{uid}/wallet.pkpass")
def wallet_pass(uid: str, session: SessionInfo = Depends(require_user),
                conn: sqlite3.Connection = Depends(get_db)) -> Response:
    from app.wallet import pkpass

    item = _item(conn, uid, session)
    doc = _doc_or_404(conn, int(item["id"]))
    if doc["doc_group"] != "pass":
        raise HTTPException(status_code=422, detail="not_a_pass")
    if not pkpass.configured()[0]:
        raise HTTPException(status_code=409, detail="wallet_not_configured")
    full = conn.execute("SELECT * FROM items WHERE id = ?", (int(item["id"]),)).fetchone()
    try:
        data = pkpass.build_pass(conn, full, doc)
    except pkpass.WalletNotConfigured:
        raise HTTPException(status_code=409, detail="wallet_not_configured") from None
    except pkpass.WalletError:
        log.exception("wallet pass for %s failed", uid)
        raise HTTPException(status_code=500, detail="wallet_signing_failed") from None
    stem = re.sub(r"[^A-Za-z0-9 _\-]+", "", (doc["title"] or "pass"))[:60].strip() or "pass"
    name = media_file_name(stem, ".pkpass")
    return Response(content=data, media_type=pkpass.MIME,
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "private, no-store"})
