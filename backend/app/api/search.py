"""GET /api/search (docs/ARCHITECTURE.md sections 12 and 13).

Contract:
- `router` carries no prefix; mount it with `app.include_router(search.router, prefix="/api")`.
- Query: q (1..200 characters after collapsing whitespace; blank gives 422
  'Type something to search for'), mode auto|semantic|text, category, kind, person (repeated, the
  item must contain all of them), from / to (ISO dates, both inclusive, on the capture date or
  the upload date when there is none), vault (1 searches the NSFW vault and needs an unlocked
  session, 403 vault_locked otherwise), limit 1..200 (60), offset 0..100000.
- Answer: SearchPage{query, mode, items: [SearchHit = ItemCard + score, match_time_s, snippet,
  source], total, next_offset, indexed, indexing}.
- 503 search_unavailable when the engines asked for cannot answer: mode semantic without the
  model, or both engines failing. Full-text search needs no model, so auto and text keep working.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.common import ITEM_CARD_COLUMNS, item_card
from app.api.deps import get_db, require_user, want_vault
from app.schemas import Category, ItemKind, SearchHit, SearchPage
from app.search import service as search_service

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_user)])

_IN_CHUNK = 500


def _card_rows(conn: sqlite3.Connection, ids: list[int]) -> dict[int, sqlite3.Row]:
    rows: dict[int, sqlite3.Row] = {}
    for i in range(0, len(ids), _IN_CHUNK):
        chunk = ids[i:i + _IN_CHUNK]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT {ITEM_CARD_COLUMNS} FROM items i WHERE i.id IN ({ph})", chunk):
            rows[int(r["id"])] = r
    return rows


@router.get("/search", response_model=SearchPage)
def search_items(
    q: str = Query(default="", max_length=200),
    mode: Literal["auto", "semantic", "text"] = "auto",
    category: Category | None = None,
    kind: ItemKind | None = None,
    person: list[int] = Query(default=[]),
    tag: list[str] = Query(default=[]),
    date_from: date | None = Query(default=None, alias="from"),
    date_to: date | None = Query(default=None, alias="to"),
    limit: int = Query(default=60, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=100_000),
    in_vault: bool = Depends(want_vault),
    conn: sqlite3.Connection = Depends(get_db),
) -> SearchPage:
    query = " ".join(q.split())
    if not query:
        raise HTTPException(status_code=422, detail="Type something to search for")
    try:
        result = search_service.search(
            conn, query, mode=mode, category=category, kind=kind, person_ids=person, date_from=date_from,
            date_to=date_to, vault=in_vault, limit=limit, offset=offset, tag_names=tag or None,
        )
    except search_service.SearchUnavailable:
        raise HTTPException(status_code=503, detail="search_unavailable") from None
    rows = _card_rows(conn, [int(h["id"]) for h in result.hits])
    items: list[SearchHit] = []
    for h in result.hits:
        row = rows.get(int(h["id"]))
        if row is None:
            continue  # purged between the search and this read
        items.append(SearchHit(
            **item_card(row).model_dump(), score=h["score"], match_time_s=h["match_time_s"],
            snippet=h["snippet"], source=h["source"],
        ))
    return SearchPage(
        query=query, mode=mode, items=items, total=result.total, next_offset=result.next_offset,
        indexed=result.indexed, indexing=result.indexing,
    )
