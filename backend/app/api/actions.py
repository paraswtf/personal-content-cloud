"""Undo of owner actions (docs/ARCHITECTURE.md section 33, app.undo).

  POST /api/actions/{action_id}/undo -> UndoOut {undo_id, changes, partial, items}
    404 not_found, 410 expired, 409 already_undone, 409 later_changes ({detail, blocking: [ids]}),
    409 conflict, 403 vault_locked when the action touched a vault item and the vault is locked.
    The undo is an action of its own: undoing undo_id is a redo.
"""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app import undo as undo_mod
from app.api.deps import get_db, require_user
from app.auth.sessions import SessionInfo
from app.schemas import UndoOut

router = APIRouter(prefix="/api/actions", tags=["undo"], dependencies=[Depends(require_user)])


@router.post("/{action_id}/undo", response_model=UndoOut)
def undo_action(action_id: str, conn: sqlite3.Connection = Depends(get_db),
                session: SessionInfo = Depends(require_user)):
    if not undo_mod.ACTION_ID_RE.fullmatch(action_id):
        raise HTTPException(status_code=404, detail="not_found")
    from app.api.people import _check_items_vault

    _check_items_vault(conn, session, undo_mod.item_ids(conn, action_id))
    try:
        out = undo_mod.undo(conn, action_id)
    except undo_mod.UndoError as e:
        if e.blocking:
            return JSONResponse(status_code=e.status, content={"detail": e.detail, "blocking": e.blocking})
        raise HTTPException(status_code=e.status, detail=e.detail) from None
    if out["items"]:
        from app.analysis import guess

        guess.refresh_fts(conn, set(out["items"]))
        rows = conn.execute(
            f"SELECT uid FROM items WHERE id IN ({','.join('?' * len(out['items']))})", out["items"]).fetchall()
        out["items"] = [r["uid"] for r in rows]
    return UndoOut(**out)
