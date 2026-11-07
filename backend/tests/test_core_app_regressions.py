"""Regression tests for the app-level review fixes: the session cookie slides with the row, the
OpenAPI schema needs a session, oversized simple uploads are refused before they are read, and the
listing and job indexes exist."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app import db
from app.auth import sessions


def test_openapi_needs_a_session(anon_client):
    assert anon_client.get("/api/openapi.json").status_code == 401


def test_openapi_is_served_to_the_owner(client):
    r = client.get("/api/openapi.json")
    assert r.status_code == 200
    assert "/api/items" in r.json()["paths"]


def test_session_cookie_is_reissued_when_the_session_slides(client):
    # Fresh login: the row was touched within the minute, so nothing is re-sent.
    r = client.get("/api/auth/state")
    assert r.status_code == 200
    assert "set-cookie" not in r.headers
    # Age the row past the touch interval: the next request slides it and sends the cookie again.
    old = (datetime.now(timezone.utc) - timedelta(seconds=sessions.TOUCH_INTERVAL_S + 5)).isoformat()
    with db.connect() as conn, db.tx(conn):
        conn.execute("UPDATE sessions SET last_seen_at = ?", (old,))
    r = client.get("/api/auth/state")
    assert r.status_code == 200
    cookie = r.headers.get("set-cookie", "")
    assert cookie.startswith(sessions.COOKIE + "=")
    assert f"Max-Age={sessions.settings.session_days * 86400}" in cookie
    assert "HttpOnly" in cookie
    # The same token is kept: the client stays signed in with its existing cookie value.
    assert client.get("/api/auth/state").status_code == 200


def test_logout_response_is_not_overwritten_by_the_refresh(client):
    old = (datetime.now(timezone.utc) - timedelta(seconds=sessions.TOUCH_INTERVAL_S + 5)).isoformat()
    with db.connect() as conn, db.tx(conn):
        conn.execute("UPDATE sessions SET last_seen_at = ?", (old,))
    r = client.post("/api/auth/logout", headers={"X-Requested-With": "pcc"})
    assert r.status_code in (200, 204)
    cookies = r.headers.get_list("set-cookie")
    assert len(cookies) == 1 and ("Max-Age=0" in cookies[0] or "expires=" in cookies[0].lower())
    assert client.get("/api/auth/state").json()["authenticated"] is False


def test_simple_upload_rejects_oversized_bodies_before_reading_them(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "simple_upload_max_bytes", 1024)
    headers = {"X-Requested-With": "pcc", "content-type": "multipart/form-data; boundary=x", "content-length": str(10 * 1024 * 1024)}
    r = client.post("/api/uploads/simple", headers=headers, content=b"")
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]
    # Without a declared length the route does not spool an unbounded body.
    r = client.post("/api/uploads/simple", headers={"X-Requested-With": "pcc", "content-type": "multipart/form-data; boundary=x", "transfer-encoding": "chunked"}, content=iter([b""]))
    assert r.status_code == 411


def test_indexes_for_listing_order_and_unique_jobs(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"items_order", "items_added", "jobs_dedup"} <= names
    plan = " ".join(
        r[3] for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM items i WHERE i.deleted_at IS NULL AND i.nsfw = 0 "
            "ORDER BY COALESCE(i.taken_at, i.created_at) DESC, i.id DESC LIMIT 60"
        )
    )
    assert "items_order" in plan and "TEMP B-TREE" not in plan
    plan = " ".join(
        r[3] for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM items i WHERE i.deleted_at IS NULL AND i.nsfw = 0 AND i.misclick = 0 "
            "ORDER BY i.created_at DESC, i.id DESC LIMIT 60"
        )
    )
    assert "items_added" in plan and "TEMP B-TREE" not in plan
    plan = " ".join(
        r[3] for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM jobs WHERE type = 'analyze' AND payload = '{}' AND status IN ('queued', 'running')"
        )
    )
    assert "jobs_dedup" in plan
