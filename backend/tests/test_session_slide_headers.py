"""A response that re-issues the session cookie (the row slid, section 4) carries exactly one
Content-Length. The middleware once copied every header of a blank probe response, including its
"content-length: 0", and browsers, URLSession and h11 drop a response with two Content-Length values:
the PWA's first request after a minute's pause (an upload create, for one) failed with it."""
from __future__ import annotations

from app.auth import sessions
from app.db import tx


def test_sliding_session_response_has_one_content_length(client, conn):
    with tx(conn):
        conn.execute("UPDATE sessions SET last_seen_at = '2026-01-01T00:00:00.000Z'")   # older than TOUCH_INTERVAL_S
    r = client.get("/api/auth/state")
    assert r.status_code == 200
    assert r.headers.get_list("content-length") == [str(len(r.content))]
    cookies = r.headers.get_list("set-cookie")
    assert len(cookies) == 1 and cookies[0].startswith(sessions.COOKIE + "=")
    # the next request, inside the touch interval, re-issues nothing
    r2 = client.get("/api/auth/state")
    assert r2.headers.get_list("content-length") == [str(len(r2.content))]
    assert r2.headers.get_list("set-cookie") == []
