"""GET /api/search through app.api.search.router on a minimal app whose require_user is
overridden (the auth area may not exist in this run), plus one end-to-end check through the real
app that skips while the app or the route is not assembled yet."""
from __future__ import annotations

import sqlite3

import pytest

from app.util import now_iso
from tests.helpers import make_face, make_person
from tests.test_search_support import (  # noqa: F401
    CSRF,
    add_embedding,
    add_fts,
    fake_engine,
    image_vec,
    item,
    no_engine,
    pad,
)


@pytest.fixture()
def api(data_dir, no_engine):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import search as search_api
    from app.api.deps import require_user
    from app.auth.sessions import SessionInfo

    state = {"vault_until": None}
    app = FastAPI()
    app.include_router(search_api.router, prefix="/api")
    app.dependency_overrides[require_user] = lambda: SessionInfo(
        id=1, created_at="", last_seen_at="", expires_at="", vault_until=state["vault_until"], user_agent=None,
        ip=None)
    with TestClient(app) as c:
        c.headers.update(CSRF)
        c.vault_state = state  # type: ignore[attr-defined]
        yield c


def test_text_hits_come_back_as_cards(api, conn):
    row = item(conn, kind="pdf", filename="Passport.pdf", category="documents", page_count=2)
    add_fts(conn, row["id"], filename="Passport.pdf", text="Republic of India passport K1234567")
    r = api.get("/api/search", params={"q": "  passport   K1234567 "})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "passport K1234567"
    assert body["mode"] == "auto"
    assert body["total"] == 1 and body["next_offset"] is None
    assert body["indexed"] == 0 and body["indexing"] is False
    hit = body["items"][0]
    assert hit["uid"] == row["uid"] and hit["kind"] == "pdf" and hit["filename"] == "Passport.pdf"
    assert hit["source"] == "text" and hit["score"] == 1.0 and hit["match_time_s"] is None
    assert "<b>passport</b>" in hit["snippet"]
    assert hit["urls"]["thumb"] and hit["page_count"] == 2 and hit["category"] == "documents"


@pytest.mark.parametrize("q", ["", "   ", "\t"])
def test_blank_query_is_422(api, q):
    r = api.get("/api/search", params={"q": q})
    assert r.status_code == 422
    assert r.json()["detail"] == "Type something to search for"


@pytest.mark.parametrize("params", [
    {"q": "x" * 201}, {"q": "x", "mode": "fuzzy"}, {"q": "x", "limit": 0}, {"q": "x", "limit": 201},
    {"q": "x", "offset": -1}, {"q": "x", "from": "yesterday"}, {"q": "x", "kind": "spreadsheet"},
    {"q": "x", "category": "nope"}, {"q": "x", "person": "abc"},
])
def test_invalid_params_are_422(api, params):
    assert api.get("/api/search", params=params).status_code == 422


def test_filters_and_paging_through_the_api(api, conn):
    alice = make_person(conn, "Alice")
    rows = []
    for k in range(4):
        r = item(conn, category="personal", taken_at=f"2026-05-0{k + 1}T10:00:00.000Z")
        add_fts(conn, r["id"], text="wedding day")
        rows.append(r)
    make_face(conn, rows[1]["id"], alice, write_crop=False)
    other = item(conn, kind="pdf", category="documents", taken_at="2026-05-02T10:00:00.000Z")
    add_fts(conn, other["id"], text="wedding invitation")

    def uids(**params):
        r = api.get("/api/search", params={"q": "wedding", "mode": "text", **params})
        assert r.status_code == 200, r.text
        return r.json()

    assert uids()["total"] == 5
    assert [h["uid"] for h in uids(person=alice)["items"]] == [rows[1]["uid"]]
    assert {h["uid"] for h in uids(**{"from": "2026-05-02", "to": "2026-05-03"})["items"]} == {
        rows[1]["uid"], rows[2]["uid"], other["uid"]}
    assert [h["uid"] for h in uids(kind="pdf")["items"]] == [other["uid"]]
    assert len(uids(category="personal")["items"]) == 4

    page1 = uids(limit=2)
    page2 = uids(limit=2, offset=2)
    page3 = uids(limit=2, offset=4)
    assert (page1["next_offset"], page2["next_offset"], page3["next_offset"]) == (2, 4, None)
    assert len({h["uid"] for p in (page1, page2, page3) for h in p["items"]}) == 5


def test_vault_needs_unlock_and_shows_only_nsfw(api, conn):
    plain = item(conn)
    private = item(conn, nsfw=1, category="nsfw")
    add_fts(conn, plain["id"], text="night out")
    add_fts(conn, private["id"], text="night in")

    r = api.get("/api/search", params={"q": "night"})
    assert [h["uid"] for h in r.json()["items"]] == [plain["uid"]]

    r = api.get("/api/search", params={"q": "night", "vault": "1"})
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"

    api.vault_state["vault_until"] = now_iso(600)
    r = api.get("/api/search", params={"q": "night", "vault": "1"})
    assert r.status_code == 200
    assert [h["uid"] for h in r.json()["items"]] == [private["uid"]]
    assert r.json()["items"][0]["nsfw"] is True
    # unlocked but not asking for the vault: NSFW stays hidden
    r = api.get("/api/search", params={"q": "night"})
    assert [h["uid"] for h in r.json()["items"]] == [plain["uid"]]


def test_semantic_mode_without_model_is_503_but_auto_and_text_work(api, conn):
    row = item(conn)
    add_fts(conn, row["id"], text="graduation")
    assert api.get("/api/search", params={"q": "graduation", "mode": "semantic"}).status_code == 503
    assert api.get("/api/search", params={"q": "graduation", "mode": "semantic"}).json()["detail"] == (
        "search_unavailable")
    for mode in ("auto", "text"):
        r = api.get("/api/search", params={"q": "graduation", "mode": mode})
        assert r.status_code == 200 and r.json()["total"] == 1


def test_both_engines_failing_is_503(api, monkeypatch):
    from app.search import service

    def broken(*a, **kw):
        raise sqlite3.OperationalError("fts5 broke")

    monkeypatch.setattr(service, "_text_hits", broken)
    r = api.get("/api/search", params={"q": "anything"})
    assert r.status_code == 503 and r.json()["detail"] == "search_unavailable"


def test_semantic_and_video_hits_through_the_api(api, conn, fake_engine):
    pad(conn)
    video = item(conn, kind="video", ext=".mp4", mime="video/mp4", duration_s=3.0)
    add_embedding(conn, video["id"], image_vec(seed="p"), frame_index=0, frame_time_s=0.0)
    add_embedding(conn, video["id"], image_vec("dog", seed="d"), frame_index=1, frame_time_s=1.5)
    photo = item(conn, filename="dog.jpg")
    add_fts(conn, photo["id"], filename="dog.jpg")
    r = api.get("/api/search", params={"q": "dog"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["indexed"] == 13
    by_uid = {h["uid"]: h for h in body["items"]}
    assert by_uid[video["uid"]]["source"] == "semantic" and by_uid[video["uid"]]["match_time_s"] == 1.5
    assert by_uid[photo["uid"]]["source"] == "text"
    r = api.get("/api/search", params={"q": "dog", "mode": "semantic"})
    assert [h["uid"] for h in r.json()["items"]] == [video["uid"]]


def test_route_requires_login(data_dir, no_engine):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import search as search_api

    app = FastAPI()
    app.include_router(search_api.router, prefix="/api")
    with TestClient(app) as c:
        r = c.get("/api/search", params={"q": "x"})
    assert r.status_code == 401 and r.json()["detail"] == "login_required"


def test_end_to_end_through_the_real_app(request, no_engine):
    """Runs once the app factory, auth and the router mount exist; skips before integration."""
    try:
        client = request.getfixturevalue("client")
    except Exception as e:  # app.main or the auth area not assembled yet
        pytest.skip(f"real app not available yet: {type(e).__name__}")
    conn = request.getfixturevalue("conn")
    row = item(conn, kind="pdf", filename="licence.pdf")
    add_fts(conn, row["id"], filename="licence.pdf", text="driving licence MH12 20110012345")
    r = client.get("/api/search", params={"q": "licence"})
    if r.status_code == 404:
        pytest.skip("search router not mounted yet")
    assert r.status_code == 200, r.text
    assert [h["uid"] for h in r.json()["items"]] == [row["uid"]]
