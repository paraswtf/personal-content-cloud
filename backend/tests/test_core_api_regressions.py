"""Regression tests for the API layer review findings (C1, C2, C3, C4, C12, C15, C21, C25, C26, C27)."""
from __future__ import annotations

import pytest

from tests.helpers import (
    H,
    jpeg_bytes,
    make_face,
    make_item,
    make_person,
    noise_image,
    run_jobs,
    upload_bytes,
    upload_status,
)

RESTORED = "Restored from the trash (identical file)."


def noop(conn, *args) -> None:
    """Stand-in handler for jobs of other areas (analysis, documents)."""


@pytest.fixture(autouse=True)
def quiet_jobs(monkeypatch):
    """Only core's own handlers run here (no model is loaded by these tests)."""
    from app import worker

    for kind in ("analyze", "text", "guess"):
        monkeypatch.setitem(worker.HANDLERS, kind, (__name__, "noop", worker.HANDLERS[kind][2]))
    monkeypatch.setitem(worker.HANDLERS, "recluster", (__name__, "noop", lambda p: ()))


def conn_():
    from app.db import connect

    return connect()


def group(conn, best_id: int, copy_ids: list[int]) -> None:
    """Make a duplicate group directly (dedup ignores the vault flag, so mixed groups happen)."""
    from app.db import tx

    with tx(conn):
        conn.execute("UPDATE items SET dup_group_id = ?, dup_role = 'best' WHERE id = ?", (best_id, best_id))
        for cid in copy_ids:
            conn.execute("UPDATE items SET dup_group_id = ?, dup_role = 'copy' WHERE id = ?", (best_id, cid))


def unlock(client) -> None:
    from tests.conftest import ACCOUNT

    r = client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]})
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ C1, C2: duplicate groups across the vault
def test_c1_duplicates_and_dup_count_ignore_vault_copies_when_locked(client):
    c = conn_()
    try:
        best = make_item(c, filename="beach.jpg")
        normal_copy = make_item(c, filename="beach-small.jpg")
        vault_copy = make_item(c, nsfw=1, filename="secret.jpg")
        group(c, best["id"], [normal_copy["id"], vault_copy["id"]])
    finally:
        c.close()
    page = client.get("/api/duplicates").json()
    assert page["total"] == 1
    assert [x["uid"] for x in page["groups"][0]["copies"]] == [normal_copy["uid"]]
    cards = {x["uid"]: x for x in client.get("/api/items").json()["items"]}
    assert set(cards) == {best["uid"]}
    assert cards[best["uid"]]["dup_count"] == 1  # the vault copy is not counted


def test_c1_vault_only_copy_gives_no_group_in_the_library(client):
    c = conn_()
    try:
        best = make_item(c, filename="beach.jpg")
        vault_copy = make_item(c, nsfw=1, filename="secret.jpg")
        group(c, best["id"], [vault_copy["id"]])
    finally:
        c.close()
    page = client.get("/api/duplicates").json()
    assert page["total"] == 0 and page["groups"] == []
    assert client.get("/api/items").json()["items"][0]["dup_count"] == 0


def test_c2_mixed_groups_collapse_within_each_side(client):
    c = conn_()
    try:
        vault_best = make_item(c, nsfw=1, filename="burst1.jpg")
        normal_copy = make_item(c, filename="burst2.jpg")
        group(c, vault_best["id"], [normal_copy["id"]])
        normal_best = make_item(c, filename="beach.jpg")
        vault_copy = make_item(c, nsfw=1, filename="secret.jpg")
        group(c, normal_best["id"], [vault_copy["id"]])
    finally:
        c.close()
    page = client.get("/api/items").json()
    cards = {x["uid"]: x for x in page["items"]}
    assert set(cards) == {normal_copy["uid"], normal_best["uid"]} and page["total"] == 2
    # The normal copy under a vault best carries no vault uid.
    assert cards[normal_copy["uid"]]["dup_group"] is None and cards[normal_copy["uid"]]["dup_count"] == 0
    detail = client.get(f"/api/items/{normal_copy['uid']}").json()
    assert detail["dup_group"] is None
    unlock(client)
    vault = client.get("/api/items", params={"vault": 1}).json()
    assert {x["uid"] for x in vault["items"]} == {vault_best["uid"], vault_copy["uid"]}


def test_c2_search_filter_uses_the_same_collapse(data_dir, conn):
    from app.api.common import visible_sql

    vault_best = make_item(conn, nsfw=1)
    normal_copy = make_item(conn)
    group(conn, vault_best["id"], [normal_copy["id"]])
    ids = {int(r[0]) for r in conn.execute(
        f"SELECT i.id FROM items i WHERE {visible_sql(vault=False, misclicks='include', dups='collapse')}")}
    assert ids == {int(normal_copy["id"])}


def test_c1_vault_items_get_their_own_url_key(data_dir, conn):
    from app.api.common import item_urls

    normal = make_item(conn)
    flagged = make_item(conn, nsfw=1)
    assert item_urls(normal).thumb.endswith("?v=1")
    assert item_urls(flagged).thumb.endswith("?v=1n")
    assert item_urls(flagged, full=True).original.endswith("?v=1n")


def test_c1_files_accept_any_url_key(client):
    c = conn_()
    try:
        normal = make_item(c)
        flagged = make_item(c, nsfw=1)
    finally:
        c.close()
    url = f"/files/thumb/{normal['uid']}"
    assert client.get(url, params={"v": "1"}).headers["cache-control"] == "private, max-age=31536000, immutable"
    r = client.get(url, params={"v": "1n"})
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-cache"
    assert client.get(url, params={"v": "junk"}).status_code == 200
    assert client.get(f"/files/thumb/{flagged['uid']}", params={"v": "1n"}).status_code == 403
    unlock(client)
    r = client.get(f"/files/thumb/{flagged['uid']}", params={"v": "1n"})
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"


# ------------------------------------------------------------------ C3: upload status and the vault
def _upload_row(conn, item_id: int, client_id: str = "browser-1") -> str:
    from app.db import tx
    from app.util import new_uid, now_iso

    uid = new_uid()
    with tx(conn):
        conn.execute(
            "INSERT INTO uploads(uid, source, client_id, original_filename, size_bytes, received_bytes, status, "
            "item_id, created_at, updated_at) VALUES (?, 'web', ?, 'x.jpg', 10, 10, 'stored', ?, ?, ?)",
            (uid, client_id, item_id, now_iso(), now_iso()),
        )
    return uid


def test_c3_upload_status_hides_vault_items_while_locked(client):
    c = conn_()
    try:
        item = make_item(c, nsfw=1)
        up = _upload_row(c, item["id"])
    finally:
        c.close()
    outs = [client.get(f"/api/uploads/{up}").json(),
            client.get("/api/uploads/mine", params={"client_id": "browser-1"}).json()["uploads"][0],
            client.get("/api/uploads", params={"ids": up}).json()["uploads"][0]]
    for out in outs:
        assert out["item_uid"] is None and out["item_thumb_url"] is None and out["item_nsfw"] is False
        assert out["message"] == "Added to the library."
    unlock(client)
    out = client.get(f"/api/uploads/{up}").json()
    assert out["item_uid"] == item["uid"] and out["item_nsfw"] is True
    assert out["item_thumb_url"].endswith("?v=1n")


def test_c3_upload_status_of_a_normal_item_is_unchanged(client):
    c = conn_()
    try:
        item = make_item(c)
        up = _upload_row(c, item["id"])
    finally:
        c.close()
    out = client.get(f"/api/uploads/{up}").json()
    assert out["item_uid"] == item["uid"] and out["item_nsfw"] is False
    assert out["item_thumb_url"] == f"/files/thumb/{item['uid']}?v=1"


# ------------------------------------------------------------------ C4, C21: person covers
def test_c4_c21_person_ref_skips_a_cover_in_the_trash_or_vault(client):
    from app.urls import face_url

    c = conn_()
    try:
        pid = make_person(c, "Asha")
        shown = make_item(c)
        auto = make_face(c, shown["id"], pid, det_score=0.5)
        hidden = make_item(c)
        chosen = make_face(c, hidden["id"], pid, det_score=0.9)
        c.execute("UPDATE persons SET cover_face_id = ? WHERE id = ?", (chosen, pid))
        c.commit()
        cover = client.get(f"/api/items/{shown['uid']}").json()["faces"][0]["person"]["cover_url"]
        assert cover == face_url(chosen)  # still visible: the chosen cover is used
        for change in ("UPDATE items SET deleted_at = '2026-01-01T00:00:00.000Z' WHERE id = ?",
                       "UPDATE items SET deleted_at = NULL, nsfw = 1 WHERE id = ?"):
            c.execute(change, (hidden["id"],))
            c.commit()
            detail = client.get(f"/api/items/{shown['uid']}").json()
            assert detail["faces"][0]["person"]["cover_url"] == face_url(auto)
            assert detail["people"][0]["cover_url"] == face_url(auto)
    finally:
        c.close()


def test_c4_single_person_ref_checks_the_cover(data_dir, conn):
    from app.api.common import person_ref, person_refs
    from app.urls import face_url

    pid = make_person(conn, "Ravi")
    live = make_item(conn)
    auto = make_face(conn, live["id"], pid, det_score=0.5)
    vault = make_item(conn, nsfw=1)
    chosen = make_face(conn, vault["id"], pid, det_score=0.9)
    conn.execute("UPDATE persons SET cover_face_id = ? WHERE id = ?", (chosen, pid))
    conn.commit()
    assert person_ref(conn, pid).cover_url == face_url(auto)
    assert person_ref(conn, pid, "Ravi", chosen).cover_url == face_url(auto)
    assert person_refs(conn, [pid])[pid].cover_url == face_url(auto)


def test_c21_cover_from_trash_or_vault_is_refused(vault_client):
    c = conn_()
    try:
        pid = make_person(c, "Meera")
        live = make_item(c)
        ok_face = make_face(c, live["id"], pid)
        trashed = make_item(c, deleted_at="2026-01-01T00:00:00.000Z")
        trashed_face = make_face(c, trashed["id"], pid)
        vault = make_item(c, nsfw=1)
        vault_face = make_face(c, vault["id"], pid)
    finally:
        c.close()
    for fid in (trashed_face, vault_face):
        r = vault_client.patch(f"/api/people/{pid}", json={"cover_face_id": fid})
        assert r.status_code == 422, r.text
        assert "trash" in r.json()["detail"]
    r = vault_client.patch(f"/api/people/{pid}", json={"cover_face_id": ok_face})
    assert r.status_code == 200 and r.json()["cover_face_id"] == ok_face


# ------------------------------------------------------------------ C12: date filter edges
def test_c12_date_filter_at_the_calendar_edges(client):
    for params in ({"to": "9999-12-31"}, {"from": "0001-01-01"}, {"from": "0001-01-01", "to": "9999-12-31"}):
        r = client.get("/api/items", params=params)
        assert r.status_code == 200, (params, r.text)
    assert client.get("/api/items", params={"from": "2024-13-40"}).status_code == 422


def test_c12_library_uses_the_shared_day_helper(data_dir):
    from app.api.library import local_day_start_utc

    assert local_day_start_utc("2024-03-05") == "2024-03-04T18:30:00.000Z"
    assert local_day_start_utc("9999-12-31", next_day=True) > "9998"


# ------------------------------------------------------------------ C15: re-uploading a trashed file
def _trashed_upload_case(client, seed: int):
    from app import storage

    data = jpeg_bytes(noise_image(200, 160, seed=seed))
    up = upload_bytes(client, data, "photo.jpg")
    run_jobs()
    st = upload_status(client, up["id"])
    assert st["status"] == "stored"
    uid = st["item_uid"]
    assert client.delete(f"/api/items/{uid}").status_code == 200
    c = conn_()
    try:
        row = c.execute("SELECT version, ext FROM items WHERE uid = ?", (uid,)).fetchone()
    finally:
        c.close()
    original = storage.original_path(uid, int(row["version"]), row["ext"])
    assert original.exists()
    return data, uid, original


def _assert_restored(client, out: dict, uid: str, original) -> None:
    assert out["status"] == "duplicate", out
    assert out["message"] == RESTORED
    assert out["item_uid"] == uid and out["item_thumb_url"]
    assert client.get(f"/api/items/{uid}").json()["deleted_at"] is None
    assert uid in {x["uid"] for x in client.get("/api/items").json()["items"]}
    assert original.exists()


def test_c15_reupload_with_client_hash_restores_from_trash(client):
    data, uid, original = _trashed_upload_case(client, 11)
    out = upload_bytes(client, data, "again.jpg", send_hash=True)
    assert out["received_bytes"] == 0
    _assert_restored(client, out, uid, original)


def test_c15_chunked_reupload_restores_from_trash(client):
    data, uid, original = _trashed_upload_case(client, 12)
    out = upload_bytes(client, data, "again.jpg")
    _assert_restored(client, out, uid, original)
    run_jobs()
    _assert_restored(client, upload_status(client, out["id"]), uid, original)


def test_c15_simple_reupload_restores_from_trash(client):
    data, uid, original = _trashed_upload_case(client, 13)
    r = client.post("/api/uploads/simple", files={"file": ("again.jpg", data, "image/jpeg")}, data={"source": "api"},
                    headers=H)
    assert r.status_code == 201, r.text
    _assert_restored(client, r.json(), uid, original)


# ------------------------------------------------------------------ C25, C26: listing queries
def _plan(conn, sql: str, params: list) -> list[tuple[int, int, str]]:
    return [(int(r[0]), int(r[1]), str(r[3])) for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params)]


@pytest.mark.parametrize("sort,index", [("taken", "items_order"), ("added", "items_added")])
@pytest.mark.parametrize("vault", [False, True])
@pytest.mark.parametrize("with_cursor", [False, True])
def test_c25_listing_plan_uses_the_order_index(data_dir, conn, sort, index, vault, with_cursor):
    from app.api.common import page_sql, visible_sql
    from app.api.library import SORT_KEYS

    for _ in range(3):
        make_item(conn, write_files=False, taken_at="2024-01-01T00:00:00.000Z")
    for where in ([visible_sql(vault=vault, misclicks="exclude", dups="collapse")],
                  [visible_sql(vault=vault, misclicks="only", dups="all")]):
        after = ["2024-01-01T00:00:00.000Z", 5] if with_cursor else None
        sql, extra = page_sql(where, SORT_KEYS[sort], after)
        plan = _plan(conn, sql, [*extra, 61])
        text = "\n".join(p[2] for p in plan)
        assert "TEMP B-TREE" not in text, text
        assert f"USING INDEX {index}" in text, text
        if with_cursor:  # the cursor seeks into the index instead of walking it from the top
            assert "<?" in next(p[2] for p in plan if index in p[2]), text


def test_c25_person_items_plan_uses_the_order_index(data_dir, conn):
    from app.api.common import page_sql, visible_sql

    sql, extra = page_sql(
        [visible_sql(vault=False, misclicks="exclude", dups="all"),
         "i.id IN (SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
         "UNION SELECT item_id FROM item_tags WHERE person_id = ?)"],
        "COALESCE(i.taken_at, i.created_at)", ["2024", 3])
    plan = _plan(conn, sql, [1, 1, 1, *extra, 61])
    assert any("USING INDEX items_order" in p[2] for p in plan)


def test_c25_total_is_counted_on_the_first_page_only(data_dir, conn):
    from app.api.common import visible_sql
    from app.api.library import SORT_KEYS, _page

    for k in range(5):
        make_item(conn, write_files=False, taken_at=f"2024-01-0{k + 1}T00:00:00.000Z")
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        first = _page(conn, [visible_sql()], [], SORT_KEYS["taken"], None, 2)
        assert first.total == 5 and first.next_cursor
        assert any(s.startswith("SELECT COUNT(*) FROM items i") for s in seen)
        seen.clear()
        second = _page(conn, [visible_sql()], [], SORT_KEYS["taken"], first.next_cursor, 2)
        assert not any(s.startswith("SELECT COUNT(*) FROM items i") for s in seen)
        assert second.total == 0 and len(second.items) == 2
        third = _page(conn, [visible_sql()], [], SORT_KEYS["taken"], second.next_cursor, 2)
        assert [x.uid for x in first.items + second.items + third.items] == [
            r["uid"] for r in conn.execute("SELECT uid FROM items ORDER BY taken_at DESC, id DESC")]
    finally:
        conn.set_trace_callback(None)


def _page_steps(conn, n_more: int) -> int:
    """VM steps of the default first page after adding n_more ungrouped items and ANALYZE."""
    from app.api.common import page_sql, visible_sql
    from app.api.library import SORT_KEYS
    from app.db import tx

    with tx(conn):
        for _ in range(n_more):
            conn.execute(
                "INSERT INTO items(uid, kind, version, original_filename, ext, mime, sha256, size_bytes, category, "
                "nsfw, misclick, analysis_status, created_at, updated_at) "
                "VALUES (lower(hex(randomblob(7))), 'image', 1, 'a.jpg', '.jpg', 'image/jpeg', "
                "lower(hex(randomblob(32))), 1, 'misc', 0, 0, 'done', '2024-01-01T00:00:00.000Z', "
                "'2024-01-01T00:00:00.000Z')")
    conn.execute("ANALYZE")
    sql, extra = page_sql([visible_sql()], SORT_KEYS["taken"], None)
    steps = [0]

    def tick() -> int:
        steps[0] += 1
        return 0

    conn.set_progress_handler(tick, 100)
    try:
        assert len(conn.execute(sql, [*extra, 61]).fetchall()) == 61
    finally:
        conn.set_progress_handler(None, 0)
    return steps[0]


def test_c26_card_columns_cost_does_not_grow_with_the_library(data_dir, conn):
    # After ANALYZE on a library with no duplicates, items_dup looks useless (every dup_group_id is
    # NULL), so an unguarded dup_count subquery scans the table for each card.
    small = _page_steps(conn, 400)
    large = _page_steps(conn, 1600)
    assert large < small * 1.5, (small, large)


def test_c26_card_subqueries_run_outside_the_page_query(data_dir, conn):
    from app.api.common import page_sql, visible_sql
    from app.api.library import SORT_KEYS

    sql, extra = page_sql([visible_sql()], SORT_KEYS["taken"], None)
    plan = _plan(conn, sql, [*extra, 61])
    co = next(p[0] for p in plan if p[2].startswith("CO-ROUTINE"))
    inside: set[int] = {co}
    for node, parent, _ in plan:
        if parent in inside:
            inside.add(node)
    details = [p[2] for p in plan if p[0] in inside]
    assert not any("documents" in d or " d " in d or "items_dup" in d for d in details), details


def test_c26_dup_count_values(client):
    c = conn_()
    try:
        alone = make_item(c)
        best = make_item(c)
        copy = make_item(c)
        group(c, best["id"], [copy["id"]])
    finally:
        c.close()
    cards = {x["uid"]: x for x in client.get("/api/items", params={"dups": "all"}).json()["items"]}
    assert cards[alone["uid"]]["dup_count"] == 0 and cards[alone["uid"]]["dup_group"] is None
    assert cards[best["uid"]]["dup_count"] == 1 and cards[copy["uid"]]["dup_count"] == 1
    assert cards[copy["uid"]]["dup_group"] == best["uid"]


# ------------------------------------------------------------------ C27: person lookups per page
def _statements(conn, fn) -> int:
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        fn()
    finally:
        conn.set_trace_callback(None)
    return len(seen)


def _item_with_people(conn, n: int):
    item = make_item(conn)
    for k in range(n):
        pid = make_person(conn, f"P{n}-{k}")
        other = make_person(conn, f"G{n}-{k}")
        make_face(conn, item["id"], pid, write_crop=False)
        make_face(conn, item["id"], None, write_crop=False, guess_person_id=other, guess_score=0.7,
                  assigned_by="guess")
    return item


def test_c27_item_detail_looks_people_up_once(data_dir, conn):
    from app.api.library import DETAIL_COLUMNS, item_detail

    small = _item_with_people(conn, 2)
    large = _item_with_people(conn, 20)

    def detail(item):
        row = conn.execute(f"SELECT {DETAIL_COLUMNS} FROM items i WHERE i.id = ?", (item["id"],)).fetchone()
        out = item_detail(conn, row)
        assert len(out.faces) == len([f for f in out.faces if f.person or f.guess])
        return out

    assert _statements(conn, lambda: detail(small)) == _statements(conn, lambda: detail(large))
    out = detail(large)
    assert len(out.people) == 20 and all(f.guess.person.name.startswith("G") for f in out.faces if f.guess)


def test_c27_review_guesses_looks_people_up_once(data_dir, conn):
    from app.api.people import review_guesses

    _item_with_people(conn, 2)
    first = _statements(conn, lambda: review_guesses(cursor=None, limit=200, person=None, conn=conn))
    _item_with_people(conn, 20)
    second = _statements(conn, lambda: review_guesses(cursor=None, limit=200, person=None, conn=conn))
    assert first == second
    page = review_guesses(cursor=None, limit=200, person=None, conn=conn)
    assert len(page.items) == 22
    for g in page.items:
        assert g.box.guess is not None and g.box.guess.person.id == g.guess.person.id
        assert g.box.url == f"/files/face/{g.id}"
