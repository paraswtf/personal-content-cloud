"""Body traits through the API (section 34): traits on a photo's person boxes, the crop route, a person's traits
and trait count, hiding and showing a trait again (journalled, undoable, limited to the photo), the guess evidence
URLs, the basis filter of the review page, and the traits runner woken by box changes. Models are off
(PCC_NO_MODELS=1).

The hide rules themselves (a, b, c) and person_groups belong to app.analysis.trait_match (group C). Where a case
needs them, small fakes below read the hides the route wrote, so these tests check the route's wiring; the
rules are tested in test_trait_match.py."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from app import undo
from app.analysis import trait_match
from app.analysis import traits as atraits
from app.api import common
from app.api.deps import get_db
from app.db import connect, tx
from tests.helpers import (
    make_body,
    make_face,
    make_item,
    make_person,
    make_trait,
    set_traits_ready,
    tattoo_crop,
)

REGION = (0.2, 0.4, 0.05, 0.1)
BODY = (0.1, 0.1, 0.3, 0.6)


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(384).astype(np.float32)


def _photo(conn, **kw):
    return make_item(conn, width=800, height=600, **kw)


def _tattoo(conn, body_id: int, seed: int = 1, **kw) -> int:
    return make_trait(conn, body_id, "tattoo", "forearm_l", vec=_vec(seed), detail=tattoo_crop(seed), **kw)


def _nails(conn, body_id: int, **kw) -> int:
    attrs = {"color": "red", "hex": "#c0203a", "finish": "glitter", "length": "long", "lab": [45.0, 60.0, 30.0]}
    return make_trait(conn, body_id, "nails", "hand_r", attrs=attrs, box=(0.3, 0.6, 0.04, 0.04), **kw)


def _hides(conn, item_id: int | None = None) -> list[sqlite3.Row]:
    if item_id is None:
        return conn.execute("SELECT * FROM trait_hides ORDER BY id").fetchall()
    return conn.execute("SELECT * FROM trait_hides WHERE item_id = ? ORDER BY id", (item_id,)).fetchall()


# ------------------------------------------------------------------ fakes for group C's hide rules
def _fake_hidden_ids(conn, rows, hides=None):
    """Hidden: a not_trait / not_theirs hide made on the row, or a look-alike hit without a 'shown' hide."""
    out = set()
    for r in rows:
        tid = int(r["id"])
        own = conn.execute("SELECT 1 FROM trait_hides WHERE src_trait_id = ? AND verdict != 'shown'", (tid,)).fetchone()
        if own or _fake_lookalike(conn, tid):
            out.add(tid)
    return out


def _fake_lookalike(conn, trait_id):
    hit = conn.execute("SELECT 1 FROM trait_hide_hits WHERE trait_id = ?", (trait_id,)).fetchone()
    shown = conn.execute("SELECT 1 FROM trait_hides WHERE src_trait_id = ? AND verdict = 'shown'", (trait_id,)).fetchone()
    return bool(hit) and not shown


def _fake_covering(conn, trait_id):
    item = conn.execute("SELECT item_id FROM body_traits WHERE id = ?", (trait_id,)).fetchone()[0]
    return [int(r[0]) for r in conn.execute(
        "SELECT id FROM trait_hides WHERE item_id = ? AND src_trait_id = ? AND verdict != 'shown'", (item, trait_id))]


@pytest.fixture()
def hide_rules(monkeypatch):
    calls: list[list[int]] = []
    monkeypatch.setattr(trait_match, "hidden_ids", _fake_hidden_ids)
    monkeypatch.setattr(trait_match, "hides_covering", _fake_covering)
    monkeypatch.setattr(trait_match, "hidden_by_lookalike", _fake_lookalike)
    monkeypatch.setattr(trait_match, "after_hide_change", lambda conn, ids: calls.append(sorted(ids)) or 0)
    return calls


@pytest.fixture()
def journaled(conn, monkeypatch):
    """trait_hides is journalled. Group B adds it to undo.JOURNALED along with the after_undo hook; until that
    lands this fixture adds it, so the undo cases test this route either way."""
    if "trait_hides" not in undo.JOURNALED:
        monkeypatch.setattr(undo, "JOURNALED", (*undo.JOURNALED, "trait_hides"))
        undo.install_triggers(conn)
        conn.commit()


@pytest.fixture()
def traced(app):
    """Every SQL statement the API runs, from the request connections."""
    statements: list[str] = []

    def db():
        c = connect()
        c.set_trace_callback(statements.append)
        try:
            yield c
        finally:
            c.close()

    app.dependency_overrides[get_db] = db
    yield statements
    app.dependency_overrides.pop(get_db, None)


# ------------------------------------------------------------------ item detail
def test_a_photos_person_boxes_carry_their_traits_without_blobs(client, conn, traced, hide_rules):
    item = _photo(conn)
    body = make_body(conn, item["id"], box=BODY)
    bare = make_body(conn, item["id"], box=(0.6, 0.1, 0.3, 0.6))
    tattoo = _tattoo(conn, body)
    nails = _nails(conn, body)
    set_traits_ready(conn)
    traced.clear()
    r = client.get(f"/api/items/{item['uid']}")
    assert r.status_code == 200, r.text
    bodies = {b["id"]: b for b in r.json()["bodies"]}
    assert bodies[bare]["traits"] == []
    got = {t["id"]: t for t in bodies[body]["traits"]}
    assert set(got) == {tattoo, nails}
    assert got[tattoo] == {"id": tattoo, "kind": "tattoo", "part": "forearm_l", "x": 0.2, "y": 0.4, "w": 0.05,
                           "h": 0.1, "label": "Tattoo on the left forearm", "crop_url": f"/files/trait/{tattoo}",
                           "hidden": False, "hex": None, "item_uid": None, "taken_at": None}
    assert got[nails]["hex"] == "#c0203a" and got[nails]["label"] == "Red nail polish (glitter), long"
    trait_sql = [s for s in traced if "body_traits" in s]
    assert trait_sql, "the traits were read"
    for s in trait_sql:
        for blob in ("crop", "detail", "vec", "*"):
            assert f"t.{blob}" not in s and f" {blob}," not in s, s


def test_detail_reads_the_rows_once_and_keeps_hidden_ones_marked(client, conn, monkeypatch, hide_rules):
    item = _photo(conn)
    a, b = make_body(conn, item["id"], box=BODY), make_body(conn, item["id"], box=(0.6, 0.1, 0.3, 0.6))
    ta, tb = _tattoo(conn, a), _tattoo(conn, b, seed=2)
    with tx(conn):
        conn.execute("INSERT INTO trait_hides(item_id, kind, verdict, x, y, w, h, src_trait_id, created_at) "
                     "VALUES (?, 'tattoo', 'not_trait', 0.2, 0.4, 0.05, 0.1, ?, 'now')", (item["id"], tb))
    calls = {"rows": 0, "hidden": 0}
    real_rows = atraits.rows_for_bodies

    def rows(conn, ids):
        calls["rows"] += 1
        return real_rows(conn, ids)

    def hidden(conn, rs, hides=None):
        calls["hidden"] += 1
        return _fake_hidden_ids(conn, rs, hides)

    monkeypatch.setattr(atraits, "rows_for_bodies", rows)
    monkeypatch.setattr(trait_match, "hidden_ids", hidden)
    out = {x["id"]: x for x in client.get(f"/api/items/{item['uid']}").json()["bodies"]}
    assert calls == {"rows": 1, "hidden": 1}
    assert [t["hidden"] for t in out[a]["traits"]] == [False] and out[a]["traits"][0]["id"] == ta
    assert [t["hidden"] for t in out[b]["traits"]] == [True]


def test_a_row_measured_on_an_old_box_is_not_shown(client, conn, hide_rules):
    item = _photo(conn)
    body = make_body(conn, item["id"], box=BODY)
    _tattoo(conn, body)
    with tx(conn):
        conn.execute("UPDATE bodies SET x = 0.15 WHERE id = ?", (body,))
    assert client.get(f"/api/items/{item['uid']}").json()["bodies"][0]["traits"] == []


def test_a_trashed_photo_shows_no_traits_and_a_video_none_at_all(client, conn, hide_rules):
    item = _photo(conn)
    _tattoo(conn, make_body(conn, item["id"], box=BODY))
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-11-01T00:00:00Z' WHERE id = ?", (item["id"],))
    assert client.get(f"/api/items/{item['uid']}").json()["bodies"][0]["traits"] == []
    clip = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", write_files=False, data=None)
    make_body(conn, clip["id"], box=BODY, frame_time_s=1.0)
    assert client.get(f"/api/items/{clip['uid']}").json()["bodies"][0]["traits"] is None


# ------------------------------------------------------------------ crop route
def test_the_crop_route_serves_the_stored_crop_with_the_crop_headers(client, conn):
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    stored = conn.execute("SELECT crop FROM body_traits WHERE id = ?", (tid,)).fetchone()[0]
    r = client.get(f"/files/trait/{tid}")
    assert r.status_code == 200 and r.content == bytes(stored)
    assert r.headers["content-type"] == "image/webp"
    assert r.headers["cache-control"] == "private, max-age=86400"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert client.head(f"/files/trait/{tid}").status_code == 200


def test_the_crop_route_answers_404_for_a_missing_row_or_a_trashed_photo(client, conn):
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    assert client.get(f"/files/trait/{tid + 100}").status_code == 404
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-11-01T00:00:00Z' WHERE id = ?", (item["id"],))
    assert client.get(f"/files/trait/{tid}").status_code == 404


def test_the_crop_route_keeps_the_vault(client, conn):
    item = _photo(conn, nsfw=1, category="nsfw")
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    r = client.get(f"/files/trait/{tid}")
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    assert client.post("/api/auth/vault/unlock", json={"password": "correct horse battery staple"}).status_code == 200
    r = client.get(f"/files/trait/{tid}")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_the_crop_route_needs_a_login(anon_client, conn):
    tid = _tattoo(conn, make_body(conn, _photo(conn)["id"], box=BODY))
    assert anon_client.get(f"/files/trait/{tid}").status_code == 401


# ------------------------------------------------------------------ a person's traits
def test_a_persons_traits_come_from_person_groups_with_the_vault_state(client, conn, monkeypatch):
    pid = make_person(conn, "Kabir")
    item = _photo(conn)
    make_face(conn, item["id"], pid, assigned_by="user")
    seen: list[tuple[int, bool]] = []
    tattoo = {"id": 7, "kind": "tattoo", "part": "forearm_l", "x": 0.2, "y": 0.4, "w": 0.05, "h": 0.1,
              "label": "Tattoo on the left forearm", "crop_url": "/files/trait/7"}

    def groups(conn, person_id, *, vault_open):
        seen.append((person_id, vault_open))
        return {"tattoos": [{"kind": "tattoo", "part": "forearm", "label": "Tattoo on the forearm", "photos": 3,
                             "first_seen": "2025-03-01T10:00:00Z", "last_seen": "2026-01-11T07:09:00Z",
                             "best": tattoo, "item_uids": [item["uid"]]}],
                "mehndi": [], "nails": [dict(tattoo, id=9, kind="nails", part="hand_r", hex="#c0203a",
                                             label="Red nail polish", crop_url="/files/trait/9")]}

    monkeypatch.setattr(trait_match, "person_groups", groups)
    r = client.get(f"/api/people/{pid}/traits")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["tattoos"][0]["photos"] == 3 and out["tattoos"][0]["best"]["crop_url"] == "/files/trait/7"
    assert out["tattoos"][0]["best"]["hidden"] is False and out["nails"][0]["hex"] == "#c0203a"
    assert seen == [(pid, False)]
    client.post("/api/auth/vault/unlock", json={"password": "correct horse battery staple"})
    client.get(f"/api/people/{pid}/traits")
    assert seen[-1] == (pid, True)


def test_person_traits_404_for_unknown_or_vault_only_people_and_empty_for_pets(client, conn, monkeypatch):
    monkeypatch.setattr(trait_match, "person_groups", lambda *a, **k: pytest.fail("not for these"))
    assert client.get("/api/people/999/traits").status_code == 404
    hidden = make_person(conn, "Private")
    make_face(conn, _photo(conn, nsfw=1, category="nsfw")["id"], hidden, assigned_by="user")
    assert client.get(f"/api/people/{hidden}/traits").status_code == 404
    pet = make_person(conn, "Bruno")
    with tx(conn):
        conn.execute("UPDATE persons SET kind = 'pet', species = 'dog' WHERE id = ?", (pet,))
    r = client.get(f"/api/people/{pet}/traits")
    assert r.status_code == 200 and r.json() == {"tattoos": [], "mehndi": [], "nails": []}


def test_trait_count_is_on_one_person_only(client, conn, monkeypatch):
    pid = make_person(conn, "Kabir")
    make_face(conn, _photo(conn)["id"], pid, assigned_by="user")
    seen: list[bool] = []

    def count(conn, person_id, *, vault_open=False):
        seen.append(vault_open)
        return 4 if person_id == pid else 0

    monkeypatch.setattr(trait_match, "trait_count", count)
    assert client.get(f"/api/people/{pid}").json()["trait_count"] == 4
    assert seen == [False]   # the vault is locked: its photos do not count
    listed = client.get("/api/people").json()["items"]
    assert [p["trait_count"] for p in listed] == [None]


# ------------------------------------------------------------------ hide, show again, undo
def test_not_a_tattoo_hides_the_row_keeps_its_descriptors_and_undo_puts_it_back(client, conn, hide_rules, journaled):
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    src = conn.execute("SELECT vec, detail FROM body_traits WHERE id = ?", (tid,)).fetchone()
    r = client.put(f"/api/traits/{tid}", json={"hidden": True})
    assert r.status_code == 200, r.text
    assert r.json()["hidden"] is True and r.json()["id"] == tid
    action = r.headers["x-action-id"]
    [hide] = _hides(conn)
    assert (hide["item_id"], hide["kind"], hide["verdict"], hide["src_trait_id"]) == (item["id"], "tattoo", "not_trait", tid)
    assert (hide["x"], hide["y"], hide["w"], hide["h"]) == pytest.approx(REGION)
    assert hide["bx"] is None
    desc = conn.execute("SELECT * FROM trait_hide_desc WHERE hide_id = ?", (hide["id"],)).fetchone()
    assert bytes(desc["vec"]) == bytes(src["vec"]) and bytes(desc["detail"]) == bytes(src["detail"])
    # the request only queues the photo; the traits runner recomputes look-alikes and re-guesses
    assert hide_rules == []
    assert [r[0] for r in conn.execute("SELECT item_id FROM trait_requeue")] == [item["id"]]
    assert atraits.drain_requeue(conn) == 1
    assert hide_rules == [[item["id"]]]
    # the journal holds the hide only: no descriptor table, no BLOB hex
    logged = conn.execute("SELECT tbl, old, new FROM undo_log WHERE action_id = ?", (action,)).fetchall()
    assert {r["tbl"] for r in logged} == {"trait_hides"}
    assert all(not {"vec", "detail", "crop"} & set(json.loads(r["new"] or "{}")) for r in logged)

    u = client.post(f"/api/actions/{action}/undo")
    assert u.status_code == 200, u.text
    assert _hides(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM trait_hide_desc").fetchone()[0] == 1   # kept for a redo
    assert client.get(f"/api/items/{item['uid']}").json()["bodies"][0]["traits"][0]["hidden"] is False
    assert client.post(f"/api/actions/{u.json()['undo_id']}/undo").status_code == 200
    [again] = _hides(conn)
    assert again["id"] == hide["id"]
    assert conn.execute("SELECT hide_id FROM trait_hide_desc").fetchone()[0] == again["id"]


def test_show_again_and_its_undo(client, conn, hide_rules, journaled):
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    client.put(f"/api/traits/{tid}", json={"hidden": True})
    [hide] = _hides(conn)
    r = client.put(f"/api/traits/{tid}", json={"hidden": False})
    assert r.status_code == 200 and r.json()["hidden"] is False
    assert _hides(conn) == []
    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    [back] = _hides(conn)
    assert back["id"] == hide["id"] and back["verdict"] == "not_trait"
    assert conn.execute("SELECT hide_id FROM trait_hide_desc").fetchone()[0] == hide["id"]   # the descriptor too


def test_not_theirs_keeps_the_body_box_and_no_descriptor(client, conn, hide_rules):
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    r = client.put(f"/api/traits/{tid}", json={"hidden": True, "verdict": "not_theirs"})
    assert r.status_code == 200 and r.json()["hidden"] is True
    [hide] = _hides(conn)
    assert hide["verdict"] == "not_theirs"
    assert (hide["bx"], hide["by"], hide["bw"], hide["bh"]) == pytest.approx(BODY)
    assert conn.execute("SELECT COUNT(*) FROM trait_hide_desc").fetchone()[0] == 0


def test_nails_and_rows_without_descriptors_hide_without_one(client, conn, hide_rules):
    item = _photo(conn)
    body = make_body(conn, item["id"], box=BODY)
    nails = _nails(conn, body)
    bare = make_trait(conn, body, "tattoo", "calf_r", box=(0.2, 0.8, 0.05, 0.1))   # no vec, no detail
    for tid in (nails, bare):
        assert client.put(f"/api/traits/{tid}", json={"hidden": True}).status_code == 200
    assert len(_hides(conn)) == 2
    assert conn.execute("SELECT COUNT(*) FROM trait_hide_desc").fetchone()[0] == 0


def test_hiding_twice_writes_one_hide(client, conn, hide_rules):
    tid = _tattoo(conn, make_body(conn, _photo(conn)["id"], box=BODY))
    for _ in range(2):
        assert client.put(f"/api/traits/{tid}", json={"hidden": True}).status_code == 200
    assert len(_hides(conn)) == 1


def test_show_again_never_removes_a_hide_on_another_photo(client, conn, hide_rules, monkeypatch):
    here, there = _photo(conn), _photo(conn)
    tid = _tattoo(conn, make_body(conn, here["id"], box=BODY))
    src = _tattoo(conn, make_body(conn, there["id"], box=BODY), seed=1)
    client.put(f"/api/traits/{tid}", json={"hidden": True})
    client.put(f"/api/traits/{src}", json={"hidden": True})
    mine, theirs = _hides(conn, here["id"])[0], _hides(conn, there["id"])[0]
    # even if the rule answered with a hide of another photo, the route only deletes this photo's
    monkeypatch.setattr(trait_match, "hides_covering", lambda conn, t: [int(mine["id"]), int(theirs["id"])])
    assert client.put(f"/api/traits/{tid}", json={"hidden": False}).status_code == 200
    assert _hides(conn, here["id"]) == [] and [h["id"] for h in _hides(conn, there["id"])] == [theirs["id"]]


def test_show_again_on_a_look_alike_adds_a_shown_hide_here_only(client, conn, hide_rules):
    here, there = _photo(conn), _photo(conn)
    tid = _tattoo(conn, make_body(conn, here["id"], box=BODY))
    src = _tattoo(conn, make_body(conn, there["id"], box=BODY), seed=3)
    client.put(f"/api/traits/{src}", json={"hidden": True})
    [theirs] = _hides(conn)
    with tx(conn):   # group C writes these hits; here the look-alike is given
        conn.execute("INSERT INTO trait_hide_hits(trait_id, hide_id) VALUES (?, ?)", (tid, theirs["id"]))
    detail = client.get(f"/api/items/{here['uid']}").json()
    assert detail["bodies"][0]["traits"][0]["hidden"] is True
    r = client.put(f"/api/traits/{tid}", json={"hidden": False})
    assert r.status_code == 200 and r.json()["hidden"] is False
    [shown] = _hides(conn, here["id"])
    assert (shown["verdict"], shown["kind"], shown["src_trait_id"]) == ("shown", "tattoo", tid)
    assert (shown["x"], shown["y"], shown["w"], shown["h"]) == pytest.approx(REGION)
    assert [h["id"] for h in _hides(conn, there["id"])] == [theirs["id"]]   # the source photo stays hidden
    assert here["id"] in {r[0] for r in conn.execute("SELECT item_id FROM trait_requeue")}
    atraits.drain_requeue(conn)
    assert [here["id"]] in hide_rules


def test_show_again_after_the_rows_own_hide_listed_it_adds_no_shown_hide(client, conn, hide_rules):
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    client.put(f"/api/traits/{tid}", json={"hidden": True})
    [own] = _hides(conn)
    with tx(conn):   # its own descriptor verifies against itself
        conn.execute("INSERT INTO trait_hide_hits(trait_id, hide_id) VALUES (?, ?)", (tid, own["id"]))
    assert client.put(f"/api/traits/{tid}", json={"hidden": False}).json()["hidden"] is False
    assert _hides(conn) == []


def test_hide_route_refusals(client, conn, hide_rules):
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    assert client.put(f"/api/traits/{tid + 50}", json={"hidden": True}).status_code == 404
    assert client.put(f"/api/traits/{tid}", json={"hidden": True, "verdict": "shown"}).status_code == 422
    vault = _photo(conn, nsfw=1, category="nsfw")
    vid = _tattoo(conn, make_body(conn, vault["id"], box=BODY))
    r = client.put(f"/api/traits/{vid}", json={"hidden": True})
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-11-01T00:00:00Z' WHERE id = ?", (item["id"],))
    assert client.put(f"/api/traits/{tid}", json={"hidden": True}).status_code == 404
    assert _hides(conn) == []


def test_a_hide_outlives_the_rows_of_a_rescan_or_a_reprocess(client, conn, hide_rules):
    item = _photo(conn)
    body = make_body(conn, item["id"], box=BODY)
    tid = _tattoo(conn, body)
    client.put(f"/api/traits/{tid}", json={"hidden": True})
    with tx(conn):   # a re-analysis drops the old bodies (and with them their trait rows and scans)
        conn.execute("DELETE FROM bodies WHERE id = ?", (body,))
    assert conn.execute("SELECT COUNT(*) FROM body_traits").fetchone()[0] == 0
    [hide] = _hides(conn)
    assert (hide["item_id"], hide["src_trait_id"]) == (item["id"], tid)
    assert conn.execute("SELECT COUNT(*) FROM trait_hide_desc").fetchone()[0] == 1


def test_a_failing_re_guess_does_not_fail_the_hide(client, conn, hide_rules, monkeypatch):
    tid = _tattoo(conn, make_body(conn, _photo(conn)["id"], box=BODY))

    def boom(conn, ids):
        raise RuntimeError("runner gone")

    monkeypatch.setattr(trait_match, "after_hide_change", boom)
    r = client.put(f"/api/traits/{tid}", json={"hidden": True})
    assert r.status_code == 200 and len(_hides(conn)) == 1


# ------------------------------------------------------------------ guesses
TRAIT_EVIDENCE = {
    "basis": "trait", "score": 0.93, "trait_llr": 5.6, "ref_item_uid": "abc", "ref_body_id": 301,
    "traits": [
        {"kind": "tattoo", "part": "forearm_l", "trait_id": 812, "ref_trait_id": 77, "inliers": 57, "llr": 4.5},
        {"kind": "nails", "part": "hand_r", "trait_id": 815, "ref_trait_id": 640, "color": "blue", "llr": 1.1},
        {"kind": "tattoo", "part": "calf_r"},
    ],
    "against": [{"kind": "tattoo", "part": "upper_arm", "llr": -0.7, "why": "bare"}],
}


def test_guess_out_adds_crop_urls_to_a_copy_without_a_query(conn):
    pid = make_person(conn, "Kabir")
    item = _photo(conn)
    bid = make_body(conn, item["id"], guess_person_id=pid, guess_score=0.93, guess_evidence=json.dumps(TRAIT_EVIDENCE))
    row = conn.execute("SELECT * FROM bodies WHERE id = ?", (bid,)).fetchone()
    refs = common.person_refs(conn, [pid])
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        g = common.guess_out(conn, row, refs)
    finally:
        conn.set_trace_callback(None)
    assert statements == []
    t = g.evidence["traits"]
    assert (t[0]["crop_url"], t[0]["ref_crop_url"]) == ("/files/trait/812", "/files/trait/77")
    assert (t[1]["crop_url"], t[1]["ref_crop_url"]) == ("/files/trait/815", "/files/trait/640")
    assert "crop_url" not in t[2] and g.evidence["against"] == TRAIT_EVIDENCE["against"]
    assert json.loads(row["guess_evidence"]) == TRAIT_EVIDENCE


def test_guess_out_leaves_clothes_evidence_as_it_is(conn):
    pid = make_person(conn, "Asha")
    ev = {"clothes": 0.81, "scene_term": 0.2, "ref_item_uid": "x", "score": 0.7}
    bid = make_body(conn, _photo(conn)["id"], guess_person_id=pid, guess_score=0.7, guess_evidence=json.dumps(ev))
    row = conn.execute("SELECT * FROM bodies WHERE id = ?", (bid,)).fetchone()
    assert common.guess_out(conn, row, common.person_refs(conn, [pid])).evidence == ev


def _guessed(conn, pid: int, basis: str | None, score: float) -> int:
    ev = {"clothes": 0.8, "score": score}
    if basis:
        ev.update(basis=basis, traits=[{"kind": "tattoo", "part": "forearm_l", "trait_id": 5, "ref_trait_id": 6}])
    return make_body(conn, _photo(conn)["id"], guess_person_id=pid, guess_score=score, guess_evidence=json.dumps(ev))


def test_the_review_page_filters_by_basis(client, conn):
    pid = make_person(conn, "Kabir")
    clothes = _guessed(conn, pid, None, 0.9)
    trait = _guessed(conn, pid, "trait", 0.8)
    both = _guessed(conn, pid, "clothes+trait", 0.7)

    def ids(basis: str | None):
        r = client.get("/api/review/guesses", params={"basis": basis} if basis else {})
        assert r.status_code == 200, r.text
        return [x["id"] for x in r.json()["items"]], r.json()["total"]

    assert ids(None) == ([clothes, trait, both], 3) == ids("all")
    assert ids("clothes") == ([clothes], 1)
    assert ids("trait") == ([trait, both], 2)
    page = client.get("/api/review/guesses", params={"basis": "trait"}).json()["items"]
    assert page[0]["guess"]["evidence"]["traits"][0]["crop_url"] == "/files/trait/5"
    assert client.get("/api/review/guesses", params={"basis": "faces"}).status_code == 422


def test_the_review_page_runs_no_query_per_trait(client, conn, traced):
    pid = make_person(conn, "Kabir")
    for i in range(3):
        _guessed(conn, pid, "trait", 0.8 - i / 10)
    traced.clear()
    client.get("/api/review/guesses")
    assert not [s for s in traced if "body_traits" in s]


# ------------------------------------------------------------------ the runner is woken by box changes
@pytest.fixture()
def kicks(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(atraits, "kick", lambda *a, **k: calls.append(a))
    return calls


def test_deleting_a_person_box_wakes_the_traits_runner(client, conn, kicks):
    item = _photo(conn)
    body = make_body(conn, item["id"], box=BODY)
    face = make_face(conn, item["id"], None)
    assert client.delete(f"/api/faces/{face}").status_code == 200
    assert kicks == []
    assert client.delete(f"/api/bodies/{body}").status_code == 200
    assert len(kicks) == 1


def test_moving_a_person_box_wakes_the_traits_runner(client, conn, kicks):
    item = _photo(conn)
    body = make_body(conn, item["id"], box=BODY)
    r = client.put(f"/api/bodies/{body}/box", json={"x": 0.12, "y": 0.1, "w": 0.3, "h": 0.4})
    assert r.status_code == 200, r.text
    assert len(kicks) == 1


def test_drawing_a_person_box_wakes_the_traits_runner(client, conn, kicks):
    item = _photo(conn)
    r = client.post(f"/api/items/{item['uid']}/people-boxes",
                    json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.8, "new_person_name": "Meera"})
    assert r.status_code == 201, r.text
    assert len(kicks) == 1


def test_a_refused_box_change_does_not_wake_the_runner(client, conn, kicks):
    assert client.delete("/api/bodies/999").status_code == 404
    item = _photo(conn)
    r = client.post(f"/api/items/{item['uid']}/people-boxes", json={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.8})
    assert r.status_code == 400
    assert kicks == []


def test_the_hide_route_verifies_nothing_in_the_request(client, conn, monkeypatch):
    """The look-alikes of a hide (SIFT) and the re-guess belong to the traits runner: the route answers even when
    they would fail, and leaves the photo in the requeue."""
    def boom(*a, **k):
        raise AssertionError("no verification inside the request")

    monkeypatch.setattr(trait_match, "after_hide_change", boom)
    monkeypatch.setattr(trait_match, "hide_hits_for_hide", boom)
    item = _photo(conn)
    tid = _tattoo(conn, make_body(conn, item["id"], box=BODY))
    r = client.put(f"/api/traits/{tid}", json={"hidden": True})
    assert r.status_code == 200 and r.json()["hidden"] is True
    assert [x[0] for x in conn.execute("SELECT item_id FROM trait_requeue")] == [item["id"]]
    r = client.put(f"/api/traits/{tid}", json={"hidden": False})
    assert r.status_code == 200 and r.json()["hidden"] is False


def test_the_review_basis_filter_survives_bad_evidence_json(client, conn):
    item = _photo(conn)
    body = make_body(conn, item["id"], box=BODY)
    pid = make_person(conn, "Kabir")
    with tx(conn):
        conn.execute("UPDATE bodies SET assigned_by = 'guess', guess_person_id = ?, guess_score = 0.9, "
                     "guess_evidence = 'not json' WHERE id = ?", (pid, body))
    for basis in ("all", "clothes", "trait"):
        r = client.get(f"/api/review/guesses?basis={basis}")
        assert r.status_code == 200, (basis, r.text)
    assert client.get("/api/review/guesses?basis=trait").json()["items"] == []
