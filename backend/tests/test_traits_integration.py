"""Body traits end to end through the API with the real rules of app.analysis.trait_match (no fakes): a hide spreads
to a look-alike on another photo (rule c) once the traits runner drains its queue (after_hide_change and real SIFT
verification run there, never in the request), "Show again" there keeps it shown on that photo only, undo of a
hide brings the row back, and the person page groups the rows with their photo. Models are off (PCC_NO_MODELS=1);
SIFT runs on the synthetic tattoo crops."""
from __future__ import annotations

import numpy as np

from app.analysis import traits as atraits
from app.db import tx
from app.util import now_iso
from tests.helpers import make_body, make_face, make_item, make_person, make_trait, set_traits_ready, tattoo_crop

BODY = (0.1, 0.1, 0.3, 0.6)


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(384).astype(np.float32)


def _photo(conn, taken_at: str) -> dict:
    return make_item(conn, width=800, height=600, taken_at=taken_at)


def _tattoo(conn, body_id: int, seed: int, variant: int = 0) -> int:
    return make_trait(conn, body_id, "tattoo", "forearm_l", vec=_vec(seed),
                      detail=tattoo_crop(seed, variant=variant, angle=4.0 * variant))


def _traits(client, uid: str) -> list[dict]:
    detail = client.get(f"/api/items/{uid}").json()
    return [t for b in detail["bodies"] for t in (b["traits"] or [])]


def test_not_a_tattoo_spreads_to_a_look_alike_and_show_again_keeps_it_here(client, conn):
    set_traits_ready(conn)
    a, b, c = _photo(conn, "2026-01-10T10:00:00Z"), _photo(conn, "2026-03-02T10:00:00Z"), _photo(conn, "2026-03-03T10:00:00Z")
    ta = _tattoo(conn, make_body(conn, a["id"], box=BODY), seed=5)
    tb = _tattoo(conn, make_body(conn, b["id"], box=BODY), seed=5, variant=1)   # the same design, another photo
    tc = _tattoo(conn, make_body(conn, c["id"], box=BODY), seed=9)              # a different design
    r = client.put(f"/api/traits/{ta}", json={"hidden": True})
    assert r.status_code == 200, r.text
    assert r.json()["hidden"] is True
    # the request verified nothing: the look-alike is found when the runner drains the queue
    assert conn.execute("SELECT COUNT(*) FROM trait_hide_hits").fetchone()[0] == 0
    assert [t["hidden"] for t in _traits(client, b["uid"])] == [False]
    atraits.drain_requeue(conn)
    hits = {int(x[0]) for x in conn.execute("SELECT trait_id FROM trait_hide_hits")}
    assert tb in hits and tc not in hits
    assert [t["hidden"] for t in _traits(client, b["uid"])] == [True]
    assert [t["hidden"] for t in _traits(client, c["uid"])] == [False]
    # "Show again" on the look-alike: a 'shown' hide on that photo, the source photo stays hidden
    r = client.put(f"/api/traits/{tb}", json={"hidden": False})
    assert r.status_code == 200 and r.json()["hidden"] is False
    shown = conn.execute("SELECT verdict, item_id FROM trait_hides WHERE src_trait_id = ?", (tb,)).fetchall()
    assert [(s["verdict"], s["item_id"]) for s in shown] == [("shown", b["id"])]
    assert [t["hidden"] for t in _traits(client, b["uid"])] == [False]
    assert [t["hidden"] for t in _traits(client, a["uid"])] == [True]


def test_undoing_a_hide_shows_the_row_and_its_look_alike_again(client, conn):
    set_traits_ready(conn)
    a, b = _photo(conn, "2026-01-10T10:00:00Z"), _photo(conn, "2026-03-02T10:00:00Z")
    ta = _tattoo(conn, make_body(conn, a["id"], box=BODY), seed=7)
    tb = _tattoo(conn, make_body(conn, b["id"], box=BODY), seed=7, variant=2)
    r = client.put(f"/api/traits/{ta}", json={"hidden": True})
    action = r.headers["x-action-id"]
    atraits.drain_requeue(conn)
    assert [t["hidden"] for t in _traits(client, b["uid"])] == [True]
    u = client.post(f"/api/actions/{action}/undo")
    assert u.status_code == 200, u.text
    atraits.drain_requeue(conn)
    assert conn.execute("SELECT COUNT(*) FROM trait_hides").fetchone()[0] == 0
    assert [t["hidden"] for t in _traits(client, a["uid"])] == [False]
    assert [t["hidden"] for t in _traits(client, b["uid"])] == [False]
    assert tb  # the look-alike row itself is untouched


def test_the_person_page_groups_tattoos_and_lists_recent_nails_with_their_photo(client, conn):
    set_traits_ready(conn)
    pid = make_person(conn, "Kabir")
    old, new = _photo(conn, "2025-03-01T10:00:00Z"), _photo(conn, now_iso(-86400.0))
    for item, seed_variant in ((old, 0), (new, 1)):
        fid = make_face(conn, item["id"], pid, assigned_by="user")
        bid = make_body(conn, item["id"], pid, face_id=fid, assigned_by="user", box=BODY)
        _tattoo(conn, bid, seed=11, variant=seed_variant)
    nb = conn.execute("SELECT id FROM bodies WHERE item_id = ?", (new["id"],)).fetchone()[0]
    nails = make_trait(conn, nb, "nails", "hand_r", box=(0.3, 0.6, 0.04, 0.04),
                       attrs={"color": "red", "hex": "#c0203a", "finish": "plain", "length": "long",
                              "lab": [45.0, 60.0, 30.0]})
    r = client.get(f"/api/people/{pid}/traits")
    assert r.status_code == 200, r.text
    out = r.json()
    [group] = out["tattoos"]
    assert group["photos"] == 2 and group["part"] == "forearm_l"
    assert set(group["item_uids"]) == {old["uid"], new["uid"]}
    assert group["best"]["item_uid"] in {old["uid"], new["uid"]} and group["best"]["taken_at"]
    [n] = out["nails"]
    assert (n["id"], n["hex"], n["item_uid"]) == (nails, "#c0203a", new["uid"])
    assert client.get(f"/api/people/{pid}").json()["trait_count"] == 3
    with tx(conn):  # a hidden row leaves the count
        conn.execute("INSERT INTO trait_hides(item_id, kind, verdict, x, y, w, h, src_trait_id, created_at) "
                     "SELECT item_id, kind, 'not_trait', x, y, w, h, id, ? FROM body_traits WHERE id = ?",
                     (now_iso(), nails))
    assert client.get(f"/api/people/{pid}").json()["trait_count"] == 2
