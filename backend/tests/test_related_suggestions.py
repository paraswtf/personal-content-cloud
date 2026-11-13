"""Related tags suggested (section 10, app.analysis.related): from the items the owner marked as related to someone,
the items that look like one stand out against its own baseline ('alike'), or somewhat like it at the same moment or
place ('moment', 'place'); never one already tagged, one they are in, one the owner turned down, a trashed or vault
item; the routes, "Not related" with undo, the viewer's list, and merges carrying the turn-downs. SigLIP vectors are
written straight into item_embeddings (the models are off)."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import cluster, related
from app.db import bump_search_version, tx
from app.search import service
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person

RNG = np.random.default_rng(23)
DIM = 768
SEED_DIR = RNG.standard_normal(DIM).astype(np.float32)
SEED_DIR /= np.linalg.norm(SEED_DIR)


def toward(sim: float, seed: int) -> np.ndarray:
    """A unit vector with cosine about `sim` to SEED_DIR."""
    noise = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    noise -= noise.dot(SEED_DIR) * SEED_DIR
    noise /= np.linalg.norm(noise)
    v = sim * SEED_DIR + float(np.sqrt(1 - sim * sim)) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


def item(conn, emb: np.ndarray | None, **kw):
    row = make_item(conn, write_files=False, **kw)
    if emb is not None:
        with tx(conn):
            conn.execute("INSERT INTO item_embeddings(item_id, version, model, frame_index, embedding, created_at) "
                         "VALUES (?, 1, ?, 0, ?, ?)", (row["id"], service.current_model(), emb.tobytes(), now_iso()))
            bump_search_version(conn)
    return row


@pytest.fixture
def world(conn):
    service.reset_matrix()
    asha = make_person(conn, "Asha")
    spread = np.random.default_rng(5).normal(0.30, 0.05, 80)   # the rest of the library: 0.30 +- 0.05 alike to it
    for k, sv in enumerate(spread):
        item(conn, toward(float(np.clip(sv, 0.05, 0.45)), 100 + k), taken_at="2025-05-05T05:05:00.000Z")
    seed = item(conn, SEED_DIR, taken_at="2026-08-06T19:00:00.000Z", gps_lat=18.5244, gps_lon=73.8265)
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', 'related', ?)",
                     (seed["id"], asha, now_iso()))
    other_day = "2026-01-01T10:00:00.000Z"
    w = {
        "asha": asha, "seed": seed,
        "alike": item(conn, toward(0.62, 1), taken_at=other_day),                        # far above its baseline
        "moment": item(conn, toward(0.48, 2), taken_at="2026-08-06T20:30:00.000Z"),       # somewhat, same evening
        "place": item(conn, toward(0.48, 3), taken_at=other_day, gps_lat=18.52445, gps_lon=73.82655),
        "neither": item(conn, toward(0.48, 4), taken_at=other_day, gps_lat=19.0, gps_lon=72.8),
        "tagged": item(conn, toward(0.62, 5), taken_at=other_day),
        "in_it": item(conn, toward(0.62, 6), taken_at=other_day),
        "trashed": item(conn, toward(0.62, 7), taken_at=other_day, deleted_at=now_iso()),
        "vault": item(conn, toward(0.62, 8), taken_at=other_day, nsfw=1),
    }
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', 'in', ?)",
                     (w["tagged"]["id"], asha, now_iso()))
    make_face(conn, w["in_it"]["id"], asha, write_crop=False)
    return w


def _visible():
    from app.api.common import visible_sql

    return visible_sql(vault=False, misclicks="exclude", dups="collapse")


def test_alike_moment_and_place(conn, world):
    w = world
    got = {s.item_id: s.reason for s in related.suggest(conn, w["asha"], visible=_visible())}
    assert got == {w["alike"]["id"]: "alike", w["moment"]["id"]: "moment", w["place"]["id"]: "place"}
    first = related.suggest(conn, w["asha"], visible=_visible())[0]
    assert first.item_id == w["alike"]["id"] and first.because == w["seed"]["id"]


def test_the_same_batch_counts_as_the_same_moment_without_capture_times(conn, world):
    w = world
    loose = item(conn, toward(0.48, 9))  # no capture time, a few ids after the related item
    seed_created = conn.execute("SELECT created_at FROM items WHERE id = ?", (w["seed"]["id"],)).fetchone()[0]
    from datetime import datetime, timedelta

    def saved(seconds: int) -> None:
        t = datetime.fromisoformat(seed_created.replace("Z", "+00:00")) + timedelta(seconds=seconds)
        with tx(conn):
            conn.execute("UPDATE items SET created_at = ? WHERE id = ?", (t.isoformat().replace("+00:00", "Z"), loose["id"]))

    saved(3600)                                               # saved an hour later: another batch
    assert loose["id"] not in {s.item_id for s in related.suggest(conn, w["asha"], visible=_visible())}
    saved(30)                                                 # saved with it
    got = {s.item_id: s.reason for s in related.suggest(conn, w["asha"], visible=_visible())}
    assert got.get(loose["id"]) == "moment"


def test_routes_dismiss_and_undo(client, conn, world):
    w = world
    r = client.get(f"/api/people/{w['asha']}/related-suggestions")
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert [i["item"]["uid"] for i in items][0] == w["alike"]["uid"] and items[0]["because_uid"] == w["seed"]["uid"]
    d = client.post(f"/api/people/{w['asha']}/related-suggestions/dismiss", json={"uids": [w["alike"]["uid"]]})
    assert d.status_code == 200, d.text
    uids = [i["item"]["uid"] for i in client.get(f"/api/people/{w['asha']}/related-suggestions").json()["items"]]
    assert w["alike"]["uid"] not in uids
    assert client.post(f"/api/actions/{d.headers['x-action-id']}/undo").status_code == 200
    uids = [i["item"]["uid"] for i in client.get(f"/api/people/{w['asha']}/related-suggestions").json()["items"]]
    assert w["alike"]["uid"] in uids
    # added as related: it becomes one of hers, and no longer a suggestion
    t = client.post("/api/items/people/tag", json={"uids": [w["alike"]["uid"]], "person_id": w["asha"], "relation": "related"})
    assert t.status_code == 200 and t.json()["updated"] == 1
    uids = [i["item"]["uid"] for i in client.get(f"/api/people/{w['asha']}/related-suggestions").json()["items"]]
    assert w["alike"]["uid"] not in uids


def test_the_viewer_lists_who_an_item_may_be_related_to(client, conn, world):
    w = world
    out = client.get(f"/api/items/{w['alike']['uid']}/related-suggestions").json()["people"]
    assert [(p["person"]["id"], p["reason"], p["because_uid"]) for p in out] == [(w["asha"], "alike", w["seed"]["uid"])]
    assert client.get(f"/api/items/{w['neither']['uid']}/related-suggestions").json() == {"people": []}
    assert client.get(f"/api/items/{w['in_it']['uid']}/related-suggestions").json() == {"people": []}
    assert client.get(f"/api/items/{w['vault']['uid']}/related-suggestions").status_code == 403
    client.post(f"/api/people/{w['asha']}/related-suggestions/dismiss", json={"uids": [w["alike"]["uid"]]})
    assert client.get(f"/api/items/{w['alike']['uid']}/related-suggestions").json() == {"people": []}


def test_a_merge_carries_the_turn_downs(conn, world):
    w = world
    other = make_person(conn, "Asha K")
    with tx(conn):
        related.dismiss(conn, other, [w["alike"]["id"]])
        cluster.merge_persons(conn, w["asha"], [other])
    assert conn.execute("SELECT person_id FROM related_dismissals WHERE item_id = ?", (w["alike"]["id"],)).fetchall()[0][0] == w["asha"]
    assert w["alike"]["id"] not in {s.item_id for s in related.suggest(conn, w["asha"], visible=_visible())}


def test_nothing_without_related_items_or_vectors(conn):
    service.reset_matrix()
    asha = make_person(conn, "Asha")
    assert related.suggest(conn, asha, visible=_visible()) == []
    seed = item(conn, None)
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', 'related', ?)",
                     (seed["id"], asha, now_iso()))
    assert related.suggest(conn, asha, visible=_visible()) == []


def test_meters_and_moments():
    assert related.meters(18.5244, 73.8265, 18.52445, 73.82655) < 10
    assert 1000 < related.meters(18.5244, 73.8265, 18.5344, 73.8265) < 1200
    a = related._Meta(taken=1000.0, created=None, lat=None, lon=None)
    b = related._Meta(taken=1000.0 + related.MOMENT_S + 1, created=None, lat=None, lon=None)
    assert not related.same_moment(a, b, 1, 2)


def test_pets_get_suggestions_and_a_pet_in_the_picture_none(conn, world):
    w = world
    with tx(conn):
        chutney = cluster.new_person(conn, "Chutney", kind="pet", species="cat")
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', 'related', ?)",
                     (w["seed"]["id"], chutney, now_iso()))
        conn.execute("INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, created_at) "
                     "VALUES (?, ?, 'user', 0.1, 0.1, 0.5, 0.5, 0.9, 'cat', ?)", (w["alike"]["id"], chutney, now_iso()))
    got = {s.item_id for s in related.suggest(conn, chutney, visible=_visible())}
    assert w["alike"]["id"] not in got and w["moment"]["id"] in got     # she is in that one, through her pet box


def test_mis_clicks_and_duplicate_copies_are_not_suggested(conn, world):
    w = world
    copy = item(conn, toward(0.62, 11), taken_at="2026-01-01T10:00:00.000Z")
    with tx(conn):
        conn.execute("UPDATE items SET misclick = 1 WHERE id = ?", (w["alike"]["id"],))
        conn.execute("UPDATE items SET dup_group_id = ?, dup_role = 'best' WHERE id = ?", (w["place"]["id"], w["place"]["id"]))
        conn.execute("UPDATE items SET dup_group_id = ?, dup_role = 'copy' WHERE id = ?", (w["place"]["id"], copy["id"]))
    got = {s.item_id for s in related.suggest(conn, w["asha"], visible=_visible())}
    assert w["alike"]["id"] not in got and copy["id"] not in got and w["place"]["id"] in got


def test_a_forwarded_related_item_still_counts(client, conn, world):
    w = world
    with tx(conn):
        conn.execute("UPDATE items SET forward = 1 WHERE id = ?", (w["seed"]["id"],))
    uids = [i["item"]["uid"] for i in client.get(f"/api/people/{w['asha']}/related-suggestions").json()["items"]]
    assert w["alike"]["uid"] in uids


def test_the_vault_side_counts_only_for_itself(vault_client, conn, world):
    w = world
    vault_seed = item(conn, toward(0.95, 12), taken_at="2026-03-03T03:03:00.000Z", nsfw=1)
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', 'related', ?)",
                     (vault_seed["id"], w["asha"], now_iso()))
    # the vault listing: vault items only, from the related items in the vault
    r = vault_client.get(f"/api/people/{w['asha']}/related-suggestions?vault=1")
    assert r.status_code == 200, r.text
    assert [i["item"]["uid"] for i in r.json()["items"]] == [w["vault"]["uid"]]
    # the vault item in the viewer: only the related items in the vault count
    out = vault_client.get(f"/api/items/{w['vault']['uid']}/related-suggestions").json()["people"]
    assert [(p["person"]["id"], p["because_uid"]) for p in out] == [(w["asha"], vault_seed["uid"])]
    # and the usual listing never names the vault's related item
    becauses = {i["because_uid"] for i in vault_client.get(f"/api/people/{w['asha']}/related-suggestions").json()["items"]}
    assert vault_seed["uid"] not in becauses
