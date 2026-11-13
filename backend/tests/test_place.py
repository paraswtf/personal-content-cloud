"""Photos taken together, lined up (section 35): the links table and the runner (app.analysis.place)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis import place, place_align
from app.config import settings
from app.db import tx
from app.util import now_iso
from tests.helpers import make_body, make_face, make_item, make_person
from tests.test_place_align import scene, warp

T0 = "2026-05-01T10:00:%02d.000Z"


def photo(conn, second: int, *, box: bool = True, **extra):
    """A dated photo without files; a loose face makes it take part."""
    item = make_item(conn, taken_at=T0 % second if second < 60 else "2026-05-01T10:%02d:%02d.000Z" % divmod(second, 60),
                     write_files=False, data=None, **extra)
    if box:
        make_face(conn, item["id"], None, write_crop=False)
    return item


@pytest.fixture()
def pictures(monkeypatch):
    """{uid: grey picture} read in place of the previews."""
    store: dict[str, np.ndarray] = {}
    monkeypatch.setattr(place_align, "load_gray", lambda row: store.get(row["uid"]))
    return store


def links(conn) -> dict[tuple[int, int], tuple[str, str]]:
    return {(int(r["a_item"]), int(r["b_item"])): (r["kind"], r["m"]) for r in conn.execute("SELECT * FROM place_links")}


def scene_vec(conn, item_id: int, v: np.ndarray) -> None:
    with tx(conn):
        conn.execute("INSERT INTO item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding, created_at) "
                     "VALUES (?, 1, 'test', 0, NULL, ?, ?)", (item_id, v.astype(np.float32).tobytes(), now_iso()))


def test_which_photos_are_pending_and_with_which_next_photos(conn):
    a = photo(conn, 0)
    b = photo(conn, 5)
    photo(conn, 7, box=False)                       # nobody in it: not part of a sequence
    c = photo(conn, 9)
    d = photo(conn, 9 + 400)                        # more than MAX_GAP_S after c
    photo(conn, 12, kind="video")
    photo(conn, 13, category="screenshots")
    photo(conn, 14, nsfw=1)
    photo(conn, 15, deleted_at=now_iso())
    photo(conn, 16, analysis_status="pending")
    undated = make_item(conn, write_files=False, data=None)
    make_face(conn, undated["id"], None, write_crop=False)
    body_only = photo(conn, 20, box=False)
    make_body(conn, body_only["id"], None)
    got = dict(place.pending_items(conn, 50))
    assert got == {a["id"]: [b["id"], c["id"], body_only["id"]], b["id"]: [c["id"], body_only["id"]],
                   c["id"]: [body_only["id"]], body_only["id"]: [], d["id"]: []}
    by_time = [d["id"], body_only["id"], c["id"], b["id"], a["id"]]                     # newest capture first
    assert [i for i, _ in place.pending_items(conn, 50)] == by_time
    assert [i for i, _ in place.pending_items(conn, 50, skip=[d["id"]])] == by_time[1:]


def test_the_sub_second_orders_photos_of_one_second(conn):
    late = photo(conn, 3, exif_subsec="80")
    early = photo(conn, 3, exif_subsec="15")
    assert dict(place.pending_items(conn, 10)) == {early["id"]: [late["id"]], late["id"]: []}
    assert place.capture_time(late) - place.capture_time(early) == pytest.approx(0.65)


def test_run_pending_stores_links_and_scans(conn, pictures):
    a, b = photo(conn, 0), photo(conn, 4)
    pictures[a["uid"]] = scene(1)
    pictures[b["uid"]] = warp(scene(1), np.array([[1, 0, 50], [0, 1, 0]]))
    r = place.run_pending(conn)
    assert r == {"items": 2, "pairs": 1, "aligned": 1, "complete": True}
    (kind, m), = links(conn).values()
    assert kind == "moved"
    assert place_align.apply(json.loads(m), 0.5, 0.5) == pytest.approx((0.55, 0.5), abs=0.004)
    (ln,) = place.links_for(conn, [b["id"]])
    assert (ln.a_item, ln.b_item, ln.kind) == (a["id"], b["id"], "moved") and ln.dt_s == pytest.approx(4.0)
    assert place.links_for(conn, [a["id"], b["id"]]) == [ln]          # each link once
    assert {r["item_id"]: r["sig"] for r in conn.execute("SELECT * FROM place_scans")} == {a["id"]: f"{b['id']},", b["id"]: ""}
    assert place.pending_items(conn, 10) == []
    assert place.run_pending(conn) == {"items": 0, "pairs": 0, "aligned": 0, "complete": True}
    assert not place.ready(conn)
    place.start(conn)
    assert place.ready(conn)
    p = place.progress(conn)
    assert (p["scanned"], p["eligible"], p["ready"], p["enabled"]) == (2, 2, True, True)


def test_a_pair_that_does_not_line_up_is_stored_and_not_tried_again(conn, pictures):
    a, b = photo(conn, 0), photo(conn, 4)
    pictures[a["uid"]], pictures[b["uid"]] = scene(1), scene(2)
    assert place.run_pending(conn)["aligned"] == 0
    assert links(conn) == {(a["id"], b["id"]): ("none", "")}
    assert place.links_for(conn, [a["id"], b["id"]]) == []
    described = place_align.stats()["described"]
    assert place.run_pending(conn)["items"] == 0 and place_align.stats()["described"] == described


def test_a_photo_without_pixels_is_stored_as_not_lined_up(conn, pictures):
    a, b = photo(conn, 0), photo(conn, 4)
    pictures[a["uid"]] = scene(1)                    # b has no preview at all
    assert place.run_pending(conn)["complete"]
    assert links(conn)[(a["id"], b["id"])][0] == "none"


def test_a_photo_arriving_in_the_middle_makes_its_neighbours_pending_again(conn, pictures):
    a, c = photo(conn, 0), photo(conn, 8)
    for it in (a, c):
        pictures[it["uid"]] = scene(3)
    place.run_pending(conn)
    assert set(links(conn)) == {(a["id"], c["id"])}
    b = photo(conn, 4)
    pictures[b["uid"]] = scene(3)
    assert dict(place.pending_items(conn, 10)) == {a["id"]: [b["id"], c["id"]], b["id"]: [c["id"]]}
    described = place_align.stats()["described"]
    r = place.run_pending(conn)
    assert r["pairs"] == 2 and r["complete"]                    # (a, c) is kept, not measured again
    assert place_align.stats()["described"] == described + 1    # only the new photo was read
    assert set(links(conn)) == {(a["id"], c["id"]), (a["id"], b["id"]), (b["id"], c["id"])}


def test_the_same_moment_is_no_movement(conn, pictures):
    a = photo(conn, 0)
    b = photo(conn, 1, dup_group_id=a["id"])
    with tx(conn):
        conn.execute("UPDATE items SET dup_group_id = ? WHERE id = ?", (a["id"], a["id"]))
    c = photo(conn, 2, content_id="live-1")
    d = photo(conn, 3, content_id="live-1")
    for it in (a, b, c, d):
        pictures[it["uid"]] = scene(4)
    place.run_pending(conn)
    got = links(conn)
    assert got[(a["id"], b["id"])][0] == "none" and got[(c["id"], d["id"])][0] == "none"
    assert got[(a["id"], c["id"])][0] == "still"


def test_a_burst_of_plain_pictures_counts_without_a_fit(conn, pictures):
    a, b, c = photo(conn, 0), photo(conn, 2), photo(conn, 30)
    wall = np.full((750, 1000), 150, np.uint8)
    v = np.ones(768, np.float32) / np.sqrt(768)
    for it in (a, b, c):
        pictures[it["uid"]] = wall
        scene_vec(conn, it["id"], v)
    place.run_pending(conn)
    got = links(conn)
    assert got[(a["id"], b["id"])] == ("burst", json.dumps(list(place_align.IDENTITY)))
    assert got[(a["id"], c["id"])][0] == "none" and got[(b["id"], c["id"])][0] == "none"   # too far apart for a burst


def test_photos_with_an_open_box_are_queued_for_a_fresh_guess(conn, pictures):
    p = make_person(conn, "Asha")
    a = make_item(conn, taken_at=T0 % 0, write_files=False, data=None)
    make_face(conn, a["id"], p, assigned_by="user", write_crop=False)
    b = photo(conn, 3)                               # a loose face: an open box
    c = photo(conn, 50)
    pictures[a["uid"]] = pictures[b["uid"]] = scene(5)
    pictures[c["uid"]] = scene(6)                    # lines up with nothing
    place.run_pending(conn)
    assert {int(r[0]) for r in conn.execute("SELECT item_id FROM trait_requeue")} == {b["id"]}


def test_off_means_nothing_is_measured(conn, pictures, monkeypatch):
    a, b = photo(conn, 0), photo(conn, 4)
    pictures[a["uid"]] = pictures[b["uid"]] = scene(7)
    monkeypatch.setattr(settings, "place_enabled", False)
    assert place.run_pending(conn) == {"items": 0, "pairs": 0, "aligned": 0, "complete": False}
    assert links(conn) == {} and place.progress(conn)["enabled"] is False
    place.start(conn)
    assert not place.ready(conn)


def test_links_go_with_their_photos(conn, pictures):
    a, b = photo(conn, 0), photo(conn, 4)
    pictures[a["uid"]] = pictures[b["uid"]] = scene(8)
    place.run_pending(conn)
    with tx(conn):
        conn.execute("DELETE FROM faces WHERE item_id = ?", (b["id"],))
        conn.execute("DELETE FROM items WHERE id = ?", (b["id"],))
    assert links(conn) == {} and [r[0] for r in conn.execute("SELECT item_id FROM place_scans")] == [a["id"]]
    assert dict(place.pending_items(conn, 10)) == {a["id"]: []}       # its next photo is gone: once more


def test_a_failing_photo_stays_pending_and_the_runner_goes_on(conn, pictures, monkeypatch):
    a, b, c = photo(conn, 0), photo(conn, 4), photo(conn, 200)
    for it in (a, b, c):
        pictures[it["uid"]] = scene(9)
    real = place.measure_pair

    def flaky(conn_, x, y, cache=None):
        if x["id"] == a["id"] and y["id"] == b["id"]:
            raise RuntimeError("boom")
        return real(conn_, x, y, cache)

    monkeypatch.setattr(place, "measure_pair", flaky)
    r = place.run_pending(conn)
    assert r["complete"] is False and r["items"] == 2
    assert place.progress(conn)["last_error"].startswith("RuntimeError")
    assert [i for i, _ in place.pending_items(conn, 10)] == [a["id"]]
    monkeypatch.setattr(place, "measure_pair", real)
    assert place.run_pending(conn)["complete"] and place.progress(conn)["last_error"] is None
    assert links(conn)[(a["id"], b["id"])][0] == "still"


def test_a_photo_that_keeps_failing_is_stored_as_not_lined_up(conn, pictures, monkeypatch):
    a, b = photo(conn, 0), photo(conn, 4)
    pictures[a["uid"]] = pictures[b["uid"]] = scene(9)

    def broken(conn_, x, y, cache=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(place, "measure_pair", broken)
    for _ in range(place.ABORTS_PER_ITEM - 1):
        assert place.run_pending(conn)["complete"] is False
    assert place.run_pending(conn)["complete"] is True           # given up: stored, not tried again
    assert links(conn) == {(a["id"], b["id"]): ("none", "")}


def test_a_photo_without_previews_is_read_from_its_original(conn):
    from app import storage

    item = make_item(conn, taken_at=T0 % 0)                            # an original and its previews
    for p in (storage.preview_path(item["uid"], 1), storage.preview_small_path(item["uid"], 1)):
        p.unlink()
    gray = place_align.load_gray(place._item_row(conn, item["id"]))    # the row the runner reads
    assert gray is not None and gray.shape == (480, 640)


def test_failures_count_in_a_row_only(conn, pictures, monkeypatch):
    a, b = photo(conn, 0), photo(conn, 4)
    pictures[a["uid"]] = pictures[b["uid"]] = scene(9)
    real = place.measure_pair
    boom = {"on": True}

    def flaky(conn_, x, y, cache=None):
        if boom["on"] and x["id"] == a["id"]:
            raise RuntimeError("boom")
        return real(conn_, x, y, cache)

    monkeypatch.setattr(place, "measure_pair", flaky)
    for _ in range(place.ABORTS_PER_ITEM - 1):
        assert place.run_pending(conn)["complete"] is False
    boom["on"] = False
    assert place.run_pending(conn)["complete"] is True and links(conn)[(a["id"], b["id"])][0] == "still"
    c = photo(conn, 2)                                                 # arrives in the middle: a is pending again
    pictures[c["uid"]] = scene(9)
    boom["on"] = True
    assert place.run_pending(conn)["complete"] is False                # one new failure: not given up
    assert (a["id"], c["id"]) not in links(conn)
    boom["on"] = False
    assert place.run_pending(conn)["complete"] is True and links(conn)[(a["id"], c["id"])][0] == "still"


def queued(conn) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT item_id FROM trait_requeue")}


def test_photos_a_chain_can_reach_are_queued_too(conn, pictures):
    p = make_person(conn, "Asha")
    first = make_item(conn, taken_at=T0 % 0, write_files=False, data=None)
    make_face(conn, first["id"], p, assigned_by="user", write_crop=False)
    rest = [photo(conn, 4 * k) for k in range(1, 5)]                   # four photos with an open box each
    for it in (first, *rest):
        pictures[it["uid"]] = scene(5)
    assert place.run_pending(conn, limit=4)["items"] == 4              # newest first: the known person's photo is last
    with tx(conn):
        conn.execute("DELETE FROM trait_requeue")
    assert place.run_pending(conn)["complete"]
    assert (first["id"], rest[3]["id"]) not in links(conn)             # the fifth photo is not linked with the first,
    assert queued(conn) == {it["id"] for it in rest}                   # but a chain reaches it: guessed again too


def test_trashing_or_restoring_a_photo_queues_the_photos_lined_up_with_it(conn, pictures):
    from app import ingest

    p = make_person(conn, "Asha")
    a = make_item(conn, taken_at=T0 % 0, write_files=False, data=None)
    make_face(conn, a["id"], p, assigned_by="user", write_crop=False)
    b = photo(conn, 3)
    pictures[a["uid"]] = pictures[b["uid"]] = scene(5)
    place.run_pending(conn)
    for change in (ingest.soft_delete, ingest.restore):
        with tx(conn):
            conn.execute("DELETE FROM trait_requeue")
        change(conn, a["id"])
        assert queued(conn) == {b["id"]}


def test_measuring_runs_with_the_set_number_of_opencv_threads(monkeypatch):
    import cv2

    before = cv2.getNumThreads()
    for want in (1, 2):
        monkeypatch.setattr(settings, "place_threads", want)
        with place._cv_threads():
            assert cv2.getNumThreads() == want
        assert cv2.getNumThreads() == before
