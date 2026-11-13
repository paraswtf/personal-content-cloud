"""Recognition trained on one person (section 10: app.learning.person_model, app.analysis.person_scan), with
hand-made face vectors: an identity, a look-alike identity at cosine about 0.7 to it, and strangers.

The training (examples, conflicts, the check, determinism), the scan (what may become a guess and what is never
touched), guess_item recomputing the guesses, the review actions, the automatic matching (the gate; a face the
model is sure of kept for the person, when the guess path would offer it too; what such a face becomes where no
guess can follow), the routes, and what an undo brings back (the pause, the model of a deleted or merged person).
"""
from __future__ import annotations

import json
from itertools import count

import numpy as np
import pytest

from app.analysis import cluster, guess, person_scan
from app.analysis.index import get_index
from app.db import tx
from app.learning import people as learned_people
from app.learning import person_model, store
from tests.helpers import make_body, make_face, make_item, make_person

DIM = 512


@pytest.fixture(autouse=True)
def _no_run_left_over():
    """The state of a run lives in the module (one process, one library): each test starts without one."""
    person_scan._state.clear()
    person_scan._running = None
    yield
    person_scan._state.clear()
    person_scan._running = None


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base: np.ndarray, seed: int, noise: float = 0.5) -> np.ndarray:
    """A face of the identity `base`: cosine about 0.89 to it and about 0.8 to another such face."""
    v = base + noise * unit(seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


ME = unit(1)
_other = unit(2) - float(unit(2) @ ME) * ME
TWIN = (0.7 * ME + np.sqrt(1 - 0.49) * _other / np.linalg.norm(_other)).astype(np.float32)  # cosine 0.7 to ME


class World:
    """The person with 40 faces over ten days (five tagged by hand), a stranger in each of their photos, twelve
    look-alike faces the owner took off them and gave to someone else, five faces the owner tagged as a third
    person, and two faces still on the person that look like the look-alike."""

    def __init__(self, conn, faces: int = 40, lookalike: np.ndarray = TWIN, locked: int = 5, days: int = 10) -> None:
        self.conn = conn
        self.lookalike = lookalike
        self.seed = count(1000)
        self.me = make_person(conn, "Asha")
        self.twin = make_person(conn, "Ravi")
        self.third = make_person(conn, "Meera")
        self.mine: list[int] = []
        self.minute = count(0)
        for k in range(faces):
            item = self.photo(day=1 + k % days)
            self.mine.append(self.face(item, near(ME, next(self.seed)), self.me, "user" if k < locked else "auto"))
            self.face(item, unit(next(self.seed)), box=(0.6, 0.2, 0.2, 0.3))
        self.removed: list[int] = []
        for k in range(12):
            fid = self.face(self.photo(day=1 + k % 10), self.like_twin(), self.twin, "user" if k % 2 else "auto")
            self.removed.append(fid)
        with tx(conn):
            cluster.record_rejections(conn, self.removed, self.me)
        for k in range(5):
            self.face(self.photo(day=1 + k), unit(next(self.seed)), self.third, "user")
        self.conflicts = []
        for k in range(2):
            item = self.photo(day=3 + k)
            self.conflicts.append(self.face(item, self.like_twin(), self.me))
            self.face(item, unit(next(self.seed)), box=(0.6, 0.2, 0.2, 0.3))

    def photo(self, day: int = 20, *, kind: str = "image", **extra) -> int:
        m = next(self.minute)
        item = make_item(self.conn, kind=kind, taken_at=f"2026-03-{day:02d}T{10 + m // 60:02d}:{m % 60:02d}:00.000Z",
                         write_files=False, data=None, **extra)
        return int(item["id"])

    def face(self, item_id: int, emb: np.ndarray, person: int | None = None, assigned_by: str = "auto", **extra) -> int:
        return make_face(self.conn, item_id, person, embedding=emb, assigned_by=assigned_by, write_crop=False, **extra)

    def like_me(self) -> np.ndarray:
        return near(ME, next(self.seed))

    def like_twin(self) -> np.ndarray:
        return near(self.lookalike, next(self.seed))

    def train(self) -> dict:
        return person_model.train(self.conn, self.me)


def row(conn, face_id: int):
    return conn.execute("SELECT * FROM faces WHERE id = ?", (face_id,)).fetchone()


def guessed(conn, face_id: int) -> tuple[int | None, str, dict]:
    r = row(conn, face_id)
    return r["guess_person_id"], r["assigned_by"], json.loads(r["guess_evidence"] or "{}")


def unnamed(conn) -> int:
    return make_person(conn, None)


# ------------------------------------------------------------------ training
def test_training_examples_conflicts_and_check(conn):
    w = World(conn)
    model = w.train()
    c = model["counts"]
    assert (c["positives"], c["rejected"], c["owner_others"], c["conflicts"]) == (40, 12, 5, 2)
    assert c["same_photo"] == 42 and c["negatives"] == 12 + 5 + 42 and c["ambiguous"] == 0
    assert {x["face_id"] for x in model["conflicts"]} == set(w.conflicts)
    assert all(x["like_face_id"] in w.removed and x["like_person_id"] == w.twin and x["sim"] >= person_model.CONFLICT_SIM
               for x in model["conflicts"])
    assert set(model["ref_ids"]) == set(w.mine)
    check = model["check"]
    assert check["positives"] == 40 and check["negatives"] == 59
    first = check["cuts"][0]
    assert first["p"] == 0.5 and first["found"] >= 38 and first["mistaken"] <= 1
    assert 0.5 <= model["sure_p"] < 1.0 and model["guess_p"] == 0.5
    claim = check["claim"]  # what the cut of the automatic matching found and mistook
    assert claim["p"] == person_model.CLAIM_P == 0.8 and claim["found"] <= first["found"] and claim["mistaken"] <= first["mistaken"]
    assert len(model["w"]) == DIM and len(model["stack"]["w"]) == 5
    loaded = person_model.load(w.me)
    assert loaded is not None and loaded.trained_at == model["trained_at"] and person_model.trained_ids() == [w.me]
    # it tells the person from the look-alike, which the top-3 score alone cannot do as well
    idx = get_index()
    idx.sync(conn)
    p_me = loaded.probability(np.stack([w.like_me() for _ in range(8)]), np.ones(8), conn=conn)
    p_twin = loaded.probability(np.stack([w.like_twin() for _ in range(8)]), np.ones(8), conn=conn)
    assert p_me.min() > 0.9 and p_twin.max() < 0.2


def test_a_lookalike_the_owner_locked_stays_a_positive(conn):
    w = World(conn)
    locked = w.face(w.photo(day=6), w.like_twin(), w.me, "user")  # the owner's own say, whatever it looks like
    model = w.train()
    assert locked in model["ref_ids"] and model["counts"]["positives"] == 41
    assert {x["face_id"] for x in model["conflicts"]} == set(w.conflicts)


def test_training_is_deterministic(conn):
    w = World(conn)
    a, b = w.train(), w.train()
    a.pop("trained_at")
    b.pop("trained_at")
    assert a == b


def test_too_few_examples_say_why(conn):
    w = World(conn, faces=8)
    with pytest.raises(person_model.NotEnough, match="too few clear faces"):
        w.train()
    lonely = make_person(conn, "Kiran")
    for k in range(25):
        w.face(w.photo(day=1 + k % 10), near(unit(7), next(w.seed)), lonely)
    with pytest.raises(person_model.NotEnough, match="known not to be this person"):
        person_model.train(conn, lonely)
    store.set_enabled(conn, False)
    with pytest.raises(person_model.NotEnough, match="switched off"):
        w.train()
    assert person_model.trained_ids() == []


def test_an_ambiguous_face_of_the_same_photo_is_no_negative(conn):
    w = World(conn)
    mirror = w.face(conn.execute("SELECT item_id FROM faces WHERE id = ?", (w.mine[10],)).fetchone()[0], w.like_me(),
                    box=(0.1, 0.5, 0.2, 0.3))  # the person twice in one photo (a mirror)
    other_photo = conn.execute("SELECT item_id FROM faces WHERE id = ?", (w.mine[11],)).fetchone()[0]
    special = near(ME, 33)  # a look-alike of the person: three faces of it sit on the third person...
    for _ in range(3):
        w.face(w.photo(day=8), near(special, next(w.seed), 0.2), w.third)
    there = w.face(other_photo, near(special, next(w.seed), 0.2), w.third, box=(0.1, 0.5, 0.2, 0.3))  # ...and a fourth
    # the same shape in an UNNAMED group (the person in prints, say): such a group may hold the person's own
    # faces, so it never makes the face a negative
    printed = near(ME, 44)
    group = unnamed(conn)
    for _ in range(3):
        w.face(w.photo(day=9), near(printed, next(w.seed), 0.2), group)
    print_photo = conn.execute("SELECT item_id FROM faces WHERE id = ?", (w.mine[12],)).fetchone()[0]
    w.face(print_photo, near(printed, next(w.seed), 0.2), group, box=(0.1, 0.5, 0.2, 0.3))
    model = w.train()
    # the mirror and the print are left out (no sure negatives); the look-alike matches the third person better
    assert model["counts"]["ambiguous"] == 2 and model["counts"]["same_photo"] == 42 + 1
    assert mirror not in model["ref_ids"] and there not in model["ref_ids"]


# ------------------------------------------------------------------ the scan
def test_scan_guesses_only_where_it_may(conn):
    w = World(conn)
    model = w.train()
    loose = [w.face(w.photo(), w.like_me()) for _ in range(3)]
    loose_twin = w.face(w.photo(), w.like_twin())
    stranger = w.face(w.photo(), unit(next(w.seed)))
    group_one = unnamed(conn)
    in_group = w.face(w.photo(), w.like_me(), group_one)
    body = make_body(conn, row(conn, in_group)["item_id"], group_one, face_id=in_group)
    group_two = unnamed(conn)
    in_big_group = w.face(w.photo(), w.like_me(), group_two)
    w.face(w.photo(), unit(next(w.seed)), group_two)
    twice = w.photo()
    twice_a, twice_b = w.face(twice, w.like_me()), w.face(twice, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    clip = w.photo(kind="video")
    track_a, track_b = w.face(clip, w.like_me()), w.face(clip, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    here = w.photo()
    make_body(conn, here, w.me, assigned_by="user")
    present = w.face(here, w.like_me())
    turned_down = w.face(w.photo(), w.like_me())
    with tx(conn):
        cluster.record_rejections(conn, [turned_down], w.me)
    private = w.face(w.photo(nsfw=1), w.like_me())
    trashed = w.face(w.photo(deleted_at="2026-04-01T00:00:00.000Z"), w.like_me())
    owners_nobody = w.face(w.photo(), w.like_me(), None, "user")
    owners_other = w.face(w.photo(), w.like_me(), w.third, "user")
    on_named = w.face(w.photo(), w.like_me(), w.third)
    before = {f: tuple(row(conn, f)[k] for k in ("person_id", "assigned_by", "guess_person_id"))
              for f in (present, turned_down, private, trashed, owners_nobody, owners_other, on_named, stranger, loose_twin)}

    result = person_scan.scan(conn, w.me)

    for f in [*loose, in_group, in_big_group, track_a, track_b]:
        pid, by, ev = guessed(conn, f)
        assert (pid, by, row(conn, f)["person_id"]) == (w.me, "guess", None), f
        assert ev["basis"] == "face" and ev["p"] == ev["score"] == row(conn, f)["guess_score"] >= 0.5
        assert ev["model"] == {"rev": person_model.MODEL_REV, "trained_at": model["trained_at"]}
    # taken out of their unnamed groups: the emptied group is gone, the other stays, the box around follows
    assert guessed(conn, in_group)[2]["was_person_id"] == group_one
    assert guessed(conn, in_big_group)[2]["was_person_id"] == group_two
    assert "was_person_id" not in guessed(conn, loose[0])[2]
    assert conn.execute("SELECT 1 FROM persons WHERE id = ?", (group_one,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM persons WHERE id = ?", (group_two,)).fetchone() is not None
    assert conn.execute("SELECT person_id FROM bodies WHERE id = ?", (body,)).fetchone()[0] is None
    # one per photo (the better one), a video may show them twice
    both = [guessed(conn, twice_a)[0], guessed(conn, twice_b)[0]]
    assert sorted(both, key=str) == sorted([w.me, None], key=str)
    # never: the person present, a pair turned down, the vault, the trash, the owner's own say, a named person's
    for f, was in before.items():
        assert tuple(row(conn, f)[k] for k in ("person_id", "assigned_by", "guess_person_id")) == was, f
    assert [e["face_id"] for e in result["elsewhere"]] == [on_named] and result["elsewhere"][0]["person_id"] == w.third
    assert result["guesses"] == 8 and result["detached"] == 2 and result["sure"] == 8
    stored = person_model.stored(w.me)["scan"]
    assert stored == result and stored["looked"] >= 12
    # a second look changes nothing
    version = conn.execute("SELECT value FROM kv WHERE key = 'faces_version'").fetchone()[0]
    again = person_scan.scan(conn, w.me)
    assert (again["guesses"], again["detached"]) == (8, 0)
    assert conn.execute("SELECT value FROM kv WHERE key = 'faces_version'").fetchone()[0] == version


def test_scan_skips_a_face_another_named_person_matches_better(conn):
    w = World(conn)
    w.train()
    special = near(ME, 77)
    copies = [w.face(w.photo(), near(special, next(w.seed), 0.2), w.third) for _ in range(3)]
    candidate = w.face(w.photo(), near(special, next(w.seed), 0.2))
    plain = w.face(w.photo(), w.like_me())
    result = person_scan.scan(conn, w.me)
    assert guessed(conn, candidate)[0] is None and guessed(conn, plain)[0] == w.me
    assert {e["face_id"] for e in result["elsewhere"]} == set(copies)  # listed, never moved
    assert all(row(conn, f)["person_id"] == w.third for f in copies)


def test_scan_drops_guesses_it_no_longer_makes(conn):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    keeper = w.face(w.photo(), unit(next(w.seed)), group, "user")
    face = w.face(w.photo(), w.like_me(), group)
    person_scan.scan(conn, w.me)
    assert guessed(conn, face)[0] == w.me
    make_body(conn, row(conn, face)["item_id"], w.me, assigned_by="user")  # the person is in that photo now
    result = person_scan.scan(conn, w.me)
    assert result["guesses"] == 0 and guessed(conn, face)[0] is None
    assert row(conn, keeper)["person_id"] == group


def test_a_face_whose_guess_does_not_stand_goes_back_to_its_group(conn):
    from tests.test_analysis_guess import _scene, _vec

    w = World(conn, lookalike=unit(5))
    w.train()
    scene, torso = _vec(1), _vec(2)
    ref_item = make_item(conn, taken_at="2026-05-01T10:00:00.000Z", write_files=False, data=None)
    w.face(int(ref_item["id"]), w.like_me(), w.me, "user", torso=torso)
    _scene(conn, int(ref_item["id"]), scene)
    item = make_item(conn, taken_at="2026-05-01T10:05:00.000Z", write_files=False, data=None)
    _scene(conn, int(item["id"]), scene)
    clothed = w.face(int(item["id"]), w.like_me(), torso=torso)  # the very same clothes too: a guess at 1.0
    group = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), group, "user")
    face = w.face(int(item["id"]), w.like_me(), group, box=(0.6, 0.2, 0.2, 0.3))
    result = person_scan.scan(conn, w.me)
    # one person per photo: the face with the clothes keeps them, so the other was not to leave its group
    pid, _, ev = guessed(conn, clothed)
    assert pid == w.me and ev["clothes"] == 1.0 and ev["p"] >= 0.5 and "basis" not in ev
    assert row(conn, face)["person_id"] == group and row(conn, face)["assigned_by"] == "auto"
    assert guessed(conn, face)[0] is None and (result["guesses"], result["detached"]) == (0, 0)


# ------------------------------------------------------------------ guess_item, review actions
def test_guess_item_recomputes_the_models_guesses(conn):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), group, "user")
    detached = w.face(w.photo(), w.like_me(), group)
    person_scan.scan(conn, w.me)
    stored = row(conn, detached)["guess_evidence"]
    assert json.loads(stored)["was_person_id"] == group
    assert guess.guess_item(conn, row(conn, detached)["item_id"]) == 0  # the same guess, byte for byte
    assert row(conn, detached)["guess_evidence"] == stored
    # a new photo of the person gets its guess when it is analysed, without a scan
    new = w.face(w.photo(), w.like_me())
    assert guess.guess_item(conn, row(conn, new)["item_id"]) == 1
    assert guessed(conn, new)[0] == w.me and guessed(conn, new)[2]["basis"] == "face"
    assert guess.guess_item(conn, row(conn, new)["item_id"]) == 0
    # a look-alike gets none, and a photo the person is in gets none
    twin = w.face(w.photo(), w.like_twin())
    assert guess.guess_item(conn, row(conn, twin)["item_id"]) == 0 and guessed(conn, twin)[0] is None
    make_body(conn, row(conn, new)["item_id"], w.me, assigned_by="user")
    assert guess.guess_item(conn, row(conn, new)["item_id"]) == 1 and guessed(conn, new)[0] is None


def test_clothes_and_model_guesses_merge(conn):
    from tests.test_analysis_guess import _near, _scene, _vec

    # the other person here is no look-alike: an open face that also matches someone else at 0.40 gets no clothes
    # guess at all (guess.FACE_CONTRADICTION), and this test is about the two kinds of evidence meeting
    w = World(conn, lookalike=unit(5))
    w.train()
    scene, torso = _vec(1), _vec(2)
    ref_item = make_item(conn, taken_at="2026-05-01T10:00:00.000Z", write_files=False, data=None)
    w.face(int(ref_item["id"]), w.like_me(), w.me, "user", torso=torso)
    _scene(conn, int(ref_item["id"]), scene)

    def later(minute: int, scene=scene) -> int:
        item = make_item(conn, taken_at=f"2026-05-01T10:{minute:02d}:00.000Z", write_files=False, data=None)
        _scene(conn, int(item["id"]), scene)
        return int(item["id"])

    # the same person by both: the stronger evidence stays and notes the other
    weak_scene = w.face(later(10, scene=_near(scene, 9, 0.6)), w.like_me(), torso=_near(torso, 5))
    assert guess.guess_item(conn, row(conn, weak_scene)["item_id"]) == 1
    pid, _, ev = guessed(conn, weak_scene)
    assert pid == w.me and ev["basis"] == "face" and ev["also"] == "clothes" and ev["score"] == ev["p"]
    assert "clothes" not in ev and guess.guess_item(conn, row(conn, weak_scene)["item_id"]) == 0
    same_scene = w.face(later(12), w.like_me(), torso=torso)  # the very same clothes in the very same scene: 1.0
    assert guess.guess_item(conn, row(conn, same_scene)["item_id"]) == 1
    pid, _, ev = guessed(conn, same_scene)
    assert pid == w.me and "basis" not in ev and ev["clothes"] > 0.85 and ev["p"] >= 0.5 and ev["score"] > ev["p"]
    assert guess.guess_item(conn, row(conn, same_scene)["item_id"]) == 0
    # a body by clothes and a face by the model in one photo: one person per photo, the better box keeps them
    item = later(20)
    body = make_body(conn, item, None, torso=_near(torso, 6))
    face = w.face(item, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    guess.guess_item(conn, item)
    body_guess = conn.execute("SELECT guess_person_id, guess_score FROM bodies WHERE id = ?", (body,)).fetchone()
    face_guess = row(conn, face)
    assert sorted([body_guess["guess_person_id"], face_guess["guess_person_id"]], key=str) == sorted([w.me, None], key=str)
    assert (face_guess["guess_person_id"] == w.me) == (float(face_guess["guess_score"] or 0) >= 0.5)


def test_other_paths_are_untouched_by_a_model(conn):
    """Byte for byte: a clothes world guessed before and after someone else's model exists."""
    from tests.test_analysis_guess import _body_row, _world

    p, a, b, _, body_b, _ = _world(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    before = (_body_row(conn, body_b)["guess_person_id"], _body_row(conn, body_b)["guess_score"],
              _body_row(conn, body_b)["guess_evidence"])
    w = World(conn)
    w.train()
    assert guess.guess_item(conn, b["id"]) == 0
    assert (_body_row(conn, body_b)["guess_person_id"], _body_row(conn, body_b)["guess_score"],
            _body_row(conn, body_b)["guess_evidence"]) == before and before[0] == p


def test_reject_regroups_and_confirm_locks(conn):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    was_alone = w.face(w.photo(), w.like_me(), group)
    loose = w.face(w.photo(), w.like_me())
    yes = w.face(w.photo(), w.like_me())
    person_scan.scan(conn, w.me)
    assert conn.execute("SELECT 1 FROM persons WHERE id = ?", (group,)).fetchone() is None
    assert guess.reject(conn, [was_alone, loose], []) == 2
    back = row(conn, was_alone)
    assert back["person_id"] is not None and back["person_id"] != w.me and back["assigned_by"] == "auto"  # a group again
    assert row(conn, loose)["person_id"] is None and row(conn, loose)["assigned_by"] == "auto"  # it was loose before
    assert {r[0] for r in conn.execute("SELECT ref_id FROM guess_rejections WHERE person_id = ? AND ref_id IN (?, ?)",
                                       (w.me, was_alone, loose))} == {was_alone, loose}
    assert guess.guess_item(conn, back["item_id"]) == 0 and guess.guess_item(conn, row(conn, loose)["item_id"]) == 0
    assert guess.confirm(conn, [yes], []) == 1
    r = row(conn, yes)
    assert (r["person_id"], r["assigned_by"], r["by_guess"], r["guess_person_id"]) == (w.me, "user", 1, None)
    events = [json.loads(e["features"]) for e in conn.execute(
        "SELECT features FROM learning_events WHERE kind = 'guess' ORDER BY id")]
    assert [e["basis"] for e in events] == ["face"] * 3 and all(set(e) == {"basis", "p", "face", "score"} for e in events)
    assert learned_people._clothes_events(conn) == [] and learned_people.RULES_REV >= 4


def test_withdraw_pauses_until_the_next_look(conn, monkeypatch):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), group, "user")
    special = near(ME, 55)
    for _ in range(3):
        w.face(w.photo(), near(special, next(w.seed), 0.2), group, "user")
    detached = w.face(w.photo(), near(special, next(w.seed), 0.2), group)
    loose = w.face(w.photo(), w.like_me())
    person_scan.scan(conn, w.me)
    assert guessed(conn, detached)[0] == guessed(conn, loose)[0] == w.me
    settled: list[int] = []
    real = guess.guess_item
    monkeypatch.setattr(guess, "guess_item", lambda c, item_id, cache=None: (settled.append(int(item_id)), real(c, item_id, cache))[1])
    assert person_scan.withdraw(conn, w.me) == 2
    # the items are guessed again without the model, so a guess of other evidence it had replaced comes back
    assert sorted(settled) == sorted([row(conn, detached)["item_id"], row(conn, loose)["item_id"]])
    monkeypatch.undo()
    assert row(conn, detached)["person_id"] == group and row(conn, loose)["person_id"] is None  # back in its group
    assert person_model.paused(conn, [w.me, w.twin]) == {w.me}
    assert guess.guess_item(conn, row(conn, loose)["item_id"]) == 0 and guessed(conn, loose)[0] is None  # paused
    assert person_scan.scan(conn, w.me)["guesses"] == 2 and person_model.paused(conn, [w.me]) == set()
    # a guess in the trash goes too, so none comes back with its photo; a new training un-pauses as a look does
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-06-01T00:00:00.000Z' WHERE id = ?", (row(conn, loose)["item_id"],))
    assert person_scan.withdraw(conn, w.me) == 1 and guessed(conn, loose)[0] is None
    w.train()
    assert person_model.paused(conn, [w.me]) == set()


def test_a_guess_without_its_model_waits_for_an_answer(conn):
    w = World(conn)
    w.train()
    loose = w.face(w.photo(), w.like_me())
    person_scan.scan(conn, w.me)
    stored = row(conn, loose)["guess_evidence"]
    person_model.forget(w.me)
    assert person_model.trained_ids() == []
    assert guess.guess_item(conn, row(conn, loose)["item_id"]) == 0 and row(conn, loose)["guess_evidence"] == stored
    assert guess.confirm(conn, [loose], []) == 1 and row(conn, loose)["person_id"] == w.me


def test_learning_off_switches_the_model_off(conn):
    w = World(conn)
    w.train()
    loose = w.face(w.photo(), w.like_me())
    store.set_enabled(conn, False)
    assert person_model.load(w.me) is None and person_model.stored(w.me) is not None
    assert guess.guess_item(conn, row(conn, loose)["item_id"]) == 0 and guessed(conn, loose)[0] is None
    with pytest.raises(person_model.NotEnough):
        person_scan.scan(conn, w.me)


# ------------------------------------------------------------------ the automatic matching
def test_the_gate_keeps_lookalikes_off_the_person(conn):
    w = World(conn)
    faces = [w.face(w.photo(), w.like_twin()), w.face(w.photo(), w.like_me())]
    idx = get_index()

    def options(gated: bool):
        with idx.lock:
            idx.sync(conn)
            rows = cluster._face_rows(conn, faces)
            out = cluster._score(idx, rows, frozenset(), None, threshold=0.3)  # every option, no rule applied
            embs = idx.embeddings_of([r["id"] for r in rows])
            child = idx.child_flags([r["id"] for r in rows])
            if gated:
                out = cluster._drop_gated(idx, [r["id"] for r in rows], embs, child, out)
        return {r["id"]: [pid for pid, _ in opts] for r, opts in zip(rows, out)}

    assert w.me in options(False)[faces[0]] and w.me in options(False)[faces[1]]
    assert options(True) == options(False)  # no model: nothing changes
    w.train()
    gated = options(True)
    assert w.me not in gated[faces[0]] and w.me in gated[faces[1]]
    assert gated[faces[0]] == [w.twin]  # other people's options are never the gate's business
    # only the refused person's option goes: the face takes its next option, as if they had scored under their cut
    with idx.lock:
        idx.sync(conn)
        ids = [faces[0], faces[1]]
        embs, child = idx.embeddings_of(ids), idx.child_flags(ids)
        opts = [[(w.me, 0.6), (w.third, 0.5)], [(w.me, 0.8), (w.third, 0.5)]]
        assert cluster._drop_gated(idx, ids, embs, child, opts) == [[(w.third, 0.5)], [(w.me, 0.8), (w.third, 0.5)]]
        opts = [[(w.third, 0.7), (w.me, 0.6), (w.twin, 0.5)], []]
        assert cluster._drop_gated(idx, ids, embs, child, opts) == [[(w.third, 0.7), (w.twin, 0.5)], []]


def test_new_faces_are_matched_through_the_gate(conn, monkeypatch):
    w = World(conn)
    with tx(conn):  # forget the removals, so only the trained model can keep the look-alike off the person
        conn.execute("DELETE FROM guess_rejections")
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    lookalike, real = w.face(w.photo(), w.like_twin()), w.face(w.photo(), w.like_me())
    with tx(conn):
        cluster.record_rejections(conn, w.removed, w.me)
    w.train()
    with tx(conn):
        conn.execute("DELETE FROM guess_rejections")
    got = cluster.assign_new_faces(conn, [lookalike, real])
    assert got[real] == w.me and got[lookalike] != w.me


def test_recluster_leaves_a_model_guess_alone(conn):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    special = near(ME, 91)
    for _ in range(3):
        w.face(w.photo(), near(special, next(w.seed), 0.2), group, "user")
    face = w.face(w.photo(), near(special, next(w.seed), 0.2), group)
    strangers = unnamed(conn)
    someone = unit(next(w.seed))
    for _ in range(3):
        w.face(w.photo(), near(someone, next(w.seed), 0.2), strangers, "user")
    plain = w.face(w.photo(), near(someone, next(w.seed), 0.2))
    with tx(conn):  # an ordinary loose face with a clothes guess: recluster may still take it
        conn.execute("UPDATE faces SET assigned_by = 'guess', guess_person_id = ?, guess_score = 0.6, "
                     "guess_evidence = '{\"clothes\":0.9}' WHERE id = ?", (w.third, plain))
    person_scan.scan(conn, w.me)
    assert guessed(conn, face)[0] == w.me and row(conn, face)["person_id"] is None
    assert cluster.recluster_unassigned(conn) == 1
    assert row(conn, plain)["person_id"] == strangers  # the usual pass still works
    # the model's guess waits for the owner, although the face would score its old group at its cut
    assert guessed(conn, face)[0] == w.me and row(conn, face)["person_id"] is None


# ------------------------------------------------------------------ routes
def test_routes_train_look_settle_and_take_back(client, conn):
    w = World(conn)
    group = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), group, "user")
    detached = w.face(w.photo(), w.like_me(), group)
    loose = w.face(w.photo(), w.like_me())
    on_named = w.face(w.photo(), w.like_me(), w.third)
    url = f"/api/people/{w.me}/recognition"
    r = client.get(url)
    assert r.status_code == 200 and r.json()["available"] is True and r.json()["trained"] is False
    assert client.post(f"{url}/scan").status_code == 409
    r = client.post(f"{url}/train", json={})
    assert r.status_code == 202 and r.json() == {"started": True} and "x-action-id" not in r.headers
    s = client.get(url).json()
    assert s["trained"] and s["running"] is None and s["error"] is None and s["withdrawn"] is False
    assert s["counts"]["positives"] == 40 and s["check"]["cuts"][0]["p"] == 0.5 and 0.5 <= s["sure_p"] < 1
    assert (s["conflicts"], s["elsewhere"], s["guesses"]) == (2, 1, 2)
    assert s["scan"]["guesses"] == 2 and s["scan"]["detached"] == 1 and s["scan"]["elsewhere"] == 1
    assert s["answers_since"] == 0
    # the lists
    page = client.get(f"{url}/conflicts").json()
    assert page["total"] == 2 and {c["face"]["id"] for c in page["items"]} == set(w.conflicts)
    assert all(c["like"]["id"] in w.removed and c["like_person"]["id"] == w.twin for c in page["items"])
    first = client.get(f"{url}/conflicts?limit=1").json()
    assert len(first["items"]) == 1 and first["next_cursor"]
    rest = client.get(f"{url}/conflicts?limit=1&cursor={first['next_cursor']}").json()
    assert len(rest["items"]) == 1 and rest["next_cursor"] is None
    assert rest["items"][0]["face"]["id"] != first["items"][0]["face"]["id"]
    assert client.get(f"{url}/conflicts?cursor=bm9wZQ").status_code == 400
    elsewhere = client.get(f"{url}/elsewhere").json()["items"]
    assert [(e["face"]["id"], e["person"]["id"]) for e in elsewhere] == [(on_named, w.third)]
    # the review list and its filter
    faces = client.get(f"/api/review/guesses?basis=face&person={w.me}").json()
    assert faces["total"] == 2 and {g["id"] for g in faces["items"]} == {detached, loose}
    assert all(g["guess"]["evidence"]["basis"] == "face" for g in faces["items"])
    assert client.get("/api/review/guesses?basis=clothes").json()["total"] == 0
    # settling a conflict: "This is them" locks it, "Not them" takes it off; both leave the list
    yes, no = w.conflicts
    assert client.post("/api/faces/assign", json={"face_ids": [yes], "person_id": w.me}).status_code == 200
    assert client.post("/api/faces/assign", json={"face_ids": [no], "person_id": None}).status_code == 200
    s = client.get(url).json()
    assert s["conflicts"] == 0 and s["answers_since"] >= 2
    # moving the face on someone else to the person empties that list
    assert client.post("/api/faces/assign", json={"face_ids": [on_named], "person_id": w.me}).status_code == 200
    assert client.get(f"{url}/elsewhere").json()["items"] == []
    # take back, undo, forget
    r = client.post(f"{url}/withdraw")
    assert r.status_code == 200 and r.json() == {"updated": 2}
    assert row(conn, detached)["person_id"] == group and guessed(conn, loose)[0] is None
    assert client.get(url).json()["withdrawn"] is True
    u = client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
    assert u.status_code == 200, u.text
    # the undo brings the guesses back and un-pauses the model: the card does not contradict itself, the model
    # answers for the restored guesses again, and a new photo of the person gets its guess
    assert guessed(conn, detached)[0] == guessed(conn, loose)[0] == w.me
    s = client.get(url).json()
    assert s["withdrawn"] is False and s["guesses"] == 2
    assert guess.guess_item(conn, row(conn, loose)["item_id"]) == 0
    new = w.face(w.photo(), w.like_me())
    assert guess.guess_item(conn, row(conn, new)["item_id"]) == 1 and guessed(conn, new)[0] == w.me
    # the redo takes them back again and pauses it again
    redo = client.post(f"/api/actions/{u.json()['undo_id']}/undo")
    assert redo.status_code == 200, redo.text
    assert client.get(url).json()["withdrawn"] is True and guessed(conn, loose)[0] is None
    assert client.post(f"{url}/scan").status_code == 202 and client.get(url).json()["withdrawn"] is False
    # forgetting is not an undoable action: nothing half comes back
    r = client.delete(url)
    assert r.status_code == 200 and "x-action-id" not in r.headers
    s = client.get(url).json()
    assert s["trained"] is False and s["guesses"] == 0 and guessed(conn, loose)[0] is None
    assert row(conn, detached)["person_id"] == group


def test_routes_refuse_what_cannot_be_trained(client, conn):
    few = make_person(conn, "Asha")
    item = make_item(conn, write_files=False, data=None)
    make_face(conn, item["id"], few, write_crop=False)
    url = f"/api/people/{few}/recognition"
    s = client.get(url).json()
    assert s["available"] is False and "about 20 clear faces" in s["reason"] and "count as" not in s["reason"]
    r = client.post(f"{url}/train", json={})
    assert r.status_code == 409 and "about 20 clear faces" in r.json()["detail"]
    with tx(conn):
        pet = int(conn.execute("INSERT INTO persons(name, name_key, created_at, updated_at, kind, species) "
                               "VALUES ('Pebble', 'pebble', '2026-01-01', '2026-01-01', 'pet', 'dog')").lastrowid)
    assert client.get(f"/api/people/{pet}/recognition").status_code == 404
    assert client.get("/api/people/999999/recognition").status_code == 404
    w = World(conn)
    store.set_enabled(conn, False)
    s = client.get(f"/api/people/{w.me}/recognition").json()
    assert s["available"] is False and s["learning"] is False and "switched off" in s["reason"]
    assert client.post(f"/api/people/{w.me}/recognition/train", json={}).status_code == 409


def test_a_failed_training_reports_its_reason(client, conn, monkeypatch):
    w = World(conn)

    def boom(*a, **k):
        raise person_model.NotEnough("There are too few clear faces on this person to train on: about 20 are needed.")

    monkeypatch.setattr(person_model, "train", boom)
    assert client.post(f"/api/people/{w.me}/recognition/train", json={}).status_code == 202
    s = client.get(f"/api/people/{w.me}/recognition").json()
    assert s["trained"] is False and s["running"] is None and "too few clear faces" in s["error"]


def test_deleting_or_merging_the_person_forgets_the_model(client, conn):
    w = World(conn)
    w.train()
    assert person_model.trained_ids() == [w.me]
    r = client.post("/api/people/merge", json={"target_id": w.third, "source_ids": [w.me]})
    assert r.status_code == 200, r.text
    assert person_model.trained_ids() == []


# ------------------------------------------------------------------ the conflict rule
@pytest.mark.parametrize("extra", [6, 12, 25])
def test_lookalike_faces_on_the_person_do_not_vouch_for_each_other(conn, extra):
    """Several faces of the look-alike sit on the person. Each one's best matches on the person are the others,
    as near as the faces the owner took off: only faces that are no suspects may clear a suspect."""
    w = World(conn)
    for k in range(extra):
        item = w.photo(day=1 + k % 10)
        w.conflicts.append(w.face(item, w.like_twin(), w.me))
        w.face(item, unit(next(w.seed)), box=(0.6, 0.2, 0.2, 0.3))
    model = w.train()
    assert {x["face_id"] for x in model["conflicts"]} == set(w.conflicts)
    assert model["counts"]["conflicts"] == extra + 2 and set(model["ref_ids"]) == set(w.mine)
    assert model["check"]["cuts"][0]["mistaken"] <= 1 and model["check"]["positives"] == 40
    order = [(-x["sim"], x["face_id"]) for x in model["conflicts"]]
    assert order == sorted(order)  # strongest first, then by face: the order the list is paged in
    loaded = person_model.load(w.me)
    get_index().sync(conn)
    assert loaded.probability(np.stack([w.like_twin() for _ in range(8)]), np.ones(8), conn=conn).max() < 0.2


def test_the_plain_conflict_rule_decides_when_no_face_is_beyond_suspicion(conn):
    """The look-alike is close to the person and the owner locked nothing: every face on the person is near a face
    taken off them, so none is trusted. Setting them all aside would keep nothing: the plain rule decides."""
    close = (0.8 * ME + 0.6 * _other / np.linalg.norm(_other)).astype(np.float32)
    w = World(conn, lookalike=close, locked=0)
    model = w.train()
    assert {x["face_id"] for x in model["conflicts"]} == set(w.conflicts)
    assert model["counts"]["positives"] == 40 and set(model["ref_ids"]) == set(w.mine)


def test_the_check_records_how_it_folded(conn):
    w = World(conn)
    assert w.train()["check"]["folds"] == "day"
    # everyone of this person's photos is from one day: no day can be held out, so photos are
    one_day = make_person(conn, "Tara")
    base = unit(9)
    for _ in range(30):
        item = w.photo(day=15)
        w.face(item, near(base, next(w.seed)), one_day)
        w.face(item, unit(next(w.seed)), box=(0.6, 0.2, 0.2, 0.3))
    model = person_model.train(conn, one_day)
    assert model["check"]["folds"] == "photo" and model["check"]["positives"] == 30


# ------------------------------------------------------------------ faces go back to their groups
def test_an_emptied_group_comes_back_as_one_group(conn):
    w = World(conn)
    w.train()

    def emptied() -> tuple[int, list[int]]:
        group = unnamed(conn)
        return group, [w.face(w.photo(), w.like_me(), group) for _ in range(3)]

    (g1, together), (g2, one_by_one), (g3, taken_back) = emptied(), emptied(), emptied()
    assert person_scan.scan(conn, w.me)["detached"] == 9
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id IN (?, ?, ?)", (g1, g2, g3)).fetchone()[0] == 0

    def group_of(faces: list[int]) -> int:
        people = {row(conn, f)["person_id"] for f in faces}
        assert len(people) == 1 and None not in people, people
        (person,) = people
        assert conn.execute("SELECT name FROM persons WHERE id = ?", (person,)).fetchone()["name"] is None
        return int(person)

    # turned down in one answer
    assert guess.reject(conn, together, []) == 3
    first = group_of(together)
    # turned down one at a time: the first answer brings the group back, and the others remember the new one
    assert guess.reject(conn, one_by_one[:1], []) == 1
    second = row(conn, one_by_one[0])["person_id"]
    assert second not in (None, first, w.me)
    for f in one_by_one[1:]:
        assert guessed(conn, f)[2]["was_person_id"] == second
        assert guess.reject(conn, [f], []) == 1
    assert group_of(one_by_one) == second
    # taken back
    assert person_scan.withdraw(conn, w.me) == 3
    third = group_of(taken_back)
    assert len({first, second, third}) == 3
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE name IS NULL").fetchone()[0] == 3


def test_a_face_goes_back_to_its_group_whenever_its_guess_goes(conn):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), group, "user")
    item_a, item_b, item_c = w.photo(), w.photo(), w.photo()
    a, b, c = (w.face(i, w.like_me(), group) for i in (item_a, item_b, item_c))
    other = w.face(item_a, unit(next(w.seed)), box=(0.6, 0.2, 0.2, 0.3))
    assert person_scan.scan(conn, w.me)["detached"] == 3
    assert all(guessed(conn, f)[0] == w.me and row(conn, f)["person_id"] is None for f in (a, b, c))
    version = conn.execute("SELECT value FROM kv WHERE key = 'faces_version'").fetchone()[0]
    # the owner puts another box of the photo on the person: the guess goes, the face is back in its group
    with tx(conn):
        conn.execute("UPDATE faces SET person_id = ?, assigned_by = 'user' WHERE id = ?", (w.me, other))
    assert guess.guess_item(conn, item_a) == 1
    assert (row(conn, a)["person_id"], row(conn, a)["assigned_by"], guessed(conn, a)[0]) == (group, "auto", None)
    assert conn.execute("SELECT value FROM kv WHERE key = 'faces_version'").fetchone()[0] != version
    # the photo goes to the vault
    with tx(conn):
        conn.execute("UPDATE items SET nsfw = 1 WHERE id = ?", (item_b,))
    assert guess.guess_item(conn, item_b) == 1 and row(conn, b)["person_id"] == group
    # a guess that cannot be confirmed (the person is in the photo already)
    make_body(conn, item_c, w.me, assigned_by="user")
    assert guess.confirm(conn, [c], []) == 0
    assert (row(conn, c)["person_id"], row(conn, c)["assigned_by"], guessed(conn, c)[0]) == (group, "auto", None)


def test_another_guess_on_such_a_face_keeps_the_memory(conn):
    from app.db import kv_set

    w = World(conn)
    w.train()
    group = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), group, "user")
    face, moved = w.face(w.photo(), w.like_me(), group), w.face(w.photo(), w.like_me(), group)
    person_scan.scan(conn, w.me)
    # other evidence holds the face now (as guess_item writes it, the memory kept)
    with tx(conn):
        conn.execute("UPDATE faces SET guess_person_id = ?, guess_score = 0.9, guess_evidence = ? WHERE id = ?",
                     (w.third, json.dumps({"clothes": 0.9, "ref_item_uid": "gone", "was_person_id": group}), face))
    # the usual matching leaves it to the owner (it would put a face like this on the person)
    assert cluster.recluster_unassigned(conn) == 0 and guessed(conn, face)[0] == w.third
    # the group is merged into another unnamed one: the faces remember that one
    target = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), target, "user")
    cluster.merge_persons(conn, target, [group])
    assert guessed(conn, face)[2]["was_person_id"] == guessed(conn, moved)[2]["was_person_id"] == target
    assert guess.guess_item(conn, row(conn, moved)["item_id"]) == 0  # the memory is written as guess_item writes it
    assert guess.reject(conn, [moved], []) == 1 and row(conn, moved)["person_id"] == target
    # the window the clothes guess was made with changes: it is withdrawn, and the face is back in the group
    with tx(conn):
        kv_set(conn, guess.WINDOW_KEY, "0.125")
    assert guess.apply_window_change(conn) == 1
    assert (row(conn, face)["person_id"], row(conn, face)["assigned_by"]) == (target, "auto")


def test_deleting_the_person_puts_the_faces_back(client, conn):
    w = World(conn)
    w.train()
    alone = unnamed(conn)
    was_alone = [w.face(w.photo(), w.like_me(), alone) for _ in range(2)]  # the look empties this group
    shared = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), shared, "user")
    in_shared = w.face(w.photo(), w.like_me(), shared)
    assert person_scan.scan(conn, w.me)["detached"] == 3
    r = client.delete(f"/api/people/{w.me}")
    assert r.status_code == 200, r.text
    assert row(conn, in_shared)["person_id"] == shared
    back = {row(conn, f)["person_id"] for f in was_alone}
    assert len(back) == 1 and None not in back and back != {alone}
    assert all(guessed(conn, f)[0] is None and row(conn, f)["assigned_by"] == "auto" for f in [*was_alone, in_shared])


def test_a_group_the_owner_deleted_is_not_remembered(client, conn):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), group, "user")
    face = w.face(w.photo(), w.like_me(), group)
    person_scan.scan(conn, w.me)
    assert guessed(conn, face)[2]["was_person_id"] == group
    assert client.delete(f"/api/people/{group}").status_code == 200
    pid, _, ev = guessed(conn, face)
    assert pid == w.me and "was_person_id" not in ev
    assert guess.guess_item(conn, row(conn, face)["item_id"]) == 0  # written as guess_item writes it
    assert guess.reject(conn, [face], []) == 1 and row(conn, face)["person_id"] is None  # no group to go back to


# ------------------------------------------------------------------ what the owner said about a group
def test_scan_leaves_a_group_the_owner_said_is_not_the_person(conn):
    w = World(conn)
    w.train()
    group = unnamed(conn)
    faces = [w.face(w.photo(), w.like_me(), group) for _ in range(3)]
    other = unnamed(conn)
    w.face(w.photo(), unit(next(w.seed)), other, "user")
    later = w.face(w.photo(), w.like_me(), other)

    def dismiss(a: int, b: int) -> None:
        with tx(conn):
            conn.execute("INSERT INTO merge_dismissals(person_a, person_b, created_at) VALUES (?, ?, '2026-01-01')",
                         (min(a, b), max(a, b)))

    dismiss(group, w.me)
    result = person_scan.scan(conn, w.me)
    assert all(row(conn, f)["person_id"] == group and guessed(conn, f)[0] is None for f in faces)
    assert conn.execute("SELECT COUNT(*) FROM merge_dismissals").fetchone()[0] == 1  # the answer is still there
    assert (result["guesses"], result["detached"]) == (1, 1) and guessed(conn, later)[0] == w.me
    # said after the look: the guess goes the next time the photo is guessed, and the face is back in its group
    dismiss(other, w.me)
    assert guess.guess_item(conn, row(conn, later)["item_id"]) == 1
    assert row(conn, later)["person_id"] == other and guessed(conn, later)[0] is None
    assert person_scan.scan(conn, w.me)["guesses"] == 0 and row(conn, later)["person_id"] == other


# ------------------------------------------------------------------ the automatic matching: the gate's price
def test_a_face_the_gate_takes_off_the_person_goes_to_its_next_option(conn, monkeypatch):
    """The gate removes the refused person's option and nothing else, so it can change who a face is tagged as:
    here a look-alike's face whose best raw score is the person goes to the runner-up, as if the person had
    scored under their cut. (A face of the person that the model misses pays the same price.)"""
    w = World(conn)
    runner = make_person(conn, "Lena")
    for _ in range(3):
        w.face(w.photo(), near(w.lookalike, next(w.seed), 1.5), runner)  # looks a little like the look-alike
    w.train()
    with tx(conn):  # forget the removals, so only the trained model keeps the look-alike off the person
        conn.execute("DELETE FROM guess_rejections")
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    face = w.face(w.photo(), w.like_twin())
    without = set(w.removed)  # the look-alike's own faces left out: the person scores best, the runner-up next
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        plain = cluster._score(idx, cluster._face_rows(conn, [face]), frozenset(without), None)[0]
    assert [pid for pid, _ in plain] == [w.me, runner]
    assert cluster.assign_new_faces(conn, [face], exclude_face_ids=without)[face] == runner


# ------------------------------------------------------------------ the automatic matching: a face the model is sure of
def _options(conn, faces: list[int], exclude: frozenset[int] = frozenset()) -> tuple[dict[int, list[int]], set[int]]:
    """The people each face may join on the default path (cluster._score), and the faces a model kept."""
    idx = get_index()
    claimed: dict[int, cluster.Kept] = {}
    with idx.lock:
        idx.sync(conn)
        rows = cluster._face_rows(conn, faces)
        out = cluster._score(idx, rows, exclude, conn, claimed=claimed)
    return {int(r["id"]): [pid for pid, _ in opts] for r, opts in zip(rows, out)}, set(claimed)


def _claims(conn, faces: list[int], exclude: frozenset[int] = frozenset()) -> dict[int, cluster.Kept]:
    """What the models kept among these faces: face id -> (person, probability, unnamed options)."""
    idx = get_index()
    claimed: dict[int, cluster.Kept] = {}
    with idx.lock:
        idx.sync(conn)
        cluster._score(idx, cluster._face_rows(conn, faces), exclude, conn, claimed=claimed)
    return claimed


def _guess_jobs(conn) -> set[tuple[int, int | None]]:
    """(person, item) of the queued 'guess' jobs."""
    rows = conn.execute("SELECT payload FROM jobs WHERE type = 'guess' AND status = 'queued'").fetchall()
    return {(int(p["person_id"]), p["item_id"]) for p in (json.loads(r["payload"]) for r in rows)}


def _is_unnamed(conn, person_id: int | None) -> bool:
    r = conn.execute("SELECT name FROM persons WHERE id = ?", (person_id,)).fetchone()
    return r is not None and r["name"] is None


def _cut_keeps_them_off(monkeypatch, w: World) -> None:
    """The person's own cut is above what their new faces score (on the owner's library the faces taken off the
    person raised it), so the plain matching gives such a face to the look-alike, whose cut it clears."""
    monkeypatch.setattr(cluster, "_learned", lambda pid: 0.95 if pid == w.me else None)


def test_a_face_the_model_is_sure_of_is_not_given_to_a_lookalike(conn, monkeypatch):
    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    face = w.face(w.photo(), w.like_me())
    clip = w.photo(kind="video")
    tracks = [w.face(clip, w.like_me()), w.face(clip, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))]
    lookalike = w.face(w.photo(), w.like_twin())
    everyone = [face, *tracks, lookalike]
    # without the claim the look-alike takes all of them
    monkeypatch.setattr(person_model, "CLAIM_P", 1.01)
    plain, kept = _options(conn, everyone)
    assert all(plain[f][0] == w.twin for f in everyone) and kept == set()
    monkeypatch.undo()
    _cut_keeps_them_off(monkeypatch, w)
    options, kept = _options(conn, everyone)
    assert kept == {face, *tracks} and all(options[f] == [] for f in kept)
    assert options[lookalike] == plain[lookalike]  # the model is not sure of this one: nothing changes for it
    sure = _claims(conn, tracks)
    track, other = sorted(tracks, key=lambda f: (-sure[f].p, f))  # the one the model is surest of waits for the guess
    people = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    got = cluster.assign_new_faces(conn, everyone)
    assert got[lookalike] == w.twin and [got[f] for f in (face, track)] == [None, None]
    # kept loose: no look-alike, and no new unnamed person either (photo and video)
    assert all((row(conn, f)["person_id"], row(conn, f)["assigned_by"]) == (None, "auto") for f in (face, track))
    # a person is confirmed once in an item, so one face of the clip waits for the guess: the other track starts
    # a person of its own, as it would without a model, and is never the look-alike's
    assert _is_unnamed(conn, got[other]) and conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == people + 1
    # nobody but the analysis guesses by itself, so the guesses are queued
    assert _guess_jobs(conn) == {(w.me, row(conn, face)["item_id"]), (w.me, clip)}
    # the guess offers them to the person
    assert guess.guess_item(conn, row(conn, face)["item_id"]) == 1 and guess.guess_item(conn, clip) == 1
    for f in (face, track):
        pid, by, ev = guessed(conn, f)
        assert (pid, by, ev["basis"]) == (w.me, "guess", "face") and ev["p"] >= person_model.CLAIM_P


def test_a_kept_face_is_the_persons_when_their_cut_lets_it(conn, monkeypatch):
    """The person is among the face's options: nobody else is any more, so the face is theirs even where a
    look-alike scored higher. Nothing the plain rules would refuse the person is tagged."""
    w = World(conn)
    w.train()
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    face = w.face(w.photo(), w.like_me())
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        ids = [face]
        embs, child = idx.embeddings_of(ids), idx.child_flags(ids)
        kept: dict[int, cluster.Kept] = {}
        opts = [[(w.twin, 0.9), (w.me, 0.8), (w.third, 0.5)]]
        assert cluster._claim(conn, idx, ids, embs, child, frozenset(), opts, kept) == [[(w.me, 0.8)]]
        assert set(kept) == {face} and kept[face].person_id == w.me and kept[face].rest == ()  # no unnamed option
    assert cluster.assign_new_faces(conn, [face])[face] == w.me


def test_a_kept_face_starts_a_group_when_the_person_is_in_the_photo(conn, monkeypatch):
    """No guess names someone who is in the photo already, so a second face the model keeps for them could never
    be answered: it starts a person of its own (never the look-alike the plain matching would give it to)."""
    w = World(conn)
    w.train()
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    here = w.photo()
    w.face(here, w.like_me(), w.me, "user")
    second = w.face(here, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))  # a mirror, a print: the person twice
    monkeypatch.setattr(person_model, "CLAIM_P", 1.01)
    assert cluster.assign_new_faces(conn, [second])[second] == w.twin  # the plain matching: the next option
    with tx(conn):
        conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'auto' WHERE id = ?", (second,))
    monkeypatch.undo()
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    assert _options(conn, [second]) == ({second: [w.me]}, {second})  # kept: nobody else is an option
    people = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    own = cluster.assign_new_faces(conn, [second])[second]
    assert _is_unnamed(conn, own) and conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == people + 1
    assert _guess_jobs(conn) == set() and guess.guess_item(conn, here) == 0  # nothing waits for a guess here


def test_the_claim_cut_and_who_may_claim(conn, monkeypatch):
    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    a, b = w.face(w.photo(), w.like_me()), w.face(w.photo(), w.like_me())
    monkeypatch.setattr(person_model, "CLAIM_P", 1.01)
    plain, _ = _options(conn, [a, b])
    monkeypatch.undo()
    _cut_keeps_them_off(monkeypatch, w)
    said: dict[int, list[float]] = {}
    monkeypatch.setattr(person_model.Model, "probability",
                        lambda self, embs, weights, **k: np.asarray(said[self.person_id][: len(embs)], dtype=np.float64))
    # under the cut nothing changes, at the cut the face is kept
    said[w.me] = [0.79, 0.8]
    options, kept = _options(conn, [a, b])
    assert kept == {b} and options == {a: plain[a], b: []}
    # a face the owner said is not the person is never kept for them
    said[w.me] = [0.99, 0.99]
    with tx(conn):
        cluster.record_rejections(conn, [a], w.me)
    options, kept = _options(conn, [a, b])
    assert kept == {b} and options[a] == plain[a]
    # a paused model keeps nothing (it would make no guess for the face), nor does one switched off
    person_model.pause(conn, w.me)
    assert _options(conn, [a, b]) == (plain, set())
    person_model.resume(conn, w.me)
    assert _options(conn, [a, b])[1] == {b}
    store.set_enabled(conn, False)
    assert _options(conn, [a, b]) == (plain, set())


def test_two_models_sure_of_one_face(conn, monkeypatch):
    import dataclasses

    w = World(conn)
    w.train()
    mine = person_model.load(w.me)
    theirs = dataclasses.replace(mine, person_id=w.third)
    monkeypatch.setattr(person_model, "trained_ids", lambda: [w.me, w.third])
    monkeypatch.setattr(person_model, "load", lambda pid: {w.me: mine, w.third: theirs}.get(int(pid)))
    said: dict[int, float] = {}
    monkeypatch.setattr(person_model.Model, "probability",
                        lambda self, embs, weights, **k: np.full(len(embs), said[self.person_id]))
    # the second model is a copy without faces of its own: here only who has the face counts, not whether the
    # guess path would offer it (the tests below)
    monkeypatch.setattr(person_scan, "offerable",
                        lambda conn, idx, model, face_ids, *a, **k: np.ones(len(face_ids), dtype=bool))
    face = w.face(w.photo(), w.like_me())
    idx = get_index()
    opts = [[(w.twin, 0.9), (w.me, 0.8), (w.third, 0.7)]]

    def kept_for() -> list[list[tuple[int, float]]]:
        with idx.lock:
            idx.sync(conn)
            return cluster._claim(conn, idx, [face], idx.embeddings_of([face]), idx.child_flags([face]), frozenset(), opts, None)

    said.update({w.me: 0.85, w.third: 0.95})
    assert kept_for() == [[(w.third, 0.7)]]  # the higher probability has the face
    said.update({w.me: 0.95, w.third: 0.85})
    assert kept_for() == [[(w.me, 0.8)]]
    said.update({w.me: 0.9, w.third: 0.7})
    assert kept_for() == [[(w.me, 0.8)]]
    said.update({w.me: 0.7, w.third: 0.7})
    assert kept_for() == opts


def test_the_claim_leaves_a_child_band_face_to_the_child_rules(conn, monkeypatch):
    from app.analysis import ages
    from tests.test_age_matching import CHILD, ready

    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    kid = w.face(w.photo(), w.like_me(), age_groups=ages.encode(CHILD))
    grown = w.face(w.photo(), w.like_me())
    ready(conn)  # the ages of the library are read: the child band is in force
    idx = get_index()
    idx.sync(conn)
    assert idx.child_flags([kid, grown]).tolist() == [True, False]
    monkeypatch.setattr(person_model, "CLAIM_P", 1.01)
    plain, _ = _options(conn, [kid, grown])
    monkeypatch.undo()
    _cut_keeps_them_off(monkeypatch, w)
    options, kept = _options(conn, [kid, grown])
    assert kept == {grown} and options[kid] == plain[kid] and options[grown] == []


def test_without_a_model_the_matching_is_what_it_was(conn, monkeypatch):
    w = World(conn)
    _cut_keeps_them_off(monkeypatch, w)
    faces = [w.face(w.photo(), w.like_me()), w.face(w.photo(), w.like_twin()), w.face(w.photo(), unit(next(w.seed)))]
    idx = get_index()

    def scored(**more) -> list[list[tuple[int, float]]]:
        with idx.lock:
            idx.sync(conn)
            return cluster._score(idx, cluster._face_rows(conn, faces), frozenset(), conn, **more)

    kept: dict[int, cluster.Kept] = {}
    before = scored()
    assert scored(claimed=kept) == before and not kept  # nobody trained
    w.train()
    assert scored(claimed=kept) != before and set(kept) == {faces[0]}
    store.set_enabled(conn, False)  # learning off: the trained recognition has no say
    kept.clear()
    assert scored(claimed=kept) == before and not kept


# ------------------------------------------------------------------ the claim asks what the guess path asks
def _someone_named_later(conn, w: World) -> tuple[int, list[int], list[int]]:
    """A person the owner named after the training, so none of their faces was a negative for the model, and it
    takes their faces for the trained person's (cosine about 0.8 to that identity). Returns (the person, their
    six tagged faces, five new faces of theirs)."""
    o = unit(77) - float(unit(77) @ ME) * ME
    base = (0.8 * ME + 0.6 * o / np.linalg.norm(o)).astype(np.float32)
    other = make_person(conn, "Lena")
    tagged = [w.face(w.photo(), near(base, next(w.seed), 0.3), other, "user") for _ in range(6)]
    return other, tagged, [w.face(w.photo(), near(base, next(w.seed), 0.3)) for _ in range(5)]


def _model_says(conn, person_id: int, faces: list[int]) -> np.ndarray:
    idx = get_index()
    model = person_model.load(person_id)
    with idx.lock:
        idx.sync(conn)
        return person_scan._Judge(conn, model).score(idx.embeddings_of(faces), idx.weights_of(faces))[0]


@pytest.mark.parametrize("cut", [None, 0.95])
def test_a_face_another_named_person_matches_better_is_not_kept(conn, monkeypatch, cut):
    """With the trained person among the face's options (their cut lets it) and without: someone else with a name
    matches the face better, the guess path would not offer it (person_scan.offerable), so the claim leaves it
    alone. Before, the first case tagged the trained person on the other person's new faces."""
    w = World(conn)
    w.train()
    monkeypatch.setattr(cluster, "_learned", lambda pid: cut if pid == w.me else None)
    other, tagged, fresh = _someone_named_later(conn, w)
    assert (_model_says(conn, w.me, fresh) >= person_model.CLAIM_P).all()  # the model is sure of every one
    options, kept = _options(conn, fresh)
    assert kept == set() and all(options[f][0] == other for f in fresh)
    assert (w.me in options[fresh[0]]) == (cut is None)
    # the faces left out of the scores are left out of that question too: without the other person's own faces
    # nobody matches these better than the trained person, and the model has them
    assert _options(conn, fresh, frozenset(tagged))[1] == set(fresh)
    got = cluster.assign_new_faces(conn, fresh)
    assert [got[f] for f in fresh] == [other] * 5 and _guess_jobs(conn) == set()


def test_a_face_that_looks_like_one_taken_off_the_person_is_not_kept(conn, monkeypatch):
    """The other refusal of the guess path: a face the owner took off the person is as near as their own faces
    are. A model that is sure all the same keeps nothing, and the face goes where the plain rules send it."""
    w = World(conn)
    w.train()
    with tx(conn):  # the look-alike has no name here: only the faces taken off the person speak against them
        conn.execute("UPDATE persons SET name = NULL, name_key = NULL WHERE id = ?", (w.twin,))
    face = w.face(w.photo(), w.like_twin())
    monkeypatch.setattr(person_model, "CLAIM_P", 1.01)
    plain, _ = _options(conn, [face])
    monkeypatch.undo()
    assert plain[face] == [w.twin]
    monkeypatch.setattr(person_model.Model, "probability", lambda self, embs, weights, **k: np.full(len(embs), 0.99))
    assert _options(conn, [face]) == (plain, set())
    assert cluster.assign_new_faces(conn, [face])[face] == w.twin and _guess_jobs(conn) == set()


def test_what_the_claim_never_keeps(conn, monkeypatch):
    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    face = w.face(w.photo(), w.like_me())
    blurred = w.face(w.photo(), w.like_me(), clarity=1.0)
    idx = get_index()
    idx.sync(conn)
    assert idx.weights_of([face, blurred]).tolist() == [1.0, 0.0]
    # a face of weight 0, though the model is as sure of it: merge_item makes no guess for one
    assert (_model_says(conn, w.me, [face, blurred]) >= person_model.CLAIM_P).all()
    assert _options(conn, [face, blurred])[1] == {face}
    # the faces left out of the scores are left out of the model's reference: with none of the person's own
    # faces to compare with, it is not sure of this one
    assert _options(conn, [face], frozenset(w.mine))[1] == set()
    # a pet (a person turned into one keeps the model file): the recognition is trained on people
    with tx(conn):
        conn.execute("UPDATE persons SET kind = 'pet' WHERE id = ?", (w.me,))
    assert _options(conn, [face])[1] == set()


# ------------------------------------------------------------------ a kept face and its guess
def test_lookalikes_that_follow_an_unassign_get_their_guess(client, conn, monkeypatch):
    """Only the analysis guesses for its own item. On every other path (here the look-alikes that follow an
    unassign) a face kept loose gets a guess job for its item: without one it would wait with no person, no group
    and no guess, and every later recluster would keep it loose again."""
    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    seeds = [w.face(w.photo(), w.like_me(), w.twin) for _ in range(2)]  # the person's faces, wrongly on the look-alike
    more = [w.face(w.photo(), w.like_me(), w.twin) for _ in range(3)]
    assert set(cluster.followers(conn, seeds, w.twin)) == set(more)
    assert client.post("/api/faces/assign", json={"face_ids": seeds, "person_id": None}).json()["followed"] == 3
    assert all((row(conn, f)["person_id"], row(conn, f)["assigned_by"]) == (None, "auto") for f in more)
    items = {int(row(conn, f)["item_id"]) for f in more}
    assert {(w.me, i) for i in items} <= _guess_jobs(conn)
    assert cluster.recluster_unassigned(conn) == 0  # they wait for the guess
    for i in sorted(items):  # what the worker does with such a job
        guess.propagate(conn, w.me, i)
    assert all(guessed(conn, f)[:2] == (w.me, "guess") for f in more)


def test_the_analysis_asks_for_no_guess_job(conn, monkeypatch):
    """The analysis runs guess_item for its item itself (pipeline._after), so it turns the job off."""
    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    face = w.face(w.photo(), w.like_me())
    assert cluster.assign_new_faces(conn, [face], guess_kept=False)[face] is None and _guess_jobs(conn) == set()
    assert guess.guess_item(conn, row(conn, face)["item_id"]) == 1 and guessed(conn, face)[:2] == (w.me, "guess")


# ------------------------------------------------------------------ a kept face no guess can reach
def test_two_faces_of_one_photo_the_model_keeps(conn, monkeypatch):
    """A person is in a photo once. Their cut lets them have a face: the better one is theirs and the other,
    which the plain matching would give to the look-alike, starts a person of its own. Their cut keeps both off:
    the one the model is surest of waits for the guess, the other joins an unnamed person it matches."""
    w = World(conn)
    w.train()
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    here = w.photo()
    a, b = w.face(here, near(ME, 4242, 0.1)), w.face(here, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    monkeypatch.setattr(person_model, "CLAIM_P", 1.01)
    assert _options(conn, [a, b])[0] == {a: [w.me, w.twin], b: [w.me, w.twin]}  # the plain rules: b to the look-alike
    monkeypatch.undo()
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    got = cluster.assign_new_faces(conn, [a, b])
    own = got[b]
    assert got[a] == w.me and _is_unnamed(conn, own) and _guess_jobs(conn) == set()
    _cut_keeps_them_off(monkeypatch, w)
    there = w.photo()
    c, d = w.face(there, w.like_me()), w.face(there, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    sure = _claims(conn, [c, d])
    assert all(k.person_id == w.me and [pid for pid, _ in k.rest] == [own] for k in sure.values())
    first, second = sorted((c, d), key=lambda f: (-sure[f].p, f))
    got = cluster.assign_new_faces(conn, [c, d])
    assert (got[first], got[second]) == (None, own) and _guess_jobs(conn) == {(w.me, there)}
    assert guess.guess_item(conn, there) == 1 and guessed(conn, first)[:2] == (w.me, "guess")


def test_a_kept_track_of_a_clip_the_person_is_in(conn, monkeypatch):
    """A clip may show one person several times, but a guess never names someone who is in it, and a confirmation
    names them once. So a track kept for a person who is tagged on another track (before, or in this very pass)
    could never be answered: it starts or joins an unnamed person, and is never the look-alike's."""
    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    clip = w.photo(kind="video")
    w.face(clip, w.like_me(), w.me, "user")
    track = w.face(clip, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    assert _options(conn, [track]) == ({track: []}, {track})
    own = cluster.assign_new_faces(conn, [track])[track]
    assert _is_unnamed(conn, own) and _guess_jobs(conn) == set()
    # their cut lets them have the nearer face of a new clip only: once it is theirs, the other track cannot wait
    monkeypatch.setattr(cluster, "_learned", lambda pid: 0.86 if pid == w.me else None)
    again = w.photo(kind="video")
    nearer, other = w.face(again, near(ME, 4243, 0.1)), w.face(again, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    assert _options(conn, [nearer, other]) == ({nearer: [w.me], other: []}, {nearer, other})
    got = cluster.assign_new_faces(conn, [nearer, other])
    assert (got[nearer], got[other]) == (w.me, own) and _guess_jobs(conn) == set()


@pytest.mark.parametrize("how", ["vault", "person box", "their own person box", "tag"])
def test_a_kept_face_where_no_guess_is_made(conn, monkeypatch, how):
    """guess_item makes no guesses in the vault, and names nobody who is in the item through a person box or a
    tag: a face kept for the person there starts a person of its own instead of waiting for nothing. (A person
    box around the face itself counts when the owner tagged it; one the app tagged follows its face.)"""
    w = World(conn)
    w.train()
    _cut_keeps_them_off(monkeypatch, w)
    item = w.photo(nsfw=1 if how == "vault" else 0)
    if how == "person box":
        make_body(conn, item, w.me, assigned_by="user")
    elif how == "tag":
        with tx(conn):
            conn.execute("INSERT INTO item_tags(item_id, person_id, relation, source, created_at) "
                         "VALUES (?, ?, 'in', 'user', ?)", (item, w.me, "2026-03-20T10:00:00.000Z"))
    face = w.face(item, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    if how == "their own person box":
        make_body(conn, item, w.me, assigned_by="user", face_id=face)
    assert _options(conn, [face]) == ({face: []}, {face})
    own = cluster.assign_new_faces(conn, [face])[face]
    assert _is_unnamed(conn, own) and _guess_jobs(conn) == set()
    assert guess.guess_item(conn, item) == 0


def test_recluster_lets_a_kept_face_join_an_unnamed_group(conn, monkeypatch):
    """Recluster starts no people, so a kept face that no guess can reach stays loose there unless an unnamed
    person is among the options it had: that one it may join (never the look-alike)."""
    w = World(conn)
    w.train()
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)
    here = w.photo()
    w.face(here, w.like_me(), w.me, "user")
    second = w.face(here, w.like_me(), box=(0.6, 0.2, 0.2, 0.3))
    assert cluster.recluster_unassigned(conn) == 0 and row(conn, second)["person_id"] is None
    group = unnamed(conn)
    for _ in range(3):
        w.face(w.photo(), w.like_me(), group)
    kept = _claims(conn, [second])[second]
    assert kept.person_id == w.me and [pid for pid, _ in kept.rest] == [group]
    assert cluster.recluster_unassigned(conn) == 1 and row(conn, second)["person_id"] == group


def test_the_models_reference_leaves_faces_out_on_request(conn):
    w = World(conn)
    w.train()
    model = person_model.load(w.me)
    get_index().sync(conn)
    emb = np.frombuffer(row(conn, w.mine[7])["embedding"], dtype=np.float32)[None, :]
    assert model.top3(emb, conn=conn)[0] > 0.86  # the face itself is among its three best
    alone = model.top3(emb, exclude_ids={w.mine[7]}, conn=conn)[0]
    assert alone < model.top3(emb, conn=conn)[0] and alone > 0.7
    assert model.top3(emb, exclude_ids=frozenset(w.mine), conn=conn)[0] == 0.0  # no reference left
    assert (model.probability(emb, [1.0], exclude_ids=frozenset(w.mine), conn=conn)[0]
            < model.probability(emb, [1.0], exclude_ids={w.mine[7]}, conn=conn)[0])


# ------------------------------------------------------------------ routes, second part
def test_a_run_lost_to_a_restart_is_reported(client, conn):
    from app.db import kv_set

    w = World(conn)
    url = f"/api/people/{w.me}/recognition"
    person_scan._mark_run(conn, w.me, True, True)  # the app stopped under a training: its mark, and no run
    s = client.get(url).json()
    assert s["running"] is None and s["trained"] is False and "training stopped" in s["error"]
    assert client.get(f"/api/people/{w.third}/recognition").json()["error"] is None
    # a run that ends takes its mark away
    assert client.post(f"{url}/train", json={}).status_code == 202
    s = client.get(url).json()
    assert s["trained"] and s["error"] is None and s["check"]["folds"] == "day"
    assert conn.execute("SELECT COUNT(*) FROM kv WHERE key LIKE 'recognition_run:%'").fetchone()[0] == 0
    # stopped under the look, the training done; and under a look alone
    with tx(conn):
        kv_set(conn, person_scan.RUN_KEY.format(w.me),
               json.dumps({"train": True, "look": True, "started_at": "2000-01-01T00:00:00.000Z"}))
    assert "It was trained, but the look" in client.get(url).json()["error"]
    person_scan._mark_run(conn, w.me, False, True)
    assert client.get(url).json()["error"].startswith("The look at other faces stopped")
    # the next run starts clean
    assert client.post(f"{url}/scan").status_code == 202 and client.get(url).json()["error"] is None


def test_a_failed_run_leaves_no_mark(client, conn, monkeypatch):
    w = World(conn)

    def boom(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(person_model, "train", boom)
    assert client.post(f"/api/people/{w.me}/recognition/train", json={}).status_code == 202
    assert client.get(f"/api/people/{w.me}/recognition").json()["error"] == "It stopped on an error. Try again."
    assert conn.execute("SELECT COUNT(*) FROM kv WHERE key LIKE 'recognition_run:%'").fetchone()[0] == 0
    assert person_scan.busy_with() is None


def test_the_conflicts_cursor_skips_nothing_when_one_is_settled(client, conn):
    w = World(conn)
    for k in range(3):
        w.conflicts.append(w.face(w.photo(day=5 + k), w.like_twin(), w.me))
    w.train()
    url = f"/api/people/{w.me}/recognition/conflicts"
    everything = [c["face"]["id"] for c in client.get(url).json()["items"]]
    assert sorted(everything) == sorted(w.conflicts) and len(everything) == 5
    first = client.get(f"{url}?limit=2").json()
    assert [c["face"]["id"] for c in first["items"]] == everything[:2] and first["total"] == 5
    # the owner settles one of the page just read; the next page still starts where that page ended
    assert client.post("/api/faces/assign", json={"face_ids": [everything[0]], "person_id": w.me}).status_code == 200
    rest = client.get(f"{url}?limit=2&cursor={first['next_cursor']}").json()
    assert [c["face"]["id"] for c in rest["items"]] == everything[2:4] and rest["total"] == 4
    last = client.get(f"{url}?limit=2&cursor={rest['next_cursor']}").json()
    assert [c["face"]["id"] for c in last["items"]] == everything[4:] and last["next_cursor"] is None


def test_training_is_offered_by_face_weight_not_by_count(client, conn):
    from app.analysis.faces import face_weight

    size = next(px for px in range(25, 48) if 0.3 <= face_weight(0.9, px) <= 0.6)
    small = make_person(conn, "Omar")
    base = unit(11)
    for k in range(25):  # 25 faces, but small ones: they count for less than 20
        item = make_item(conn, write_files=False, data=None)
        make_face(conn, item["id"], small, embedding=near(base, 500 + k), size_px=size, write_crop=False)
    weight, faces = person_model.clear_weight(conn, small)
    assert faces == 25 and 5 < weight < 20
    url = f"/api/people/{small}/recognition"
    s = client.get(url).json()
    assert s["available"] is False and f"Their 25 faces count as about {int(weight)}" in s["reason"]
    assert client.post(f"{url}/train", json={}).status_code == 409
    for k in range(20):  # clear faces added: now a training has enough to start from
        item = make_item(conn, write_files=False, data=None)
        make_face(conn, item["id"], small, embedding=near(base, 600 + k), write_crop=False)
    s = client.get(url).json()
    assert s["available"] is True and s["reason"] is None and person_model.clear_weight(conn, small)[0] >= 20


def test_leave_is_remembered(client, conn):
    w = World(conn)
    w.train()
    on_named = [w.face(w.photo(), w.like_me(), w.third) for _ in range(2)]
    person_scan.scan(conn, w.me)
    url = f"/api/people/{w.me}/recognition"
    assert client.get(url).json()["elsewhere"] == 2
    r = client.post(f"{url}/leave", json={"face_ids": [on_named[0]]})
    assert r.status_code == 200 and r.json() == {"updated": 1} and r.headers.get("x-action-id")
    s = client.get(url).json()
    assert (s["elsewhere"], s["answers_since"]) == (1, 1)
    assert [e["face"]["id"] for e in client.get(f"{url}/elsewhere").json()["items"]] == [on_named[1]]
    kept = row(conn, on_named[0])
    assert (kept["person_id"], kept["assigned_by"]) == (w.third, "auto")  # the face stays where it is
    assert client.post(f"{url}/leave", json={"face_ids": [on_named[0]]}).json() == {"updated": 0}
    # only a face on another person can be left there
    assert client.post(f"{url}/leave", json={"face_ids": [w.mine[0]]}).status_code == 400
    # the next look does not list it again
    assert [e["face_id"] for e in person_scan.scan(conn, w.me)["elsewhere"]] == [on_named[1]]
    # the answer is an owner action: its undo forgets it
    assert client.post(f"/api/actions/{r.headers['x-action-id']}/undo").status_code == 200
    assert sorted(e["face_id"] for e in person_scan.scan(conn, w.me)["elsewhere"]) == on_named
    # and the next training takes a face that was left as one that is not the person
    assert client.post(f"{url}/leave", json={"face_ids": [on_named[0]]}).json() == {"updated": 1}
    assert w.train()["counts"]["rejected"] == 13


def test_take_back_and_forget_say_whose_run_is_going(client, conn, monkeypatch):
    w = World(conn)
    w.train()
    url = f"/api/people/{w.me}/recognition"
    monkeypatch.setattr(person_scan, "_running", w.me)  # a run for this very person
    for r in (client.post(f"{url}/withdraw"), client.delete(url)):
        assert r.status_code == 409 and r.json()["detail"] == "running"
    assert client.post(f"{url}/scan").json() == {"started": False}
    monkeypatch.setattr(person_scan, "_running", w.third)  # another person's run
    r = client.post(f"{url}/scan")
    assert r.status_code == 409 and r.json()["detail"] == "busy"
    assert client.get(url).json()["busy"] is True
    assert client.post(f"{url}/withdraw").status_code == 200


def test_taking_an_existing_name_merges_and_forgets_the_model(client, conn):
    w = World(conn)
    w.train()
    assert person_model.trained_ids() == [w.me]
    r = client.patch(f"/api/people/{w.me}", json={"name": "meera"})
    assert r.status_code == 200 and r.json()["id"] == w.third, r.text
    assert person_model.trained_ids() == [] and person_model.stored(w.me) is None


def test_undo_brings_the_model_of_a_deleted_or_merged_person_back(client, conn):
    w = World(conn)
    w.train()
    loose = w.face(w.photo(), w.like_me())
    person_scan.scan(conn, w.me)
    trained_at = person_model.stored(w.me)["trained_at"]

    def undo(response) -> dict:
        action = response.headers["x-action-id"] if "x-action-id" in response.headers else response.json()["undo_id"]
        u = client.post(f"/api/actions/{action}/undo")
        assert u.status_code == 200, u.text
        return u

    def back() -> None:
        assert person_model.trained_ids() == [w.me] and person_model.load(w.me).trained_at == trained_at
        assert guessed(conn, loose)[0] == w.me
        assert guess.guess_item(conn, row(conn, loose)["item_id"]) == 0  # the model answers for its guess again

    r = client.delete(f"/api/people/{w.me}")
    assert r.status_code == 200 and person_model.trained_ids() == [] and guessed(conn, loose)[0] is None
    first = undo(r)
    back()
    again = undo(first)  # a redo: the person and the model go again
    assert person_model.trained_ids() == [] and conn.execute("SELECT 1 FROM persons WHERE id = ?", (w.me,)).fetchone() is None
    undo(again)
    back()
    r = client.post("/api/people/merge", json={"target_id": w.third, "source_ids": [w.me]})
    assert r.status_code == 200 and person_model.trained_ids() == []
    undo(r)
    back()
    r = client.patch(f"/api/people/{w.me}", json={"name": "Meera"})
    assert r.status_code == 200 and person_model.trained_ids() == []
    undo(r)
    back()
