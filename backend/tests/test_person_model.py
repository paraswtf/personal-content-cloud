"""Recognition trained on one person (section 10: app.learning.person_model, app.analysis.person_scan), with
hand-made face vectors: an identity, a look-alike identity at cosine about 0.7 to it, and strangers.

The training (examples, conflicts, the check, determinism), the scan (what may become a guess and what is never
touched), guess_item recomputing the guesses, the review actions, the gate in the automatic matching, and the routes.
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

    def __init__(self, conn, faces: int = 40, lookalike: np.ndarray = TWIN) -> None:
        self.conn = conn
        self.lookalike = lookalike
        self.seed = count(1000)
        self.me = make_person(conn, "Asha")
        self.twin = make_person(conn, "Ravi")
        self.third = make_person(conn, "Meera")
        self.mine: list[int] = []
        self.minute = count(0)
        for k in range(faces):
            item = self.photo(day=1 + k % 10)
            self.mine.append(self.face(item, near(ME, next(self.seed)), self.me, "user" if k < 5 else "auto"))
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
    model = w.train()
    # the mirror is left out (not a sure negative); the look-alike matches the third person better: a negative
    assert model["counts"]["ambiguous"] == 1 and model["counts"]["same_photo"] == 42 + 1
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
    assert person_model.load(w.me).withdrawn
    assert guess.guess_item(conn, row(conn, loose)["item_id"]) == 0 and guessed(conn, loose)[0] is None  # paused
    assert person_scan.scan(conn, w.me)["guesses"] == 2 and not person_model.load(w.me).withdrawn


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
    assert w.twin in gated[faces[0]]  # other people's options are never the gate's business


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
    assert guessed(conn, detached)[0] == guessed(conn, loose)[0] == w.me
    assert guess.guess_item(conn, row(conn, loose)["item_id"]) == 0  # the paused model leaves the restored guess
    assert client.post(f"{url}/scan").status_code == 202 and client.get(url).json()["withdrawn"] is False
    assert client.delete(url).status_code == 200
    s = client.get(url).json()
    assert s["trained"] is False and s["guesses"] == 0 and guessed(conn, loose)[0] is None


def test_routes_refuse_what_cannot_be_trained(client, conn):
    few = make_person(conn, "Asha")
    item = make_item(conn, write_files=False, data=None)
    make_face(conn, item["id"], few, write_crop=False)
    url = f"/api/people/{few}/recognition"
    s = client.get(url).json()
    assert s["available"] is False and "at least 20 faces" in s["reason"]
    r = client.post(f"{url}/train", json={})
    assert r.status_code == 409 and "at least 20 faces" in r.json()["detail"]
    with tx(conn):
        pet = int(conn.execute("INSERT INTO persons(name, name_key, created_at, updated_at, kind, species) "
                               "VALUES ('Chikki', 'chikki', '2026-01-01', '2026-01-01', 'pet', 'dog')").lastrowid)
    assert client.get(f"/api/people/{pet}/recognition").status_code == 404
    assert client.get("/api/people/999999/recognition").status_code == 404
    w = World(conn)
    store.set_enabled(conn, False)
    s = client.get(f"/api/people/{w.me}/recognition").json()
    assert s["available"] is False and "switched off" in s["reason"]
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
