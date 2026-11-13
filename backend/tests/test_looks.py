"""The looks of a person (app.analysis.looks, docs/ARCHITECTURE.md section 36): the geometry, the bins and the
choice with hand-made readings, the sample over the years, and the plan and the generator's picture on rows without
any model (PCC_NO_MODELS: the hair and facial hair are unknown, the bins are age bands)."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import ages, looks
from app.analysis.looks import Reading
from app.db import kv_set, tx
from tests.helpers import make_body, make_face, make_item, make_person


def reading(face_id: int, *, age: float = 30.0, hair: float | None = 0.6, facial: float | None = 0.5,
            faults: tuple[str, ...] = (), day: str = "2024-05-01", size: int = 400, yaw: float | None = 0.0,
            weight: float = 1.0, child: bool = False, year: float | None = 2024.3, expected: float | None = None,
            sharpness: float = 40.0) -> Reading:
    return Reading(face_id=face_id, item_id=1000 + face_id, item_uid=f"u{face_id}", taken_at=f"{day}T10:00:00.000Z",
                   year=year, box=(0.4, 0.3, 0.2, 0.3), width=1000, height=800, weight=weight, size_px=size,
                   sharpness=sharpness, expected_age=age if expected is None else expected, child=child, age=age,
                   facial_hair=facial, hair=hair, yaw=yaw, pitch=0.55, roll=0.0, faults=list(faults))


def groups(index: int) -> bytes:
    g = np.zeros(9, dtype=np.float32)
    g[index] = 1.0
    return ages.encode(g)


# ------------------------------------------------------------------ geometry
def test_square_sits_around_the_face_a_little_low():
    left, top, side = looks.square_for((0.4, 0.3, 0.2, 0.25), 1000, 800, 2.6)
    assert side == pytest.approx(2.6 * 200)                    # face 200 x 200 px: the larger side counts
    assert left + side / 2 == pytest.approx(500)               # centred on the face across
    assert top + side / 2 == pytest.approx(340 + looks.FACE_ABOVE * side)  # the face is above the middle
    assert looks.outside_share((left, top, side), 1000, 800) == 0.0
    # a face at the edge: the square leaves the picture
    sq = looks.square_for((0.0, 0.0, 0.2, 0.25), 1000, 800, 2.6)
    assert 0.3 < looks.outside_share(sq, 1000, 800) < 0.7
    assert looks.outside_share((0.0, 0.0, 0.0), 10, 10) == 1.0


def test_another_face_in_the_square_counts_unless_it_is_far_away():
    sq = looks.square_for((0.4, 0.3, 0.2, 0.25), 1000, 800)
    own = (400.0, 240.0, 200.0, 200.0)
    assert looks.other_face_inside(sq, own, [(620.0, 260.0, 150.0, 150.0)])       # a neighbour, wholly inside
    assert not looks.other_face_inside(sq, own, [(900.0, 260.0, 150.0, 150.0)])   # outside the square
    assert not looks.other_face_inside(sq, own, [(620.0, 260.0, 40.0, 40.0)])     # small: someone in the background
    assert not looks.other_face_inside(sq, own, [(730.0, 260.0, 150.0, 150.0)])   # only a sliver of it is inside
    assert not looks.other_face_inside(sq, own, [])


def test_head_pose_from_the_five_points():
    front = np.array([[40, 50], [80, 50], [60, 72], [45, 90], [75, 90]], dtype=np.float32)
    yaw, pitch, roll = looks.head_pose(front)
    assert abs(yaw) < 1e-6 and pitch == pytest.approx(0.55) and abs(roll) < 1e-6
    turned = front.copy()
    turned[2, 0] = 76  # the nose towards one eye
    assert abs(looks.head_pose(turned)[0]) > looks.YAW_MAX
    chin_up = front.copy()
    chin_up[2, 1] = 58  # the nose near the eye line
    assert looks.head_pose(chin_up)[1] < looks.PITCH_MIN
    leaning = np.array([[40, 40], [75, 60], [52, 70], [35, 80], [62, 95]], dtype=np.float32)
    assert abs(looks.head_pose(leaning)[2]) > looks.ROLL_MAX
    assert looks.head_pose(np.zeros((5, 2))) == (0.0, 0.55, 0.0)  # no points, no claim


def test_cut_repeats_the_edge_where_the_square_leaves_the_picture():
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[:, :100] = (255, 0, 0)
    rgb[:, 100:] = (0, 0, 255)
    inside = looks.cut(rgb, (50.0, 0.0, 100.0), 50)
    assert inside.shape == (50, 50, 3) and tuple(inside[25, 5]) == (255, 0, 0) and tuple(inside[25, 45]) == (0, 0, 255)
    past = looks.cut(rgb, (-50.0, -50.0, 100.0), 40)
    assert tuple(past[2, 2]) == (255, 0, 0) and tuple(past[38, 38]) == (255, 0, 0)  # padded with the edge colour
    assert looks.cut(rgb, (500.0, 500.0, 50.0), 16).shape == (16, 16, 3)            # wholly outside: black


def test_inside_is_the_part_of_the_cut_that_is_the_photo():
    """Where the square leaves the photo, cut() repeats the edge; inside() says where the photo itself is, as
    (left, top, right, bottom) in the cut picture's pixels."""
    # wholly in the photo: all of the picture, at any size
    assert looks.inside((50.0, 0.0, 100.0), 200, 100, 50) == (0, 0, 50, 50)
    assert looks.inside((50.0, 0.0, 100.0), 200, 100, 400) == (0, 0, 400, 400)
    # past the top left corner by half the square
    assert looks.inside((-50.0, -50.0, 100.0), 200, 100, 40) == (20, 20, 40, 40)
    # past the right edge by half and past the bottom by two fifths
    assert looks.inside((150.0, 40.0, 100.0), 200, 100, 100) == (0, 0, 50, 60)
    # larger than the photo on every side
    assert looks.inside((-100.0, -100.0, 400.0), 200, 100, 400) == (100, 100, 300, 200)
    # rounded inwards when the sizes do not divide: 10 px of 300 are 3.33 px of 100
    assert looks.inside((-10.0, 0.0, 300.0), 200, 1000, 100) == (4, 0, 70, 100)
    # the square misses the photo
    assert looks.inside((500.0, 500.0, 50.0), 200, 100, 16) == (0, 0, 0, 0)
    assert looks.inside((-300.0, 0.0, 100.0), 200, 100, 16) == (0, 0, 0, 0)

    # and cut() agrees: past the rectangle the picture is the photo's last column and last row again
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(200, dtype=np.uint8)[None, :]              # every column its own red
    rgb[..., 1] = np.arange(100, dtype=np.uint8)[:, None]              # every row its own green
    square = (150.0, 40.0, 100.0)
    out = looks.cut(rgb, square, 100)
    _left, _top, right, bottom = looks.inside(square, 200, 100, 100)
    assert (out[:, right:, 0] == 199).all() and (out[bottom:, :, 1] == 99).all()
    assert out[0, right - 1, 0] == 199 and out[0, right - 2, 0] == 198   # the photo's own last columns
    assert out[bottom - 1, 0, 1] == 99 and out[bottom - 2, 0, 1] == 98   # and rows


# ------------------------------------------------------------------ bins, ages, the choice
def test_bins_and_labels():
    assert [looks.age_band(a) for a in (0, 9.9, 10, 17.9, 18, 24.9, 30, 40, 50, 60, 99)] == [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6]
    assert looks.age_band(None) is None and looks.age_band(float("nan")) is None
    assert looks.bin_of(reading(1, age=30, hair=0.2, facial=0.3)) == (3, 0, 0)
    assert looks.bin_of(reading(1, age=30, hair=0.45, facial=0.5)) == (3, 1, 1)
    assert looks.bin_of(reading(1, age=30, hair=0.7, facial=0.6)) == (3, 2, 2)
    assert looks.bin_of(reading(1, age=30, hair=0.9, facial=0.99)) == (3, 2, 3)
    assert looks.bin_of(reading(1, age=30, hair=None, facial=None)) == (3, None, None)   # no SigLIP: age bands only
    assert looks.bin_of(reading(1, age=30, faults=("covered_head",))) is None            # the hair under a cap is unknown
    r = reading(1)
    r.age = None
    assert looks.bin_of(r) is None
    assert looks.look_label(24.4, "18 to 24", "short hair", "a beard") == "About 24, short hair, a beard"
    assert looks.look_label(None, "25 to 34", None, None) == "25 to 34"


def test_ages_come_from_the_year_once_the_birth_year_is_known():
    rs = [reading(k, year=2020.0 + (k % 5), expected=28.0 + (k % 5) + (0.5 if k % 2 else -0.5)) for k in range(12)]
    rs.append(reading(50, year=2026.0, expected=6.0))                 # a re-shot childhood print: read young
    rs.append(reading(51, year=None, expected=31.0))                  # a forward: no date to trust
    rs.append(reading(52, year=2026.0, expected=5.0, child=True))
    birth = looks.settle_ages(rs)
    assert birth == pytest.approx(1992.0, abs=0.6)
    assert rs[0].age == pytest.approx(2020.0 - birth) and rs[4].age == pytest.approx(2024.0 - birth)
    assert rs[12].age == 6.0 and rs[13].age == 31.0 and rs[14].age == 5.0
    few = [reading(k, year=2020.0, expected=30.0) for k in range(5)]
    assert looks.settle_ages(few) is None and few[0].age == 30.0      # too few dated faces: the read age


def test_a_day_shares_its_levels():
    a, b, c = reading(1, hair=0.60), reading(2, hair=0.64), reading(3, hair=0.20, faults=("turned",))
    other = reading(4, hair=0.9, day="2024-06-09")
    undated = reading(5, hair=0.1, year=None)
    looks.share_day_levels([a, b, c, other, undated])
    assert a.hair == b.hair == c.hair == 0.62        # the median of the day's photos without a fault
    assert other.hair == 0.9 and undated.hair == 0.1  # alone on its day, or no date: its own reading
    x, y = reading(6, facial=0.4, faults=("turned",), day="2024-07-01"), reading(7, facial=0.6, faults=("cut_off",), day="2024-07-01")
    looks.share_day_levels([x, y])
    assert x.facial_hair == y.facial_hair == 0.5      # no clean photo that day: the median of all


def test_choose_takes_the_fullest_bin_of_every_age_band_first():
    rs = []
    rs += [reading(k, age=30, hair=0.6, facial=0.5) for k in range(1, 9)]            # 25 to 34, medium, short beard: 8
    rs += [reading(k, age=30, hair=0.9, facial=0.5) for k in range(20, 25)]          # 25 to 34, long: 5
    rs += [reading(k, age=30, hair=0.6, facial=0.7) for k in range(30, 34)]          # 25 to 34, medium, beard: 4
    rs += [reading(k, age=20, hair=0.4, facial=0.3) for k in range(40, 43)]          # 18 to 24: 3
    rs += [reading(k, age=45, hair=0.4, facial=0.3) for k in range(50, 52)]          # 45 to 59: only 2, not a look
    rs += [reading(60, age=6, hair=0.4, facial=0.2, child=True)]                     # a child: one photo is enough
    chosen = looks.choose(rs, 6)
    assert [(lk.age_band, lk.hair_label, lk.facial_hair_label, lk.photos) for lk in chosen] == [
        ("under 10", "short hair", "little or no facial hair", 1),
        ("18 to 24", "short hair", "little or no facial hair", 3),
        ("25 to 34", "medium hair", "stubble or a short beard", 8),
        ("25 to 34", "medium hair", "a beard", 4),
        ("25 to 34", "long hair", "stubble or a short beard", 5),
    ]
    assert [lk.key for lk in chosen] == ["l1", "l2", "l3", "l4", "l5"]
    assert chosen[2].label == "About 30, medium hair, stubble or a short beard" and chosen[2].age == 30.0
    # fewer looks than age bands: the fullest bands stay
    assert [(lk.age_band, lk.photos) for lk in looks.choose(rs, 2)] == [("18 to 24", 3), ("25 to 34", 8)]
    assert [(lk.age_band, lk.photos) for lk in looks.choose(rs, 4)] == [
        ("under 10", 1), ("18 to 24", 3), ("25 to 34", 8), ("25 to 34", 5)]
    assert len(looks.choose(rs, 99)) == 5 and len(looks.choose(rs, 0)) == 1  # the cap is kept inside 1..12


def test_a_look_is_made_of_photos_without_a_fault_and_takes_the_best_of_them():
    sharp_but_crowded = reading(1, size=500, faults=("other_face",))
    cut_off = reading(2, size=480, faults=("cut_off",))
    covered = reading(3, size=470, faults=("covered_face",))
    turned = reading(4, size=460, faults=("turned",))
    small_clean = reading(5, size=160)
    clean = [reading(k, size=300 + k) for k in (6, 7, 8)]
    chosen = looks.choose([sharp_but_crowded, cut_off, covered, turned, small_clean, *clean], 6)
    assert len(chosen) == 1 and chosen[0].photos == 4            # only the four without a fault count
    assert chosen[0].face_id == 8 and chosen[0].faults == []      # the largest of them, not the larger faulty ones
    assert chosen[0].runners_up == [7, 6, 5]
    for name in ("other_face", "cut_off", "covered_face", "raised_hand", "turned", "sunglasses", "covered_head",
                 "odd_light"):
        assert name in looks.FAULTS and name in looks.FAULT_WORDS
    # a face turned a little is fine but ranks below one facing the camera
    assert reading(1, yaw=0.0).score() > reading(2, yaw=0.2).score() > reading(3, yaw=0.5).score()
    assert reading(1, size=400).score() > reading(2, size=150).score()
    lit, dim = reading(1), reading(2)
    lit.light, dim.light = (45.0, 8.0, 10.0), (30.0, 8.0, 10.0)
    assert lit.score() > dim.score()                              # a well-lit face ranks above a dim one


def test_with_no_clean_bin_the_best_photo_of_the_fullest_bin_still_gives_one_head():
    rs = [reading(1, faults=("cut_off", "turned")), reading(2, faults=("other_face",)), reading(3, age=50, faults=("turned",))]
    chosen = looks.choose(rs, 6)
    assert len(chosen) == 1 and chosen[0].face_id == 2 and chosen[0].faults == ["other_face"] and chosen[0].photos == 2
    # two clean photos are not a look of their own (three are needed), so the same fallback applies
    assert len(looks.choose([reading(1), reading(2)], 6)) == 1
    assert looks.choose([], 6) == []
    nowhere = reading(9)
    nowhere.age = None
    assert looks.choose([nowhere], 6) == []


def test_faults_from_the_readings():
    r = reading(1)
    r.outside, r.yaw, r.covered_face, r.sunglasses, r.covered_head, r.light = 0.2, 0.5, 0.9, 0.9, 0.9, (40.0, -3.0, 9.0)
    r.hand_up = True
    assert looks._faults(r) == ["cut_off", "turned", "covered_face", "raised_hand", "sunglasses", "covered_head",
                                "odd_light"]
    ok = reading(2)
    ok.outside, ok.covered_face, ok.sunglasses, ok.covered_head, ok.light = 0.1, 0.1, 0.2, 0.1, (45.0, 8.0, 11.0)
    ok.hand_up = False
    assert looks._faults(ok) == []
    dark, blue = reading(3), reading(4)
    dark.light, blue.light = (10.0, 8.0, 11.0), (45.0, 8.0, -4.0)
    assert looks._faults(dark) == ["odd_light"] and looks._faults(blue) == ["odd_light"]
    down = reading(5)
    down.pitch = 0.8
    assert looks._faults(down) == ["turned"]
    unknown = reading(6, yaw=None)
    unknown.pitch = unknown.roll = None
    assert looks._faults(unknown) == []


def _pose(**points) -> np.ndarray:
    """133 pose points, unseen but for the named ones: name=(x, y)."""
    index = {"l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10}
    k = np.zeros((133, 3), dtype=np.float32)
    for name, (x, y) in points.items():
        k[index[name]] = (x, y, 0.9)
    return k


def test_a_hand_at_the_face_or_an_arm_up_shows_in_the_pose():
    face = (400.0, 200.0, 100.0, 120.0)  # centre (450, 260), size 120
    shoulders = {"l_shoulder": (380, 400), "r_shoulder": (520, 400)}
    assert not looks.hand_at_head(_pose(**shoulders, l_wrist=(360, 620), r_wrist=(540, 620)), face)   # arms down
    assert looks.hand_at_head(_pose(**shoulders, r_wrist=(470, 340)), face)                           # a hand under the chin
    assert looks.hand_at_head(_pose(**shoulders, l_wrist=(330, 150), l_elbow=(300, 300)), face)       # an arm raised
    assert looks.hand_at_head(_pose(**shoulders, r_elbow=(600, 330)), face)                           # an elbow up
    assert not looks.hand_at_head(_pose(**shoulders, l_wrist=(100, 150)), face)                       # far to the side
    assert not looks.hand_at_head(_pose(**shoulders, l_wrist=(430, 470)), face)                       # a phone at the chest
    unseen = _pose(**shoulders, r_wrist=(470, 340))
    unseen[10, 2] = 0.2
    assert not looks.hand_at_head(unseen, face)                                                       # a point not seen
    assert not looks.hand_at_head(np.zeros((133, 3), np.float32), face)
    assert not looks.hand_at_head(_pose(r_wrist=(470, 340)), (0.0, 0.0, 0.0, 0.0))


def test_face_light_reads_skin_and_coloured_light():
    from PIL import Image

    skin = looks.face_light(Image.new("RGB", (64, 64), (190, 140, 110)))
    assert skin[1] > looks.LIGHT_A_MIN and skin[2] > looks.LIGHT_B_MIN and skin[0] > looks.LIGHT_L_MIN
    blue = looks.face_light(Image.new("RGB", (64, 64), (60, 90, 200)))
    assert blue[2] < looks.LIGHT_B_MIN
    assert looks.face_light(Image.new("RGB", (64, 64), (8, 8, 8)))[0] < looks.LIGHT_L_MIN


def test_the_sample_keeps_every_year():
    rows = [{"id": k, "year": 2025.5, "child": False, "quality": 1.0 - k / 1000} for k in range(300)]
    rows += [{"id": 1000 + k, "year": 2019.1, "child": False, "quality": 0.2} for k in range(5)]
    rows += [{"id": 2000 + k, "year": None, "child": False, "quality": 0.3} for k in range(30)]
    rows += [{"id": 3000 + k, "year": 2026.0, "child": True, "quality": 0.1} for k in range(4)]
    picked = looks.spread(rows, 60)
    assert len(picked) <= 60
    ids = {r["id"] for r in picked}
    assert {1000 + k for k in range(5)} <= ids                    # a thin year is kept whole
    assert {3000 + k for k in range(4)} <= ids                    # childhood photos are all kept
    assert sum(1 for r in picked if r["year"] is None) >= looks.YEAR_FLOOR
    assert sum(1 for r in picked if r["year"] == 2025.5) >= looks.YEAR_FLOOR
    assert {0, 1, 2} <= ids and 299 not in ids                    # the best of a full year
    assert looks.spread(rows[:10], 60) == rows[:10]               # room for all: all


def test_the_sample_never_passes_its_limit():
    kids = [{"id": k, "year": 2010 + (k % 6), "child": True, "quality": 1.0 - k / 10000} for k in range(3000)]
    picked = looks.spread(kids, 240)                              # someone who is a child in every photo
    assert len(picked) == 240 and {r["year"] for r in picked} == set(range(2010, 2016))
    assert {0, 1, 2, 3, 4, 5} <= {r["id"] for r in picked}         # the best of each year
    adults = [{"id": 5000 + k, "year": 2024.5, "child": False, "quality": 0.9} for k in range(500)]
    prints = [{"id": 9000 + k, "year": 2024.2, "child": True, "quality": 0.1} for k in range(20)]
    picked = looks.spread(adults + prints, 60)                    # childhood prints photographed again that year
    assert len(picked) == 60 and sum(1 for r in picked if r["child"]) >= looks.YEAR_FLOOR
    many = [{"id": k, "year": 1900 + k, "child": False, "quality": 0.5} for k in range(100)]
    assert len(looks.spread(many, 30)) == 30 and looks.spread(many, 0) == []   # more years than room


# ------------------------------------------------------------------ rows, the plan, the picture (no models)
def _person_with_faces(conn, n_adult=4, n_older=3):
    person = make_person(conn, "Asha")
    faces = []
    for k in range(n_adult):
        item = make_item(conn, taken_at=f"2024-03-{k + 1:02d}T10:00:00.000Z", exif_datetime=f"2024-03-{k + 1:02d}T15:30:00")
        faces.append(make_face(conn, item["id"], person, box=(0.35, 0.3, 0.25, 0.3), size_px=200, age_groups=groups(3)))
    for k in range(n_older):
        item = make_item(conn, taken_at=f"2025-06-{k + 1:02d}T10:00:00.000Z")
        faces.append(make_face(conn, item["id"], person, box=(0.35, 0.3, 0.25, 0.3), size_px=300, age_groups=groups(5)))
    return person, faces


def test_source_rows_are_clear_large_faces_on_photos_in_the_library(conn):
    person, faces = _person_with_faces(conn)
    small = make_face(conn, make_item(conn)["id"], person, size_px=80, age_groups=groups(3))
    weak = make_face(conn, make_item(conn)["id"], person, size_px=200, det_score=0.55, age_groups=groups(3))
    video = make_face(conn, make_item(conn, kind="video", ext=".mp4", mime="video/mp4")["id"], person, size_px=200)
    private = make_face(conn, make_item(conn, nsfw=1)["id"], person, size_px=200)
    trashed_item = make_item(conn)
    trashed = make_face(conn, trashed_item["id"], person, size_px=200)
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-01-01T00:00:00.000Z' WHERE id = ?", (trashed_item["id"],))
    someone_else = make_face(conn, make_item(conn)["id"], make_person(conn, "Ravi"), size_px=200)
    rows = looks.source_rows(conn, person)
    assert {r["id"] for r in rows} == set(faces)
    assert not {small, weak, video, private, trashed, someone_else} & {r["id"] for r in rows}
    assert looks.usable_photos(conn, person) == len(faces)
    dated = [r for r in rows if r["year"] is not None]
    assert len(dated) == 4 and dated[0]["year"] == pytest.approx(2024.16, abs=0.03)  # only a camera time counts
    assert all(not r["child"] for r in rows)


def test_plan_without_models_bins_by_age_band(conn, monkeypatch):
    monkeypatch.setenv("PCC_NO_MODELS", "1")
    person, faces = _person_with_faces(conn)
    crowd_item = conn.execute("SELECT item_id FROM faces WHERE id = ?", (faces[0],)).fetchone()[0]
    make_face(conn, crowd_item, None, box=(0.62, 0.32, 0.2, 0.25), size_px=160)   # a neighbour next to the first face
    edge_item = make_item(conn, taken_at="2024-03-09T10:00:00.000Z")
    edge = make_face(conn, edge_item["id"], person, box=(0.0, 0.0, 0.25, 0.3), size_px=200, age_groups=groups(3))
    body_item = conn.execute("SELECT item_id FROM faces WHERE id = ?", (faces[1],)).fetchone()[0]
    make_body(conn, body_item, person, face_id=faces[1], box=(0.2, 0.2, 0.55, 0.8))
    plan = looks.plan(conn, person, 6)
    assert plan.scanned == 8 and plan.birth_year is None
    by_id = {r.face_id: r for r in plan.readings}
    assert by_id[faces[0]].faults == ["other_face"] and by_id[edge].faults == ["cut_off"]
    assert by_id[faces[1]].has_body and by_id[faces[1]].shoulders is True and by_id[faces[2]].shoulders is None
    assert by_id[faces[1]].hair is None and by_id[faces[1]].yaw is None and by_id[faces[1]].light is None
    assert by_id[faces[1]].hand_up is None
    assert plan.usable == 6
    assert [(lk.age_band, lk.hair_label, lk.facial_hair_label, lk.photos) for lk in plan.looks] == [
        ("25 to 34", None, None, 3), ("45 to 59", None, None, 3)]
    assert plan.looks[0].face_id == faces[1]                       # the one whose person box shows the shoulders
    assert plan.looks[0].label == "About 25" and plan.looks[1].label == "About 45"
    assert looks.plan(conn, person, 1).looks[0].photos == 3 and len(looks.plan(conn, person, 1).looks) == 1


def test_plan_needs_three_good_photos(conn):
    person = make_person(conn, "Asha")
    for _ in range(2):
        make_face(conn, make_item(conn)["id"], person, size_px=200, age_groups=groups(3))
    with pytest.raises(looks.NoPhotos):
        looks.plan(conn, person, 6)


def test_the_child_band_counts_once_the_ages_are_read(conn):
    person = make_person(conn, "Asha")
    for _ in range(3):
        make_face(conn, make_item(conn)["id"], person, size_px=200, age_groups=groups(3))
    kid = make_face(conn, make_item(conn)["id"], person, size_px=200, age_groups=groups(0))
    assert not any(r["child"] for r in looks.source_rows(conn, person))    # the library's ages are not all read yet
    with tx(conn):
        kv_set(conn, ages.READY_KEY, ages.MODEL_REV)
    assert [r["id"] for r in looks.source_rows(conn, person) if r["child"]] == [kid]
    plan = looks.plan(conn, person, 6)
    assert [(lk.age_band, lk.photos) for lk in plan.looks] == [("under 10", 1), ("25 to 34", 3)]


def test_the_generators_picture_is_cut_from_the_original(conn, tmp_path):
    from PIL import Image

    person = make_person(conn, "Asha")
    item = make_item(conn, width=640, height=480)
    face = make_face(conn, item["id"], person, box=(0.4, 0.3, 0.2, 0.3), size_px=200)
    rgb = looks.picture(conn, face, 2.6)
    assert rgb.shape == (looks.PICTURE_SIDE, looks.PICTURE_SIDE, 3) and rgb.dtype == np.uint8
    out = tmp_path / "work" / "l1.png"
    assert looks.write_picture(conn, face, out, 9.0)               # the framing is kept inside its range
    with Image.open(out) as im:
        assert im.size == (looks.PICTURE_SIDE, looks.PICTURE_SIDE) and im.mode == "RGB"
    with tx(conn):
        conn.execute("UPDATE items SET nsfw = 1 WHERE id = ?", (item["id"],))
    assert looks.picture(conn, face) is None and not looks.write_picture(conn, face, tmp_path / "x.png")
    assert looks.picture(conn, 99999) is None


def test_the_picture_s_side_follows_the_photo_between_its_two_ends():
    assert (looks.PICTURE_SIDE, looks.PICTURE_SIDE_MAX) == (1024, 3072)
    assert looks.picture_side(374.4) == 1024                        # a small face: enlarged to the least side
    assert looks.picture_side(1024.0) == 1024 and looks.picture_side(1024.6) == 1025
    assert looks.picture_side(1785.4) == 1785 and looks.picture_side(2719.5) == 2720
    assert looks.picture_side(3072.4) == 3072 and looks.picture_side(4160.0) == 3072
    assert looks.picture_side(0.0) == 1024 and looks.picture_side(-5.0) == 1024
    assert looks.picture_side(float("nan")) == 1024 and looks.picture_side(float("inf")) == 1024


def _gradient(width: int, height: int, tint: int) -> bytes:
    """A smooth made-up photo: red grows to the right, green downwards, so a wrong cut or size shows."""
    from tests.helpers import jpeg_bytes
    from PIL import Image

    x = np.linspace(0, 255, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None].repeat(width, axis=1)
    rgb = np.stack([x, y, np.full_like(x, tint)], axis=2).astype(np.uint8)
    return jpeg_bytes(Image.fromarray(rgb, "RGB"), quality=92)


def test_the_picture_keeps_the_photo_s_own_resolution(conn, tmp_path):
    """A small face is enlarged to 1024 as before, a middle one keeps exactly the pixels the photo has, and a
    very large one stops at 3072. All three are the same cut, only at another size."""
    import cv2
    from PIL import Image

    person = make_person(conn, "Asha")
    cases = [  # photo size, face box, the side the square has in the photo at framing 2.6, the side written
        ((640, 480), (0.4, 0.3, 0.2, 0.3), 2.6 * 144, 1024),          # face 128 x 144 px
        ((2400, 1800), (0.35, 0.3, 0.25, 0.3), 2.6 * 600, 1560),      # face 600 x 540 px
        ((4000, 3000), (0.3, 0.2, 0.4, 0.5), 2.6 * 1600, 3072),       # face 1,600 x 1,500 px
    ]
    for n, ((width, height), box, native, side) in enumerate(cases):
        item = make_item(conn, width=width, height=height, data=_gradient(width, height, 40 * n))
        face = make_face(conn, item["id"], person, box=box, size_px=int(box[2] * width))
        assert looks.square_for(box, width, height, 2.6)[2] == pytest.approx(native)
        own = looks.picture(conn, face, 2.6)
        assert own.shape == (side, side, 3) and own.dtype == np.uint8, (width, height)
        # asked for a side, it gives that side; shrunk to it, the full-size picture is the same picture
        small = looks.picture(conn, face, 2.6, side=512)
        assert small.shape == (512, 512, 3)
        shrunk = cv2.resize(own, (512, 512), interpolation=cv2.INTER_AREA)
        assert float(np.abs(shrunk.astype(np.int16) - small.astype(np.int16)).mean()) < 2.0
        out = tmp_path / f"l{n}.png"
        assert looks.write_picture(conn, face, out, 2.6)
        with Image.open(out) as im:
            assert im.size == (side, side) and im.mode == "RGB" and im.format == "PNG"
        assert looks.write_picture(conn, face, out, 2.6, side=1024)
        with Image.open(out) as im:
            assert im.size == (1024, 1024)
    # a wider framing takes more of the photo, so more pixels, up to the same end
    assert looks.picture(conn, face, 3.2).shape == (3072, 3072, 3)
    middle = make_face(conn, make_item(conn, width=2400, height=1800, data=_gradient(2400, 1800, 200))["id"], person,
                       box=(0.35, 0.3, 0.25, 0.3), size_px=600)
    assert looks.picture(conn, middle, 1.8).shape == (1080, 1080, 3)
    assert looks.picture(conn, middle, 3.2).shape == (1920, 1920, 3)


def test_the_picture_comes_with_the_part_of_it_that_is_the_photo(conn, tmp_path):
    """A face near the photo's corner: the square leaves the photo on the right and at the bottom, and the
    rectangle that comes with the picture ends where the photo does."""
    person = make_person(conn, "Asha")
    width, height = 2000, 1500
    item = make_item(conn, width=width, height=height, data=_gradient(width, height, 60))
    box = (0.7002, 0.5503, 0.25, 0.3)                                  # a face of 500 x 450 px
    face = make_face(conn, item["id"], person, box=box, size_px=500)
    # the square: 1,300 px, from 1,000.4 across and 504.45 down, so 300 px and 304 px of it are past the photo
    assert looks.square_for(box, width, height, 2.6) == pytest.approx((1000.4, 504.45, 1300.0))
    got = looks.picture_and_inside(conn, face, 2.6)
    assert got is not None
    rgb, rect = got
    assert rgb.shape == (1300, 1300, 3) and rect == (0, 0, 1000, 996)
    _left, _top, right, bottom = rect
    # past the rectangle every column is the photo's last column, every row its last row; before it they differ
    assert (rgb[:, right:] == rgb[:, right - 1:right]).all() and (rgb[bottom:] == rgb[bottom - 1:bottom]).all()
    assert (rgb[:bottom, right - 1] != rgb[:bottom, right - 60]).any()
    assert (rgb[bottom - 1, :right] != rgb[bottom - 60, :right]).any()
    assert np.array_equal(looks.picture(conn, face, 2.6), rgb)
    # written to a file, the same rectangle is handed back; at another size it is in that size's pixels
    assert looks.write_picture(conn, face, tmp_path / "l1.png", 2.6) == rect
    assert looks.write_picture(conn, face, tmp_path / "l1.png", 2.6, side=650) == (0, 0, 500, 498)
    # a square that lies in the photo: all of the picture
    whole = make_face(conn, item["id"], person, box=(0.4, 0.3, 0.1, 0.12), size_px=200)
    assert looks.write_picture(conn, whole, tmp_path / "l2.png", 2.6) == (0, 0, 1024, 1024)


def test_without_the_original_the_picture_has_the_preview_s_resolution(conn):
    """When the original does not read, the cut comes from the analysis picture, and so does its size."""
    from app import storage
    from PIL import Image

    person = make_person(conn, "Asha")
    item = make_item(conn, width=4000, height=3000, data=_gradient(4000, 3000, 90))
    face = make_face(conn, item["id"], person, box=(0.3, 0.2, 0.4, 0.5), size_px=1600)
    assert looks.picture(conn, face, 2.6).shape == (3072, 3072, 3)
    storage.original_path(item["uid"], 1, ".jpg").write_bytes(b"not a picture any more")
    with Image.open(storage.preview_path(item["uid"], 1)) as im:
        im.convert("RGB").resize((2048, 1536)).save(storage.preview_path(item["uid"], 1), "WEBP", quality=70)
    # the face is 819 px wide in the 2,048 px preview: a square of 2,130 px
    assert looks.picture(conn, face, 2.6).shape == (2130, 2130, 3)


def test_reading_reports_before_the_models_load_and_after_every_photo(conn, monkeypatch):
    person = _person_with_faces(conn)[0]
    rows = looks.source_rows(conn, person)
    order: list = []
    real = looks.engines()
    monkeypatch.setattr(looks, "engines", lambda: (order.append("models"), real)[1])
    looks.read_faces(conn, rows, None, lambda done, total: order.append((done, total)))
    assert order[:3] == [(0, 7), "models", (0, 7)] and order[3:] == [(n, 7) for n in range(1, 8)]

    class Stop(Exception):
        pass

    seen = []

    def stop(done, total):
        seen.append(done)
        if done == 2:
            raise Stop()                                          # what the caller raises ends the reading there

    with pytest.raises(Stop):
        looks.plan(conn, person, progress=stop)
    assert seen == [0, 0, 1, 2]

