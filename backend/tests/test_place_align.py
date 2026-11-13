"""Lining two photos up (section 35): app.analysis.place_align on synthetic textured pictures."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.analysis import place_align as pa


@pytest.fixture(autouse=True)
def _fresh_cache():
    pa.reset()
    yield
    pa.reset()


def scene(seed: int, w: int = 1000, h: int = 750) -> np.ndarray:
    """A grey picture with enough structure for keypoints: random blocks and discs, lightly blurred."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 128, np.uint8)
    for _ in range(260):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        s = int(rng.integers(8, 60))
        c = int(rng.integers(20, 235))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.5, 1.5))), c, -1)
        else:
            cv2.circle(img, (x, y), s // 2, c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def warp(img: np.ndarray, m: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, m.astype(np.float64), size or (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def test_a_shift_is_found_and_maps_points():
    a = scene(1)
    b = warp(a, np.array([[1, 0, 60], [0, 1, -30]]))
    al = pa.align(pa.describe(a), pa.describe(b))
    assert al is not None and al.kind == "moved" and al.background
    assert al.inliers >= pa.MIN_INLIERS
    x, y = pa.apply(al.m, 0.5, 0.5)
    assert x == pytest.approx(0.5 + 60 / 1000, abs=0.004) and y == pytest.approx(0.5 - 30 / 750, abs=0.004)
    back = pa.invert(al.m)
    assert back is not None
    bx, by = pa.apply(back, x, y)
    assert (bx, by) == pytest.approx((0.5, 0.5), abs=1e-6)


def test_a_small_shift_is_still():
    a = scene(2)
    b = warp(a, np.array([[1, 0, 4], [0, 1, 3]]))
    al = pa.align(pa.describe(a), pa.describe(b))
    assert al is not None and al.kind == "still"


def test_zoom_and_roll():
    a = scene(3)
    h, w = a.shape
    zoom = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.3)
    al = pa.align(pa.describe(a), pa.describe(warp(a, zoom)))
    assert al is not None and al.kind == "moved"
    assert pa.scale_of(al.m, w, h, w, h) == pytest.approx(1.3, abs=0.02)
    assert pa.apply(al.m, 0.5, 0.5) == pytest.approx((0.5, 0.5), abs=0.004)
    roll = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)
    al = pa.align(pa.describe(a), pa.describe(warp(a, roll)))
    assert al is not None and al.kind == "moved"
    # a point right of the centre goes up when the picture turns counter-clockwise
    x, y = pa.apply(al.m, 0.8, 0.5)
    dx, dy = 0.3 * w, 0.0
    c, s = math.cos(math.radians(8)), math.sin(math.radians(8))
    assert x == pytest.approx(0.5 + (dx * c + dy * s) / w, abs=0.005)
    assert y == pytest.approx(0.5 + (-dx * s + dy * c) / h, abs=0.005)


def test_another_size_and_shape_of_the_same_scene():
    a = scene(4)
    h, w = a.shape
    small = cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    al = pa.align(pa.describe(a), pa.describe(small))
    assert al is not None
    assert pa.apply(al.m, 0.3, 0.7) == pytest.approx((0.3, 0.7), abs=0.004)  # normalised: the same place
    crop = a[100:650, 200:900]  # another aspect ratio: 700 x 550 of 1000 x 750
    al = pa.align(pa.describe(a), pa.describe(np.ascontiguousarray(crop)))
    assert al is not None
    x, y = pa.apply(al.m, 0.55, 0.5)  # pixel (550, 375) of a is pixel (350, 275) of the crop
    assert (x, y) == pytest.approx((350 / 700, 275 / 550), abs=0.004)


def test_unrelated_pictures_and_plain_walls_do_not_line_up():
    assert pa.align(pa.describe(scene(5)), pa.describe(scene(6))) is None
    wall = np.full((750, 1000), 140, np.uint8)
    fw = pa.describe(wall)
    assert len(fw.xy) < pa.MIN_KP
    assert pa.align(fw, fw) is None
    assert pa.align(None, pa.describe(scene(5))) is None


def test_the_burst_fallback():
    al = pa.burst(1.2, 0.985)
    assert al is not None and al.kind == "burst" and al.m == pa.IDENTITY and al.inliers == 0
    assert pa.burst(pa.BURST_S + 0.5, 0.99) is None      # too far apart
    assert pa.burst(1.0, pa.BURST_COS - 0.01) is None    # not the same picture
    assert pa.burst(None, 0.99) is None and pa.burst(1.0, None) is None


def test_the_background_decides_not_a_person_who_moved():
    a = scene(7)
    h, w = a.shape
    b = warp(a, np.array([[1, 0, 30], [0, 1, 0]]))
    # a "person": a textured patch that sits elsewhere in the second picture
    patch = scene(8, 260, 420)
    a2, b2 = a.copy(), b.copy()
    a2[200:620, 300:560] = patch
    b2[250:670, 620:880] = patch
    box_a = (300 / w, 200 / h, 260 / w, 420 / h)
    box_b = (620 / w, 250 / h, 260 / w, 420 / h)
    fa, fb = pa.describe(a2, [box_a]), pa.describe(b2, [box_b])
    assert fa.people.any() and fb.people.any() and not fa.people.all()
    al = pa.align(fa, fb)
    assert al is not None and al.background
    assert pa.apply(al.m, 0.1, 0.1) == pytest.approx((0.1 + 30 / w, 0.1), abs=0.004)


def test_features_are_cached_by_key_and_a_missing_picture_too():
    calls = []

    def loader():
        calls.append(1)
        return scene(9), []

    f1 = pa.features((1, 1), loader)
    f2 = pa.features((1, 1), loader)
    assert f1 is f2 and len(calls) == 1
    assert pa.features((2, 1), lambda: None) is None
    assert pa.features((2, 1), lambda: (_ for _ in ()).throw(AssertionError("cached"))) is None
    st = pa.stats()
    assert st["features"] == 2 and st["described"] == 1 and st["bytes"] > 0
    pa.reset()
    assert pa.stats()["features"] == 0


def test_the_cache_is_bounded_in_bytes(monkeypatch):
    f = pa.describe(scene(10))
    monkeypatch.setattr(pa, "FEATURE_CACHE_BYTES", int(f.nbytes * 2.5))
    for k in range(5):
        pa.features((k, 1), lambda: (scene(10), []))
    assert pa.stats()["features"] == 2  # the two newest stay
    assert pa.features((4, 1), lambda: None) is not None  # still cached


def test_a_fit_on_one_walking_person_alone_is_refused():
    # a plain wall, the camera fixed, a textured person who walks 400 px: every match lies on the walker
    h, w = 750, 1000
    wall = np.full((h, w), 128, np.uint8)
    walker = scene(21, 200, 400)
    a, b = wall.copy(), wall.copy()
    a[200:600, 100:300] = walker
    b[200:600, 500:700] = walker
    still = (0.75, 0.3, 0.1, 0.5)                                      # someone else, who did not move
    fa = pa.describe(a, [(100 / w, 200 / h, 200 / w, 400 / h), still])
    fb = pa.describe(b, [(500 / w, 200 / h, 200 / w, 400 / h), still])
    assert len(fa.xy) >= pa.MIN_INLIERS and fa.people.all() and fb.people.all()     # nothing on the wall
    assert fa.groups == 2 and set(fa.owner.tolist()) == {0}            # every keypoint is the walker's
    assert pa.align(fa, fb) is None                                    # not a camera move of 400 px
    assert pa.stats()["refused"] == 1 and pa.stats()["on_people"] == 0
    # alone in both pictures, the same fit puts nobody else in the wrong place: it stands
    alone = pa.align(pa.describe(a, [(100 / w, 200 / h, 200 / w, 400 / h)]), pa.describe(b, [(500 / w, 200 / h, 200 / w, 400 / h)]))
    assert alone is not None and not alone.background and pa.apply(alone.m, 0.2, 0.5)[0] == pytest.approx(0.6, abs=0.01)


def test_two_people_who_stood_still_carry_a_fit_without_a_background():
    h, w = 750, 1000
    a = np.full((h, w), 128, np.uint8)
    a[200:600, 100:300] = scene(22, 200, 400)
    a[250:650, 600:800] = scene(23, 200, 400)
    b = warp(a, np.array([[1, 0, 40], [0, 1, 0]]))                     # the camera turned a little
    boxes_a = [(100 / w, 200 / h, 200 / w, 400 / h), (600 / w, 250 / h, 200 / w, 400 / h),
               (150 / w, 220 / h, 80 / w, 100 / h)]                    # the first person's face, inside their box
    boxes_b = [(bx + 40 / w, by, bw, bh) for bx, by, bw, bh in boxes_a]
    fa, fb = pa.describe(a, boxes_a), pa.describe(b, boxes_b)
    assert fa.groups == 2 and fa.people.all() and set(fa.owner.tolist()) == {0, 1}
    al = pa.align(fa, fb)
    assert al is not None and not al.background and pa.stats()["on_people"] == 1
    assert pa.apply(al.m, 0.5, 0.5)[0] == pytest.approx(0.54, abs=0.005)
