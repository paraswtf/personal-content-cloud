"""Tattoo verification (section 34, app.analysis.trait_sift): real SIFT on synthetic glyphs (tests.helpers.tattoo_crop).

The glyphs stand in for the native-resolution limb crops: a fixed design per seed, turned, scaled or mirrored
between "photos", on skin whose faint texture differs every time."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from app.analysis import trait_sift as ts
from tests.helpers import tattoo_crop

SHARE = 0.6  # forearm


@pytest.fixture(autouse=True)
def _fresh_caches():
    ts.reset()
    yield
    ts.reset()


def _f(img: np.ndarray, share: float | None = SHARE) -> ts.Features:
    return ts.describe(ts.encode_detail(img), share)


def test_describe_shapes_and_dtypes():
    f = _f(tattoo_crop(1))
    for xy, desc in (f.plain, f.mirror):
        assert xy.dtype == np.float32 and xy.ndim == 2 and xy.shape[1] == 2
        assert desc.dtype == np.uint8 and desc.shape == (xy.shape[0], 128)
        assert xy.shape[0] >= 150
    assert f.side == 448


def test_the_strip_mask_keeps_keypoints_on_the_limb():
    img = tattoo_crop(2)
    half = int(448 * SHARE / ts.LIMB_STRIP / 2)
    lo, hi = 448 // 2 - half, 448 // 2 + half
    masked = _f(img)
    for xy, _ in (masked.plain, masked.mirror):
        assert xy[:, 0].min() >= lo - 1 and xy[:, 0].max() <= hi + 1
    # a hand crop (no strip) is searched everywhere: the skin texture and glyph edges near the border count
    whole = _f(img, None)
    assert len(whole.plain[0]) >= len(masked.plain[0])


def test_a_flat_crop_has_no_features_and_verifies_nothing():
    flat = np.full((300, 300, 3), (196, 150, 122), np.uint8)
    f = _f(flat)
    assert f.plain[0].shape == (0, 2) and f.plain[1].shape == (0, 128)
    assert ts.verify(f, _f(tattoo_crop(1))) is None


def test_a_small_mark_has_few_keypoints():
    f = _f(tattoo_crop(3, small=True))
    assert len(f.plain[0]) < 60


@pytest.mark.parametrize("kw", [{"angle": 12}, {"angle": -20, "scale": 1.3}, {"scale": 0.75}, {"variant": 1}])
def test_the_same_tattoo_in_another_photo_verifies(kw):
    v = ts.verify(_f(tattoo_crop(1)), _f(tattoo_crop(1, **kw)))
    assert v is not None and not v.mirrored
    assert v.inliers >= 60 and v.ratio >= 0.6
    for n in (12, 20, 30):
        assert ts.passes(v, n)


def test_a_mirrored_photo_verifies_as_mirrored():
    a = _f(tattoo_crop(4))
    v = ts.verify(a, _f(tattoo_crop(4, mirror=True)))
    assert v is not None and v.mirrored and ts.passes(v, 30)
    # either side may be the mirrored one
    w = ts.verify(_f(tattoo_crop(4, angle=8, scale=1.1, mirror=True)), a)
    assert w is not None and w.mirrored and ts.passes(w, 30)


def test_different_tattoos_never_pass():
    base = [_f(tattoo_crop(s)) for s in range(10, 16)]
    turned = [_f(tattoo_crop(s, angle=10, scale=1.1)) for s in range(10, 16)]
    worst = 0
    for i, a in enumerate(base):
        for j in range(len(base)):
            if i == j:
                continue
            for b in (base[j], turned[j]):
                v = ts.verify(a, b)
                assert not ts.passes(v, 12), (i, j, v)
                worst = max(worst, v.inliers if v else 0)
    assert worst < 12  # measured: at most 11 over 4,100 pairs of seeds 0 to 40


def test_a_fit_outside_the_scale_range_is_rejected(monkeypatch):
    a, b = _f(tattoo_crop(1)), _f(tattoo_crop(1, scale=0.4))
    assert ts.verify(a, b) is None
    # the same pair fits once the range allows it: the rejection is the scale rule, not a lack of matches
    monkeypatch.setattr(ts, "SCALE_MIN", 0.2)
    v = ts.verify(a, b)
    assert v is not None and v.inliers >= 20


def test_verify_is_symmetric_and_one_cache_entry_serves_both_orders():
    a, b = _f(tattoo_crop(5)), _f(tattoo_crop(5, angle=15))
    v1 = ts.verify(a, b, key_a=("t", 9), key_b=("t", 2))
    v2 = ts.verify(b, a, key_a=("t", 2), key_b=("t", 9))
    assert v1 == v2
    s = ts.stats()
    assert s["results"] == 1 and s["verified"] == 1
    assert ts.cached(("t", 2), ("t", 9)) and ts.cached(("t", 9), ("t", 2))


def test_cache_keys_name_the_table():
    a, b = _f(tattoo_crop(5)), _f(tattoo_crop(6))
    ts.verify(a, b, key_a=("t", 5), key_b=("t", 6))
    ts.verify(a, b, key_a=("h", 5), key_b=("t", 6))
    assert ts.stats()["results"] == 2 and ts.stats()["verified"] == 2


def test_a_cached_none_is_not_computed_again():
    a, b = _f(tattoo_crop(1)), _f(tattoo_crop(1, scale=0.4))
    assert ts.verify(a, b, key_a=("t", 1), key_b=("t", 2)) is None
    assert ts.verify(a, b, key_a=("t", 1), key_b=("t", 2)) is None
    assert ts.stats()["verified"] == 1


def test_without_keys_nothing_is_cached():
    a, b = _f(tattoo_crop(5)), _f(tattoo_crop(5, angle=5))
    ts.verify(a, b)
    ts.verify(a, b)
    assert ts.stats()["results"] == 0 and ts.stats()["verified"] == 2


def test_the_result_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(ts, "RESULT_CACHE", 2)
    a, b = _f(tattoo_crop(7)), _f(tattoo_crop(7, angle=5))
    for i in range(4):
        ts.verify(a, b, key_a=("t", 100 + i), key_b=("t", 200 + i))
    assert ts.stats()["results"] == 2
    assert ts.cached(("t", 103), ("t", 203)) and not ts.cached(("t", 100), ("t", 200))


def test_features_load_once_and_the_lru_is_bounded(monkeypatch):
    monkeypatch.setattr(ts, "FEATURE_CACHE", 2)
    blob = ts.encode_detail(tattoo_crop(8))
    calls: list[int] = []

    def loader(i: int):
        def load():
            calls.append(i)
            return blob, SHARE
        return load

    f1 = ts.features(("t", 1), loader(1))
    f1b = ts.features(("t", 1), loader(1))
    assert calls == [1] and f1b is f1
    ts.features(("t", 2), loader(2))
    ts.features(("t", 3), loader(3))  # evicts ("t", 1), the least recently used
    assert ts.stats()["features"] == 2
    ts.features(("t", 1), loader(1))
    assert calls == [1, 2, 3, 1]


def test_passes_needs_the_inliers_and_the_ratio():
    V = ts.Verification
    assert not ts.passes(None, 12)
    assert ts.passes(V(12, 0.3, False), 12) and not ts.passes(V(12, 0.3, False), 20)
    assert ts.passes(V(20, 0.25, True), 20) and ts.passes(V(30, 0.5, False), 30)
    assert not ts.passes(V(40, 0.2, False), 12)  # many inliers among many more chance matches


def test_detail_crops_are_capped_at_512_px():
    big = np.zeros((700, 700, 3), np.uint8)
    big[:, :, 0] = 200
    out = ts.decode_detail(ts.encode_detail(big))
    assert out.shape == (512, 512, 3)
    assert abs(int(out[256, 256, 0]) - 200) <= 3  # RGB order kept


def test_verify_from_two_threads():
    a, b = _f(tattoo_crop(9)), _f(tattoo_crop(9, angle=10))
    got: list[ts.Verification | None] = []

    def run(k: int):
        got.append(ts.verify(a, b, key_a=("t", k), key_b=("t", k + 1000)))

    threads = [threading.Thread(target=run, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 4 and len({g for g in got}) == 1 and ts.passes(got[0], 30)
