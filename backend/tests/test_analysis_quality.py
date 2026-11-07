"""Mis-click measures and decisions (app.analysis.quality): pure, no models."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.analysis import quality
from app.util import json_dumps
from tests.helpers import flat_image, make_item, noise_image


def test_blur_variance_sharp_vs_blurred():
    sharp = noise_image(800, 600, seed=1)
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=6))
    v_sharp = quality.blur_variance(sharp)
    v_blur = quality.blur_variance(blurred)
    assert v_sharp > 1000
    assert v_blur < 40 < v_sharp
    # BGR arrays give the same answer as PIL images.
    bgr = np.ascontiguousarray(np.asarray(sharp)[:, :, ::-1])
    assert abs(quality.blur_variance(bgr) - v_sharp) / v_sharp < 0.01


def test_blur_variance_downscales_large_images():
    big = noise_image(3000, 2000, seed=2)
    assert quality.blur_variance(big) > 100


def test_exposure_dark_normal_bright():
    dark = flat_image(640, 480, (10, 10, 12))
    normal = noise_image(640, 480, seed=3)
    bright = flat_image(640, 480, (255, 255, 255))
    m, p95, clipped = quality.exposure(dark)
    assert m < 28 and p95 < 70 and clipped == 0
    m, p95, clipped = quality.exposure(normal)
    assert 100 < m < 160
    m, p95, clipped = quality.exposure(bright)
    assert m > 235 and clipped > 0.99


def test_measure_keys():
    s = quality.measure(noise_image(320, 240, seed=4))
    assert set(s) == {"blur_var", "mean_lum", "p95_lum", "clipped_frac"}


def test_decide_rules():
    good = {"blur_var": 500.0, "mean_lum": 120.0, "p95_lum": 220.0, "clipped_frac": 0.01}
    assert quality.decide("image", "misc", good, good_face=False) == (False, {}, good)

    blurry = {**good, "blur_var": 12.0}
    mis, reasons, _ = quality.decide("image", "misc", blurry, good_face=False)
    assert mis and "blurry" in reasons

    shaky = {**good, "blur_var": 60.0, "motion_blur_p": 0.8}
    mis, reasons, _ = quality.decide("image", "personal", shaky, good_face=True)
    assert mis and reasons["blurry"]["motion_blur_p"] == 0.8
    # Motion blur alone on a sharp picture is not enough.
    mis, _, _ = quality.decide("image", "misc", {**good, "motion_blur_p": 0.9}, good_face=False)
    assert not mis

    dark = {**good, "mean_lum": 10.0, "p95_lum": 30.0}
    mis, reasons, _ = quality.decide("image", "misc", dark, good_face=False)
    assert mis and "dark" in reasons

    blown = {**good, "mean_lum": 240.0, "clipped_frac": 0.7}
    mis, reasons, _ = quality.decide("image", "misc", blown, good_face=False)
    assert mis and "blown_out" in reasons

    pocket = {**good, "accident_p": 0.9}
    mis, reasons, _ = quality.decide("image", "misc", pocket, good_face=False)
    assert mis and "pocket" in reasons
    mis, reasons, _ = quality.decide("image", "personal", pocket, good_face=True)
    assert not mis

    # Documents and screenshots are never mis-clicks; a video only gets 'dark'.
    assert quality.decide("image", "documents", blurry, good_face=False)[0] is False
    assert quality.decide("image", "screenshots", dark, good_face=False)[0] is False
    assert quality.decide("pdf", "documents", dark, good_face=False)[0] is False
    mis, reasons, _ = quality.decide("video", "misc", {**blurry, "mean_lum": 10.0, "p95_lum": 20.0}, good_face=False)
    assert mis and set(reasons) == {"dark"}
    assert quality.decide("video", "misc", blurry, good_face=False)[0] is False


def test_burst_entries():
    good = {"blur_var": 500.0, "mean_lum": 120.0, "p95_lum": 220.0, "clipped_frac": 0.01}
    mis, reasons, _ = quality.decide("image", "misc", good, good_face=False, burst={"burst": {"best_uid": "x"}})
    assert mis and reasons["burst"]["best_uid"] == "x"
    mis, reasons, _ = quality.decide("image", "misc", good, good_face=False, burst={"burst_best": {"size": 2}})
    assert not mis and "burst_best" in reasons
    r = quality.with_burst({"burst": {"best_uid": "x"}, "dark": {}}, None)
    assert r == {"dark": {}}
    assert quality.is_misclick({"burst_best": {}}) is False


def _shot(conn, taken_at: str, phash: str, blur: float | None, camera: str | None = "Pixel 8", status: str = "done"):
    scores = json_dumps({"blur_var": blur}) if blur is not None else None
    return make_item(conn, taken_at=taken_at, phash=phash, camera=camera, quality_scores=scores,
                     analysis_status=status, write_files=False, data=None)


def test_burst_group_and_roles(conn):
    ph = "a5a5a5a5a5a5a5a5"
    near = "a5a5a5a5a5a5a5a4"  # 1 bit away
    far = "5a5a5a5a5a5a5a5a"
    a = _shot(conn, "2026-05-01T10:00:00.000Z", ph, 300.0)
    b = _shot(conn, "2026-05-01T10:00:01.500Z", near, 80.0)
    _shot(conn, "2026-05-01T10:00:02.000Z", far, 900.0)  # different picture
    _shot(conn, "2026-05-01T10:00:09.000Z", ph, 900.0)  # outside the window
    _shot(conn, "2026-05-01T10:00:01.000Z", ph, 900.0, camera="Other camera")
    _shot(conn, "2026-05-01T10:00:01.000Z", ph, None, status="pending")  # not analysed yet
    new = make_item(conn, taken_at="2026-05-01T10:00:02.500Z", phash=ph, camera="Pixel 8", write_files=False,
                    data=None, analysis_status="pending")
    members = quality.burst_group(conn, new, 150.0)
    assert sorted(m.id for m in members) == sorted([a["id"], b["id"], new["id"]])
    roles = quality.burst_roles(members)
    assert roles[a["id"]] is None  # sharpest
    assert roles[b["id"]] == a["uid"] and roles[new["id"]] == a["uid"]


def test_burst_group_needs_time_and_hash(conn):
    item = make_item(conn, taken_at=None, phash="a5a5a5a5a5a5a5a5", write_files=False, data=None)
    assert quality.burst_group(conn, item, 100.0) == []
    alone = make_item(conn, taken_at="2026-05-01T10:00:00.000Z", phash="a5a5a5a5a5a5a5a5", write_files=False, data=None)
    assert quality.burst_group(conn, alone, 100.0) == []


def test_exposure_many_frames():
    frames = [np.full((90, 160, 3), 8, np.uint8), np.full((90, 160, 3), 12, np.uint8)]
    m, p95, clipped = quality.exposure_many(frames)
    assert m < 28 and p95 < 70
    assert quality.exposure_many([]) == (0.0, 0.0, 0.0)


def test_accident_probability_uses_engine():
    class Engine:  # duck-typed SigLIP engine: first group wins for row 0, second for row 1
        def zero_shot(self, embs, groups):
            assert list(groups) == ["pos", "neg"]
            return np.array([[0.9, 0.1], [0.2, 0.8]][: len(embs)], dtype=np.float32)

    p = quality.accident_probability(Engine(), np.zeros((2, 768), np.float32))
    assert p.tolist() == [np.float32(0.9), np.float32(0.2)]
    assert quality.motion_blur_probability(Engine(), np.zeros(768, np.float32)).shape == (1,)


def test_blur_on_small_image_not_upscaled():
    img = Image.fromarray(np.zeros((40, 40, 3), np.uint8))
    assert quality.blur_variance(img) == 0.0
