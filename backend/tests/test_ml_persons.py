"""app.ml.persons: YOLOX-S person boxes, plus the decoder on synthetic output."""
from __future__ import annotations

import cv2
import numpy as np

from tests.conftest import FIXTURES, skip_without_models


def test_persons_on_crew_photo():
    skip_without_models("persons")
    from app.ml import persons

    assert persons.available()
    img = cv2.imread(str(FIXTURES / "apollo11_crew.jpg"))
    h, w = img.shape[:2]
    boxes = persons.detect(img, 0.5)
    assert len(boxes) >= 1
    for x0, y0, x1, y1, s in boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and 0.5 <= s <= 1
    assert [b[4] for b in boxes] == sorted((b[4] for b in boxes), reverse=True)
    assert persons.detect(np.full((480, 640, 3), 128, np.uint8)) == []


def test_decode_synthetic():
    from app.ml import persons

    raw = np.zeros((1, 8400, 85), np.float32)
    # cell (x=10, y=5) of the stride-8 grid: index 5*80+10; centre (10.5+... ) -> ((0.5+10)*8, (0.5+5)*8)
    i = 5 * 80 + 10
    raw[0, i, :4] = [0.5, 0.5, np.log(4.0), np.log(8.0)]  # w = 32 px, h = 64 px in the 640 canvas
    raw[0, i, 4] = 0.9
    raw[0, i, 5] = 0.9  # person
    j = i + 1  # an overlapping duplicate with a lower score is removed by NMS
    raw[0, j, :4] = [-0.4, 0.5, np.log(4.0), np.log(8.0)]
    raw[0, j, 4] = 0.8
    raw[0, j, 5] = 0.8
    k = 6400 + 3  # stride-16 cell, a non-person class only
    raw[0, k, 4] = 0.99
    raw[0, k, 6] = 0.99
    boxes = persons.decode(raw, r=0.5, width=2000, height=2000, threshold=0.5)
    assert len(boxes) == 1
    x0, y0, x1, y1, s = boxes[0]
    cx, cy = (10.5 * 8) / 0.5, (5.5 * 8) / 0.5
    assert np.allclose([x0, y0, x1, y1], [cx - 32, cy - 64, cx + 32, cy + 64], atol=1e-3)
    assert abs(s - 0.81) < 1e-5


def test_letterbox_pads_bottom_right():
    from app.ml import persons

    img = np.zeros((100, 200, 3), np.uint8)
    blob, r = persons.letterbox(img)
    assert blob.shape == (3, 640, 640) and r == 3.2
    assert blob[0, 0, 0] == 0 and blob[0, 639, 0] == 114
