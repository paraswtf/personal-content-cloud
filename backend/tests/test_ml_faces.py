"""app.ml.faces on the hub's public-domain portraits."""
from __future__ import annotations

import cv2
import numpy as np

from tests.conftest import FIXTURES, skip_without_models


def test_faces_detect_and_embed():
    skip_without_models("faces")
    from app.ml import faces

    assert faces.available()
    img = cv2.imread(str(FIXTURES / "obama_2009.jpg"))
    found = faces.get_engine().analyze(img, (640, 1280), 0.5, 0)
    assert len(found) >= 1
    f = found[0]
    assert f.det_score > 0.7 and f.size_px > 40
    assert f.embedding.shape == (512,)
    assert abs(float(np.linalg.norm(f.embedding)) - 1.0) < 1e-3
    assert faces.loaded() and faces.provider() in ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_same_person_closer_than_other():
    skip_without_models("faces")
    from app.ml import faces

    eng = faces.get_engine()

    def emb(name):
        fs = eng.analyze(cv2.imread(str(FIXTURES / name)), (640, 1280), 0.5, 0)
        return max(fs, key=lambda f: f.size_px).embedding

    a, b, c = emb("obama_2009.jpg"), emb("obama_2012.jpg"), emb("harris.jpg")
    assert faces.cosine(a, b) > 0.45
    assert faces.cosine(a, c) < 0.3


def test_group_photo_and_blank():
    skip_without_models("faces")
    from app.ml import faces

    eng = faces.get_engine()
    group = eng.analyze(cv2.imread(str(FIXTURES / "apollo11_crew.jpg")), (640, 1280), 0.5, 0)
    assert len(group) >= 3
    assert eng.analyze(np.full((480, 640, 3), 128, np.uint8), (640,), 0.5, 0) == []
    assert eng.detect(np.zeros((4, 4, 3), np.uint8)) == []
