"""Body trait models (docs/ARCHITECTURE.md section 34): DWPose, handpose and DINOv2 on the real model files.
Skipped without the models (and under PCC_NO_MODELS=1). Run alone with ORT_THREADS=2."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FIXTURES, skip_without_models

cv2 = pytest.importorskip("cv2")


@pytest.fixture(autouse=True)
def _fresh_engines():
    from app.ml import dino, hands, pose

    yield
    for mod in (pose, hands, dino):
        mod.reset()


def _photo() -> np.ndarray:
    img = cv2.imread(str(FIXTURES / "apollo11_crew.jpg"))
    assert img is not None
    return img


def test_pose_returns_133_points_per_box():
    skip_without_models("pose")
    from app.ml import pose

    assert pose.available() and not pose.loaded()
    assert pose.estimate(_photo(), []) == []
    assert not pose.loaded()
    img = _photo()
    h, w = img.shape[:2]
    boxes = [(0.0, 0.0, float(w), float(h)), (w * 0.2, h * 0.1, w * 0.8, h * 0.9), (5.0, 5.0, 60.0, 90.0)]
    res = pose.estimate(img, boxes)
    assert len(res) == 3 and pose.loaded() and pose.provider()
    for kp in res:
        assert kp.shape == (133, 3) and kp.dtype == np.float32
        assert np.isfinite(kp).all()
    # batching does not change a result: the same box alone gives the same points
    alone = pose.estimate(img, [boxes[1]])[0]
    np.testing.assert_allclose(alone, res[1], atol=1e-3)


def test_pose_finds_a_standing_person_inside_the_box():
    skip_without_models("pose")
    from app.ml import pose

    img = cv2.imread(str(FIXTURES / "obama_2009.jpg"))
    h, w = img.shape[:2]
    kp = pose.estimate(img, [(0.0, 0.0, float(w), float(h))])[0]
    shoulders = kp[[5, 6]]
    assert (shoulders[:, 2] > 0.4).all()
    assert (shoulders[:, 0] > 0).all() and (shoulders[:, 0] < w).all()
    assert shoulders[0, 0] != shoulders[1, 0]


def test_hands_shapes_and_low_confidence_on_grey():
    skip_without_models("hands")
    from app.ml import hands

    grey = np.full((224, 224, 3), 128, np.uint8)
    noise = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    assert hands.landmarks([]) == []
    res = hands.landmarks([grey, noise, np.full((96, 96, 3), 128, np.uint8)])
    assert len(res) == 3 and hands.loaded()
    for lm, conf, right_p in res:
        assert lm.shape == (21, 2) and np.isfinite(lm).all()
        assert 0.0 <= conf <= 1.0 and 0.0 <= right_p <= 1.0
    assert res[0][1] < 0.5
    assert res[2][1] < 0.5


def test_dino_embeddings_are_normalised_and_mirror_alike():
    skip_without_models("dino")
    from app.ml import dino

    assert dino.embed([]).shape == (0, 384)
    img = cv2.cvtColor(_photo(), cv2.COLOR_BGR2RGB)
    crop = img[: img.shape[0] // 2, : img.shape[1] // 2]
    other = cv2.cvtColor(cv2.imread(str(FIXTURES / "biden.jpg")), cv2.COLOR_BGR2RGB)
    crops = [crop, crop[:, ::-1].copy(), other] + [crop] * 8      # 11 crops: one full batch of 8 plus 3
    v = dino.embed(crops)
    assert v.shape == (11, 384) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert float(v[0] @ v[1]) >= 0.8              # a mirrored crop is the same picture to DINOv2
    assert float(v[0] @ v[2]) < float(v[0] @ v[1])
    np.testing.assert_allclose(v[3], v[0], atol=1e-4)    # batch position does not matter
    assert dino.provider()
