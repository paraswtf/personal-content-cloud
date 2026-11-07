"""app.ml.siglip: search relevance on the fixture photos and zero-shot probabilities."""
from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np
import pytest
from PIL import Image

from tests.conftest import FIXTURES, skip_without_models

SEARCH = FIXTURES / "search"
NAMES = ["dog.jpg", "cat_snow.jpg", "birthday_cake.jpg", "green_tshirt.jpg", "tractor_men.jpg"]


def _imgs():
    return [cv2.imread(str(SEARCH / n)) for n in NAMES]


def test_dog_query_finds_dog():
    skip_without_models("siglip")
    from app.ml import siglip

    assert siglip.available()
    eng = siglip.get_engine()
    embs = eng.embed_images(_imgs())
    assert embs.shape == (len(NAMES), 768) and embs.dtype == np.float32
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-3)
    q = eng.embed_text("a photo of a dog")
    rel = eng.relevance(embs @ q)
    assert int(np.argmax(rel)) == 0
    # calibration of tests/fixtures/search/README.md: dog 0.090, non-matches far below 0.001
    assert rel[0] > 0.01
    assert all(r < 0.001 for r in rel[1:])
    assert np.allclose(siglip.relevance(embs @ q), rel)


def test_bgr_array_matches_pil_rgb():
    skip_without_models("siglip")
    from app.ml import siglip

    eng = siglip.get_engine()
    bgr = cv2.imread(str(SEARCH / "dog.jpg"))
    pil = Image.open(SEARCH / "dog.jpg").convert("RGB")
    a = eng.embed_images([bgr])[0]
    b = eng.embed_images([pil])[0]
    assert float(a @ b) > 0.999
    grey = eng.embed_images([cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)])
    assert grey.shape == (1, 768)
    assert eng.embed_images([]).shape == (0, 768)


def test_zero_shot_rows_sum_to_one():
    skip_without_models("siglip")
    from app.ml import siglip

    eng = siglip.get_engine()
    embs = eng.embed_images(_imgs())
    groups = {
        "pets": ["a photo of a dog", "a photo of a cat", "a photo of a pet"],
        "food": ["a photo of food", "a photo of a cake"],
        "document": ["a scanned document", "a photo of an identity card"],
    }
    p = eng.zero_shot(embs, groups)
    assert p.shape == (len(NAMES), 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert int(np.argmax(p[0])) == 0  # dog -> pets
    assert int(np.argmax(p[1])) == 0  # cat -> pets
    assert int(np.argmax(p[2])) == 1  # cake -> food
    one = eng.zero_shot(embs[0], groups)
    assert one.shape == (1, 3) and np.allclose(one[0], p[0], atol=1e-5)


def test_zero_shot_math_without_model():
    """Prompt averaging and the scaled softmax, with fixed prompt vectors (no model needed)."""
    from app.ml.siglip import LOGIT_SCALE, SiglipEngine

    eng = SiglipEngine.__new__(SiglipEngine)
    eng._cache_lock = threading.Lock()
    eng._groups = OrderedDict()
    vecs = {"a1": np.array([1, 0, 0], np.float32), "a2": np.array([0.8, 0.6, 0], np.float32),
            "b": np.array([0, 0, 1], np.float32)}
    calls = []

    def fake_text(text, prompt=True):
        calls.append((text, prompt))
        return vecs[text]

    eng.embed_text = fake_text
    img = np.array([[1, 0, 0], [0, 0, 1]], np.float32)
    p = eng.zero_shot(img, {"A": ["a1", "a2"], "B": ["b"]})
    ga = np.array([1.8, 0.6, 0]) / np.linalg.norm([1.8, 0.6, 0])
    la = np.array([ga @ img[0], 0.0]) * LOGIT_SCALE
    expect0 = np.exp(la - la.max()) / np.exp(la - la.max()).sum()
    assert np.allclose(p[0], expect0, atol=1e-5)
    assert p[1, 1] > 0.99
    assert all(prompt is False for _, prompt in calls)
    n = len(calls)
    eng.zero_shot(img, {"A": ["a1", "a2"], "B": ["b"]})
    assert len(calls) == n  # cached
    with pytest.raises(ValueError):
        eng.zero_shot(img, {})
