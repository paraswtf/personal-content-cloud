"""Category rules, screenshot and document heuristics (app.analysis.categories): pure, no models."""
from __future__ import annotations

import numpy as np

from app.analysis import categories as c
from app.config import settings
from tests.helpers import flat_image, noise_image, text_image


def test_decide_order():
    # 1. kind wins over everything.
    assert c.decide("pdf", nsfw_score=0.99).category == "documents"
    assert c.decide("document", has_person=True).category == "documents"
    # 2. nsfw by classifier, or classifier >= 0.4 with the explicit zero-shot as second signal.
    assert c.decide("image", nsfw_score=0.95, has_person=True).category == "nsfw"
    d = c.decide("image", nsfw_score=0.5, explicit_p=0.7)
    assert d.category == "nsfw" and d.nsfw
    assert c.decide("image", nsfw_score=0.5, explicit_p=0.2).category == "misc"
    assert c.decide("image", nsfw_score=None, explicit_p=0.99).category == "misc"  # explicit alone never flags
    # 3. screenshots by heuristic or zero-shot.
    assert c.decide("image", screenshot_heur=True, has_person=True).category == "screenshots"
    assert c.decide("image", screenshot_p=0.7).category == "screenshots"
    assert c.decide("image", screenshot_p=0.5).category == "misc"
    # 4. document-like images.
    assert c.decide("image", doc_like=settings.doc_likeness_min, has_person=True).category == "documents"
    # 5. people.
    assert c.decide("image", doc_like=0.1, has_person=True).category == "personal"
    # 6. scene groups.
    d = c.decide("image", scores={"places": 0.6, "food": 0.1, "pets": 0.1, "objects": 0.1, "personal": 0.1})
    assert d.category == "places"
    assert c.decide("image", scores={"places": 0.2, "food": 0.3, "personal": 0.5}).category == "misc"
    assert c.decide("video", scores={"pets": 0.8}).category == "pets"
    assert c.decide("other").category == "misc"


def test_nsfw_override():
    assert c.decide("image", nsfw_score=0.99, nsfw_override=False, has_person=True).category == "personal"
    d = c.decide("image", nsfw_score=0.0, nsfw_override=True)
    assert d.category == "nsfw" and d.nsfw


def test_screenshot_heuristic():
    assert c.screenshot_heuristic(".png", None, 1170, 2532)
    assert c.screenshot_heuristic(".webp", None, 2532, 1170)
    assert c.screenshot_heuristic(".png", None, 1920, 1080)
    assert not c.screenshot_heuristic(".jpg", None, 1170, 2532)
    assert not c.screenshot_heuristic(".png", "iPhone 15", 1170, 2532)
    assert not c.screenshot_heuristic(".png", None, 1170, 2532, exif_datetime="2026:01:01 10:00:00")
    assert not c.screenshot_heuristic(".png", None, 1000, 777)
    assert not c.is_screen_size(None, 100)


def test_text_density():
    page = text_image([f"Line {i} of a printed page with words" for i in range(12)], w=1000, h=1100)
    assert c.text_density(page) >= 0.55
    assert c.text_density(noise_image(800, 600, seed=5)) < 0.1
    assert c.text_density(flat_image(800, 600, (250, 250, 250))) == 0.0
    assert c.text_density(flat_image(8, 8)) == 0.0
    bgr = np.ascontiguousarray(np.asarray(page)[:, :, ::-1])
    assert c.text_density(bgr) >= 0.55


def test_doc_likeness():
    assert c.doc_likeness(0.3, 0.8) == 0.8
    assert c.doc_likeness(None, None) == 0.0
    assert c.doc_likeness(0.9, None) == 0.9


class _Engine:
    """Duck-typed SigLIP engine for the scoring helpers: deterministic softmax rows."""

    def zero_shot(self, embs, groups):
        n, k = np.asarray(embs).shape[0], len(groups)
        rows = np.full((n, k), 1.0 / k, dtype=np.float32)
        rows[:, 0] += 0.5
        return rows / rows.sum(axis=1, keepdims=True)


def test_scoring_helpers():
    e = _Engine()
    scores = c.category_scores(e, np.zeros((3, 768), np.float32))
    assert list(scores) == list(c.CATEGORY_PROMPTS)
    assert max(scores, key=scores.get) == "personal"
    assert abs(sum(scores.values()) - 1.0) < 1e-3
    labels = c.top_labels(e, np.zeros(768, np.float32), k=5)
    assert labels[0][0] == c.LABELS[0] and len(labels) <= 5
    assert c.binary_p(e, np.zeros((2, 768), np.float32), ["a"], ["b"]) > 0.5
    assert c.category_scores(e, np.zeros((0, 768), np.float32)) == {}


def test_prompt_groups_present():
    for group in ("personal", "places", "food", "pets", "objects", "documents", "screenshots"):
        assert len(c.CATEGORY_PROMPTS[group]) >= 3
    assert len(c.ACCIDENT_PROMPTS) >= 5 and len(c.EXPLICIT_PROMPTS) >= 3


def test_text_density_ignores_white_photos():
    from pathlib import Path

    from PIL import Image

    snow = Path(__file__).parent / "fixtures" / "search" / "cat_snow.jpg"
    with Image.open(snow) as im:
        assert c.text_density(im) < settings.doc_likeness_min
