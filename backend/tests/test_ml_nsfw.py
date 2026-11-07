"""app.ml.nsfw: probabilities in [0, 1], low on ordinary photos and a flat grey image."""
from __future__ import annotations

import cv2
import numpy as np

from tests.conftest import FIXTURES, skip_without_models


def test_nsfw_scores_low_on_safe_images():
    skip_without_models("nsfw")
    from app.ml import nsfw

    assert nsfw.available()
    names = ["obama_2009.jpg", "harris.jpg", "apollo11_crew.jpg", "search/dog.jpg", "search/birthday_cake.jpg",
             "search/cat_snow.jpg", "search/green_tshirt.jpg", "search/tractor_men.jpg", "aldrin.jpg"]
    imgs = [cv2.imread(str(FIXTURES / n)) for n in names]
    imgs.append(np.full((300, 400, 3), 128, np.uint8))
    scores = nsfw.score(imgs)  # 10 images: two batches
    assert len(scores) == len(imgs)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all(s < 0.2 for s in scores), scores
    assert scores[-1] < 0.1
    assert nsfw.score([]) == []
    assert nsfw.get_engine().nsfw_index == 1
    # batch order does not change results
    single = nsfw.score([imgs[3]])[0]
    assert abs(single - scores[3]) < 1e-3
