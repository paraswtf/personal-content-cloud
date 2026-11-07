"""Zero-shot categories, screenshot and document heuristics, and the category decision
(docs/ARCHITECTURE.md section 9).

CONTRACT.
- CATEGORY_PROMPTS: {group: [prompts]} for personal, places, food, pets, objects, documents,
  screenshots. DOCUMENT_PROMPTS / SCREENSHOT_PROMPTS / EXPLICIT_PROMPTS are binary groups scored
  against their NOT_* counterparts; EXPLICIT is a second NSFW signal only. ACCIDENT_PROMPTS lives in
  quality.py and is re-exported here. LABELS: vocabulary for the top-5 item labels.
- category_scores(engine, embs) -> {group: p} (mean over the embeddings), plus binary_p(...).
- top_labels(engine, embs, k=5) -> [(label, score)].
- is_screen_size(w, h) / screenshot_heuristic(ext, camera, w, h, exif_datetime) -> bool.
- text_density(img) -> 0..1: text lines (rows of strong edges separated by blank rows) on a
  white-ish background with little mid-grey (print is paper plus a little dark ink).
- doc_likeness(doc_p, density) = max of the two (None counts as 0).
- decide(...) -> Decision(category, nsfw, reason) in the section 9 order; user overrides are applied
  by the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.analysis.quality import ACCIDENT_PROMPTS  # noqa: F401  (re-exported, section 9)
from app.config import settings

log = logging.getLogger(__name__)

SCENE_GROUPS = ("places", "food", "pets", "objects")
# What the owner's labelled look-alikes may decide (section 18): every category but the vault.
LEARNABLE_CATEGORIES = ("documents", "screenshots", "personal", "places", "food", "pets", "objects", "misc")
SCENE_MIN = 0.35
NSFW_SECOND_SIGNAL = 0.4  # classifier score that, with the explicit zero-shot, still flags
EXPLICIT_MIN = 0.5
SCREENSHOT_P_MIN = 0.6

CATEGORY_PROMPTS: dict[str, list[str]] = {
    "personal": [
        "a photo of a person",
        "a selfie",
        "a group photo of friends",
        "a portrait of a man",
        "a portrait of a woman",
        "a photo of a family",
        "a photo of a child",
    ],
    "places": [
        "a photo of a landscape",
        "a photo of a city street",
        "a photo of a building",
        "a photo of the beach",
        "a photo of mountains",
        "a photo of a room interior",
        "a travel photo of a landmark",
    ],
    "food": [
        "a photo of food",
        "a photo of a meal on a plate",
        "a photo of a dessert",
        "a photo of a drink",
        "a photo of a restaurant dish",
    ],
    "pets": [
        "a photo of a dog",
        "a photo of a cat",
        "a photo of a pet",
        "a photo of an animal",
        "a photo of a bird",
    ],
    "objects": [
        "a photo of an object",
        "a photo of a product",
        "a photo of a gadget",
        "a photo of a car",
        "a photo of furniture",
        "a photo of clothes",
    ],
    "documents": [
        "a scan of a document",
        "a photo of a paper document",
        "a photo of an identity card",
        "a photo of a receipt",
        "a page of text",
        "a photo of a form",
    ],
    "screenshots": [
        "a screenshot of a phone screen",
        "a screenshot of a website",
        "a screenshot of a chat conversation",
        "a screenshot of an app",
        "a screenshot of a computer desktop",
    ],
}

DOCUMENT_PROMPTS = [
    "a scan of a document",
    "a photo of a printed page with text",
    "a photo of an identity card",
    "a photo of a passport",
    "a photo of a driving licence",
    "a photo of a receipt",
    "a photo of an invoice",
    "a photo of a certificate",
    "a photo of a ticket",
    "a photo of a handwritten note",
]
NOT_DOCUMENT_PROMPTS = [
    "a photo of people",
    "a photo of a place",
    "a photo of food",
    "a photo of an animal",
    "a photo of an object",
    "a screenshot of a phone screen",
]
SCREENSHOT_PROMPTS = [
    "a screenshot of a phone screen",
    "a screenshot of a website",
    "a screenshot of a chat conversation",
    "a screenshot of a mobile app",
    "a screenshot of a computer desktop",
    "a screenshot of a video game",
]
NOT_SCREENSHOT_PROMPTS = [
    "a photo taken with a camera",
    "a photo of people",
    "a photo of a place",
    "a photo of a printed document",
    "a photo of an object",
]
EXPLICIT_PROMPTS = [
    "a nude photo",
    "an explicit sexual photo",
    "a photo of naked people",
    "a pornographic image",
    "a topless photo",
]
NOT_EXPLICIT_PROMPTS = [
    "a photo of clothed people",
    "a family photo",
    "a photo of a person wearing clothes",
    "a photo of a landscape",
    "a photo of an object",
]

LABELS: tuple[str, ...] = (
    "person", "selfie", "group of people", "child", "baby", "wedding", "party", "birthday cake", "concert",
    "beach", "mountain", "forest", "lake", "river", "sunset", "snow", "desert", "city", "street", "building",
    "temple", "church", "bridge", "park", "garden", "flowers", "tree", "sky", "night", "room", "kitchen",
    "office", "classroom", "restaurant", "food", "dessert", "coffee", "drink", "fruit", "vegetables",
    "dog", "cat", "bird", "horse", "cow", "fish", "car", "motorcycle", "bicycle", "bus", "train", "airplane",
    "boat", "road", "laptop", "phone", "computer screen", "book", "document", "receipt", "identity card",
    "ticket", "certificate", "handwriting", "whiteboard", "screenshot", "chart", "map", "painting", "sports",
    "football", "cricket", "swimming pool", "gym", "shopping", "clothes", "shoes", "toy", "furniture",
    "festival", "fireworks", "selfie stick", "stage", "graduation", "hospital",
)


# ------------------------------------------------------------------ zero-shot
def _as_matrix(embs: np.ndarray) -> np.ndarray:
    embs = np.asarray(embs, dtype=np.float32)
    return embs[None, :] if embs.ndim == 1 else embs


def category_scores(engine: Any, embs: np.ndarray) -> dict[str, float]:
    """Softmax over CATEGORY_PROMPTS groups, averaged over the rows of embs (frames of a video)."""
    m = _as_matrix(embs)
    if m.shape[0] == 0:
        return {}
    probs = np.asarray(engine.zero_shot(m, CATEGORY_PROMPTS), dtype=np.float64).mean(axis=0)
    return {k: round(float(p), 4) for k, p in zip(CATEGORY_PROMPTS, probs)}


def binary_p(engine: Any, embs: np.ndarray, pos: list[str], neg: list[str], reduce: str = "mean") -> float:
    """p(pos group) vs the neg group, reduced over rows ('mean' or 'max')."""
    m = _as_matrix(embs)
    if m.shape[0] == 0:
        return 0.0
    probs = np.asarray(engine.zero_shot(m, {"pos": pos, "neg": neg}), dtype=np.float64)[:, 0]
    return float(probs.max() if reduce == "max" else probs.mean())


def top_labels(engine: Any, embs: np.ndarray, k: int = 5, min_score: float = 0.02) -> list[tuple[str, float]]:
    """The k most likely LABELS (softmax over the vocabulary, mean over rows), score >= min_score."""
    m = _as_matrix(embs)
    if m.shape[0] == 0:
        return []
    groups = {label: [f"a photo of {label}"] for label in LABELS}
    probs = np.asarray(engine.zero_shot(m, groups), dtype=np.float64).mean(axis=0)
    order = np.argsort(-probs)[:k]
    return [(LABELS[i], round(float(probs[i]), 4)) for i in order if probs[i] >= min_score]


# ------------------------------------------------------------------ screenshots
_SCREENS = {
    # phones (portrait; landscape is checked too)
    (720, 1280), (720, 1520), (720, 1560), (720, 1600), (750, 1334), (828, 1792), (1080, 1920), (1080, 2160),
    (1080, 2220), (1080, 2240), (1080, 2280), (1080, 2310), (1080, 2340), (1080, 2400), (1080, 2408),
    (1080, 2412), (1080, 2460), (1125, 2436), (1170, 2532), (1179, 2556), (1206, 2622), (1220, 2712),
    (1240, 2772), (1242, 2208), (1242, 2688), (1260, 2800), (1284, 2778), (1290, 2796), (1320, 2868),
    (1344, 2992), (1440, 2560), (1440, 2960), (1440, 3040), (1440, 3088), (1440, 3120), (1440, 3200),
    (640, 1136), (1176, 2480), (1080, 2520),
    # tablets
    (1536, 2048), (1620, 2160), (1640, 2360), (1668, 2224), (1668, 2388), (2048, 2732), (1200, 1920),
    (1600, 2560), (1800, 2880), (1752, 2800), (1488, 2266),
    # desktops and laptops (landscape stored portrait-first like the rest)
    (768, 1366), (800, 1280), (900, 1440), (864, 1536), (900, 1600), (1050, 1680), (1080, 1920),
    (1200, 1920), (1440, 2560), (1600, 2560), (1800, 2880), (1964, 3024), (2234, 3456), (2160, 3840),
    (1824, 2736), (2000, 3000), (1080, 2560), (1440, 3440), (1440, 3840), (2160, 5120), (1280, 2048),
    (720, 1280), (1024, 1280), (768, 1024), (1117, 1728), (982, 1512), (956, 1470), (1964, 3024),
}


def is_screen_size(width: int | None, height: int | None) -> bool:
    if not width or not height:
        return False
    a, b = sorted((int(width), int(height)))
    return (a, b) in _SCREENS


def screenshot_heuristic(ext: str | None, camera: str | None, width: int | None, height: int | None,
                         exif_datetime: str | None = None) -> bool:
    """PNG or WebP without camera EXIF whose dimensions match a phone, tablet or desktop screen."""
    if (ext or "").lower() not in (".png", ".webp"):
        return False
    if camera or exif_datetime:
        return False
    return is_screen_size(width, height)


# ------------------------------------------------------------------ text density
def text_density(img: Image.Image | np.ndarray) -> float:
    """0..1: how much the image looks like lines of text on paper. Product of a white-ish background
    share, the number of text lines (row bands of strong edges separated by blank rows) and an edge
    density sanity check."""
    if isinstance(img, Image.Image):
        g = np.asarray(img.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(img)
        g = arr if arr.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    s = 1024 / max(h, w)
    if s < 1:
        g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    bg = float((g >= 170).mean())
    s_bg = float(np.clip((bg - 0.45) / 0.35, 0.0, 1.0))
    if s_bg <= 0:
        return 0.0
    # Print is bimodal: paper plus a little dark ink. Photos of white scenes (snow, walls) carry
    # many mid tones and large dark areas.
    ink = float((g < 110).mean())
    mid = float(((g >= 110) & (g < 170)).mean())
    s_tone = float(np.clip((0.30 - ink) / 0.12, 0.0, 1.0)) * float(np.clip((0.20 - mid) / 0.12, 0.0, 1.0))
    if s_tone <= 0:
        return 0.0
    edges = cv2.Canny(g, 80, 200) > 0
    density = float(edges.mean())
    s_edge = 1.0 if 0.005 <= density <= 0.25 else 0.3
    rows = edges.mean(axis=1)
    texty = rows > 0.02
    # Count runs of text rows (each run is one line of text) separated by blank stretches.
    runs = int(np.count_nonzero(texty[1:] & ~texty[:-1])) + (1 if texty.size and texty[0] else 0)
    blank_share = float((rows < 0.005).mean())
    s_lines = float(np.clip(runs / 8.0, 0.0, 1.0)) * (1.0 if blank_share >= 0.1 else 0.5)
    return round(s_bg * s_tone * s_lines * s_edge, 4)


def doc_likeness(doc_p: float | None, density: float | None) -> float:
    return round(max(float(doc_p or 0.0), float(density or 0.0)), 4)


# ------------------------------------------------------------------ decision
@dataclass
class Decision:
    category: str
    nsfw: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def nsfw_decision(nsfw_score: float | None, explicit_p: float | None) -> bool:
    """nsfw_score >= settings.nsfw_threshold, or >= 0.4 with the explicit zero-shot >= 0.5."""
    if nsfw_score is None:
        return False
    if nsfw_score >= settings.nsfw_threshold:
        return True
    return nsfw_score >= NSFW_SECOND_SIGNAL and (explicit_p or 0.0) >= EXPLICIT_MIN


def decide(kind: str, *, nsfw_score: float | None = None, explicit_p: float | None = None,
           nsfw_override: bool | None = None, screenshot_heur: bool = False, screenshot_p: float | None = None,
           doc_like: float | None = None, has_person: bool = False,
           scores: dict[str, float] | None = None, learned: tuple[str, float] | None = None,
           nsfw_learned: bool | None = None) -> Decision:
    """Section 9 order: documents by kind, nsfw, screenshots, document-like images, people, then the
    best scene group (places, food, pets, objects; probability from the joint CATEGORY_PROMPTS softmax)
    when >= SCENE_MIN, else misc.
    nsfw_override (the user's choice) replaces the classifier decision; nsfw_learned (the vault
    learner, section 18) replaces it when there is no override; learned (category, score) from the
    owner's labelled look-alikes wins over the rules below, except documents by kind and the vault."""
    if kind in ("pdf", "document"):
        return Decision("documents", bool(nsfw_override) if nsfw_override is not None else False, "kind")
    if nsfw_override is not None:
        nsfw = nsfw_override
    elif nsfw_learned is not None:
        nsfw = nsfw_learned
    else:
        nsfw = nsfw_decision(nsfw_score, explicit_p)
    if nsfw:
        return Decision("nsfw", True, "nsfw")
    if kind == "other":
        return Decision("misc", False, "kind")
    if learned is not None and learned[0] in LEARNABLE_CATEGORIES:
        return Decision(learned[0], False, "learned", {"score": round(float(learned[1]), 4)})
    if screenshot_heur or (screenshot_p is not None and screenshot_p >= SCREENSHOT_P_MIN):
        return Decision("screenshots", False, "screenshot")
    if doc_like is not None and doc_like >= settings.doc_likeness_min:
        return Decision("documents", False, "doc_likeness")
    if has_person:
        return Decision("personal", False, "people")
    scores = scores or {}
    scene = {k: float(scores.get(k, 0.0)) for k in SCENE_GROUPS}
    best = max(scene, key=lambda k: scene[k])
    if scene[best] >= SCENE_MIN:
        return Decision(best, False, "scene", {"p": round(scene[best], 4)})
    return Decision("misc", False, "none")
