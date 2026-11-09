"""Nail polish on one hand (section 34, design 3.5): colour, finish and length, as weak short-term evidence.

CONTRACT.
- describe_hand(crop_rgb, lm, conf, side, probs, *, src) -> attrs | None. crop_rgb is handpose's 224 px hand crop
  (app.analysis.trait_regions.nail_crop, wrist down), lm its 21 landmarks in crop pixels, conf handpose's
  confidence, side 'l' or 'r' (the person's hand, from DWPose), probs(crops_rgb, groups) -> (n, len(groups))
  the SigLIP zero-shot through trait_regions (PIL RGB inside), src 'original' or 'preview'.
- None (no row) unless every gate holds: conf >= MIN_CONF; the back of the hand faces the camera; at least
  MIN_FINGERS fingers whose nail differs from their own skin by NAIL_SKIN_DE or more and whose nail colour does
  not run more than MAX_RUN past the tip (a held object); the tip crop's p(polish) >= TIP_P; the polished nails
  agree within SPREAD (or form two such groups, 'multi'); the tip crop's sharpness >= MIN_SHARP.
- attrs: {lab (the polish after von Kries white balance on the hand's skin), color (one of COLOR_NAMES), hex,
  finish (plain, french, glitter, art, multi), finish_p, length (short, medium, long, from the median run past
  the tip), r, fingers, p_tip, back, sharp, src}.
- The back-of-hand rule in image coordinates (y down) with the crop upright: cross(p5 - p0, p17 - p0) > 0 for
  the right hand and < 0 for the left (the thumb on the image's left for a right hand seen from the back). Not
  checked on real photos.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable

import cv2
import numpy as np

TIP_GROUPS: dict[str, list[str]] = {
    "polish": ["close-up of fingernails painted with nail polish", "painted fingernails",
               "fingertips with coloured nail polish", "nail art"],
    "natural": ["close-up of natural unpainted fingernails", "bare fingertips without nail polish",
                "plain natural fingernails"],
}
FINISH_GROUPS: dict[str, list[str]] = {
    "plain": ["fingernails painted in one plain colour"],
    "french": ["french manicure with white nail tips"],
    "glitter": ["glitter nail polish", "sparkly fingernails"],
    "art": ["nail art designs on fingernails", "fingernails with rhinestones"],
}
FINGERS = ((4, 3, 2), (8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18))  # tip, dip, pip
MIN_CONF = 0.8
MIN_PIXELS = 12
NAIL_SKIN_DE = 12.0
MIN_FINGERS = 2
SPREAD = 15.0
MULTI_APART = 20.0
RUN_DE = 8.0          # the nail colour "runs" past the tip while within this of the nail
MAX_RUN = 0.6         # more than this (of DIP to TIP) past the tip: a held object, the finger is rejected
RUN_STEP = 0.05
TIP_P = 0.5
TIP_PAD = 0.35
FINISH_P = 0.7
MIN_SHARP = 80.0
LENGTHS = ((0.15, "short"), (0.35, "medium"))  # else long
SKIN_TARGET = (190.0, 140.0, 115.0)  # sRGB the hand's skin is mapped to before the polish colour is kept
COLOR_NAMES: dict[str, tuple[int, int, int]] = {
    "red": (200, 30, 40), "maroon": (110, 20, 35), "pink": (240, 130, 170), "coral": (250, 110, 90),
    "orange": (240, 130, 30), "yellow": (240, 210, 60), "green": (40, 150, 70), "blue": (40, 80, 200),
    "purple": (120, 50, 150), "brown": (120, 70, 40), "nude": (215, 165, 140), "white": (240, 240, 235),
    "black": (25, 25, 30), "grey": (130, 130, 135),
}


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIELAB float32 (L 0..100)."""
    a = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(a, cv2.COLOR_RGB2Lab).reshape(np.shape(rgb)[:-1] + (3,))


_NAMES = list(COLOR_NAMES)
_NAME_LABS = _lab(np.array([COLOR_NAMES[n] for n in _NAMES], np.uint8))


def color_name(lab: np.ndarray) -> str:
    return _NAMES[int(np.argmin(np.linalg.norm(_NAME_LABS - np.asarray(lab, np.float32), axis=1)))]


def _to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def white_balance(nail_rgb: np.ndarray, skin_rgb: np.ndarray) -> np.ndarray:
    """von Kries in linear RGB: the gains that map the median skin to SKIN_TARGET, applied to the nail colour.
    Returns sRGB float (3,)."""
    skin = np.maximum(_to_linear(skin_rgb), 1e-4)
    gains = _to_linear(np.array(SKIN_TARGET)) / skin
    return _to_srgb(_to_linear(nail_rgb) * gains)


def strip(img: np.ndarray, a: np.ndarray, b: np.ndarray, s0: float, s1: float, half: float,
          step: float = 0.5) -> np.ndarray | None:
    """Pixels of `img` along a -> b from s0 to s1 (fractions of |ab|), half-width `half` x |ab| (probe4's patch);
    None under MIN_PIXELS."""
    u = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    L = float(np.hypot(*u))
    if L < 3:
        return None
    u /= L
    n = np.array([-u[1], u[0]])
    ss = np.arange(s0, s1, step / L)
    ws = np.arange(-half, half + 1e-6, step / L)
    if not len(ss) or not len(ws):
        return None
    pts = np.asarray(a)[None, None, :] + ss[:, None, None] * L * u + ws[None, :, None] * L * n
    xs, ys = np.rint(pts[..., 0]).astype(int).ravel(), np.rint(pts[..., 1]).astype(int).ravel()
    ok = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
    if ok.sum() < MIN_PIXELS:
        return None
    return img[ys[ok], xs[ok]]


def run_past_tip(lab: np.ndarray, dip: np.ndarray, tip: np.ndarray, nail: np.ndarray) -> float:
    """How far (a fraction of DIP to TIP) the nail colour goes on past the tip: the first slice beyond the tip
    whose median is more than RUN_DE from the nail ends it. Above MAX_RUN means it never ended."""
    t = 0.0
    while t <= MAX_RUN:
        sl = strip(lab, dip, tip, 1.0 + t, 1.0 + t + RUN_STEP, 0.18)
        if sl is None or float(np.linalg.norm(np.median(sl, 0) - nail)) > RUN_DE:
            return t
        t += RUN_STEP
    return t


def is_back(lm: np.ndarray, side: str) -> bool:
    v5, v17 = lm[5, :2] - lm[0, :2], lm[17, :2] - lm[0, :2]
    cross = float(v5[0] * v17[1] - v5[1] * v17[0])
    return cross > 0 if side == "r" else cross < 0


def _spread(labs: np.ndarray) -> float:
    m = np.median(labs, 0)
    return float(np.median(np.linalg.norm(labs - m, axis=1)))


def _two_groups(labs: np.ndarray) -> tuple[list[int], list[int]] | None:
    """The split of the polished nails into two groups of 2 or more, each within SPREAD, medians MULTI_APART
    apart (the best such split), or None."""
    n = len(labs)
    best = None
    for k in range(2, n - 1):
        for g in itertools.combinations(range(n), k):
            h = [i for i in range(n) if i not in g]
            if len(h) < 2 or min(g) > min(h):
                continue
            a, b = labs[list(g)], labs[h]
            if _spread(a) > SPREAD or _spread(b) > SPREAD:
                continue
            apart = float(np.linalg.norm(np.median(a, 0) - np.median(b, 0)))
            if apart > MULTI_APART and (best is None or apart > best[0]):
                best = (apart, list(g), h)
    return None if best is None else (best[1], best[2])


def tip_box(lm: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    pts = lm[[3, 4, 7, 8, 11, 12, 15, 16, 19, 20], :2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = TIP_PAD * max(x1 - x0, y1 - y0, 10)
    H, W = shape[:2]
    return (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(W, x1 + pad)), int(min(H, y1 + pad)))


def describe_hand(crop_rgb: np.ndarray, lm: np.ndarray, conf: float, side: str,
                  probs: Callable[[list[np.ndarray], dict[str, list[str]]], np.ndarray], *,
                  src: str) -> dict[str, Any] | None:
    lm = np.asarray(lm, np.float64).reshape(21, -1)[:, :2]
    if conf < MIN_CONF or not is_back(lm, side):
        return None
    rgb = np.ascontiguousarray(crop_rgb, dtype=np.uint8)
    lab = _lab(rgb)
    fingers = []
    for t, d, p in FINGERS:
        nl = strip(lab, lm[d], lm[t], 0.5, 1.0, 0.18)
        sk = strip(lab, lm[p], lm[d], 0.25, 0.75, 0.15)
        if nl is None or sk is None:
            continue
        nail, skin = np.median(nl, 0), np.median(sk, 0)
        de = float(np.linalg.norm(nail - skin))
        if de < NAIL_SKIN_DE:
            continue
        r = run_past_tip(lab, lm[d], lm[t], nail)
        if r > MAX_RUN:
            continue  # the colour runs on: a cup, a phone or a pole, not a nail
        fingers.append({"lab": nail, "r": r, "rgb": np.median(strip(rgb, lm[d], lm[t], 0.5, 1.0, 0.18), 0),
                        "skin_rgb": np.median(strip(rgb, lm[p], lm[d], 0.25, 0.75, 0.15), 0)})
    if len(fingers) < MIN_FINGERS:
        return None
    x0, y0, x1, y1 = tip_box(lm, rgb.shape)
    tip = rgb[y0:y1, x0:x1]
    if tip.size == 0:
        return None
    sharp = float(cv2.Laplacian(cv2.cvtColor(tip, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    if sharp < MIN_SHARP:
        return None
    labs = np.array([f["lab"] for f in fingers])
    groups = _two_groups(labs)  # two colours first: a robust spread would hide a 3 and 2 split
    if groups is None and _spread(labs) > SPREAD:
        return None
    tip224 = cv2.resize(tip, (224, 224), interpolation=cv2.INTER_AREA if tip.shape[0] > 224 else cv2.INTER_LINEAR)
    p_tip = float(probs([tip224], TIP_GROUPS)[0][0])
    if p_tip < TIP_P:
        return None
    keep = list(range(len(fingers))) if groups is None else max(groups, key=len)
    nail_rgb = np.median(np.array([fingers[i]["rgb"] for i in keep]), 0)
    skin_rgb = np.median(np.array([f["skin_rgb"] for f in fingers]), 0)
    balanced = white_balance(nail_rgb, skin_rgb)
    blab = _lab(np.clip(np.rint(balanced), 0, 255).astype(np.uint8))
    if groups is not None:
        finish, finish_p = "multi", 1.0
    else:
        fp = probs([tip224], FINISH_GROUPS)[0]
        k = int(np.argmax(fp))
        finish, finish_p = list(FINISH_GROUPS)[k], float(fp[k])
        if finish != "plain" and finish_p < FINISH_P:
            finish = "plain"
    r_med = float(np.median([f["r"] for f in fingers]))
    length = next((name for cut, name in LENGTHS if r_med < cut), "long")
    hexs = "#" + "".join(f"{int(v):02x}" for v in np.clip(np.rint(balanced), 0, 255))
    return {"lab": [round(float(v), 1) for v in blab], "color": color_name(blab), "hex": hexs, "finish": finish,
            "finish_p": round(finish_p, 3), "length": length, "r": round(r_med, 3), "fingers": len(fingers),
            "p_tip": round(p_tip, 3), "back": True, "sharp": round(sharp, 1), "src": src}
