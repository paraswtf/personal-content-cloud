"""Where traits are read on a photo (section 34): pose, limb and hand crops, ownership, skin and ink, the SigLIP
gates, the native-resolution crops and their descriptors. Called by the traits runner (app.analysis.traits).

CONTRACT.
- measure_item(conn, item_id, engines) -> Measured | None: measures every body of one item as a whole, outside
  any transaction. Measured = (item_id, fingerprint, scans {body id: (status, bare)}, rows [TraitRow]); the
  fingerprint is the item's box fingerprint read before measuring (the writer compares it again). None when the
  item or its bodies are gone.
- Statuses without a decode, first match wins: 'video' (not an image, or a video frame's box), 'vault'
  (nsfw = 1), 'doc' (category screenshots or documents, or doc_likeness >= settings.doc_likeness_min), 'small'
  (the box's long side at 2048 px under settings.trait_min_body_px), 'child' (ages.ready and the body's face in
  the child band). The other bodies are measured on the preview (falling back to animals.open_photo): 'ok', or
  'no_pixels' when neither can be read, or 'error' when our own code failed on that body (it keeps no rows).
- Pose runs once over the measured bodies. Limbs: forearm (elbow to FOREARM_END of the way to the wrist), upper
  arm and calf, kept with both keypoint scores >= KP_MIN and the full segment >= settings.trait_min_limb_px.
  Hands: DWPose hand points with a mean score >= HAND_MIN and a span >= HAND_MIN_SPAN. Ownership drops a region
  (never flags it): under OWN_INSIDE of it inside the body box grown by OWN_EXPAND, more than OWN_OTHER_ALONE
  of it over another box, another body's same-family claim (scores >= CLAIM_KP_MIN) within CLAIM_DIST of its length,
  or a forearm / upper arm length ratio outside ARM_RATIO. Boxes at IoU >= TWIN_IOU with the body are its twin
  and ignored. Every box of the item counts for the overlap test, measured or not.
- The gate: probe8's crops on the preview (a limb: a square of LIMB_SIDE x the segment, upright, 224 px, the
  sides outside the limb strip set to grey; a hand: HAND_GATE_SPAN x its span, wrist down), as PIL RGB through
  traits.siglip_images, zero-shot over LIMB_GROUPS, one batch per item; skin, ink and sharpness on the strip
  (decide()). Rules in order: mehndi (forearm or hand), tattoo, unclear, bare, nothing.
- A tattoo or mehndi crop is cut again from the original (decoded once per item, only when needed) at
  out = clamp(LIMB_SIDE x L x s, 224, 512) px, s = original side / preview side; with s < ORIG_MIN_SCALE or an
  unreadable original, from the preview (attrs.src 'preview'). Row: detail = trait_sift.encode_detail(native),
  vec = DINOv2 CLS of the native crop at 224 and of its mirror (float32[768]), crop = 192 px WebP q80, attrs
  {p (the five groups), kp (SIFT keypoints of the stored detail), small (kp < SMALL_KP), share, out, src, skin,
  ink, sharp, len}.
- A hand with no tattoo or mehndi and p_jewel < HAND_JEWEL_P whose span in the source is at least
  settings.nail_min_hand_px goes to app.analysis.trait_nails (the original when the photo's long side is
  NAIL_ORIG_SIDE or more). A polished hand gives a 'nails' row (no vec, no detail).
- Engines: every call to pose, hands, dino or SigLIP goes through _call; any exception there raises EngineError
  (the runner stops and writes nothing for the item). Crops reach SigLIP as PIL RGB, never as arrays.
"""
from __future__ import annotations

import io
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from app.analysis import trait_sift
from app.config import settings

log = logging.getLogger(__name__)

# The SigLIP groups (frozen by the design: the cuts below were measured with exactly these, probe8).
LIMB_GROUPS: dict[str, list[str]] = {
    "tattoo": ["a photo of a tattoo on a person's skin", "a tattooed arm with black ink", "a leg with a tattoo design"],
    "bare": ["a photo of bare skin", "plain skin of an arm", "a leg without tattoos"],
    "cloth": ["the sleeve of a shirt", "trousers covering a leg", "printed fabric with a pattern"],
    "henna": ["henna mehndi design on a hand or arm", "a hand decorated with mehndi"],
    "jewel": ["a wristwatch on a wrist", "a bracelet or bangle on a wrist", "a hand holding a phone"],
}
GROUP_KEYS = tuple(LIMB_GROUPS)
# Gate cuts, measured by probe8 on the final groups (TATTOO_GATE_P is settings.tattoo_gate_p).
UNCLEAR_P = 0.60
HENNA_P = 0.50
CLOTH_P = 0.10
JEWEL_P = 0.15
BARE_P = 0.80
HAND_JEWEL_P = 0.50   # a phone or a watch in the hand: no nails row
# Skin, ink and sharpness (not measured; design 3.4).
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)
FACE_SKIN_RANGE = 10  # with a face, the skin range is its median Cr / Cb plus or minus this (20 read inked arms as skin)
INK_DL = 20.0         # ink: 20 or more below the skin L*...
INK_DC = 25.0         # ...or 25 or more away in chroma
TATTOO_SKIN = 0.35
TATTOO_INK = (0.02, 0.60)
SHARP_MIN = 60.0      # bare limbs (tattoo absence) need this sharpness
TATTOO_SHARP_MIN = 30.0  # a tattoo row needs this; skin and ink fractions are not used for it (they dropped real sleeves)
BARE_SKIN = 0.60
BARE_INK = 0.01
BARE_KP_MIN = 0.6
MEHNDI_HUE = (30.0, 75.0)  # median Lab hue (degrees) of the darkest DARK_SHARE of the dark pixels...
MEHNDI_CHROMA = 15.0       # ...with this chroma or more
DARK_SHARE = 0.20
# Geometry (COCO-WholeBody points: 5/6 shoulders, 7/8 elbows, 9/10 wrists, 13/14 knees, 15/16 ankles,
# 91-111 left hand, 112-132 right hand).
LIMBS: dict[str, tuple[int, int]] = {
    "forearm_l": (7, 9), "forearm_r": (8, 10), "upper_arm_l": (5, 7), "upper_arm_r": (6, 8),
    "calf_l": (13, 15), "calf_r": (14, 16),
}
HANDS: dict[str, int] = {"hand_l": 91, "hand_r": 112}
FOREARM_END = 0.8      # the forearm crop stops here on the way to the wrist (watches and bangles stay out)
KP_MIN = 0.5
CLAIM_KP_MIN = 0.3
HAND_MIN = 0.4
HAND_MIN_SPAN = 64    # step 0 measured the gate on hands of 64 px or more
LIMB_SIDE = trait_sift.LIMB_STRIP  # crop side = 1.15 x the segment
GATE_SIDE = 224
HAND_GATE_SPAN = 1.4   # the gate's hand crop (probe8)
HAND_NAIL_SPAN = 1.7   # handpose's hand crop (probe4)
NATIVE_MIN = 224
NATIVE_MAX = trait_sift.DETAIL_MAX
ORIG_MIN_SCALE = 1.2
NAIL_ORIG_SIDE = 3072
UI_SIDE = 192
UI_QUALITY = 80
SMALL_KP = 60
QUALITY_SIDE = 96.0
QUALITY_SHARP = 150.0
# Ownership.
OWN_EXPAND = 0.10
OWN_INSIDE = 0.80
OWN_OTHER_ALONE = 0.5   # a region more than half over another box goes; less only when another pose claims it
TWIN_IOU = 0.7
CLAIM_DIST = 0.5
ARM_RATIO = (0.5, 1.6)


class EngineError(RuntimeError):
    """A model call failed (including session creation or running out of GPU memory): the run stops."""


@dataclass
class TraitRow:
    body_id: int
    kind: str
    part: str
    x: float
    y: float
    w: float
    h: float
    quality: float
    vec: bytes | None
    detail: bytes | None
    attrs: dict[str, Any]
    crop: bytes


@dataclass
class Measured:
    item_id: int
    fingerprint: str
    scans: dict[int, tuple[str, dict[str, Any]]]
    rows: list[TraitRow] = field(default_factory=list)


@dataclass(eq=False)
class Region:
    """A limb or hand of one body, in preview pixels."""
    body_id: int
    part: str
    a: np.ndarray            # segment start (the elbow, shoulder or knee); for a hand the centre
    b: np.ndarray            # segment end (shortened for the forearm); for a hand the centre too
    length: float            # the full segment (or the hand span) in preview px
    crop_len: float          # the length the crop is cut around (the forearm's shortened one)
    score: float             # min keypoint score (mean for a hand)
    bound: tuple[float, float, float, float]  # x0, y0, x1, y1
    pts: np.ndarray | None = None              # a hand's 21 points (x, y, score)

    @property
    def family(self) -> str:
        return self.part.rsplit("_", 1)[0]

    @property
    def mid(self) -> np.ndarray:
        return (self.a + self.b) / 2.0

    @property
    def is_hand(self) -> bool:
        return self.pts is not None


@dataclass(eq=False)
class Look:
    """What the gate saw on one region."""
    region: Region
    gate: np.ndarray
    skin: float = 0.0
    ink: float = 0.0
    sharp: float = 0.0
    henna_hue: bool = False
    p: dict[str, float] = field(default_factory=dict)
    verdict: str | None = None


# ------------------------------------------------------------------ engines
def _call(name: str, fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception as err:  # noqa: BLE001 - any engine failure stops the run, it is never a body's error
        raise EngineError(f"{name}: {err}") from err


def siglip_probs(siglip: Any, crops_rgb: list[np.ndarray], groups: dict[str, list[str]]) -> np.ndarray:
    """Zero-shot probabilities (n, len(groups)) of RGB crops, sent to SigLIP as PIL RGB images."""
    from app.analysis.traits import siglip_images

    if not crops_rgb:
        return np.zeros((0, len(groups)), np.float32)
    embs = _call("siglip", siglip.embed_images, siglip_images(crops_rgb))
    probs = np.asarray(_call("siglip", siglip.zero_shot, embs, groups), dtype=np.float32)
    if probs.shape != (len(crops_rgb), len(groups)):
        raise EngineError(f"siglip: {probs.shape} probabilities for {len(crops_rgb)} crops")
    return probs


# ------------------------------------------------------------------ geometry
def _box_px(r: Any, W: int, H: int) -> tuple[float, float, float, float]:
    return float(r["x"]) * W, float(r["y"]) * H, (float(r["x"]) + float(r["w"])) * W, (float(r["y"]) + float(r["h"])) * H


def _area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _rect_bound(c: np.ndarray, u: np.ndarray, along: float, across: float) -> tuple[float, float, float, float]:
    """Axis-aligned bound of a rectangle centred on c, `along` long in direction u and `across` wide."""
    n = np.array([-u[1], u[0]])
    pts = [c + sa * along / 2 * u + sb * across / 2 * n for sa in (-1, 1) for sb in (-1, 1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def limb_regions(body_id: int, kp: np.ndarray, min_limb_px: float) -> tuple[list[Region], list[Region]]:
    """(candidates, claims) of one body's pose: candidates pass the score and length cuts; claims are every
    segment or hand seen with scores >= CLAIM_KP_MIN (what the double-claim rule compares against)."""
    from app.analysis.traits import share_for

    cands: list[Region] = []
    claims: list[Region] = []
    for part, (ia, ib) in LIMBS.items():
        a, b = kp[ia, :2].astype(np.float64), kp[ib, :2].astype(np.float64)
        score = float(min(kp[ia, 2], kp[ib, 2]))
        full = float(np.hypot(*(b - a)))
        if score < CLAIM_KP_MIN or full < 1.0:
            continue
        end = a + FOREARM_END * (b - a) if part.startswith("forearm") else b
        crop_len = float(np.hypot(*(end - a)))
        u = (end - a) / max(crop_len, 1e-6)
        bound = _rect_bound((a + end) / 2.0, u, LIMB_SIDE * crop_len, float(share_for(part)) * crop_len)
        r = Region(body_id, part, a, end, full, crop_len, score, bound)
        claims.append(r)
        if score >= KP_MIN and full >= min_limb_px:
            cands.append(r)
    for part, lo in HANDS.items():
        pts = kp[lo:lo + 21].astype(np.float64)
        score = float(pts[:, 2].mean())
        if score < CLAIM_KP_MIN:
            continue
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        c = pts[:, :2].mean(0)
        pad = 0.15 * span
        bound = (float(pts[:, 0].min() - pad), float(pts[:, 1].min() - pad),
                 float(pts[:, 0].max() + pad), float(pts[:, 1].max() + pad))
        r = Region(body_id, part, c, c, span, span, score, bound, pts)
        claims.append(r)
        if score >= HAND_MIN and span >= HAND_MIN_SPAN:
            cands.append(r)
    return cands, claims


def plausible(cands: list[Region]) -> list[Region]:
    """Drops a side's forearm and upper arm when their full lengths are out of proportion (a mixed-up pose)."""
    by = {r.part: r for r in cands}
    drop: set[str] = set()
    for side in ("l", "r"):
        fa, ua = by.get(f"forearm_{side}"), by.get(f"upper_arm_{side}")
        if fa is not None and ua is not None and ua.length > 0:
            ratio = fa.length / ua.length
            if not ARM_RATIO[0] <= ratio <= ARM_RATIO[1]:
                drop |= {fa.part, ua.part}
    return [r for r in cands if r.part not in drop]


def owned(r: Region, own_box: tuple[float, float, float, float], others: list[tuple[float, float, float, float]],
          other_claims: list[Region]) -> bool:
    """The ownership rules for one region: inside its own box, not over another box, not claimed by another
    body's pose. `others` and `other_claims` already leave out the body's twins."""
    area = _area(r.bound)
    if area <= 0:
        return False
    x0, y0, x1, y1 = own_box
    ex, ey = OWN_EXPAND * (x1 - x0), OWN_EXPAND * (y1 - y0)
    if _inter(r.bound, (x0 - ex, y0 - ey, x1 + ex, y1 + ey)) / area < OWN_INSIDE:
        return False
    claimed = any(c.family == r.family and float(np.hypot(*(c.mid - r.mid))) < CLAIM_DIST * max(r.length, 1.0)
                  for c in other_claims)
    if claimed:
        return False
    # over another box: dropped when mostly over it; a little overlap (people standing close) only with a claim
    return not any(_inter(r.bound, o) / area > OWN_OTHER_ALONE for o in others)


def _upright(rgb: np.ndarray, c: np.ndarray, angle_deg: float, side_src: float, out: int, interp: int) -> np.ndarray:
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), angle_deg, out / max(side_src, 1e-6))
    M[0, 2] += out / 2 - c[0]
    M[1, 2] += out / 2 - c[1]
    return cv2.warpAffine(rgb, M, (out, out), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def _limb_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) - 90.0


def _hand_angle(pts: np.ndarray) -> float:
    wrist, mid = pts[0, :2], pts[9, :2]
    return math.degrees(math.atan2(mid[1] - wrist[1], mid[0] - wrist[0])) + 90.0


def gate_crop(rgb: np.ndarray, r: Region) -> np.ndarray:
    """The 224 px crop the gate was measured on (probe8): a limb strip with grey sides, or a hand."""
    from app.analysis.traits import share_for

    if r.is_hand:
        side = max(HAND_GATE_SPAN * r.length, 24.0)
        return _upright(rgb, r.pts[:, :2].mean(0), _hand_angle(r.pts), side, GATE_SIDE, cv2.INTER_LINEAR)
    crop = _upright(rgb, r.mid, _limb_angle(r.a, r.b), LIMB_SIDE * r.crop_len, GATE_SIDE, cv2.INTER_LINEAR)
    lo, hi = strip_cols(GATE_SIDE, share_for(r.part))
    out = np.full_like(crop, 127)
    out[:, lo:hi] = crop[:, lo:hi]
    return out


def strip_cols(side: int, share: float | None) -> tuple[int, int]:
    """The columns of a crop's limb strip (the whole crop for a hand), as trait_sift.strip_mask has them."""
    if share is None:
        return 0, side
    half = int(side * float(share) / LIMB_SIDE / 2)
    return max(0, side // 2 - half), side // 2 + half


def native_crop(src: np.ndarray, r: Region, s: float) -> np.ndarray:
    """The region cut again from `src` (the original, or the preview with s = 1): upright, unmasked, INTER_AREA,
    clamp(LIMB_SIDE x L x s, 224, 512) px (a hand: HAND_GATE_SPAN x its span)."""
    if r.is_hand:
        side = max(HAND_GATE_SPAN * r.length * s, 24.0)
        out = int(min(NATIVE_MAX, max(NATIVE_MIN, side)))
        return _upright(src, r.pts[:, :2].mean(0) * s, _hand_angle(r.pts), side, out, cv2.INTER_AREA)
    side = LIMB_SIDE * r.crop_len * s
    out = int(min(NATIVE_MAX, max(NATIVE_MIN, side)))
    return _upright(src, r.mid * s, _limb_angle(r.a, r.b), side, out, cv2.INTER_AREA)


def nail_crop(src: np.ndarray, r: Region, s: float) -> np.ndarray:
    """handpose's 224 px input (probe4): HAND_NAIL_SPAN x the span, wrist down."""
    side = max(HAND_NAIL_SPAN * r.length * s, 24.0)
    return _upright(src, r.pts[:, :2].mean(0) * s, _hand_angle(r.pts), side, GATE_SIDE, cv2.INTER_LINEAR)


# ------------------------------------------------------------------ skin and ink
def face_skin(rgb: np.ndarray, face_box_px: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    """Median (Cr, Cb) of the central half of a face box, or None."""
    if face_box_px is None:
        return None
    x0, y0, x1, y1 = face_box_px
    cx0, cy0 = int(x0 + (x1 - x0) / 4), int(y0 + (y1 - y0) / 4)
    cx1, cy1 = int(x1 - (x1 - x0) / 4), int(y1 - (y1 - y0) / 4)
    H, W = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, cx0), max(0, cy0), min(W, cx1), min(H, cy1)
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    ycc = cv2.cvtColor(np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1]), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    return float(np.median(ycc[:, 1])), float(np.median(ycc[:, 2]))


def skin_ink(strip_rgb: np.ndarray, face: tuple[float, float] | None) -> tuple[float, float, float, bool]:
    """(skin_frac, ink_frac, sharpness, henna_hue) of a limb strip (or a hand crop).
    Skin: YCrCb within SKIN_CR / SKIN_CB, or the face's median plus or minus FACE_SKIN_RANGE. Ink: inside the
    convex hull of the skin, not skin, and INK_DL below the skin L* or INK_DC from it in chroma; ink_frac is its
    share of the hull. Sharpness: variance of the Laplacian of the grey strip. henna_hue: the darkest DARK_SHARE
    of the pixels INK_DL or more below the skin L* has a median Lab hue in MEHNDI_HUE and chroma of
    MEHNDI_CHROMA or more (the design said "of the pixels": with a small tattoo those are mostly skin, whose hue
    is in the same range)."""
    rgb = np.ascontiguousarray(strip_rgb, dtype=np.uint8)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    if face is not None:
        cr_lo, cr_hi = face[0] - FACE_SKIN_RANGE, face[0] + FACE_SKIN_RANGE
        cb_lo, cb_hi = face[1] - FACE_SKIN_RANGE, face[1] + FACE_SKIN_RANGE
    else:
        (cr_lo, cr_hi), (cb_lo, cb_hi) = SKIN_CR, SKIN_CB
    cr, cb = ycc[..., 1].astype(np.float32), ycc[..., 2].astype(np.float32)
    skin = (cr >= cr_lo) & (cr <= cr_hi) & (cb >= cb_lo) & (cb <= cb_hi)
    n = skin.size
    skin_frac = float(skin.sum()) / n if n else 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    ink_frac = 0.0
    dark = np.zeros(skin.shape, bool)
    if skin.sum() >= 16:
        ys, xs = np.nonzero(skin)
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
        hull_mask = np.zeros(skin.shape, np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        inside = hull_mask.astype(bool)
        sl = np.median(lab[skin], axis=0)
        dl = sl[0] - lab[..., 0]
        dc = np.hypot(lab[..., 1] - sl[1], lab[..., 2] - sl[2])
        ink = inside & ~skin & ((dl >= INK_DL) | (dc >= INK_DC))
        ink_frac = float(ink.sum()) / max(1, int(inside.sum()))
        dark = dl >= INK_DL  # henna's brown passes the YCrCb skin test, so its colour is read on the dark pixels
    henna = False
    flat = lab[dark]
    if len(flat) >= 12:  # the ink's own colour: black and blue ink is grey, henna is brown to orange
        k = max(1, int(len(flat) * DARK_SHARE))
        dark = flat[np.argsort(flat[:, 0])[:k]]
        a, b = float(np.median(dark[:, 1])), float(np.median(dark[:, 2]))
        hue = (math.degrees(math.atan2(b, a)) + 360.0) % 360.0
        henna = MEHNDI_HUE[0] <= hue <= MEHNDI_HUE[1] and math.hypot(a, b) >= MEHNDI_CHROMA
    return skin_frac, ink_frac, sharp, henna


def decide(family: str, p: dict[str, float], skin: float, ink: float, sharp: float, henna_hue: bool, score: float,
           length: float) -> str | None:
    """The gate's verdict for one crop: 'mehndi', 'tattoo', 'unclear', 'bare' or None (no evidence).

    A tattoo is SigLIP's verdict (the step 0 cuts) on a sharp enough crop. The skin and ink fractions are not used
    for it: on the owner's library they kept 345 of 874 crops that passed the cuts, dropping mostly real sleeves
    (dense ink leaves little skin); sharpness alone keeps 852. They still decide bare limbs. Henna (a forearm or a
    hand that SigLIP calls henna, or tattoo-like ink with a henna hue) is mehndi only with TRAITS_MEHNDI; off, it
    is no evidence at all, never a long-term tattoo."""
    gate = float(settings.tattoo_gate_p)
    sharp_ok = sharp >= TATTOO_SHARP_MIN
    tattoo = p["tattoo"] >= gate and p["cloth"] < CLOTH_P and p["jewel"] < JEWEL_P and sharp_ok
    if family in ("forearm", "hand") and (p["henna"] >= HENNA_P or (tattoo and henna_hue)):
        return "mehndi" if settings.traits_mehndi else None
    if tattoo:
        return "tattoo"
    jewel_only = p["tattoo"] >= gate and p["cloth"] < CLOTH_P and p["jewel"] >= JEWEL_P and sharp_ok
    if UNCLEAR_P <= p["tattoo"] < gate or jewel_only:
        return "unclear"
    if (p["bare"] >= BARE_P and skin >= BARE_SKIN and ink < BARE_INK and sharp >= SHARP_MIN and score >= BARE_KP_MIN
            and length >= 2 * settings.trait_min_limb_px):
        return "bare"
    return None


# ------------------------------------------------------------------ pixels
def _webp(rgb: np.ndarray, side: int = UI_SIDE, quality: int = UI_QUALITY) -> bytes:
    img = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB")
    img.thumbnail((side, side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality)
    return buf.getvalue()


def load_preview(item: sqlite3.Row) -> np.ndarray | None:
    """The 2048 px preview as RGB, falling back to the original as the analysis saw it; None when neither reads."""
    from app import storage

    p = storage.preview_path(item["uid"], int(item["version"]))
    try:
        if p.is_file():
            with Image.open(p) as im:
                return np.asarray(im.convert("RGB"))
    except Exception as e:  # noqa: BLE001 - a broken preview falls back to the original
        log.warning("traits: preview of %s unreadable: %s", item["uid"], e)
    try:
        from app.analysis.animals import open_photo

        img, _bgr = open_photo(item)
        return np.asarray(img.convert("RGB"))
    except Exception as e:  # noqa: BLE001 - no pixels at all: the bodies get 'no_pixels'
        log.info("traits: no pixels for %s: %s", item["uid"], e)
        return None


def load_original(item: sqlite3.Row) -> np.ndarray | None:
    from app import storage
    from app.media.images import load_oriented

    try:
        img = load_oriented(storage.original_path(item["uid"], int(item["version"]), item["ext"]))
        return np.asarray(img.convert("RGB"))
    except Exception as e:  # noqa: BLE001 - the preview stands in (attrs.src 'preview')
        log.info("traits: original of %s unreadable, using the preview: %s", item["uid"], e)
        return None


class _Source:
    """The original, decoded once per item and only when asked for."""

    def __init__(self, item: sqlite3.Row, preview: np.ndarray):
        self.item, self.preview = item, preview
        self._orig: np.ndarray | None = None
        self._done = False
        self.s = 1.0

    def original(self) -> tuple[np.ndarray | None, float]:
        if not self._done:
            self._done = True
            o = load_original(self.item)
            if o is not None:
                self.s = max(o.shape[:2]) / max(self.preview.shape[:2])
                self._orig = o
        return self._orig, self.s

    def for_detail(self) -> tuple[np.ndarray, float, str]:
        """(pixels, scale from the preview, src) for a tattoo or mehndi crop."""
        o, s = self.original()
        if o is None or s < ORIG_MIN_SCALE:
            return self.preview, 1.0, "preview"
        return o, s, "original"

    def for_nails(self) -> tuple[np.ndarray, float, str]:
        W, H = int(self.item["width"] or 0), int(self.item["height"] or 0)
        if max(W, H) >= NAIL_ORIG_SIDE:
            o, s = self.original()
            if o is not None:
                return o, s, "original"
        return self.preview, 1.0, "preview"


# ------------------------------------------------------------------ the item
def _status_without_pixels(conn: sqlite3.Connection, item: sqlite3.Row, b: sqlite3.Row, ages_ready: bool) -> str | None:
    if item["kind"] != "image" or b["frame_time_s"] is not None:
        return "video"
    if item["nsfw"]:
        return "vault"
    if (item["category"] or "") in ("screenshots", "documents") or (
            item["doc_likeness"] is not None and float(item["doc_likeness"]) >= settings.doc_likeness_min):
        return "doc"
    W, H = int(item["width"] or 0), int(item["height"] or 0)
    if W > 0 and H > 0:
        long_side = max(float(b["w"]) * W, float(b["h"]) * H) * min(2048, max(W, H)) / max(W, H)
        if long_side < settings.trait_min_body_px:
            return "small"
    if ages_ready and b["face_id"] is not None and b["f_det"] is not None:
        from app.analysis import ages
        from app.analysis.faces import face_weight

        weight = face_weight(float(b["f_det"]), int(b["f_size"] or 0), b["f_clarity"], b["f_sharp"])
        if ages.in_child_band(b["f_age"], weight):
            return "child"
    return None


def measure_item(conn: sqlite3.Connection, item_id: int, engines: Any) -> Measured | None:
    from app.analysis import ages, traits

    item = conn.execute("SELECT * FROM items WHERE id = ?", (int(item_id),)).fetchone()
    if item is None or item["deleted_at"] is not None:
        return None
    bodies = conn.execute(
        "SELECT b.id, b.x, b.y, b.w, b.h, b.face_id, b.frame_time_s, f.age_groups AS f_age, f.det_score AS f_det, "
        "f.size_px AS f_size, f.clarity AS f_clarity, f.sharpness AS f_sharp, f.x AS fx, f.y AS fy, f.w AS fw, "
        "f.h AS fh FROM bodies b LEFT JOIN faces f ON f.id = b.face_id WHERE b.item_id = ? ORDER BY b.id",
        (int(item_id),)).fetchall()
    fp = traits.item_fingerprint(conn, int(item_id))
    if not bodies or fp is None:
        return None
    out = Measured(int(item_id), fp, {})
    ages_ready = ages.ready(conn)
    live = []
    for b in bodies:
        st = _status_without_pixels(conn, item, b, ages_ready)
        if st is None:
            live.append(b)
        else:
            out.scans[int(b["id"])] = (st, {})
    if not live:
        return out
    rgb = load_preview(item)
    if rgb is None:
        for b in live:
            out.scans[int(b["id"])] = ("no_pixels", {})
        return out
    _measure_live(item, bodies, live, rgb, engines, out)
    return out


def _measure_live(item: sqlite3.Row, bodies: list[sqlite3.Row], live: list[sqlite3.Row], rgb: np.ndarray,
                  engines: Any, out: Measured) -> None:
    from app.analysis import trait_nails
    from app.analysis.traits import share_for

    H, W = rgb.shape[:2]
    boxes = {int(b["id"]): _box_px(b, W, H) for b in bodies}
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    kps = _call("pose", engines.pose.estimate, bgr, [boxes[int(b["id"])] for b in live])
    if kps is None or len(kps) != len(live):
        raise EngineError(f"pose: {0 if kps is None else len(kps)} results for {len(live)} boxes")
    errors: set[int] = set()
    cands: dict[int, list[Region]] = {}
    claims: dict[int, list[Region]] = {}
    for b, kp in zip(live, kps):
        bid = int(b["id"])
        try:
            kp = np.asarray(kp, dtype=np.float64)
            if kp.shape != (133, 3):
                raise EngineError(f"pose: points of shape {kp.shape}")
            c, cl = limb_regions(bid, kp, float(settings.trait_min_limb_px))
            cands[bid], claims[bid] = plausible(c), cl
        except EngineError:
            raise
        except Exception:  # noqa: BLE001 - our geometry failed on this body only
            log.exception("traits: pose geometry failed on body %s", bid)
            errors.add(bid)
    # ownership
    looks: list[Look] = []
    for b in live:
        bid = int(b["id"])
        if bid in errors:
            continue
        try:
            twins = {o for o, box in boxes.items() if o != bid and _iou(boxes[bid], box) >= TWIN_IOU}
            others = [box for o, box in boxes.items() if o != bid and o not in twins]
            other_claims = [r for o, rs in claims.items() if o != bid and o not in twins for r in rs]
            face = None
            if b["face_id"] is not None and b["fx"] is not None:
                face = face_skin(rgb, (float(b["fx"]) * W, float(b["fy"]) * H, (float(b["fx"]) + float(b["fw"])) * W,
                                       (float(b["fy"]) + float(b["fh"])) * H))
            for r in cands.get(bid, []):
                if not owned(r, boxes[bid], others, other_claims):
                    continue
                g = gate_crop(rgb, r)
                lo, hi = strip_cols(GATE_SIDE, share_for(r.part))
                lk = Look(r, g)
                lk.skin, lk.ink, lk.sharp, lk.henna_hue = skin_ink(g[:, lo:hi], face)
                looks.append(lk)
        except Exception:  # noqa: BLE001
            log.exception("traits: regions failed on body %s", bid)
            errors.add(bid)
            looks = [lk for lk in looks if lk.region.body_id != bid]
    probs = siglip_probs(engines.siglip, [lk.gate for lk in looks], LIMB_GROUPS)
    for lk, p in zip(looks, probs):
        lk.p = {k: float(p[i]) for i, k in enumerate(GROUP_KEYS)}
        r = lk.region
        lk.verdict = decide(r.family, lk.p, lk.skin, lk.ink, lk.sharp, lk.henna_hue, r.score, r.length)
    src = _Source(item, rgb)
    bare: dict[int, dict[str, Any]] = {int(b["id"]): {} for b in live}
    natives: list[tuple[Look, np.ndarray, bytes, float, str]] = []
    for lk in looks:
        r = lk.region
        if r.body_id in errors:
            continue
        try:
            if lk.verdict == "bare":
                bare[r.body_id][r.part] = {"p": round(lk.p["bare"], 3), "len": round(r.length, 1)}
            elif lk.verdict in ("tattoo", "mehndi"):
                pix, s, where = src.for_detail()
                nat = native_crop(pix, r, s)
                natives.append((lk, nat, trait_sift.encode_detail(nat), s, where))
        except Exception:  # noqa: BLE001
            log.exception("traits: the native crop failed on body %s", r.body_id)
            errors.add(r.body_id)
    natives = [n for n in natives if n[0].region.body_id not in errors]
    vecs = _dino(engines.dino, [trait_sift.decode_detail(n[2]) for n in natives])
    for (lk, nat, detail, s, where), vec in zip(natives, vecs):
        r = lk.region
        try:
            out.rows.append(_ink_row(lk, nat, detail, s, where, vec, W, H))
        except Exception:  # noqa: BLE001
            log.exception("traits: the %s row failed on body %s", lk.verdict, r.body_id)
            errors.add(r.body_id)
    # nails: hands with no ink, no phone or watch in them, big enough in the source used
    hands = [lk for lk in looks if lk.region.is_hand and lk.region.body_id not in errors
             and lk.verdict not in ("tattoo", "mehndi") and lk.p.get("jewel", 0.0) < HAND_JEWEL_P]
    nail_in: list[tuple[Look, np.ndarray, float, str]] = []
    for lk in hands:
        pix, s, where = src.for_nails()
        if lk.region.length * s < settings.nail_min_hand_px:
            continue
        try:
            nail_in.append((lk, nail_crop(pix, lk.region, s), s, where))
        except Exception:  # noqa: BLE001
            log.exception("traits: the hand crop failed on body %s", lk.region.body_id)
            errors.add(lk.region.body_id)
    if nail_in:
        marks = _call("hands", engines.hands.landmarks, [n[1] for n in nail_in])
        if marks is None or len(marks) != len(nail_in):
            raise EngineError(f"hands: {0 if marks is None else len(marks)} results for {len(nail_in)} hands")
        probs_fn = lambda crops, groups: siglip_probs(engines.siglip, crops, groups)  # noqa: E731
        for (lk, crop, s, where), (lm, conf, _right_p) in zip(nail_in, marks):
            r = lk.region
            if r.body_id in errors:
                continue
            try:
                attrs = trait_nails.describe_hand(crop, np.asarray(lm, dtype=np.float64), float(conf),
                                                  r.part[-1], probs_fn, src=where)
            except EngineError:
                raise
            except Exception:  # noqa: BLE001 - nail maths failed on this body only
                log.exception("traits: nails failed on body %s", r.body_id)
                errors.add(r.body_id)
                continue
            if attrs is None:
                continue
            attrs["span"] = round(float(r.length * s), 1)  # the hand span in source px (trait_eval's histogram)
            x0, y0, x1, y1 = _clip_bound(r.bound, W, H)
            q = min(1.0, r.length * s / QUALITY_SIDE) * r.score * min(1.0, float(attrs.get("sharp", 0.0)) / QUALITY_SHARP)
            out.rows.append(TraitRow(r.body_id, "nails", r.part, x0, y0, x1 - x0, y1 - y0, q, None, None, attrs,
                                     _webp(crop)))
    out.rows = [row for row in out.rows if row.body_id not in errors]
    for b in live:
        bid = int(b["id"])
        out.scans[bid] = ("error", {}) if bid in errors else ("ok", bare[bid])


def _dino(dino: Any, crops: list[np.ndarray]) -> list[bytes]:
    """float32[768] per crop: the CLS of the crop at 224 px and of its mirror."""
    if not crops:
        return []
    batch: list[np.ndarray] = []
    for c in crops:
        small = cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)
        batch += [small, np.ascontiguousarray(small[:, ::-1])]
    v = np.asarray(_call("dino", dino.embed, batch), dtype=np.float32)
    if v.shape != (len(batch), 384):
        raise EngineError(f"dino: vectors of shape {v.shape} for {len(batch)} crops")
    return [np.concatenate([v[2 * i], v[2 * i + 1]]).astype(np.float32).tobytes() for i in range(len(crops))]


def _clip_bound(bound: tuple[float, float, float, float], W: int, H: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bound
    x0, x1 = min(max(x0 / W, 0.0), 1.0), min(max(x1 / W, 0.0), 1.0)
    y0, y1 = min(max(y0 / H, 0.0), 1.0), min(max(y1 / H, 0.0), 1.0)
    return round(x0, 4), round(y0, 4), round(max(x1, x0), 4), round(max(y1, y0), 4)


def _ink_row(lk: Look, nat: np.ndarray, detail: bytes, s: float, where: str, vec: bytes, W: int, H: int) -> TraitRow:
    from app.analysis.traits import share_for

    r = lk.region
    share = share_for(r.part)
    kp = int(len(trait_sift.describe(detail, share).plain[0]))
    side = int(max(nat.shape[:2]))
    short = r.length * s if r.is_hand else float(share) * r.crop_len * s
    q = min(1.0, short / QUALITY_SIDE) * r.score * min(1.0, lk.sharp / QUALITY_SHARP)
    attrs = {"p": {k: round(v, 3) for k, v in lk.p.items()}, "kp": kp, "small": kp < SMALL_KP, "share": share,
             "out": side, "src": where, "skin": round(lk.skin, 3), "ink": round(lk.ink, 3), "sharp": round(lk.sharp, 1),
             "len": round(r.length, 1)}
    x0, y0, x1, y1 = _clip_bound(r.bound, W, H)
    return TraitRow(r.body_id, str(lk.verdict), r.part, x0, y0, x1 - x0, y1 - y0, q, vec, detail, attrs,
                    _webp(trait_sift.decode_detail(detail)))
