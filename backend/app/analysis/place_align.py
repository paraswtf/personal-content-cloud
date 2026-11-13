"""Lining two photos of one sequence up (docs/ARCHITECTURE.md section 35).

Two photos taken a few seconds apart at one place show the same background, moved by however the camera moved in
between. This module finds that move, so a box of the first photo can be put where it would be in the second.

CONTRACT.
- describe(gray, boxes) -> Features: SIFT (SIFT_FEATURES features, contrast CONTRAST) on the CLAHE (2.0, 8x8) grey
  picture, reduced beforehand to ALIGN_SIDE px on its long side by load_gray. Keypoints are float32[n, 2] in pixels
  of that reduced picture, descriptors uint8[n, 128], and `people[n]` says whether a keypoint lies inside one of the
  photo's face or person boxes (normalised x, y, w, h, each grown by BOX_GROW of its size).
- features(key, loader) -> Features | None through an LRU bounded in bytes (FEATURE_CACHE_BYTES); key is
  (item id, version), loader() -> (gray, boxes) | None runs only on a miss. None (no pixels) is cached too.
- align(fa, fb) -> Alignment | None: BFMatcher L2 with the ratio test (RATIO_TEST), then a similarity (rotation,
  one scale, a shift: cv2.estimateAffinePartial2D, USAC_MAGSAC when this OpenCV has it, else RANSAC, RANSAC_PX at
  ALIGN_SIDE) fitted on the background matches (neither end inside a people box). When that fit has fewer than
  MIN_INLIERS inliers, once more on every match: people who stood still are good anchors, people who moved are
  outliers. That second fit must not be one person's doing: it counts with at least MIN_BG_INLIERS inliers
  outside the people boxes, or with inliers on at least two people in both photos (MIN_OWNER_INLIERS each), or
  when each photo shows one person only. Otherwise, with a plain background, it would lock onto one person who
  walked and put everybody else in the wrong place. Features.owner is the people box each keypoint lies in (the
  largest, so a face counts with the person box around it; -1 for the background) and Features.groups the number
  of people in the picture (boxes that do not lie inside a larger one).
  A fit is kept with at least MIN_INLIERS inliers, a scale within [SCALE_MIN, SCALE_MAX] and inliers
  spanning at least MIN_SPREAD of the first picture (their convex hull). Alignment.m is the 2 x 3 matrix in
  NORMALISED picture coordinates of a -> b (x and y each 0..1 of their own picture, so two aspect ratios are
  handled); kind is 'still' (the camera hardly moved: shift under STILL_SHIFT of the picture, scale within
  STILL_SCALE of 1, roll under STILL_ROLL degrees) or 'moved'.
- burst(dt_s, cos) -> Alignment | None: the fallback without a fit. Two photos taken within BURST_S seconds whose
  whole-picture vectors are at least BURST_COS alike count as the identity with kind 'burst' (a plain wall or a
  night shot has no keypoints, but nobody moved).
- apply(m, x, y) -> (x, y): a point of a in b's normalised coordinates. scale_of(m, wa, ha, wb, hb): how much
  bigger things are in b, in pixels.
- load_gray(item) -> ndarray | None: the small preview when there is one (1280 px), else the 2048 px preview, else
  the photo as the analysis saw it; grey, long side ALIGN_SIDE. Geometry is the same for all three: boxes are
  normalised to the upright picture.
- reset(): empties the cache (tests). stats() -> counts: described, aligned (pairs matched), on_people (fits
  kept from the second pass), refused (second-pass fits that were one person's doing).
Thread-safe: the cache is behind one lock, and every thread has its own SIFT, CLAHE and matcher objects.
Measured on the owner's library (2026-11-13, section 35): see the docs for the rates by gap and the timings.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
from collections import OrderedDict
from typing import Callable, NamedTuple, Sequence

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ALIGN_SIDE = 1024          # long side the pictures are reduced to
SIFT_FEATURES = 3000
CONTRAST = 0.02
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
RATIO_TEST = 0.75
MIN_KP = 8                 # fewer keypoints on either side: nothing is tried
MIN_GOOD = 6
RANSAC_PX = 3.0            # reprojection threshold in pixels at ALIGN_SIDE
RANSAC_ITERS = 3000
MIN_INLIERS = 20           # measured: see section 35
MIN_BG_INLIERS = 6         # of a fit on every keypoint: inliers that lie outside the people boxes of both photos,
MIN_OWNER_INLIERS = 3      # or this many inliers on each of two people, in both photos
SCALE_MIN = 0.4
SCALE_MAX = 2.5
MIN_SPREAD = 0.02          # the inliers' convex hull, as a share of the first picture
BOX_GROW = 0.10
STILL_SHIFT = 0.02
STILL_SCALE = 0.02
STILL_ROLL = 1.0
BURST_S = 3.0
BURST_COS = 0.97
FEATURE_CACHE_BYTES = 96 * 1024 * 1024

Box = tuple[float, float, float, float]  # normalised x, y, w, h
Key = tuple[int, int]                    # (item id, version)


class Features(NamedTuple):
    xy: np.ndarray        # float32[n, 2], pixels of the reduced picture
    desc: np.ndarray      # uint8[n, 128]
    people: np.ndarray    # bool[n]: inside a face or person box
    w: int
    h: int
    owner: np.ndarray     # int16[n]: the largest people box the keypoint lies in, -1 for the background
    groups: int           # people in the picture: boxes that do not lie inside a larger one

    @property
    def nbytes(self) -> int:
        return int(self.xy.nbytes + self.desc.nbytes + self.people.nbytes + self.owner.nbytes) + 64


class Alignment(NamedTuple):
    m: tuple[float, float, float, float, float, float]  # a -> b, normalised coordinates, row-major 2 x 3
    inliers: int
    matches: int
    spread: float
    kind: str             # 'still' | 'moved' | 'burst'
    background: bool      # fitted on background keypoints only


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

_lock = threading.Lock()
_cache: OrderedDict[Key, Features | None] = OrderedDict()
_cache_bytes = 0
_stats = {"described": 0, "aligned": 0, "on_people": 0, "refused": 0}
_local = threading.local()


def _tools() -> tuple[cv2.SIFT, cv2.CLAHE, cv2.BFMatcher]:
    t = getattr(_local, "tools", None)
    if t is None:
        t = (cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=CONTRAST),
             cv2.createCLAHE(CLAHE_CLIP, CLAHE_TILE), cv2.BFMatcher(cv2.NORM_L2))
        _local.tools = t
    return t


# ------------------------------------------------------------------ pictures
def reduce_gray(img: Image.Image) -> np.ndarray:
    """A PIL picture as a grey uint8 array with its long side at most ALIGN_SIDE."""
    g = img.convert("L")
    if max(g.size) > ALIGN_SIDE:
        g.thumbnail((ALIGN_SIDE, ALIGN_SIDE), Image.BILINEAR)
    return np.asarray(g)


def load_gray(item: sqlite3.Row) -> np.ndarray | None:
    """The photo as a reduced grey picture: the small preview, else the large one, else the photo as the analysis
    saw it; None when nothing reads."""
    from app import storage

    for path in (storage.preview_small_path(item["uid"], int(item["version"])),
                 storage.preview_path(item["uid"], int(item["version"]))):
        try:
            if path.is_file():
                with Image.open(path) as im:
                    return reduce_gray(im)
        except Exception as e:  # noqa: BLE001 - a broken preview falls through to the next source
            log.warning("place: preview %s unreadable: %s", path.name, e)
    try:
        from app.analysis.animals import open_photo

        img, _bgr = open_photo(item)
        return reduce_gray(img)
    except Exception as e:  # noqa: BLE001 - no pixels: the photo's pairs are stored as not lined up
        log.info("place: no pixels for %s: %s", item["uid"], e)
        return None


# ------------------------------------------------------------------ features
def describe(gray: np.ndarray, boxes: Sequence[Box] = ()) -> Features:
    sift, clahe, _ = _tools()
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = g.shape[:2]
    kps, desc = sift.detectAndCompute(clahe.apply(g), None)
    with _lock:
        _stats["described"] += 1
    size = [(float(b[2]) * float(b[3]), k) for k, b in enumerate(boxes)]
    # a box whose centre lies inside a larger one is part of that person (a face in its person box)
    groups = sum(1 for k, (bx, by, bw, bh) in enumerate(boxes) if not any(
        size[j] > size[k] and ox <= bx + bw / 2 <= ox + ow and oy <= by + bh / 2 <= oy + oh
        for j, (ox, oy, ow, oh) in enumerate(boxes) if j != k))
    if desc is None or not kps:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), np.zeros(0, bool), w, h,
                        np.zeros(0, np.int16), groups)
    xy = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
    owner = np.full(len(xy), -1, dtype=np.int16)
    for _area, k in sorted(size):                                     # small boxes first: a larger one takes over
        bx, by, bw, bh = boxes[k]
        gx, gy = float(bw) * BOX_GROW, float(bh) * BOX_GROW
        owner[(xy[:, 0] >= (bx - gx) * w) & (xy[:, 0] <= (bx + bw + gx) * w)
              & (xy[:, 1] >= (by - gy) * h) & (xy[:, 1] <= (by + bh + gy) * h)] = k
    return Features(xy, np.clip(np.rint(desc), 0, 255).astype(np.uint8), owner >= 0, int(w), int(h), owner, groups)


def features(key: Key, loader: Callable[[], tuple[np.ndarray, Sequence[Box]] | None]) -> Features | None:
    """describe() through the cache. loader() gives (grey picture, the photo's boxes) or None (no pixels)."""
    global _cache_bytes
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    loaded = loader()
    f = describe(*loaded) if loaded is not None else None
    with _lock:
        if key not in _cache:
            _cache[key] = f
            _cache_bytes += f.nbytes if f is not None else 64
            while _cache_bytes > FEATURE_CACHE_BYTES and len(_cache) > 1:
                _k, old = _cache.popitem(last=False)
                _cache_bytes -= old.nbytes if old is not None else 64
    return f


# ------------------------------------------------------------------ the fit
def _fit(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(a) < 3:
        return None
    m = inl = None
    method = getattr(cv2, "USAC_MAGSAC", None)
    if method is not None:
        try:
            m, inl = cv2.estimateAffinePartial2D(a, b, method=method, ransacReprojThreshold=RANSAC_PX,
                                                 maxIters=RANSAC_ITERS, confidence=0.999)
        except cv2.error:
            m = inl = None
    if m is None:
        m, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
                                             maxIters=RANSAC_ITERS)
    if m is None or inl is None:
        return None
    return m, inl.ravel().astype(bool)


def _spread(points: np.ndarray, w: int, h: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull)) / float(max(1, w * h))


def normalise(m: np.ndarray, wa: int, ha: int, wb: int, hb: int) -> tuple[float, float, float, float, float, float]:
    """A 2 x 3 pixel matrix a -> b as the matrix between normalised coordinates."""
    sa = np.array([[wa, 0, 0], [0, ha, 0], [0, 0, 1.0]])
    sb = np.array([[1.0 / wb, 0, 0], [0, 1.0 / hb, 0], [0, 0, 1.0]])
    n = (sb @ np.vstack([m, [0, 0, 1.0]]) @ sa)[:2]
    return tuple(round(float(v), 6) for v in n.ravel())  # type: ignore[return-value]


def align(fa: Features | None, fb: Features | None) -> Alignment | None:
    """The similarity that puts the first photo's background on the second's, or None (see the module contract)."""
    if fa is None or fb is None or len(fa.xy) < MIN_KP or len(fb.xy) < MIN_KP:
        return None
    _, _, bf = _tools()
    pairs = bf.knnMatch(fa.desc.astype(np.float32), fb.desc.astype(np.float32), k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    with _lock:
        _stats["aligned"] += 1
    if len(good) < MIN_GOOD:
        return None
    qi = np.fromiter((g.queryIdx for g in good), dtype=np.int64, count=len(good))
    ti = np.fromiter((g.trainIdx for g in good), dtype=np.int64, count=len(good))
    a, b = fa.xy[qi], fb.xy[ti]
    background = ~fa.people[qi] & ~fb.people[ti]
    for mask, is_bg in ((background, True), (np.ones(len(good), dtype=bool), False)):
        if int(mask.sum()) < MIN_INLIERS:
            continue
        fitted = _fit(a[mask], b[mask])
        if fitted is None:
            continue
        m, inl = fitted
        n = int(inl.sum())
        if n < MIN_INLIERS:
            continue
        if not is_bg and int((background & inl).sum()) < MIN_BG_INLIERS and not _people_agree(fa, fb, qi[inl], ti[inl]):
            with _lock:
                _stats["refused"] += 1
            continue  # nothing but one person agrees: the fit may follow them, not the camera
        scale = math.hypot(float(m[0, 0]), float(m[1, 0]))
        if not SCALE_MIN <= scale <= SCALE_MAX:
            continue
        spread = _spread(a[mask][inl], fa.w, fa.h)
        if spread < MIN_SPREAD:
            continue
        roll = abs(math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0]))))
        still = (abs(float(m[0, 2])) / fb.w < STILL_SHIFT and abs(float(m[1, 2])) / fb.h < STILL_SHIFT
                 and abs(scale - 1.0) < STILL_SCALE and roll < STILL_ROLL)
        if not is_bg:
            with _lock:
                _stats["on_people"] += 1
        return Alignment(normalise(m, fa.w, fa.h, fb.w, fb.h), n, len(good), round(spread, 4),
                         "still" if still else "moved", is_bg)
    return None


def _people_agree(fa: Features, fb: Features, qa: np.ndarray, tb: np.ndarray) -> bool:
    """Whether a fit without background inliers stands (qa, tb: the keypoints of its inliers in a and in b): its
    inliers lie on at least two people in both photos, or each photo shows one person only, so nobody else can be
    put in the wrong place by it."""
    if fa.groups <= 1 and fb.groups <= 1:
        return True

    def several(owner: np.ndarray) -> bool:
        _ids, counts = np.unique(owner[owner >= 0], return_counts=True)
        return int((counts >= MIN_OWNER_INLIERS).sum()) >= 2

    return several(fa.owner[qa]) and several(fb.owner[tb])


def burst(dt_s: float | None, cos: float | None) -> Alignment | None:
    """Two photos taken within BURST_S seconds that look the same as a whole: the identity, kind 'burst'."""
    if dt_s is None or cos is None or abs(float(dt_s)) > BURST_S or float(cos) < BURST_COS:
        return None
    return Alignment(IDENTITY, 0, 0, 0.0, "burst", False)


def apply(m: Sequence[float], x: float, y: float) -> tuple[float, float]:
    return float(m[0]) * x + float(m[1]) * y + float(m[2]), float(m[3]) * x + float(m[4]) * y + float(m[5])


def invert(m: Sequence[float]) -> tuple[float, float, float, float, float, float] | None:
    """The matrix of b -> a; None for a degenerate one."""
    a, b, c, d, e, f = (float(v) for v in m)
    det = a * e - b * d
    if abs(det) < 1e-12:
        return None
    ia, ib, id_, ie = e / det, -b / det, -d / det, a / det
    return ia, ib, -(ia * c + ib * f), id_, ie, -(id_ * c + ie * f)


def scale_of(m: Sequence[float], wa: float, ha: float, wb: float, hb: float) -> float:
    """How much bigger things are in b than in a, in pixels (1.0 when the sizes are unknown or zero)."""
    if min(wa, ha, wb, hb) <= 0:
        return 1.0
    # one step along x in a (1 / wa of the picture) becomes this many pixels in b
    return math.hypot(float(m[0]) * wb, float(m[3]) * hb) / float(wa)


def reset() -> None:
    global _cache_bytes
    with _lock:
        _cache.clear()
        _cache_bytes = 0
        _stats.update(described=0, aligned=0, on_people=0, refused=0)


def stats() -> dict[str, int]:
    with _lock:
        return {"features": len(_cache), "bytes": int(_cache_bytes), **_stats}
