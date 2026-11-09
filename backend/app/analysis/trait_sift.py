"""Geometric verification of tattoo and mehndi crops (section 34): SIFT with partial-affine RANSAC.

CONTRACT.
- A row's `detail` BLOB is its crop at native resolution (encode_detail: RGB, WebP quality DETAIL_QUALITY, at most
  DETAIL_MAX px square). Limb crops are square and upright with the limb along the vertical centre line.
- describe(detail_webp, share) -> Features: SIFT (SIFT_FEATURES features, contrast threshold CONTRAST) on the CLAHE
  (2.0, 8x8) grey of the decoded crop, and again on its mirror. When `share` is set (limb crops: 0.6 forearm,
  0.7 upper arm, 0.55 calf) only the central vertical strip of width side * share / 1.15 is searched, the mask
  the probe used; hands pass None (no mask). Keypoints are float32[n, 2] (x, y in crop pixels), descriptors
  uint8[n, 128] (SIFT values are 0..255; halves the memory).
- features(key, loader) -> Features: describe through an LRU of FEATURE_CACHE entries. `key` names the table and
  the id: ("t", body_traits.id) or ("h", trait_hides.id); loader() -> (detail_webp, share) runs only on a miss.
- verify(a, b, *, key_a=None, key_b=None) -> Verification | None: the probe's verifier, symmetric. When both keys
  are given the pair is put in sorted key order first, so verify(a, b) and verify(b, a) run the same computation
  and share one entry of a RESULT_CACHE-entry LRU. Orientations plain/plain, mirror(a)/plain(b),
  plain(a)/mirror(b) (ties: the first); BFMatcher L2, ratio test RATIO_TEST, at least MIN_GOOD good matches;
  estimateAffinePartial2D(RANSAC, 6.0 px at 224 scaled by the mean side, maxIters RANSAC_ITERS); a fit whose scale
  is outside [SCALE_MIN, SCALE_MAX] is rejected. Returns the best orientation, or None when none fits.
  Verification.ratio = inliers / good matches of that orientation. No threshold here: callers use passes().
- passes(v, min_inliers) -> bool: v is not None, v.inliers >= min_inliers and v.ratio >= MIN_RATIO.
- reset(): empties both caches (ids restart in every test database). stats() -> counts for tests.
Rows are never updated in place (a rescan writes new ids, AUTOINCREMENT), so cached entries never go stale.
Thread-safe: the caches are behind one lock, and every thread has its own SIFT, CLAHE and matcher objects.
Deterministic: partial-affine RANSAC gave the same inliers over 20 runs on OpenCV 5.0.
"""
from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from typing import Callable, NamedTuple

import cv2
import numpy as np
from PIL import Image

SIFT_FEATURES = 1500
CONTRAST = 0.02
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
RATIO_TEST = 0.8
MIN_GOOD = 6
MIN_KP = 8                 # fewer keypoints on either side: no match is tried
RANSAC_PX_224 = 6.0        # reprojection threshold at a 224 px crop, scaled by the pair's mean side
RANSAC_ITERS = 3000
SCALE_MIN = 0.5
SCALE_MAX = 2.0
MIN_RATIO = 0.25
LIMB_STRIP = 1.15          # crop side = 1.15 x limb length, so the strip is side * share / 1.15 wide
DETAIL_MAX = 512
DETAIL_QUALITY = 90
FEATURE_CACHE = 256        # about 100 MB at 1,500 keypoints with uint8 descriptors
RESULT_CACHE = 8192

Key = tuple[str, int]
FeatureSet = tuple[np.ndarray, np.ndarray]  # keypoints xy float32[n, 2], descriptors uint8[n, 128]


class Features(NamedTuple):
    plain: FeatureSet
    mirror: FeatureSet
    side: int  # crop side in px (the larger of width and height)


class Verification(NamedTuple):
    inliers: int
    ratio: float      # inliers / good matches of the chosen orientation
    mirrored: bool    # one side was mirrored (a mirror photo, or the other arm)


_lock = threading.Lock()
_features: OrderedDict[Key, Features] = OrderedDict()
_results: OrderedDict[tuple[Key, Key], Verification | None] = OrderedDict()
_stats = {"described": 0, "verified": 0}
_local = threading.local()


def _tools() -> tuple[cv2.SIFT, cv2.CLAHE, cv2.BFMatcher]:
    t = getattr(_local, "tools", None)
    if t is None:
        t = (cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=CONTRAST),
             cv2.createCLAHE(CLAHE_CLIP, CLAHE_TILE), cv2.BFMatcher(cv2.NORM_L2))
        _local.tools = t
    return t


def _empty() -> FeatureSet:
    return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)


# ------------------------------------------------------------------ detail crops
def encode_detail(rgb: np.ndarray) -> bytes:
    """The stored detail crop: RGB uint8, shrunk (INTER_AREA) to at most DETAIL_MAX px, WebP quality DETAIL_QUALITY."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = a.shape[:2]
    if max(h, w) > DETAIL_MAX:
        f = DETAIL_MAX / max(h, w)
        a = cv2.resize(a, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(a, "RGB").save(buf, "WEBP", quality=DETAIL_QUALITY)
    return buf.getvalue()


def decode_detail(blob: bytes) -> np.ndarray:
    """The detail crop as an RGB uint8 array."""
    return np.asarray(Image.open(io.BytesIO(bytes(blob))).convert("RGB"))


def strip_mask(side_w: int, side_h: int, share: float | None) -> np.ndarray | None:
    """The central vertical strip a limb crop is searched in (the probe's mask); None for hands."""
    if share is None:
        return None
    half = int(side_w * float(share) / LIMB_STRIP / 2)
    mask = np.zeros((side_h, side_w), np.uint8)
    mask[:, max(0, side_w // 2 - half): side_w // 2 + half] = 255
    return mask


def _detect(gray: np.ndarray, mask: np.ndarray | None) -> FeatureSet:
    sift, _, _ = _tools()
    kps, desc = sift.detectAndCompute(gray, mask)
    if desc is None or not kps:
        return _empty()
    xy = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
    return xy, np.clip(np.rint(desc), 0, 255).astype(np.uint8)


def describe_rgb(rgb: np.ndarray, share: float | None) -> Features:
    """describe() on an RGB array that is already decoded (callers should pass the decoded stored crop, so the
    features match what describe(detail) gives later)."""
    _, clahe, _ = _tools()
    gray = clahe.apply(cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY))
    h, w = gray.shape[:2]
    mask = strip_mask(w, h, share)
    plain = _detect(gray, mask)
    mirror = _detect(cv2.flip(gray, 1), None if mask is None else cv2.flip(mask, 1))
    with _lock:
        _stats["described"] += 1
    return Features(plain, mirror, int(max(h, w)))


def describe(detail_webp: bytes, share: float | None) -> Features:
    """SIFT of a stored detail crop and of its mirror, strip-masked when `share` is set."""
    return describe_rgb(decode_detail(detail_webp), share)


def features(key: Key, loader: Callable[[], tuple[bytes, float | None]]) -> Features:
    """describe() through the feature LRU; loader() -> (detail_webp, share) is called only on a miss."""
    with _lock:
        f = _features.get(key)
        if f is not None:
            _features.move_to_end(key)
            return f
    blob, share = loader()
    f = describe(blob, share)
    with _lock:
        _features[key] = f
        _features.move_to_end(key)
        while len(_features) > FEATURE_CACHE:
            _features.popitem(last=False)
    return f


# ------------------------------------------------------------------ verification
def _match(fa: FeatureSet, fb: FeatureSet, thresh: float) -> tuple[int, int]:
    """(inliers, good matches) of one orientation; (0, good) when no acceptable fit."""
    (ka, da), (kb, db) = fa, fb
    if len(ka) < MIN_KP or len(kb) < MIN_KP:
        return 0, 0
    _, _, bf = _tools()
    pairs = bf.knnMatch(da.astype(np.float32), db.astype(np.float32), k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
    if len(good) < MIN_GOOD:
        return 0, len(good)
    A = np.float32([ka[g.queryIdx] for g in good])
    B = np.float32([kb[g.trainIdx] for g in good])
    M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=thresh,
                                         maxIters=RANSAC_ITERS)
    if M is None or inl is None:
        return 0, len(good)
    scale = math.hypot(float(M[0, 0]), float(M[1, 0]))
    if not SCALE_MIN <= scale <= SCALE_MAX:
        return 0, len(good)
    return int(inl.sum()), len(good)


def _verify(a: Features, b: Features) -> Verification | None:
    thresh = RANSAC_PX_224 * (a.side + b.side) / 2.0 / 224.0
    best: Verification | None = None
    for fa, fb, mirrored in ((a.plain, b.plain, False), (a.mirror, b.plain, True), (a.plain, b.mirror, True)):
        inliers, good = _match(fa, fb, thresh)
        if inliers > 0 and (best is None or inliers > best.inliers):
            best = Verification(inliers, round(inliers / good, 4), mirrored)
    return best


def verify(a: Features, b: Features, *, key_a: Key | None = None, key_b: Key | None = None) -> Verification | None:
    """The best orientation of a pair (see the module contract), cached by the sorted key pair when both keys
    are given. Symmetric: with keys, verify(a, b) and verify(b, a) are one computation and one cache entry."""
    if key_a is None or key_b is None:
        with _lock:
            _stats["verified"] += 1
        return _verify(a, b)
    if key_b < key_a:
        a, b, key_a, key_b = b, a, key_b, key_a
    ck = (key_a, key_b)
    with _lock:
        if ck in _results:
            _results.move_to_end(ck)
            return _results[ck]
    v = _verify(a, b)
    with _lock:
        _stats["verified"] += 1
        _results[ck] = v
        _results.move_to_end(ck)
        while len(_results) > RESULT_CACHE:
            _results.popitem(last=False)
    return v


def cached(key_a: Key, key_b: Key) -> bool:
    """True when the pair's result is in the cache (budgets count only pairs that need a computation)."""
    ck = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
    with _lock:
        return ck in _results


def cached_result(key_a: Key, key_b: Key) -> tuple[bool, Verification | None]:
    """(hit, result) for a pair from the result cache, without its Features (the feature cache is far smaller,
    so a cached pair would otherwise often be described again only to read its cached result)."""
    ck = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
    with _lock:
        if ck in _results:
            _results.move_to_end(ck)
            return True, _results[ck]
    return False, None


def passes(v: Verification | None, min_inliers: int) -> bool:
    return v is not None and v.inliers >= int(min_inliers) and v.ratio >= MIN_RATIO


def reset() -> None:
    """Empty both caches and the counters (tests: ids restart at 1 in every database)."""
    with _lock:
        _features.clear()
        _results.clear()
        _stats["described"] = 0
        _stats["verified"] = 0


def stats() -> dict[str, int]:
    with _lock:
        return {"features": len(_features), "results": len(_results), **_stats}
