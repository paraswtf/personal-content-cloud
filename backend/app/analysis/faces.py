"""Turn a photo or video into face records ready to store (adapted from the hub's faces/pipeline.py).

CONTRACT.
- FaceRecord: normalised box, det_score, size_px (source pixels), quality, frame_time_s, embedding
  (512, L2-normalised), crop (192 px RGB), spans (videos), torso (BGR crop of head and torso for
  SigLIP clothes matching, or None), clarity (face-model feature norm) and sharpness.
- face_weight(det_score, size_px, clarity, sharpness) -> 0..1 / face_weights(arrays): how much a face
  tells about who the person is; it weighs the face as a matching example, in centroids, in the
  learned thresholds and in a video track's identity. is_clear_face: good enough to start a person.
- crop_face(img, bbox) -> 192 px square crop. person_box(bbox, W, H) -> torso box in pixels.
- torso_crop(img_bgr, bbox) -> BGR crop of the person_box (None when degenerate).
- analyze_image(img, source_scale=1.0) -> list[FaceRecord] on app.ml.faces.
- analyze_video(path, info, on_frame=None) -> list[FaceRecord]: one record per person followed through the
  clip (FaceTracker: motion and look, tracklets joined per person, section 10); on_frame(t, bgr) sees the
  frames at the plain cadence of video_plan (the pipeline keeps small copies for SigLIP, NSFW and pets).
- FaceTracker(threshold, lost_s).add_frame(t, dets) / .finish(good_min_size, good_det_score, min_count),
  identity(dets), visible_spans(times, samples, end, bridge_s), video_plan(duration_s), TRACKER_VERSION.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.config import settings

log = logging.getLogger(__name__)

CROP_SIZE = 192
CROP_QUALITY = 85
VIDEO_MAX_SIDE = 1280
TRACK_EMBED_TOPN = 8  # best views (spread in time) averaged into a video track's identity

# Face weight (section 10): how much a face tells about who the person is, from 0 (nothing: turned
# away, behind a helmet or mask, blurred, tiny, or not a face at all) to 1 (clear and fully visible).
# It is the product of linear ramps, each 0 at its first value and 1 at its second: the face-model
# feature norm, the sharpness of the aligned face, the detector score, and the size (from
# settings.face_min_size_px to face_good_min_size_px source pixels). Measured on 617 faces from the
# owner's photos: below a norm of about 14 they are turned away, masked or not faces (textures, signs,
# the back of a head), from about 22 they are clear; a sharpness below about 8 is a blurred, dark or
# tiny face.
WEIGHT_NORM = (13.0, 22.0)
WEIGHT_SHARPNESS = (5.0, 30.0)
WEIGHT_DET = (0.5, 0.8)
CLEAR_WEIGHT = 0.5  # a face at least this clear (and good) may start a new person

# Torso crop geometry around a face box (x, y = top-left, w, h = face size), in face units.
TORSO_WIDTH = 3.2
TORSO_TOP = -0.25
TORSO_BOTTOM = 4.0
TORSO_MIN_PX = 16


@dataclass
class FaceRecord:
    x: float  # normalised box in [0,1] relative to the display-oriented image/frame
    y: float
    w: float
    h: float
    det_score: float
    size_px: int  # shorter box side in source pixels (full-resolution image / frame)
    quality: float  # face_quality(); higher is a better cover/thumbnail
    frame_time_s: float | None  # video only: exact time of the frame the box belongs to
    embedding: np.ndarray  # (512,) float32, L2-normalised
    crop: Image.Image  # CROP_SIZE x CROP_SIZE RGB thumbnail
    spans: list[tuple[float, float]] | None = None  # video only: when the face is on screen
    torso: np.ndarray | None = None  # BGR crop of head and torso (person_box), for SigLIP
    bbox_px: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 in analysed-image pixels
    clarity: float | None = None  # face-model feature norm (app.ml.faces DetectedFace.norm)
    sharpness: float | None = None  # app.ml.faces.sharpness of the aligned face


def face_quality(det_score: float, size_px: int) -> float:
    """det_score * min(size_px, 160) / 160."""
    return float(det_score) * min(max(int(size_px), 0), 160) / 160.0


def face_weights(det_score: Any, size_px: Any, clarity: Any, sharpness: Any) -> np.ndarray:
    """The face weight of each face (arrays, or scalars); NaN in clarity or sharpness means not
    measured, and that ramp then counts as 1 (faces analysed before the measures existed)."""
    def ramp(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    det = np.asarray(det_score, dtype=np.float64)
    size = np.asarray(size_px, dtype=np.float64)
    norm = np.asarray(clarity, dtype=np.float64)
    sharp = np.asarray(sharpness, dtype=np.float64)
    w = ramp(det, *WEIGHT_DET)
    w = w * ramp(size, float(settings.face_min_size_px), float(max(settings.face_good_min_size_px,
                                                                    settings.face_min_size_px + 1)))
    w = w * np.where(np.isnan(norm), 1.0, ramp(np.nan_to_num(norm), *WEIGHT_NORM))
    w = w * np.where(np.isnan(sharp), 1.0, ramp(np.nan_to_num(sharp), *WEIGHT_SHARPNESS))
    return w


def face_weight(det_score: float, size_px: int, clarity: float | None = None, sharpness: float | None = None) -> float:
    """How much one face tells about who the person is, 0 to 1 (see WEIGHT_NORM)."""
    nan = float("nan")
    return float(face_weights(det_score, size_px, nan if clarity is None else clarity,
                              nan if sharpness is None else sharpness))


def is_clear_face(det_score: float, size_px: int, clarity: float | None = None,
                  sharpness: float | None = None) -> bool:
    """Good enough to start a new person: the good-face rule (size and detector score) and a face
    weight of at least CLEAR_WEIGHT."""
    if int(size_px) < settings.face_good_min_size_px or float(det_score) < settings.face_good_det_score:
        return False
    return face_weight(det_score, size_px, clarity, sharpness) >= CLEAR_WEIGHT


def crop_face(img: Image.Image, bbox: tuple[float, float, float, float], size: int = CROP_SIZE) -> Image.Image:
    """Square crop centred on the box (side = 1.6 x the longer box side, shifted to stay inside
    the image where possible, padded with black otherwise), resized to size x size.
    bbox is (x1, y1, x2, y2) in pixels of img."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    side = max(1.6 * max(x2 - x1, y2 - y1), 2.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def place(center: float, extent: int) -> float:
        lo, hi = min(0.0, extent - side), max(0.0, extent - side)
        return min(max(center - side / 2.0, lo), hi)

    left, top = place(cx, W), place(cy, H)
    box = (int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side)))
    return img.crop(box).resize((size, size), Image.Resampling.LANCZOS)


def person_box(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Head-and-torso box for a face box (x1, y1, x2, y2 pixels): TORSO_WIDTH face widths wide
    around the face centre, from y - 0.25 h to y + 4 h, clipped to the image. None when too small."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    fw, fh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    left = max(0, int(round(cx - TORSO_WIDTH * fw / 2.0)))
    right = min(int(width), int(round(cx + TORSO_WIDTH * fw / 2.0)))
    top = max(0, int(round(y1 + TORSO_TOP * fh)))
    bottom = min(int(height), int(round(y1 + TORSO_BOTTOM * fh)))
    if right - left < TORSO_MIN_PX or bottom - top < TORSO_MIN_PX:
        return None
    return left, top, right, bottom


def torso_crop(img_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """BGR crop of person_box(bbox) (a copy), or None."""
    h, w = img_bgr.shape[:2]
    box = person_box(bbox, w, h)
    if box is None:
        return None
    l, t, r, b = box
    return np.ascontiguousarray(img_bgr[t:b, l:r]).copy()


def _norm_box(bbox: Any, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = min(max(float(bbox[0]) / width, 0.0), 1.0)
    y1 = min(max(float(bbox[1]) / height, 0.0), 1.0)
    x2 = min(max(float(bbox[2]) / width, 0.0), 1.0)
    y2 = min(max(float(bbox[3]) / height, 0.0), 1.0)
    return x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)


def _to_bgr(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def analyze_image(img: Image.Image, source_scale: float = 1.0, img_bgr: np.ndarray | None = None) -> list[FaceRecord]:
    """Faces in an oriented RGB photo. source_scale = original long side / analysed long side (>= 1),
    so size_px (and the good-face rules) are in original pixels. Detector sizes
    settings.face_det_sizes, threshold settings.face_det_threshold; faces smaller than
    settings.face_min_size_px (original pixels) are dropped."""
    from app.ml.faces import get_engine

    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
    scale = max(float(source_scale or 1.0), 1.0)
    bgr = img_bgr if img_bgr is not None else _to_bgr(img)
    faces = get_engine().analyze(
        bgr,
        sizes=tuple(settings.face_det_sizes),
        threshold=settings.face_det_threshold,
        min_size=0,
    )
    out: list[FaceRecord] = []
    for f in faces:
        if f.embedding is None:
            continue
        size_px = int(round(f.size_px * scale))
        if size_px < settings.face_min_size_px:
            continue
        bbox = tuple(float(v) for v in f.bbox)
        x, y, w, h = _norm_box(f.bbox, W, H)
        out.append(FaceRecord(
            x=x, y=y, w=w, h=h,
            det_score=float(f.det_score),
            size_px=size_px,
            quality=face_quality(f.det_score, size_px),
            frame_time_s=None,
            embedding=np.asarray(f.embedding, dtype=np.float32),
            crop=crop_face(img, bbox),
            torso=torso_crop(bgr, bbox),
            bbox_px=bbox,  # type: ignore[arg-type]
            clarity=f.norm,
            sharpness=f.sharpness,
        ))
    return out


# ------------------------------------------------------------------ video tracking
# A face is followed from frame to frame by where it is (its last box moved on by its recent motion)
# and by how it looks, then the pieces of one person are joined by look (section 10). Measured on the
# owner's clips at 5 frames a second: the same face in consecutive frames looks alike at 0.43 or more
# (1st percentile), two people in one frame at 0.26 or less (99th percentile).
TRACKER_VERSION = 2       # kv video_tracker_version: videos analysed by an older tracker are analysed again
MOTION_MIN_IOU = 0.3      # a motion link: the predicted box and the detection overlap this much ...
MOTION_MIN_SIM = 0.2      # ... and the faces look at least this alike
MOTION_MAX_GAP_S = 1.0    # ... within this long of the last sighting (sparse sampling links by look only)
TRACK_LOST_S = 1.0        # a face missing this long (at least 2.5 sampling steps) ends its tracklet
TRACK_MIN_SEEN_S = 0.8    # a track is kept when seen in frames covering this long (at least 2), or once and good
TRACKLET_MIN_S = 0.3      # a tracklet in fewer frames than cover this long is a flicker (dense sampling)
IDENTITY_MIN_SIM = 0.3    # views this unlike the track as a whole stay out of its identity
IDENTITY_SPREAD_S = 0.4   # views chosen for the identity are at least this far apart in time
MAX_SPEED = 2.0           # frame widths (or heights) a second the predicted box may move
MERGE_WEIGHT_CAP = 25     # a tracklet's weight in a merged look counts at most this many views


@dataclass
class VideoDetection:
    """One face found in one sampled video frame."""

    time_s: float
    box: tuple[float, float, float, float]  # normalised x, y, w, h (frame)
    det_score: float
    size_px: int  # in source (full-resolution) pixels
    embedding: np.ndarray  # (512,) float32, L2-normalised
    clarity: float | None = None
    sharpness: float | None = None

    @property
    def quality(self) -> float:
        return face_quality(self.det_score, self.size_px)

    @property
    def weight(self) -> float:
        return face_weight(self.det_score, self.size_px, self.clarity, self.sharpness)

    def rank(self) -> tuple[float, float]:
        """Which view shows the face best: the higher face weight, then the higher quality."""
        return self.weight, self.quality


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else np.asarray(v, dtype=np.float32)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (x, y, w, h) boxes."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def identity(dets: list[VideoDetection], top_n: int = TRACK_EMBED_TOPN) -> np.ndarray:
    """The embedding a track is known by: up to top_n of its best views (VideoDetection.rank: face
    weight, then quality), at least IDENTITY_SPREAD_S apart, averaged by face weight, so a blurred,
    masked or turned-away frame counts little or nothing. Views unlike the track as a whole
    (similarity below IDENTITY_MIN_SIM to the weighted mean of all its views: a wrong link) are left
    out. When every view weighs 0 they count by quality instead; the best view alone when no view
    qualifies."""
    if not dets:
        raise ValueError("identity of a track without detections")
    E = np.stack([d.embedding for d in dets]).astype(np.float32)
    weights = np.array([d.weight for d in dets], dtype=np.float32)
    if not (weights > 0).any():
        weights = np.array([max(d.quality, 1e-3) for d in dets], dtype=np.float32)
    sims = E @ _unit((E * weights[:, None]).sum(axis=0))
    order = sorted(range(len(dets)), key=lambda i: dets[i].rank(), reverse=True)
    chosen: list[int] = []
    for i in order:
        if sims[i] < IDENTITY_MIN_SIM or weights[i] <= 0:
            continue
        if any(abs(dets[i].time_s - dets[j].time_s) < IDENTITY_SPREAD_S for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= top_n:
            break
    if not chosen:
        return _unit(E[order[0]])
    return _unit((E[chosen] * weights[chosen][:, None]).sum(axis=0))


@dataclass
class FaceTrack:
    """One face followed through consecutive sampled frames (a tracklet). FaceTracker.finish joins the
    tracklets of one person into one track."""

    dets: list[VideoDetection] = field(default_factory=list)  # in time order
    best: VideoDetection | None = None
    best_crop: Image.Image | None = None
    best_torso: np.ndarray | None = None
    recent: np.ndarray | None = None  # running look of the latest views (motion links)
    total: np.ndarray | None = None  # sum of every view (appearance links)
    velocity: tuple[float, float] = (0.0, 0.0)  # box centre, frame widths and heights a second

    @property
    def count(self) -> int:
        return len(self.dets)

    @property
    def times(self) -> list[float]:
        return [d.time_s for d in self.dets]

    def predict(self, t: float) -> tuple[float, float, float, float]:
        """Where the face should be at time t: the last box moved on by the velocity."""
        last = self.dets[-1]
        x, y, w, h = last.box
        dt = max(0.0, t - last.time_s)
        return x + self.velocity[0] * dt, y + self.velocity[1] * dt, w, h

    def look(self) -> np.ndarray:
        assert self.total is not None
        return _unit(self.total)

    def add(self, det: VideoDetection, jumped: bool = False) -> bool:
        """Add the next detection (jumped: linked by look alone, so the motion so far says nothing).
        True if it became the track's best view (VideoDetection.rank: face weight, then quality)."""
        if self.dets and not jumped:
            prev = self.dets[-1]
            dt = det.time_s - prev.time_s
            if dt > 0:
                vx = ((det.box[0] + det.box[2] / 2) - (prev.box[0] + prev.box[2] / 2)) / dt
                vy = ((det.box[1] + det.box[3] / 2) - (prev.box[1] + prev.box[3] / 2)) / dt
                vx = min(max(vx, -MAX_SPEED), MAX_SPEED)
                vy = min(max(vy, -MAX_SPEED), MAX_SPEED)
                if len(self.dets) > 1:
                    vx, vy = 0.5 * (self.velocity[0] + vx), 0.5 * (self.velocity[1] + vy)
                self.velocity = (vx, vy)
        elif jumped:
            self.velocity = (0.0, 0.0)
        self.dets.append(det)
        e = np.asarray(det.embedding, dtype=np.float32)
        self.total = e.copy() if self.total is None else self.total + e
        self.recent = e.copy() if self.recent is None else _unit(0.5 * self.recent + 0.5 * e)
        if self.best is None or det.rank() > self.best.rank():
            self.best = det
            return True
        return False

    def embedding(self) -> np.ndarray:
        return identity(self.dets)


class FaceTracker:
    """Follows faces through a video's sampled frames (pure, no I/O): add_frame for each frame in time
    order, then finish once.

    add_frame: every open tracklet predicts where its face is now. A detection continues a tracklet
    by motion (seen at most MOTION_MAX_GAP_S ago, prediction and box overlap at IoU >= MOTION_MIN_IOU
    and the faces look alike at >= MOTION_MIN_SIM, against the tracklet's latest views or all of
    them) or by look alone
    (>= threshold against all its views). Pairs are taken best first, a motion link scoring its
    similarity plus its IoU, a look-only link its similarity; a tracklet takes one detection per
    frame and a detection left over starts a tracklet. A tracklet not seen for lost_s ends.

    finish joins the tracklets of one person: the closest pair of groups by look (weighted mean of
    their identities) while it is >= threshold, never two groups seen in the same frame (two faces on
    screen at once are two people)."""

    def __init__(self, threshold: float, lost_s: float = TRACK_LOST_S) -> None:
        self.threshold = float(threshold)
        self.lost_s = float(lost_s)
        self.tracks: list[FaceTrack] = []  # every tracklet, in order of creation
        self._open: list[FaceTrack] = []

    def add_frame(self, t: float, dets: list[VideoDetection]) -> list[tuple[FaceTrack, bool]]:
        """Link one frame's detections (all at time t). Returns (tracklet, became_best) per detection,
        in order."""
        self._open = [tr for tr in self._open if t - tr.dets[-1].time_s <= self.lost_s + 1e-9]
        assigned: list[FaceTrack | None] = [None] * len(dets)
        jumped = [False] * len(dets)
        if dets and self._open:
            E = np.stack([d.embedding for d in dets]).astype(np.float32)
            s_recent = E @ np.stack([tr.recent for tr in self._open]).T
            s_look = E @ np.stack([tr.look() for tr in self._open]).T
            cands: list[tuple[float, int, int, bool]] = []
            for ti, tr in enumerate(self._open):
                pred = tr.predict(t)
                near = t - tr.dets[-1].time_s <= MOTION_MAX_GAP_S + 1e-9
                for di, det in enumerate(dets):
                    sim = max(float(s_recent[di, ti]), float(s_look[di, ti]))
                    iou = box_iou(pred, det.box) if near else 0.0
                    if iou >= MOTION_MIN_IOU and sim >= MOTION_MIN_SIM:
                        cands.append((sim + iou, di, ti, False))
                    elif float(s_look[di, ti]) >= self.threshold:
                        cands.append((float(s_look[di, ti]), di, ti, True))
            cands.sort(key=lambda c: -c[0])
            used: set[int] = set()
            for _score, di, ti, jump in cands:
                if assigned[di] is not None or ti in used:
                    continue
                assigned[di] = self._open[ti]
                jumped[di] = jump
                used.add(ti)
        result: list[tuple[FaceTrack, bool]] = []
        for di, det in enumerate(dets):
            track = assigned[di]
            if track is None:
                track = FaceTrack()
                self.tracks.append(track)
                self._open.append(track)
            result.append((track, track.add(det, jumped=jumped[di])))
        return result

    def finish(self, good_min_size: int, good_det_score: float, min_count: int = 2,
               min_tracklet: int = 1) -> list[FaceTrack]:
        """The people of the video: tracklets seen in fewer than min_tracklet frames are dropped (on
        dense sampling a face in one frame only is a detector flicker or a blurred pan), the rest are
        joined per person and kept when seen in at least min_count sampled frames or with a good face
        (source size and detector score)."""
        keep = []
        for t in self._merged(min_tracklet):
            if t.best is None:
                continue
            if t.count >= min_count or (t.best.size_px >= good_min_size and t.best.det_score >= good_det_score):
                keep.append(t)
        return keep

    def _merged(self, min_tracklet: int = 1) -> list[FaceTrack]:
        parts = [t for t in self.tracks if t.dets and t.count >= min_tracklet]
        n = len(parts)
        if n <= 1:
            return parts
        times = sorted({d.time_s for t in parts for d in t.dets})
        col = {tt: i for i, tt in enumerate(times)}
        seen = np.zeros((n, len(times)), dtype=np.float32)
        for i, t in enumerate(parts):
            seen[i, [col[d.time_s] for d in t.dets]] = 1.0
        together = (seen @ seen.T) > 0  # seen in a common frame (and every part with itself)
        weight = np.array([min(t.count, MERGE_WEIGHT_CAP) for t in parts], dtype=np.float32)
        sums = np.stack([t.embedding() for t in parts]) * weight[:, None]
        unit = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
        S = unit @ unit.T
        S[together] = -np.inf
        members = [[i] for i in range(n)]
        alive = np.ones(n, dtype=bool)
        while True:
            k = int(np.argmax(S))
            i, j = divmod(k, n)
            if not np.isfinite(S[i, j]) or S[i, j] < self.threshold:
                break
            sums[i] += sums[j]
            members[i] += members[j]
            members[j] = []
            alive[j] = False
            together[i] |= together[j]
            together[:, i] = together[i]
            unit[i] = sums[i] / max(float(np.linalg.norm(sums[i])), 1e-12)
            row = unit @ unit[i]
            row[~alive | together[i]] = -np.inf
            S[j, :] = -np.inf
            S[:, j] = -np.inf
            S[i, :] = row
            S[:, i] = row
        out: list[FaceTrack] = []
        for i in np.flatnonzero(alive):
            group = [parts[m] for m in members[i]]
            if len(group) == 1:
                out.append(group[0])
                continue
            top = max(group, key=lambda t: t.best.rank() if t.best is not None else (-1.0, -1.0))
            out.append(FaceTrack(
                dets=sorted((d for t in group for d in t.dets), key=lambda d: d.time_s),
                best=top.best, best_crop=top.best_crop, best_torso=top.best_torso,
                total=np.sum([t.total for t in group if t.total is not None], axis=0),
            ))
        return out


def visible_spans(times: list[float], samples: list[float], end: float | None,
                  bridge_s: float = 0.0) -> list[tuple[float, float]]:
    """When a face is on screen: [start, end] seconds, consecutive sampled frames merged (midpoint
    rule, never past `end`, the clip's length), then spans less than bridge_s apart joined (a face
    missed for a moment)."""
    seen = set(times)
    n = len(samples)
    spans: list[list[float]] = []
    prev = -2
    for k, s in enumerate(samples):
        if s not in seen:
            continue
        lo = 0.0 if k == 0 else (samples[k - 1] + s) / 2.0
        if k + 1 < n:
            hi = (s + samples[k + 1]) / 2.0
        elif end is not None and end > s:
            hi = end
        else:
            hi = s + (s - samples[k - 1] if k > 0 else settings.video_sample_interval_s) / 2.0
        if end is not None and end >= s:
            hi = min(hi, end)  # never past the end of the clip
        if k == prev + 1 or (spans and lo - spans[-1][1] <= bridge_s + 1e-9):
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])
        prev = k
    return [(round(a, 3), round(b, 3)) for a, b in spans]


def track_to_record(track: FaceTrack, samples: list[float] | None = None, end: float | None = None,
                    bridge_s: float = 0.0) -> FaceRecord:
    best = track.best
    assert best is not None
    x, y, w, h = best.box
    return FaceRecord(
        x=x, y=y, w=w, h=h,
        det_score=float(best.det_score),
        size_px=int(best.size_px),
        quality=best.quality,
        frame_time_s=float(best.time_s),
        embedding=track.embedding(),
        crop=track.best_crop if track.best_crop is not None else Image.new("RGB", (CROP_SIZE, CROP_SIZE)),
        spans=visible_spans(track.times, samples, end, bridge_s) if samples is not None else None,
        torso=track.best_torso,
        clarity=best.clarity,
        sharpness=best.sharpness,
    )


@dataclass
class VideoPlan:
    """How analyze_video reads a clip (see there)."""

    interval: float  # asked of iter_frames
    max_frames: int
    step: float  # the sampling step iter_frames ends up with (it grows so max_frames covers the clip)
    keep_step: float  # on_frame cadence
    keep_max: int
    lost_s: float
    min_count: int
    min_tracklet: int


def video_plan(duration_s: float | None) -> VideoPlan:
    """Sampling for a clip of this length: settings.video_face_interval_s, at most
    settings.video_face_max_frames frames; clips longer than app.media.video.SEEK_SAMPLING_MIN_S are
    read by seeking (one ffmpeg run per frame), so they keep the plain sampling
    (video_sample_interval_s, video_max_samples). on_frame keeps the plain cadence on every clip:
    max(video_sample_interval_s, duration / video_max_samples), at most video_max_samples frames. The
    lost time and the frames a tracklet and a kept track need scale with the step."""
    from app.media.video import SEEK_SAMPLING_MIN_S

    duration = float(duration_s or 0.0)
    plain_step, plain_max = float(settings.video_sample_interval_s), int(settings.video_max_samples)
    if duration > SEEK_SAMPLING_MIN_S:
        interval, max_frames = plain_step, plain_max
    else:
        interval, max_frames = float(settings.video_face_interval_s), int(settings.video_face_max_frames)
    interval = max(interval, 0.01)
    step = max(interval, duration / max_frames) if duration > 0 and max_frames > 0 else interval
    keep_step = max(plain_step, duration / plain_max) if duration > 0 and plain_max > 0 else plain_step
    return VideoPlan(
        interval=interval, max_frames=max_frames, step=step, keep_step=keep_step, keep_max=plain_max,
        lost_s=max(TRACK_LOST_S, 2.5 * step),
        min_count=max(2, math.ceil(TRACK_MIN_SEEN_S / step - 1e-9)),
        min_tracklet=max(1, math.ceil(TRACKLET_MIN_S / step - 1e-9)),
    )


def analyze_video(path: Path, info: Any, on_frame: Callable[[float, np.ndarray], None] | None = None) -> list[FaceRecord]:
    """One record per person in the video, followed through the whole clip (section 10).

    Frames come from app.media.video.iter_frames as video_plan says (5 a second by default, longer
    side <= 1280) and are detected at settings.face_video_det_size; every face of at least
    settings.face_min_size_px source pixels is embedded. FaceTracker links them by motion and look and
    joins each person's tracklets (dropping those in fewer than plan.min_tracklet frames); a track is
    kept when seen in plan.min_count frames or with a good face. Each kept track gives one record: the best view's box, crop, torso and frame time, the
    identity() embedding, and the visible spans (gaps up to plan.lost_s closed). on_frame(t, bgr) is
    called before detection for the frames at the plain cadence only (the first, then each one at
    least plan.keep_step after the last one passed, at most plan.keep_max): SigLIP, the NSFW
    classifier and the pet detector work from those."""
    from app.media.video import iter_frames
    from app.ml.faces import get_engine

    engine = get_engine()
    duration = float(getattr(info, "duration_s", 0.0) or 0.0)
    plan = video_plan(duration)
    tracker = FaceTracker(settings.video_track_threshold, plan.lost_s)
    samples: list[float] = []
    kept_at: float | None = None
    kept = 0
    frames = iter_frames(path, info, plan.interval, plan.max_frames, VIDEO_MAX_SIDE)
    width = int(getattr(info, "width", 0) or 0)
    height = int(getattr(info, "height", 0) or 0)
    try:
        for t, bgr in frames:
            t = float(t)
            samples.append(t)
            if on_frame is not None and kept < plan.keep_max and (kept_at is None or t - kept_at >= plan.keep_step * 0.999):
                on_frame(t, bgr)
                kept_at, kept = t, kept + 1
            fh, fw = bgr.shape[:2]
            src_scale = max(width, height) / max(fw, fh) if width and height else 1.0
            src_scale = max(src_scale, 1.0)
            min_frame_px = settings.face_min_size_px / src_scale
            faces = engine.analyze(
                bgr, sizes=(settings.face_video_det_size,), threshold=settings.face_det_threshold, min_size=min_frame_px
            )
            dets: list[VideoDetection] = []
            found = []
            for f in faces:
                if f.embedding is None or f.size_px < min_frame_px:
                    continue
                dets.append(VideoDetection(
                    time_s=t,
                    box=_norm_box(f.bbox, fw, fh),
                    det_score=float(f.det_score),
                    size_px=int(round(f.size_px * src_scale)),
                    embedding=np.asarray(f.embedding, dtype=np.float32),
                    clarity=f.norm,
                    sharpness=f.sharpness,
                ))
                found.append(f)
            if not dets:
                continue
            frame_img: Image.Image | None = None
            for (track, became_best), f in zip(tracker.add_frame(t, dets), found):
                if became_best:
                    if frame_img is None:
                        frame_img = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
                    bbox = tuple(float(v) for v in f.bbox)
                    track.best_crop = crop_face(frame_img, bbox)  # type: ignore[arg-type]
                    track.best_torso = torso_crop(bgr, bbox)  # type: ignore[arg-type]
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    tracks = tracker.finish(settings.face_good_min_size_px, settings.face_good_det_score, plan.min_count,
                            plan.min_tracklet)
    records = [track_to_record(tr, samples, duration or None, plan.lost_s) for tr in tracks]
    log.info("video %s: %d frames, %d tracklets, %d people kept", Path(path).name, len(samples),
             len(tracker.tracks), len(records))
    return records
