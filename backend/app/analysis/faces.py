"""Turn a photo or video into face records ready to store (adapted from the hub's faces/pipeline.py).

CONTRACT.
- FaceRecord: normalised box, det_score, size_px (source pixels), quality, frame_time_s, embedding
  (512, L2-normalised), crop (192 px RGB), spans (videos), torso (BGR crop of head and torso for
  SigLIP clothes matching, or None).
- crop_face(img, bbox) -> 192 px square crop. person_box(bbox, W, H) -> torso box in pixels.
- torso_crop(img_bgr, bbox) -> BGR crop of the person_box (None when degenerate).
- analyze_image(img, source_scale=1.0) -> list[FaceRecord] on app.ml.faces.
- analyze_video(path, info, on_frame=None) -> list[FaceRecord]: one record per FaceTracker track;
  on_frame(t, bgr) sees every sampled frame (the pipeline keeps small copies for SigLIP).
"""
from __future__ import annotations

import logging
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
TRACK_EMBED_TOPN = 5  # best-quality embeddings averaged into a video track's embedding

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


def face_quality(det_score: float, size_px: int) -> float:
    """det_score * min(size_px, 160) / 160."""
    return float(det_score) * min(max(int(size_px), 0), 160) / 160.0


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
        ))
    return out


# ------------------------------------------------------------------ video tracking
@dataclass
class VideoDetection:
    """One face found in one sampled video frame."""

    time_s: float
    box: tuple[float, float, float, float]  # normalised x, y, w, h (frame)
    det_score: float
    size_px: int  # in source (full-resolution) pixels
    embedding: np.ndarray  # (512,) float32, L2-normalised

    @property
    def quality(self) -> float:
        return face_quality(self.det_score, self.size_px)


@dataclass
class FaceTrack:
    """Detections of one person within a video."""

    sum_emb: np.ndarray
    count: int = 0
    best: VideoDetection | None = None
    best_crop: Image.Image | None = None
    best_torso: np.ndarray | None = None
    top: list[VideoDetection] = field(default_factory=list)  # up to TRACK_EMBED_TOPN best by quality
    times: list[float] = field(default_factory=list)  # every sampled frame time it was detected in

    def mean(self) -> np.ndarray:
        n = float(np.linalg.norm(self.sum_emb))
        return self.sum_emb / n if n > 0 else self.sum_emb

    def add(self, det: VideoDetection) -> bool:
        """Add a detection; True if it became the track's best-quality detection."""
        self.sum_emb = self.sum_emb + det.embedding
        self.count += 1
        self.times.append(det.time_s)
        self.top.append(det)
        self.top.sort(key=lambda d: d.quality, reverse=True)
        del self.top[TRACK_EMBED_TOPN:]
        if self.best is None or det.quality > self.best.quality:
            self.best = det
            return True
        return False

    def embedding(self) -> np.ndarray:
        v = np.sum([d.embedding for d in self.top], axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 0 else v


class FaceTracker:
    """Greedy online linking of per-frame detections into tracks (pure, no I/O)."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.tracks: list[FaceTrack] = []

    def add_frame(self, dets: list[VideoDetection]) -> list[tuple[FaceTrack, bool]]:
        """Link one frame's detections. Returns (track, became_best) per detection, in order.
        Pairs (detection, track) are taken best similarity first; each track takes at most one
        detection per frame; a detection without a track >= threshold starts a new track."""
        assigned: list[FaceTrack | None] = [None] * len(dets)
        if dets and self.tracks:
            E = np.stack([d.embedding for d in dets])
            M = np.stack([t.mean() for t in self.tracks])
            S = E @ M.T
            pairs = np.argwhere(S >= self.threshold)
            order = np.argsort(-S[pairs[:, 0], pairs[:, 1]], kind="stable") if len(pairs) else []
            used_tracks: set[int] = set()
            for k in order:
                di, ti = int(pairs[k][0]), int(pairs[k][1])
                if assigned[di] is not None or ti in used_tracks:
                    continue
                assigned[di] = self.tracks[ti]
                used_tracks.add(ti)
        result: list[tuple[FaceTrack, bool]] = []
        for det, track in zip(dets, assigned):
            if track is None:
                track = FaceTrack(sum_emb=np.zeros_like(det.embedding, dtype=np.float32))
                self.tracks.append(track)
            result.append((track, track.add(det)))
        return result

    def finished(self, good_min_size: int, good_det_score: float) -> list[FaceTrack]:
        """Tracks worth a record: at least two detections, or one good detection."""
        keep = []
        for t in self.tracks:
            if t.best is None:
                continue
            if t.count >= 2 or (t.best.size_px >= good_min_size and t.best.det_score >= good_det_score):
                keep.append(t)
        return keep


def visible_spans(times: list[float], samples: list[float], end: float | None) -> list[tuple[float, float]]:
    """When a face is on screen: [start, end] seconds, consecutive sampled frames merged (midpoint rule)."""
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
        if k == prev + 1:
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])
        prev = k
    return [(round(a, 3), round(b, 3)) for a, b in spans]


def track_to_record(track: FaceTrack, samples: list[float] | None = None, end: float | None = None) -> FaceRecord:
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
        spans=visible_spans(track.times, samples, end) if samples is not None else None,
        torso=track.best_torso,
    )


def analyze_video(path: Path, info: Any, on_frame: Callable[[float, np.ndarray], None] | None = None) -> list[FaceRecord]:
    """One record per distinct person in the video.

    Frames come from app.media.video.iter_frames (settings.video_sample_interval_s,
    settings.video_max_samples, longer side <= 1280) and are detected at
    settings.face_video_det_size. Detections are linked into tracks (FaceTracker,
    settings.video_track_threshold); each track gives one record with the best detection's box,
    crop, torso and frame time, the mean of its 5 best embeddings and its visible spans.
    Single-detection tracks are dropped unless the face is good. on_frame(t, bgr) is called for
    every sampled frame, before detection."""
    from app.media.video import iter_frames
    from app.ml.faces import get_engine

    engine = get_engine()
    tracker = FaceTracker(settings.video_track_threshold)
    samples: list[float] = []
    frames = iter_frames(path, info, settings.video_sample_interval_s, settings.video_max_samples, VIDEO_MAX_SIDE)
    width = int(getattr(info, "width", 0) or 0)
    height = int(getattr(info, "height", 0) or 0)
    try:
        for t, bgr in frames:
            samples.append(float(t))
            if on_frame is not None:
                on_frame(float(t), bgr)
            fh, fw = bgr.shape[:2]
            src_scale = max(width, height) / max(fw, fh) if width and height else 1.0
            src_scale = max(src_scale, 1.0)
            min_frame_px = settings.face_min_size_px / src_scale
            faces = engine.analyze(
                bgr, sizes=(settings.face_video_det_size,), threshold=settings.face_det_threshold, min_size=0
            )
            dets: list[VideoDetection] = []
            kept = []
            for f in faces:
                if f.embedding is None or f.size_px < min_frame_px:
                    continue
                dets.append(VideoDetection(
                    time_s=float(t),
                    box=_norm_box(f.bbox, fw, fh),
                    det_score=float(f.det_score),
                    size_px=int(round(f.size_px * src_scale)),
                    embedding=np.asarray(f.embedding, dtype=np.float32),
                ))
                kept.append(f)
            if not dets:
                continue
            linked = tracker.add_frame(dets)
            frame_img: Image.Image | None = None
            for (track, became_best), f in zip(linked, kept):
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
    tracks = tracker.finished(settings.face_good_min_size_px, settings.face_good_det_score)
    records = [track_to_record(tr, samples, getattr(info, "duration_s", None)) for tr in tracks]
    log.info("video %s: %d tracks, %d kept", Path(path).name, len(tracker.tracks), len(records))
    return records
