"""Turn a photo or video into face records ready to store.

CONTRACT (implemented by the faces agent). Uses app.faces.engine.get_engine() (already
implemented and validated: same person ~0.75-0.80 cosine, different people <= 0.12 on the
test fixtures).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.media.probe import MediaInfo

log = logging.getLogger(__name__)

CROP_SIZE = 192
VIDEO_MAX_SIDE = 1280
TRACK_EMBED_TOPN = 5  # best-quality embeddings averaged into a video track's embedding


@dataclass
class FaceRecord:
    x: float  # normalised box in [0,1] relative to the display-oriented image/frame
    y: float
    w: float
    h: float
    det_score: float
    size_px: int  # shorter box side in source pixels (full-resolution image / frame)
    quality: float  # face_quality(); higher is a better cover/thumbnail
    frame_time_s: float | None  # video only: exact time of the frame the box belongs to (0 = first frame)
    embedding: np.ndarray  # (512,) float32, L2-normalised
    crop: Image.Image  # CROP_SIZE x CROP_SIZE RGB thumbnail
    # video only: when the face is on screen, [start, end] seconds (visible_spans)
    spans: list[tuple[float, float]] | None = None


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
    # PIL fills the area outside the image with zeros (black).
    return img.crop(box).resize((size, size), Image.Resampling.LANCZOS)


def _norm_box(bbox: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = min(max(float(bbox[0]) / width, 0.0), 1.0)
    y1 = min(max(float(bbox[1]) / height, 0.0), 1.0)
    x2 = min(max(float(bbox[2]) / width, 0.0), 1.0)
    y2 = min(max(float(bbox[3]) / height, 0.0), 1.0)
    return x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)


def _to_bgr(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def analyze_image(img: Image.Image) -> list[FaceRecord]:
    """Faces in an oriented RGB photo (full resolution). Detector sizes settings.face_det_sizes,
    threshold settings.face_det_threshold, faces smaller than settings.face_min_size_px dropped."""
    from app.faces.engine import get_engine

    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
    faces = get_engine().analyze(
        _to_bgr(img),
        sizes=tuple(settings.face_det_sizes),
        threshold=settings.face_det_threshold,
        min_size=settings.face_min_size_px,
    )
    out: list[FaceRecord] = []
    for f in faces:
        if f.embedding is None:
            continue
        x, y, w, h = _norm_box(f.bbox, W, H)
        size_px = f.size_px
        out.append(FaceRecord(
            x=x, y=y, w=w, h=h,
            det_score=float(f.det_score),
            size_px=size_px,
            quality=face_quality(f.det_score, size_px),
            frame_time_s=None,
            embedding=np.asarray(f.embedding, dtype=np.float32),
            crop=crop_face(img, tuple(float(v) for v in f.bbox)),
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
    """When a face is on screen: [start, end] seconds, consecutive sampled frames merged.

    samples are the times of all sampled frames (ascending), times the ones the face was detected
    in. A detection stands for the stretch of the clip nearer to its frame than to any other
    sampled frame: from halfway after the previous sample (0 for the first) to halfway before the
    next one (for the last sample, the end of the clip, or half a gap past it when that is unknown)."""
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
    """samples: every sampled frame time of the video (for the record's spans); None leaves them out."""
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
    )


def analyze_video(path: Path, info: MediaInfo) -> list[FaceRecord]:
    """One record per distinct person in the video.

    Sample frames with app.media.video.iter_frames (settings.video_sample_interval_s,
    settings.video_max_samples, longer side <= 1280), detect at settings.face_video_det_size.
    Link detections into tracks greedily: a detection joins the track whose mean embedding is most
    similar if that similarity >= settings.video_track_threshold, else starts a new track; one
    detection per track per frame. Each track -> one record: box, crop and frame_time_s from its
    best-quality detection (iter_frames gives each frame's exact presentation time, so the viewer can
    seek to the very frame the box was found in); embedding = L2-normalised mean of its (up to) 5 best-quality embeddings;
    spans = visible_spans of the frames it was detected in, so the viewer can tell who is on screen.
    Drop single-detection tracks unless the face is "good" (size >= settings.face_good_min_size_px
    and det_score >= settings.face_good_det_score).
    """
    from app.faces.engine import get_engine
    from app.media.video import iter_frames

    engine = get_engine()
    tracker = FaceTracker(settings.video_track_threshold)
    samples: list[float] = []
    frames = iter_frames(path, info, settings.video_sample_interval_s, settings.video_max_samples, VIDEO_MAX_SIDE)
    try:
        for t, bgr in frames:
            samples.append(float(t))
            fh, fw = bgr.shape[:2]
            # size_px is reported in full-resolution pixels; the frame may be downscaled.
            src_scale = max(info.width, info.height) / max(fw, fh) if info.width and info.height else 1.0
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
                    track.best_crop = crop_face(frame_img, tuple(float(v) for v in f.bbox))
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    tracks = tracker.finished(settings.face_good_min_size_px, settings.face_good_det_score)
    records = [track_to_record(tr, samples, info.duration_s) for tr in tracks]
    log.info("video %s: %d tracks, %d kept", path.name, len(tracker.tracks), len(records))
    return records
