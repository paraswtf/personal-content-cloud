"""Face detection and recognition with the InsightFace buffalo_l models (docs/ARCHITECTURE.md section 8).

Contract:
  available() -> bool                 det_10g.onnx and w600k_r50.onnx are present (app.ml.models 'faces')
  get_engine() -> FaceEngine          process-wide engine, loaded on first use; reset() drops it
  loaded() / provider()               for app.ml.models.statuses()
  engine.detect(img_bgr, det_size=640, threshold=0.5, nms_thresh=0.4) -> list[DetectedFace] (no embeddings)
  engine.detect_multiscale(img_bgr, sizes, threshold, nms_thresh) -> list[DetectedFace]
  engine.embed(img_bgr, faces) -> None  fills face.embedding (512, float32, L2-normalised)
  engine.analyze(img_bgr, sizes, threshold, min_size) -> list[DetectedFace] detected and embedded
  DetectedFace(bbox (4,) x1 y1 x2 y2, kps (5, 2), det_score, embedding) with width, height, size_px

SCRFD-10GF (det_10g.onnx) finds faces and 5 landmarks. ArcFace R50 (w600k_r50.onnx) turns each
aligned 112x112 face into a 512-d embedding; cosine similarity of two L2-normalised embeddings says
how alike two faces are. Sessions come from app.ml.runtime (CUDA when usable, else CPU).

Pre/post-processing follows insightface's reference code (model_zoo/scrfd.py,
model_zoo/arcface_onnx.py, utils/face_align.py) without depending on the insightface package.
Swapping engines means replacing this file and re-tuning the similarity thresholds in config.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DET_MODEL = "buffalo_l/det_10g.onnx"
REC_MODEL = "buffalo_l/w600k_r50.onnx"
EMBEDDING_DIM = 512

# Reference landmark positions for a 112x112 ArcFace crop (eyes, nose, mouth corners).
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)

_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2


@dataclass
class DetectedFace:
    bbox: np.ndarray  # (4,) float32: x1, y1, x2, y2 in pixels of the analysed image
    kps: np.ndarray  # (5, 2) float32 landmarks in pixels of the analysed image
    det_score: float
    embedding: np.ndarray | None = None  # (512,) float32, L2-normalised

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def size_px(self) -> int:
        """Shorter side of the box, a proxy for how much detail the face has."""
        return int(round(min(self.width, self.height)))


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform src -> dst (same as skimage SimilarityTransform)."""
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    a = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(a) < 0:
        d[dim - 1] = -1
    t = np.eye(dim + 1, dtype=np.float64)
    u, s, v = np.linalg.svd(a)
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        return np.full_like(t, np.nan)
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(v) > 0:
            t[:dim, :dim] = u @ v
        else:
            keep = d[dim - 1]
            d[dim - 1] = -1
            t[:dim, :dim] = u @ np.diag(d) @ v
            d[dim - 1] = keep
    else:
        t[:dim, :dim] = u @ np.diag(d) @ v
    scale = 1.0 / src_demean.var(axis=0).sum() * (s @ d)
    t[:dim, dim] = dst_mean - scale * (t[:dim, :dim] @ src_mean.T)
    t[:dim, :dim] *= scale
    return t


def align_face(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp a face to the canonical 112x112 ArcFace crop using its 5 landmarks."""
    m = _umeyama(kps.astype(np.float64), ARCFACE_DST * (size / 112.0))[:2, :]
    return cv2.warpAffine(img_bgr, m, (size, size), borderValue=0.0)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class FaceEngine:
    """Holds the two onnxruntime sessions. Thread-safe; calls are serialised per model."""

    def __init__(self, models_dir: Path, threads: int = 4):
        det_path = Path(models_dir) / DET_MODEL
        rec_path = Path(models_dir) / REC_MODEL
        for p in (det_path, rec_path):
            if not p.is_file():
                raise FileNotFoundError(f"face model missing: {p} (run scripts/download_models.py)")
        from app.ml import runtime

        self._det = runtime.session(det_path, threads=threads)
        self._rec = runtime.session(rec_path, threads=threads)
        self.provider = runtime.session_provider(self._det)
        self._det_in = self._det.get_inputs()[0].name
        self._det_out = [o.name for o in self._det.get_outputs()]
        if len(self._det_out) != 9:
            raise RuntimeError(f"unexpected detector outputs ({len(self._det_out)}); expected SCRFD with landmarks")
        self._rec_in = self._rec.get_inputs()[0].name
        self._det_lock = threading.Lock()
        self._rec_lock = threading.Lock()
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    # ---------------------------------------------------------------- detection
    def detect(
        self,
        img_bgr: np.ndarray,
        det_size: int = 640,
        threshold: float = 0.5,
        nms_thresh: float = 0.4,
    ) -> list[DetectedFace]:
        """Detect faces. The image is scaled so its longer side equals det_size.

        Returns faces sorted by detection score, coordinates in img_bgr pixels.
        """
        h, w = img_bgr.shape[:2]
        if h < 8 or w < 8:
            return []
        scale = det_size / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        in_w = int(math.ceil(new_w / 32.0) * 32)
        in_h = int(math.ceil(new_h / 32.0) * 32)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        blob = canvas[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        blob = (blob - 127.5) / 128.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

        with self._det_lock:
            outs = self._det.run(self._det_out, {self._det_in: blob})
        outs = [o[0] if o.ndim == 3 else o for o in outs]

        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(_STRIDES):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + 3] * stride
            kps_preds = outs[idx + 6] * stride
            fh, fw = in_h // stride, in_w // stride
            centers = self._anchor_centers(fh, fw, stride)
            pos = np.where(scores >= threshold)[0]
            if pos.size == 0:
                continue
            c = centers[pos]
            d = bbox_preds[pos]
            boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=-1)
            k = kps_preds[pos].reshape(-1, 5, 2) + c[:, None, :]
            scores_l.append(scores[pos])
            boxes_l.append(boxes)
            kps_l.append(k)
        if not scores_l:
            return []
        scores = np.concatenate(scores_l)
        boxes = np.concatenate(boxes_l) / scale
        kpss = np.concatenate(kps_l) / scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, nms_thresh)
        faces = []
        for i in keep:
            b = boxes[i].astype(np.float32)
            b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
            b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            faces.append(DetectedFace(bbox=b, kps=kpss[i].astype(np.float32), det_score=float(scores[i])))
        faces.sort(key=lambda f: f.det_score, reverse=True)
        return faces

    def _anchor_centers(self, fh: int, fw: int, stride: int) -> np.ndarray:
        key = (fh, fw, stride)
        centers = self._centers.get(key)
        if centers is None:
            grid = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            centers = np.stack([centers] * _NUM_ANCHORS, axis=1).reshape(-1, 2)
            if len(self._centers) < 64:
                self._centers[key] = centers
        return centers

    # -------------------------------------------------------------- recognition
    def embed(self, img_bgr: np.ndarray, faces: list[DetectedFace], batch: int = 32) -> None:
        """Fill face.embedding (L2-normalised float32) for every face, in batches."""
        if not faces:
            return
        crops = [align_face(img_bgr, f.kps) for f in faces]
        for start in range(0, len(crops), batch):
            chunk = np.stack(crops[start : start + batch])  # (n, 112, 112, 3) BGR
            blob = chunk[:, :, :, ::-1].astype(np.float32)
            blob = (blob - 127.5) / 127.5
            blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2))
            with self._rec_lock:
                feats = self._rec.run(None, {self._rec_in: blob})[0]
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = (feats / np.maximum(norms, 1e-12)).astype(np.float32)
            for f, e in zip(faces[start : start + batch], feats):
                f.embedding = e

    def detect_multiscale(
        self,
        img_bgr: np.ndarray,
        sizes: tuple[int, ...] = (640, 1280),
        threshold: float = 0.5,
        nms_thresh: float = 0.4,
    ) -> list[DetectedFace]:
        """Detect at several input sizes and merge.

        SCRFD misses faces that end up much larger than ~350 px at the input scale,
        and small faces in big group photos need a larger input. The first size always
        runs; larger sizes run only when the image is big enough to benefit.
        """
        h, w = img_bgr.shape[:2]
        longest = max(h, w)
        found: list[DetectedFace] = []
        for i, size in enumerate(sizes):
            if i > 0 and longest < size * 0.75:
                continue
            found.extend(self.detect(img_bgr, size, threshold, nms_thresh))
        if len(sizes) == 1 or not found:
            return found
        dets = np.array([[*f.bbox, f.det_score] for f in found], dtype=np.float32)
        keep = _nms(dets, nms_thresh)
        merged = [found[i] for i in keep]
        merged.sort(key=lambda f: f.det_score, reverse=True)
        return merged

    def analyze(
        self,
        img_bgr: np.ndarray,
        sizes: tuple[int, ...] = (640, 1280),
        threshold: float = 0.5,
        min_size: int = 0,
    ) -> list[DetectedFace]:
        """Detect (multi-scale), drop faces smaller than min_size pixels, embed the rest."""
        faces = [f for f in self.detect_multiscale(img_bgr, sizes, threshold) if f.size_px >= min_size]
        self.embed(img_bgr, faces)
        return faces


_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def available() -> bool:
    """True when both model files are present with their pinned sizes."""
    from app.ml import models

    return models.present("faces")


def get_engine() -> FaceEngine:
    """Process-wide engine, created on first use from app settings."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import settings

                _engine = FaceEngine(settings.models_dir, settings.ort_threads)
                log.info("face models loaded on %s", _engine.provider)
    return _engine


def reset() -> None:
    """Drop the shared engine (tests, or after the model files changed)."""
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
