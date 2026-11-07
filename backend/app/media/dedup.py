"""Near-duplicate detection and quality comparison.

CONTRACT (implemented by the ingest agent). Rules are in docs/SPEC.md section "Dedup".
Exact duplicates (same sha256) never reach this module: they are caught by known_hashes.

Summary:
  images  candidates: kind='image', aspect ratio within 2%, hamming(phash) <= PHASH_MAX and
          hamming(dhash) <= DHASH_MAX; EXIF guard (both have DateTimeOriginal and they differ,
          or same second but both have SubSecTimeOriginal and it differs -> different shots);
          pixel check sig_diff(): mean <= SIG_MEAN_MAX and max block <= SIG_BLOCK_MAX.
  videos  candidates: kind='video', |duration diff| <= max(1 s, 2%), aspect within 3%;
          per-frame phash distances over VIDEO_FP_POSITIONS: >= 8 comparable frames,
          median <= 10 and >= 75% of frames <= 14; capture-time guard (both have a creation
          time and they differ by more than max(2 s, duration) -> different recordings).
  Missing EXIF on either side is NOT evidence of a different shot (messaging apps strip it).
  Hidden media are candidates too (a hidden item must not come back through a re-upload).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Literal

from app.media.hashing import Fingerprint, hamming, sig_diff
from app.media.probe import MediaInfo, parse_taken_at

log = logging.getLogger(__name__)

# Thresholds, measured on tests/fixtures (7 portraits, 661-900 px) with the variants built in
# tests/test_dedup.py (values are phash / dhash / sig mean / sig max block):
#   same photo, 55% downscale + JPEG q60 + EXIF stripped:   phash 0, dhash <= 1, mean <= 0.63, block <= 1.32
#   same photo, 25% downscale + JPEG q30:                    phash <= 2, dhash <= 1, mean <= 1.57, block <= 2.50
#   rotated pixels + EXIF orientation 6 (same view):        phash 0, dhash 0, mean <= 0.16, block <= 0.34
#   70% centre crop (new):                                   phash >= 18, dhash >= 23, mean >= 38, block >= 105
#   burst variant, 1.5% shift + one region changed (new):    phash 6-12, dhash 6-13, mean >= 17.9, block >= 55
#   one region changed, no shift (new):                      block 33-72 (one fixture 4.4, where the changed
#                                                            region is flat background and the pixels barely change)
#   different fixtures (new):                                phash >= 12 (aldrin/armstrong; next lowest 22),
#                                                            dhash >= 17, mean >= 37, block >= 80
#   colour/contrast/sharpen edit at q70 (kept separately):   mean 4.4-6.4, block 8.4-10.6
#   3 px crop + resize back (kept separately):               mean 2.6-4.3, block 7.6-16.1
# So the duplicate cases sit at block <= 2.5 and the different shots at block >= 33: the
# defaults below have a wide margin on both sides and were left unchanged.
PHASH_MAX = 10
DHASH_MAX = 12
SIG_MEAN_MAX = 4.0
SIG_BLOCK_MAX = 12.0
RES_MARGIN = 1.05  # resolution must differ by more than 5% to count as better
BYTES_MARGIN = 1.10  # same resolution & format: file must be >10% larger to count as better

IMAGE_ASPECT_TOL = 0.02
VIDEO_ASPECT_TOL = 0.03
VIDEO_DURATION_ABS_S = 1.0
VIDEO_DURATION_REL = 0.02
VIDEO_MIN_FRAMES = 8
VIDEO_TAKEN_TOL_S = 2.0  # creation times further apart than max(this, duration) mean different recordings
# Video frame fingerprints (12 frames at 256 px), measured with tests/helpers.make_video clips:
#   640x360 4 s testsrc2 vs its 320x180 crf 35 re-encode:   12 frames, median 2, max 4
#   same content as VP9 webm:                                 median 0, max 2
#   different sources of equal duration (testsrc2, testsrc, mandelbrot): median 28-31, min 22
VIDEO_FRAME_MEDIAN_MAX = 10
VIDEO_FRAME_CLOSE = 14
VIDEO_FRAME_CLOSE_SHARE = 0.75
VIDEO_BITRATE_MARGIN = 1.15
VIDEO_BYTES_MARGIN = 1.15

_FAMILIES = {
    ".jpg": "jpeg", ".jpeg": "jpeg",
    ".heic": "heif", ".heif": "heif", ".avif": "heif",
    ".png": "lossless", ".tif": "lossless", ".tiff": "lossless", ".bmp": "lossless",
    ".webp": "webp",
    ".gif": "gif",
}


@dataclass
class QualityKey:
    kind: Literal["image", "video"]
    pixels: int
    size_bytes: int
    ext: str
    bitrate: int | None = None  # video
    has_camera_exif: bool = False  # image has camera Make/Model or DateTimeOriginal


@dataclass
class DedupDecision:
    action: Literal["new", "replace", "lower_quality"]
    target_id: int | None = None  # media.id of the matched existing item
    target_uid: str | None = None
    detail: str = ""  # human-readable evidence, stored in superseded.detail and logs


def quality_key(info: MediaInfo, size_bytes: int) -> QualityKey:
    return QualityKey(
        kind=info.kind,
        pixels=int(info.width) * int(info.height),
        size_bytes=int(size_bytes),
        ext=(info.ext or "").lower(),
        bitrate=int(info.bitrate) if info.bitrate else None,
        has_camera_exif=bool(info.camera or info.exif_datetime),
    )


def quality_key_from_row(row: sqlite3.Row) -> QualityKey:
    """Build a QualityKey from a media row (uses kind, width, height, size_bytes, ext, bitrate,
    camera, exif_datetime)."""
    return QualityKey(
        kind=row["kind"],
        pixels=int(row["width"] or 0) * int(row["height"] or 0),
        size_bytes=int(row["size_bytes"] or 0),
        ext=(row["ext"] or "").lower(),
        bitrate=int(row["bitrate"]) if row["bitrate"] else None,
        has_camera_exif=bool(row["camera"] or row["exif_datetime"]),
    )


def _beyond(a: float, b: float, margin: float) -> int:
    """+1 if a > b * margin, -1 if b > a * margin, else 0."""
    if a > b * margin:
        return 1
    if b > a * margin:
        return -1
    return 0


def _compare(new: QualityKey, old: QualityKey) -> tuple[int, str]:
    """compare_quality plus the deciding criterion, for logs and superseded.detail."""
    res = _beyond(new.pixels, old.pixels, RES_MARGIN)
    if res:
        return res, "resolution"
    if new.kind == "video" or old.kind == "video":
        if new.bitrate and old.bitrate:
            res = _beyond(new.bitrate, old.bitrate, VIDEO_BITRATE_MARGIN)
            if res:
                return res, "bitrate"
        res = _beyond(new.size_bytes, old.size_bytes, VIDEO_BYTES_MARGIN)
        if res:
            return res, "file size"
        return 0, "equal"
    if new.has_camera_exif != old.has_camera_exif:
        return (1 if new.has_camera_exif else -1), "camera EXIF"
    fam_new, fam_old = _FAMILIES.get(new.ext), _FAMILIES.get(old.ext)
    if fam_new is not None and fam_new == fam_old:
        res = _beyond(new.size_bytes, old.size_bytes, BYTES_MARGIN)
        if res:
            return res, "file size"
    return 0, "equal"


def compare_quality(new: QualityKey, old: QualityKey) -> int:
    """+1 if new is clearly better, -1 if clearly worse, 0 if equivalent (ties keep the existing item).

    Images: resolution (pixels beyond RES_MARGIN) > camera EXIF presence > file size beyond
    BYTES_MARGIN when both are the same format family (jpeg / heic+avif / png+tiff+bmp / webp / gif).
    Videos: resolution > bitrate (beyond 15%) > file size (beyond 15%).
    """
    return _compare(new, old)[0]


def _aspect(width: int, height: int) -> float:
    return float(width) / float(height) if width and height else 0.0


def _aspect_close(a: float, b: float, tol: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a / b - 1.0) <= tol


def _different_shot(info: MediaInfo, row: sqlite3.Row) -> str | None:
    """EXIF evidence that two photos are different shots, or None."""
    a, b = info.exif_datetime, row["exif_datetime"]
    if not a or not b:
        return None
    if a != b:
        return f"capture times differ ({a} vs {b})"
    sa, sb = info.exif_subsec, row["exif_subsec"]
    if sa and sb and sa != sb:
        return f"sub-second capture times differ ({sa} vs {sb})"
    return None


def _different_recording(info: MediaInfo, row: sqlite3.Row) -> str | None:
    """Container creation times that show two videos were recorded separately, or None.
    Consecutive clips from a tripod look alike frame by frame; their start times do not."""
    a, b = parse_taken_at(info.taken_at), parse_taken_at(row["taken_at"])
    if a is None or b is None:
        return None
    gap = abs((a - b).total_seconds())
    if gap > max(VIDEO_TAKEN_TOL_S, float(info.duration_s or 0.0)):
        return f"capture times differ by {gap:.0f}s ({info.taken_at} vs {row['taken_at']})"
    return None


def _describe(key: QualityKey, width: int, height: int) -> str:
    return f"{width}x{height} {key.size_bytes} B{' camera EXIF' if key.has_camera_exif else ''}"


def _decide(info: MediaInfo, size_bytes: int, row: sqlite3.Row, evidence: str) -> DedupDecision:
    new_key = quality_key(info, size_bytes)
    old_key = quality_key_from_row(row)
    verdict, why = _compare(new_key, old_key)
    action: Literal["replace", "lower_quality"] = "replace" if verdict > 0 else "lower_quality"
    if verdict > 0:
        outcome = f"incoming is better ({why})"
    elif verdict < 0:
        outcome = f"kept copy is better ({why})"
    else:
        outcome = "equivalent quality, kept the existing copy"
    detail = (f"{evidence}; incoming {_describe(new_key, info.width, info.height)} vs kept "
              f"{_describe(old_key, row['width'], row['height'])}: {outcome}")
    return DedupDecision(action=action, target_id=int(row["id"]), target_uid=row["uid"], detail=detail)


def _rows_by_id(conn: sqlite3.Connection, ids: list[int], columns: str) -> list[sqlite3.Row]:
    out: list[sqlite3.Row] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        out.extend(conn.execute(f"SELECT {columns} FROM media WHERE id IN ({marks})", chunk).fetchall())
    return out


_ROW_COLUMNS = ("id, uid, kind, width, height, size_bytes, ext, bitrate, camera, exif_datetime, exif_subsec, "
                "phash, dhash, sig, frames, duration_s")


def _find_image(conn: sqlite3.Connection, info: MediaInfo, fp: Fingerprint, size_bytes: int) -> DedupDecision:
    aspect = _aspect(info.width, info.height)
    if aspect <= 0 or not fp.phash:
        return DedupDecision(action="new", detail="no usable fingerprint")
    lo, hi = aspect / (1 + IMAGE_ASPECT_TOL) * 0.999, aspect * (1 + IMAGE_ASPECT_TOL) * 1.001
    # Cheap pass: aspect in SQL, hashes in Python; only survivors load the signature.
    gated: dict[int, tuple[int, int]] = {}
    for r in conn.execute(
        "SELECT id, width, height, phash, dhash FROM media WHERE kind = 'image' AND phash IS NOT NULL "
        "AND height > 0 AND CAST(width AS REAL) / height BETWEEN ? AND ?",
        (lo, hi),
    ):
        if not _aspect_close(aspect, _aspect(r["width"], r["height"]), IMAGE_ASPECT_TOL):
            continue
        dp = hamming(fp.phash, r["phash"])
        if dp > PHASH_MAX:
            continue
        dd = hamming(fp.dhash, r["dhash"] or "")
        if dd > DHASH_MAX:
            continue
        gated[int(r["id"])] = (dp, dd)
    if not gated:
        return DedupDecision(action="new", detail="no similar photo")

    best: tuple[float, int, sqlite3.Row, str] | None = None
    for row in _rows_by_id(conn, sorted(gated), _ROW_COLUMNS):
        dp, dd = gated[int(row["id"])]
        why_not = _different_shot(info, row)
        if why_not:
            log.info("dedup: media %s passes the hash gate but %s", row["uid"], why_not)
            continue
        mean, block = sig_diff(fp.sig, row["sig"] or b"")
        if mean > SIG_MEAN_MAX or block > SIG_BLOCK_MAX:
            log.info("dedup: media %s passes the hash gate (phash %d, dhash %d) but pixels differ "
                     "(mean %.1f, block %.1f)", row["uid"], dp, dd, mean, block)
            continue
        evidence = f"phash {dp}, dhash {dd}, pixel diff mean {mean:.1f} block {block:.1f}"
        if best is None or (block, dp) < (best[0], best[1]):
            best = (block, dp, row, evidence)
    if best is None:
        return DedupDecision(action="new", detail="similar photos are different shots")
    return _decide(info, size_bytes, best[2], best[3])


def _frame_distances(a: list[str], b: list[str]) -> list[int]:
    return [hamming(x, y) for x, y in zip(a, b) if x and y]


def _find_video(conn: sqlite3.Connection, info: MediaInfo, fp: Fingerprint, size_bytes: int) -> DedupDecision:
    duration = info.duration_s or 0.0
    aspect = _aspect(info.width, info.height)
    frames = fp.frames or []
    if duration <= 0 or aspect <= 0 or sum(1 for f in frames if f) < VIDEO_MIN_FRAMES:
        return DedupDecision(action="new", detail="no usable fingerprint")
    best: tuple[float, float, sqlite3.Row, str] | None = None
    for row in conn.execute(
        "SELECT id, uid, duration_s, width, height, frames, taken_at FROM media WHERE kind = 'video' "
        "AND frames IS NOT NULL AND duration_s > 0"
    ):
        other = float(row["duration_s"])
        if abs(other - duration) > max(VIDEO_DURATION_ABS_S, VIDEO_DURATION_REL * max(other, duration)):
            continue
        if not _aspect_close(aspect, _aspect(row["width"], row["height"]), VIDEO_ASPECT_TOL):
            continue
        try:
            other_frames = json.loads(row["frames"])
        except (TypeError, ValueError):
            continue
        if not isinstance(other_frames, list):
            continue
        dists = _frame_distances(frames, [str(f or "") for f in other_frames])
        if len(dists) < VIDEO_MIN_FRAMES:
            continue
        median = float(statistics.median(dists))
        close = sum(1 for d in dists if d <= VIDEO_FRAME_CLOSE) / len(dists)
        if median > VIDEO_FRAME_MEDIAN_MAX or close < VIDEO_FRAME_CLOSE_SHARE:
            continue
        why_not = _different_recording(info, row)
        if why_not:
            log.info("dedup: video %s has matching frames but %s", row["uid"], why_not)
            continue
        mean = sum(dists) / len(dists)
        evidence = f"{len(dists)} frames compared, median distance {median:g}, {close:.0%} close"
        if best is None or (median, mean) < (best[0], best[1]):
            best = (median, mean, row, evidence)
    if best is None:
        return DedupDecision(action="new", detail="no similar video")
    full = _rows_by_id(conn, [int(best[2]["id"])], _ROW_COLUMNS)
    if not full:
        return DedupDecision(action="new", detail="match disappeared")
    return _decide(info, size_bytes, full[0], best[3])


def find_duplicate(conn: sqlite3.Connection, info: MediaInfo, fp: Fingerprint, size_bytes: int) -> DedupDecision:
    """Search stored media for a near-duplicate of the incoming file and decide what to do.

    action 'new' (no match), 'replace' (match, incoming is better) or 'lower_quality'
    (match, existing is better or equal). If several candidates pass, use the closest one.
    """
    if info.kind == "image":
        return _find_image(conn, info, fp, size_bytes)
    return _find_video(conn, info, fp, size_bytes)
