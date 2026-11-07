"""Near-duplicate detection and duplicate groups.

CONTRACT.
- Exact duplicates (same sha256) never reach this module: ingest answers them from items.sha256.
- find_similar(conn, info, fp, size_bytes) -> DedupDecision(action 'new' | 'link', target_id,
  target_uid, detail): the closest live (not trashed) image or video that is the same picture or
  recording. Nothing is ever replaced: everything uploaded is kept.
- Groups: items.dup_group_id is the id of the group's best item (the best points at itself with
  dup_role 'best'; the others have dup_role 'copy'). link_into_group(conn, new_id, target_id)
  adds an item to the target's group and re-chooses the best by the quality ranking below;
  regroup(conn, member_ids) re-chooses the best of a member set (after a delete, restore or purge)
  and dissolves groups of one; set_best(conn, item_id) is the owner's manual choice. All of them
  run inside the caller's transaction and bump search_version (copies are hidden from listings).

Matching rules (thresholds below):
  images  candidates: kind='image', aspect ratio within 2%, hamming(phash) <= PHASH_MAX and
          hamming(dhash) <= DHASH_MAX; EXIF guard (both have DateTimeOriginal and they differ,
          or same second but both have SubSecTimeOriginal and it differs -> different shots);
          pixel check sig_diff(): mean <= SIG_MEAN_MAX and max block <= SIG_BLOCK_MAX.
  videos  candidates: kind='video', |duration diff| <= max(1 s, 2%), aspect within 3%;
          per-frame phash distances over VIDEO_FP_POSITIONS: >= 8 comparable frames,
          median <= 10 and >= 75% of frames <= 14; capture-time guard (both have a creation
          time and they differ by more than max(2 s, duration) -> different recordings).
  Missing EXIF on either side is NOT evidence of a different shot (messaging apps strip it).
Quality ranking (the best of a group): images by resolution (beyond 5%), then camera EXIF, then
file size (beyond 10%) within a format family; videos by resolution, bitrate (15%), size (15%);
ties keep the current best.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Iterable, Literal

from app.db import bump_search_version
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
    action: Literal["new", "link"]
    target_id: int | None = None  # items.id of the matched existing item
    target_uid: str | None = None
    detail: str = ""  # human-readable evidence, stored in items.dup_detail and logs


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
    """Build a QualityKey from an items row (uses kind, width, height, size_bytes, ext, bitrate,
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
    """compare_quality plus the deciding criterion, for logs and items.dup_detail."""
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


def _link(info: MediaInfo, size_bytes: int, row: sqlite3.Row, evidence: str) -> DedupDecision:
    new_key = quality_key(info, size_bytes)
    old_key = quality_key_from_row(row)
    verdict, why = _compare(new_key, old_key)
    if verdict > 0:
        outcome = f"incoming is better ({why})"
    elif verdict < 0:
        outcome = f"existing copy is better ({why})"
    else:
        outcome = "equivalent quality"
    detail = (f"{evidence}; incoming {_describe(new_key, info.width or 0, info.height or 0)} vs "
              f"{_describe(old_key, row['width'] or 0, row['height'] or 0)}: {outcome}")
    return DedupDecision(action="link", target_id=int(row["id"]), target_uid=row["uid"], detail=detail[:500])


def _rows_by_id(conn: sqlite3.Connection, ids: list[int], columns: str) -> list[sqlite3.Row]:
    out: list[sqlite3.Row] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        out.extend(conn.execute(f"SELECT {columns} FROM items WHERE id IN ({marks})", chunk).fetchall())
    return out


_ROW_COLUMNS = ("id, uid, kind, width, height, size_bytes, ext, bitrate, camera, exif_datetime, exif_subsec, "
                "phash, dhash, sig, frames, duration_s, taken_at")


def _find_image(conn: sqlite3.Connection, info: MediaInfo, fp: Fingerprint, size_bytes: int) -> DedupDecision:
    aspect = _aspect(info.width or 0, info.height or 0)
    if aspect <= 0 or not fp.phash:
        return DedupDecision(action="new", detail="no usable fingerprint")
    lo, hi = aspect / (1 + IMAGE_ASPECT_TOL) * 0.999, aspect * (1 + IMAGE_ASPECT_TOL) * 1.001
    # Cheap pass: aspect in SQL, hashes in Python; only survivors load the signature.
    gated: dict[int, tuple[int, int]] = {}
    for r in conn.execute(
        "SELECT id, width, height, phash, dhash FROM items WHERE kind = 'image' AND deleted_at IS NULL "
        "AND phash IS NOT NULL AND height > 0 AND CAST(width AS REAL) / height BETWEEN ? AND ?",
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
            log.info("dedup: item %s passes the hash gate but %s", row["uid"], why_not)
            continue
        mean, block = sig_diff(fp.sig, row["sig"] or b"")
        if mean > SIG_MEAN_MAX or block > SIG_BLOCK_MAX:
            log.info("dedup: item %s passes the hash gate (phash %d, dhash %d) but pixels differ "
                     "(mean %.1f, block %.1f)", row["uid"], dp, dd, mean, block)
            continue
        evidence = f"phash {dp}, dhash {dd}, pixel diff mean {mean:.1f} block {block:.1f}"
        if best is None or (block, dp) < (best[0], best[1]):
            best = (block, dp, row, evidence)
    if best is None:
        return DedupDecision(action="new", detail="similar photos are different shots")
    return _link(info, size_bytes, best[2], best[3])


def _frame_distances(a: list[str], b: list[str]) -> list[int]:
    return [hamming(x, y) for x, y in zip(a, b) if x and y]


def _find_video(conn: sqlite3.Connection, info: MediaInfo, fp: Fingerprint, size_bytes: int) -> DedupDecision:
    duration = info.duration_s or 0.0
    aspect = _aspect(info.width or 0, info.height or 0)
    frames = fp.frames or []
    if duration <= 0 or aspect <= 0 or sum(1 for f in frames if f) < VIDEO_MIN_FRAMES:
        return DedupDecision(action="new", detail="no usable fingerprint")
    best: tuple[float, float, sqlite3.Row, str] | None = None
    for row in conn.execute(
        "SELECT id, uid, duration_s, width, height, frames, taken_at FROM items WHERE kind = 'video' "
        "AND deleted_at IS NULL AND frames IS NOT NULL AND duration_s > 0"
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
    return _link(info, size_bytes, full[0], best[3])


def find_similar(conn: sqlite3.Connection, info: MediaInfo, fp: Fingerprint | None, size_bytes: int) -> DedupDecision:
    """Search live items for a near-duplicate of the incoming file. 'link' with the closest match,
    or 'new'. Only images and videos are compared (documents and other files are never linked)."""
    if fp is None or info.kind not in ("image", "video"):
        return DedupDecision(action="new", detail="not compared")
    if info.kind == "image":
        return _find_image(conn, info, fp, size_bytes)
    return _find_video(conn, info, fp, size_bytes)


# ------------------------------------------------------------------ groups
_GROUP_COLUMNS = ("id, uid, kind, width, height, size_bytes, ext, bitrate, camera, exif_datetime, deleted_at, "
                  "dup_group_id, dup_role")


def group_member_ids(conn: sqlite3.Connection, item_id: int) -> list[int]:
    """Ids of every item in the item's duplicate group (the item itself included; trashed ones too)."""
    row = conn.execute("SELECT dup_group_id FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return []
    if row["dup_group_id"] is None:
        return [int(item_id)]
    ids = {int(r[0]) for r in conn.execute("SELECT id FROM items WHERE dup_group_id = ?", (row["dup_group_id"],))}
    ids.add(int(item_id))
    return sorted(ids)


def _apply(conn: sqlite3.Connection, ids: list[int], best_id: int | None) -> None:
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        if best_id is None:
            conn.execute(f"UPDATE items SET dup_group_id = NULL, dup_role = NULL WHERE id IN ({marks})", chunk)
        else:
            conn.execute(
                f"UPDATE items SET dup_group_id = ?, dup_role = CASE WHEN id = ? THEN 'best' ELSE 'copy' END "
                f"WHERE id IN ({marks})",
                [best_id, best_id, *chunk],
            )


def regroup(conn: sqlite3.Connection, member_ids: Iterable[int]) -> int | None:
    """Re-choose the best item of a duplicate group made of member_ids (call inside a tx). The
    best is chosen among live members (all members when every one is in the trash): the current
    best stays unless another is clearly better. A group of fewer than two items is dissolved.
    Returns the best id, or None when there is no group."""
    ids = sorted({int(i) for i in member_ids})
    rows = _rows_by_id(conn, ids, _GROUP_COLUMNS) if ids else []
    ids = [int(r["id"]) for r in rows]
    if len(rows) < 2:
        if ids:
            _apply(conn, ids, None)
            bump_search_version(conn)
        return None
    live = [r for r in rows if r["deleted_at"] is None] or rows
    best = next((r for r in live if r["dup_role"] == "best" and r["dup_group_id"] == r["id"]), None)
    if best is None:
        best = min(live, key=lambda r: int(r["id"]))
    for r in live:
        if r is not best and compare_quality(quality_key_from_row(r), quality_key_from_row(best)) > 0:
            best = r
    _apply(conn, ids, int(best["id"]))
    bump_search_version(conn)
    return int(best["id"])


def link_into_group(conn: sqlite3.Connection, new_id: int, target_id: int, detail: str = "") -> int | None:
    """Put new_id into target_id's duplicate group (creating it) and re-choose the best."""
    members = set(group_member_ids(conn, target_id))
    members.add(int(new_id))
    if detail:
        conn.execute("UPDATE items SET dup_detail = ? WHERE id = ?", (detail[:500], new_id))
    return regroup(conn, members)


def set_best(conn: sqlite3.Connection, item_id: int) -> bool:
    """The owner's choice of the group's best item. False when the item is in no group."""
    members = group_member_ids(conn, item_id)
    if len(members) < 2:
        return False
    _apply(conn, members, int(item_id))
    bump_search_version(conn)
    return True
