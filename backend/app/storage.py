"""File layout under settings.data_dir (docs/ARCHITECTURE.md section 3).

  originals/<sh>/<uid>.v<version><ext>     untouched original (sh = first 2 chars of uid)
  thumbs/<sh>/<uid>.v<version>.webp        grid thumbnail (settings.thumb_size max side)
  previews/<sh>/<uid>.v<version>.webp      large view (settings.preview_size); video poster; page 1
  previews/<sh>/<uid>.v<version>.sm.webp   smaller view (settings.preview_small_size)
  pages/<sh>/<uid>.v<version>/<n>.webp     document page renders (settings.page_render_size), n from 1
  videos/<sh>/<uid>.v<version>.mp4         compressed H.264 streaming copy
  faces/<id // 1000>/<id>.webp             face crop (192x192)
  bodies/<id // 1000>/<id>.webp            person crop
  docphoto/<sh>/<uid>.v<version>.webp      photo cropped out of an ID document
  staging/<upload_uid>.part                upload in progress / waiting for ingest
  failed/<upload_uid>.part                 upload whose ingest failed for good
  wallet/                                  Apple Wallet certificates placed by the owner
  tmp/                                     scratch files (same filesystem, so renames are atomic)
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from app.config import settings
from app.util import new_uid

DIRS = ("originals", "thumbs", "previews", "pages", "videos", "faces", "bodies", "docphoto", "staging", "failed",
        "wallet", "tmp", "learned")
# Directories the orphan sweeper walks (names must match the patterns in app.ingest).
SWEPT_DIRS = ("originals", "thumbs", "previews", "pages", "videos", "faces", "bodies", "docphoto")


def root() -> Path:
    return settings.data_dir


def ensure_dirs() -> None:
    for d in DIRS:
        (root() / d).mkdir(parents=True, exist_ok=True)


def _shard(uid: str) -> str:
    return uid[:2]


def original_path(uid: str, version: int, ext: str) -> Path:
    return root() / "originals" / _shard(uid) / f"{uid}.v{version}{ext}"


def thumb_path(uid: str, version: int) -> Path:
    return root() / "thumbs" / _shard(uid) / f"{uid}.v{version}.webp"


def preview_path(uid: str, version: int) -> Path:
    return root() / "previews" / _shard(uid) / f"{uid}.v{version}.webp"


def preview_small_path(uid: str, version: int) -> Path:
    return root() / "previews" / _shard(uid) / f"{uid}.v{version}.sm.webp"


def pages_dir(uid: str, version: int) -> Path:
    return root() / "pages" / _shard(uid) / f"{uid}.v{version}"


def page_path(uid: str, version: int, page_no: int) -> Path:
    return pages_dir(uid, version) / f"{page_no}.webp"


def video_preview_path(uid: str, version: int) -> Path:
    return root() / "videos" / _shard(uid) / f"{uid}.v{version}.mp4"


def live_clip_path(uid: str, version: int) -> Path:
    """The clip embedded in a motion photo (section 20), a derivative next to the streaming copies."""
    return root() / "videos" / _shard(uid) / f"{uid}.v{version}.live.mp4"


def face_crop_path(face_id: int) -> Path:
    return root() / "faces" / str(face_id // 1000) / f"{face_id}.webp"


def body_crop_path(body_id: int) -> Path:
    return root() / "bodies" / str(body_id // 1000) / f"{body_id}.webp"


def animal_crop_path(animal_id: int) -> Path:
    return root() / "animals" / str(animal_id // 1000) / f"{animal_id}.webp"


def doc_photo_path(uid: str, version: int) -> Path:
    return root() / "docphoto" / _shard(uid) / f"{uid}.v{version}.webp"


def staging_path(upload_uid: str) -> Path:
    return root() / "staging" / f"{upload_uid}.part"


def failed_path(upload_uid: str) -> Path:
    return root() / "failed" / f"{upload_uid}.part"


def wallet_dir() -> Path:
    return settings.wallet_path


def tmp_path(suffix: str = "") -> Path:
    p = root() / "tmp" / f"{new_uid(16)}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def rel(path: Path) -> str:
    """Path relative to the data dir, for logs and the sweeper."""
    return str(Path(path).resolve().relative_to(root().resolve()))


def from_rel(relpath: str) -> Path:
    """Inverse of rel(); refuses anything that escapes the data dir."""
    base = root().resolve()
    p = (base / relpath).resolve()
    if base not in p.parents and p != base:
        raise ValueError(f"path escapes data dir: {relpath}")
    return p


def move_into_place(src: Path, dst: Path) -> None:
    """Atomic rename within the data dir filesystem (creates parent dirs)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".{new_uid(6)}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def remove_tree(path: Path | None) -> None:
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


def disk_usage() -> tuple[int, int]:
    """(free_bytes, total_bytes) of the filesystem holding the data dir."""
    root().mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(root())
    return u.free, u.total


def has_room_for(nbytes: int) -> bool:
    """True if writing nbytes still leaves settings.min_free_disk_bytes free."""
    free, _ = disk_usage()
    return free - int(nbytes) >= settings.min_free_disk_bytes


def dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total
