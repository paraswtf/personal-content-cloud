"""File layout (docs/ARCHITECTURE.md sections 3 and 31).

Two roots. settings.originals_root (ORIGINALS_DIR, a fast disk; DATA_DIR when it is not set) holds the
bytes the owner uploaded; settings.data_dir (DATA_DIR) holds the database and everything the app makes
from them. The paths below are the same under either root.

  originals/<sh>/<uid>.v<version><ext>     untouched original (sh = first 2 chars of uid)   [originals root]
  staging/<upload_uid>.part                upload in progress / waiting for ingest           [originals root]
  failed/<upload_uid>.part                 upload whose ingest failed for good              [originals root]
  thumbs/<sh>/<uid>.v<version>.webp        grid thumbnail (settings.thumb_size max side)
  previews/<sh>/<uid>.v<version>.webp      large view (settings.preview_size); video poster; page 1
  previews/<sh>/<uid>.v<version>.sm.webp   smaller view (settings.preview_small_size)
  pages/<sh>/<uid>.v<version>/<n>.webp     document page renders (settings.page_render_size), n from 1
  videos/<sh>/<uid>.v<version>.mp4         compressed H.264 streaming copy
  faces/<id // 1000>/<id>.webp             face crop (192x192)
  bodies/<id // 1000>/<id>.webp            person crop
  animals/<id // 1000>/<id>.webp           pet crop
  docphoto/<sh>/<uid>.v<version>.webp      photo cropped out of an ID document
  learned/                                 the learned models (section 18)
  wallet/                                  Apple Wallet certificates placed by the owner
  tmp/                                     scratch files on the data disk (wiped at startup)

With two roots, an original that an earlier layout left under DATA_DIR/originals is still found
(original_path falls back to it while app.tiers moves those files over), and the sweeper walks both.
"""
from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

from app.config import settings
from app.util import new_uid

# Folders under the originals root (ORIGINALS_DIR when set): the uploaded bytes and the uploads on their way in.
TIER_DIRS = ("originals", "staging", "failed")
# Folders under DATA_DIR: the database's companions, everything derived, and scratch space.
DERIVED_DIRS = ("thumbs", "previews", "pages", "videos", "faces", "bodies", "animals", "docphoto", "wallet", "tmp",
                "learned")
DIRS = TIER_DIRS + DERIVED_DIRS
# Directories the orphan sweeper walks (names must match the patterns in app.ingest).
SWEPT_DIRS = ("originals", "thumbs", "previews", "pages", "videos", "faces", "bodies", "animals", "docphoto")


def root() -> Path:
    """DATA_DIR: the database and the derived files."""
    return settings.data_dir


def originals_root() -> Path:
    """ORIGINALS_DIR when set, else DATA_DIR: originals, staging and failed uploads."""
    return settings.originals_root


def tiered() -> bool:
    """True when the originals live on their own root (ORIGINALS_DIR set and different from DATA_DIR)."""
    return settings.originals_dir is not None


def base_for(top: str) -> Path:
    """The root a top-level folder lives under."""
    return originals_root() if top in TIER_DIRS else root()


def legacy_dir(top: str) -> Path | None:
    """DATA_DIR/<top> for a tier folder when ORIGINALS_DIR is set: where an earlier layout kept it."""
    if top in TIER_DIRS and tiered():
        return root() / top
    return None


def ensure_dirs() -> None:
    for d in DIRS:
        (base_for(d) / d).mkdir(parents=True, exist_ok=True)


def _shard(uid: str) -> str:
    return uid[:2]


def original_path(uid: str, version: int, ext: str) -> Path:
    """The original's path on the originals root. While ORIGINALS_DIR is new, a file that only exists
    under DATA_DIR/originals (the earlier layout) is returned from there until app.tiers has moved it."""
    name = f"{uid}.v{version}{ext}"
    p = originals_root() / "originals" / _shard(uid) / name
    if tiered() and not p.exists():
        legacy = root() / "originals" / _shard(uid) / name
        if legacy.exists():
            return legacy
    return p


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
    return originals_root() / "staging" / f"{upload_uid}.part"


def failed_path(upload_uid: str) -> Path:
    return originals_root() / "failed" / f"{upload_uid}.part"


def wallet_dir() -> Path:
    return settings.wallet_path


def tmp_path(suffix: str = "") -> Path:
    """A scratch path under DATA_DIR/tmp (derived files are renamed from here into place)."""
    p = root() / "tmp" / f"{new_uid(16)}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def scratch_beside(dest: Path, suffix: str = ".tmp") -> Path:
    """A scratch path in dest's own folder, so a rename onto dest never crosses filesystems (used to
    place originals, whose folder may be on another disk than tmp/)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest.with_name(f"{dest.name}.{new_uid(6)}{suffix}")


def _bases() -> list[Path]:
    """The roots, deepest first, so a path under a nested root resolves against that root."""
    bases = {root().resolve()}
    if tiered():
        bases.add(originals_root().resolve())
    return sorted(bases, key=lambda b: len(b.parts), reverse=True)


def rel(path: Path) -> str:
    """Path relative to the root that holds it (the top folder names the kind), for logs and the sweeper."""
    p = Path(path).resolve()
    for base in _bases():
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    raise ValueError(f"path is outside the data folders: {path}")


def from_rel(relpath: str) -> Path:
    """Inverse of rel(); refuses anything that escapes its root."""
    parts = Path(relpath).parts
    base = base_for(parts[0]).resolve() if parts else root().resolve()
    p = (base / relpath).resolve()
    if base not in p.parents and p != base:
        raise ValueError(f"path escapes data dir: {relpath}")
    return p


def move_into_place(src: Path, dst: Path) -> None:
    """Atomic rename (creates parent dirs). Across filesystems the bytes are copied to a scratch file
    beside dst first and renamed from there, so dst never appears half-written; src is then removed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        tmp = scratch_beside(dst)
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except BaseException:
            safe_unlink(tmp)
            raise
        safe_unlink(src)


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


def disk_usage(path: Path | None = None) -> tuple[int, int]:
    """(free_bytes, total_bytes) of the filesystem holding path; by default the originals root, the
    disk that receives uploads."""
    base = path if path is not None else originals_root()
    base.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(base)
    return u.free, u.total


def same_filesystem(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def tiers() -> list[dict]:
    """The storage tiers as GET /api/system reports them (section 31): the originals root, the data
    root (left out when it is the same folder) and the backup root when backups are on. `shared_with`
    names an earlier tier on the same filesystem, so the UI does not count that disk twice."""
    out: list[dict] = []
    def add(key: str, path: Path, holds: str) -> None:
        try:
            free, total = disk_usage(path)
        except OSError:
            free, total = 0, 0
        shared = next((t["key"] for t in out if same_filesystem(Path(t["path"]), path)), None)
        out.append({"key": key, "path": str(path), "holds": holds, "free_bytes": free, "total_bytes": total,
                    "shared_with": shared})
    add("originals", originals_root(), "originals and uploads on their way in")
    if tiered():
        add("data", root(), "database, previews, thumbnails, streaming copies, crops, the served site")
    if settings.backup_dir is not None:
        add("backup", settings.backup_dir, "nightly snapshots")
    return out


def swept_bases() -> list[tuple[str, Path]]:
    """(top folder, directory) pairs the orphan sweeper walks: each swept folder under its root, plus
    DATA_DIR/originals while an earlier layout's files are still being moved from there."""
    out: list[tuple[str, Path]] = []
    for top in SWEPT_DIRS:
        out.append((top, base_for(top) / top))
        legacy = legacy_dir(top)
        if legacy is not None:
            out.append((top, legacy))
    return out


def has_room_for(nbytes: int) -> bool:
    """True if writing nbytes to the originals root still leaves settings.min_free_disk_bytes free
    there, and the data root (when it is another disk) keeps that much for the derived files."""
    free, _ = disk_usage(originals_root())
    if free - int(nbytes) < settings.min_free_disk_bytes:
        return False
    if tiered():
        free_data, _ = disk_usage(root())
        return free_data >= settings.min_free_disk_bytes
    return True


def dir_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total
