"""URL builders for files served by app/api/files.py. `v` busts caches after a replacement: an item's version,
or for a crop the time its file was written (a moved box gets a new crop, and an undo brings the old file back
with its old time, so the old URL and its cached copy are right again)."""
from __future__ import annotations

import os
from pathlib import Path

from app import storage


def _written(path: Path) -> str:
    try:
        return f"?v={os.stat(path).st_mtime_ns // 1_000_000:x}"
    except OSError:
        return ""


def thumb_url(uid: str, version: int) -> str:
    return f"/files/thumb/{uid}?v={version}"


def preview_url(uid: str, version: int) -> str:
    return f"/files/preview/{uid}?v={version}"


def preview_small_url(uid: str, version: int) -> str:
    return f"/files/preview-small/{uid}?v={version}"


def original_url(uid: str, version: int) -> str:
    return f"/files/original/{uid}?v={version}"


def download_url(uid: str, version: int) -> str:
    return f"/files/original/{uid}?v={version}&download=1"


def video_url(uid: str, version: int) -> str:
    """Playable stream: the compressed H.264 copy when one exists, otherwise the original."""
    return f"/files/video/{uid}?v={version}"


def playable_video_url(uid: str, version: int, preview_status: str, stream_friendly: bool) -> str | None:
    """The stream URL once the compressed copy is done, or when there is none to wait for ('none',
    'error': the original is served). While the copy is 'pending' only a stream_friendly original is
    offered; otherwise None, and the viewer shows "preparing"."""
    if preview_status == "pending" and not stream_friendly:
        return None
    return video_url(uid, version)


def page_url(uid: str, version: int, page_no: int) -> str:
    return f"/files/page/{uid}/{page_no}?v={version}"


def face_url(face_id: int) -> str:
    return f"/files/face/{face_id}" + _written(storage.face_crop_path(face_id))


def body_url(body_id: int) -> str:
    return f"/files/body/{body_id}" + _written(storage.body_crop_path(body_id))


def animal_url(animal_id: int) -> str:
    return f"/files/animal/{animal_id}" + _written(storage.animal_crop_path(animal_id))


def doc_photo_url(uid: str, version: int) -> str:
    return f"/files/doc-photo/{uid}?v={version}"


def wallet_url(uid: str) -> str:
    return f"/api/documents/{uid}/wallet.pkpass"
