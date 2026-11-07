"""URL builders for files served by app/api/files.py. `v` busts caches after a replacement."""
from __future__ import annotations


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
    return f"/files/face/{face_id}"


def body_url(body_id: int) -> str:
    return f"/files/body/{body_id}"


def doc_photo_url(uid: str, version: int) -> str:
    return f"/files/doc-photo/{uid}?v={version}"


def wallet_url(uid: str) -> str:
    return f"/api/documents/{uid}/wallet.pkpass"
