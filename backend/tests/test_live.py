"""Live Photos and motion photos (docs/ARCHITECTURE.md section 20): the identifiers, the embedded
clip, the pairing rules, and the whole flow through the upload API: one live item, a hidden
companion, playback and download URLs, the vault and the trash following the still."""
from __future__ import annotations

import io
import struct

from PIL import Image

from app import db
from app.db import tx
from app.media import live
from app.util import now_iso
from tests.helpers import jpeg_bytes, make_item, make_video, noise_image, run_jobs

H = {"X-Requested-With": "pcc"}
UUID = "0D2E8E3F-9B2C-4C1E-8B2A-1234567890AB"
FAKE_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 200 + b"moov" + b"\x00" * 64


def apple_exif() -> Image.Exif:
    exif = Image.Exif()
    note = b"Apple iOS\x00\x00\x01MM" + b"\x00" * 20 + UUID.encode() + b"\x00"
    exif.get_ifd(0x8769)[0x927C] = note
    return exif


def test_content_identifier_from_makernote_and_quicktime_tags():
    assert live.content_identifier(apple_exif()) == UUID
    plain = Image.Exif()
    plain.get_ifd(0x8769)[0x927C] = b"Not Apple" + UUID.encode()
    assert live.content_identifier(plain) is None
    assert live.content_identifier(Image.Exif()) is None
    assert live.video_content_identifier({"com.apple.quicktime.content.identifier": UUID.lower()}) == UUID
    assert live.video_content_identifier({"creation_time": "2026-01-01"}) is None
    assert live.stem_of("IMG_1234.HEIC") == "img_1234" == live.stem_of("IMG_1234.mov")


def motion_photo_bytes(with_xmp: bool = True) -> bytes:
    jpeg = jpeg_bytes(noise_image(320, 240, seed=5))
    xmp = (b'<x:xmpmeta><rdf:Description Item:Semantic="MotionPhoto" Item:Mime="video/mp4" '
           + b'Item:Length="' + str(len(FAKE_MP4)).encode() + b'"/></x:xmpmeta>')
    if not with_xmp:
        return jpeg + FAKE_MP4
    com = b"\xff\xfe" + struct.pack(">H", len(xmp) + 2) + xmp
    return jpeg[:2] + com + jpeg[2:] + FAKE_MP4


def test_extract_motion_clip(tmp_path):
    p = tmp_path / "motion.jpg"
    p.write_bytes(motion_photo_bytes())
    assert live.extract_motion_clip(p) == FAKE_MP4
    with Image.open(p) as im:  # still a readable JPEG
        assert im.size == (320, 240)
    p2 = tmp_path / "samsung.jpg"
    p2.write_bytes(jpeg_bytes(noise_image(320, 240, seed=6)) + b"SEFT" + b"MotionPhoto_Data" + FAKE_MP4)
    assert live.extract_motion_clip(p2) == FAKE_MP4
    p3 = tmp_path / "plain.jpg"
    p3.write_bytes(jpeg_bytes(noise_image(320, 240, seed=7)))
    assert live.extract_motion_clip(p3) is None


def test_find_partner_rules(conn):
    still = make_item(conn, filename="IMG_0001.HEIC", content_id=UUID, write_files=False)
    long_video = make_item(conn, kind="video", filename="IMG_0001.MOV", duration_s=12.0, write_files=False)
    assert live.find_partner(conn, still, None) is None  # too long to be a clip
    clip = make_item(conn, kind="video", filename="IMG_0001.MOV", duration_s=2.9, write_files=False)
    assert live.find_partner(conn, still, None)["id"] == clip["id"]       # same stem, short clip
    other = make_item(conn, kind="video", filename="VID_9.mov", duration_s=2.0, content_id=UUID, write_files=False)
    assert live.find_partner(conn, still, None)["id"] == other["id"]      # the content identifier wins
    assert live.find_partner(conn, long_video, None) is None
    assert live.find_partner(conn, clip, None)["id"] == still["id"]


def test_live_photo_pair_through_the_api(client, tmp_path):
    still = jpeg_bytes(noise_image(640, 480, seed=8))
    clip_path = tmp_path / "IMG_0002.mov"
    make_video(clip_path, seconds=2)
    r = client.post("/api/uploads/simple", files={"file": ("IMG_0002.JPG", io.BytesIO(still))}, headers=H)
    assert r.status_code == 201, r.text
    r = client.post("/api/uploads/simple", files={"file": ("IMG_0002.MOV", clip_path.read_bytes())}, headers=H)
    assert r.status_code == 201, r.text
    run_jobs()
    items = client.get("/api/items", params={"misclicks": "include"}).json()["items"]
    assert [it["filename"] for it in items] == ["IMG_0002.JPG"]   # the clip is a hidden companion
    card = items[0]
    assert card["live"] is True
    det = client.get(f"/api/items/{card['uid']}").json()
    assert det["live_video"] and det["live_video"]["uid"] and det["urls"]["live"] == det["live_video"]["url"]
    comp_uid = det["live_video"]["uid"]
    assert client.get(f"/files/video/{comp_uid}").status_code == 200
    assert client.get(f"/files/original/{comp_uid}", params={"download": 1}).status_code == 200
    assert client.get("/api/stats").json()["by_kind"]["video"] == 0
    assert client.get("/api/search", params={"q": "IMG_0002", "mode": "text"}).json()["total"] == 1
    assert [it["filename"] for it in client.get("/api/items", params={"live": 1}).json()["items"]] == ["IMG_0002.JPG"]
    # the trash and the vault carry the companion along
    assert client.delete(f"/api/items/{card['uid']}", headers=H).status_code == 200
    with db.connect() as conn:
        comp = conn.execute("SELECT deleted_at, companion_of FROM items WHERE uid = ?", (comp_uid,)).fetchone()
    assert comp["deleted_at"] and comp["companion_of"]
    assert client.post(f"/api/items/{card['uid']}/restore", headers=H).status_code == 200
    with db.connect() as conn:
        assert conn.execute("SELECT deleted_at FROM items WHERE uid = ?", (comp_uid,)).fetchone()["deleted_at"] is None
    assert client.patch(f"/api/items/{card['uid']}", json={"nsfw": True}, headers=H).status_code == 200
    with db.connect() as conn:
        assert conn.execute("SELECT nsfw FROM items WHERE uid = ?", (comp_uid,)).fetchone()["nsfw"] == 1
    assert client.get(f"/files/video/{comp_uid}").status_code == 403  # vault locked: the clip is private too


def test_pair_key_pairs_files_with_different_names(client, tmp_path):
    still = jpeg_bytes(noise_image(320, 240, seed=9))
    clip_path = tmp_path / "clip.mp4"
    make_video(clip_path, seconds=2)
    client.post("/api/uploads/simple", files={"file": ("photo.jpg", io.BytesIO(still))}, data={"pair_key": "asset-1"}, headers=H)
    client.post("/api/uploads/simple", files={"file": ("movie.mp4", clip_path.read_bytes())}, data={"pair_key": "asset-1"}, headers=H)
    run_jobs()
    items = client.get("/api/items").json()["items"]
    assert [it["filename"] for it in items] == ["photo.jpg"] and items[0]["live"] is True


def test_motion_photo_through_the_api(client):
    data = motion_photo_bytes()
    r = client.post("/api/uploads/simple", files={"file": ("PXL_0003.jpg", io.BytesIO(data))}, headers=H)
    assert r.status_code == 201, r.text
    run_jobs()
    card = client.get("/api/items").json()["items"][0]
    assert card["live"] is True
    det = client.get(f"/api/items/{card['uid']}").json()
    assert det["live_video"] == {"uid": None, "url": f"/files/live/{card['uid']}?v=1", "duration_s": None}
    r = client.get(det["live_video"]["url"])
    assert r.status_code == 200 and r.content == FAKE_MP4
    r = client.get(f"/files/live/{card['uid']}", params={"v": 1, "download": 1})
    assert r.headers["content-disposition"].startswith("attachment")
    # the original stays byte for byte
    assert client.get(det["urls"]["original"]).content == data
