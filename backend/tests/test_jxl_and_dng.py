"""JPEG XL photos and the DNGs LibRaw cannot open (section 7, 2026-11-09): recognised, probed and decoded like
other photos; a stored one kept as a plain file is probed again; an iPhone ProRAW DNG whose raw image is JPEG XL
compressed is read from its embedded full-size JPEG, turned upright once."""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
from PIL import Image

from app.media import images as images_mod  # noqa: F401 - registers the JPEG XL opener
from app.media import probe as probe_mod
from app.media import raw as raw_mod
from tests.helpers import jpeg_bytes, make_item, noise_image, run_jobs, upload_bytes, upload_status

JXL_CONTAINER_HEAD = b"\x00\x00\x00\x0cJXL \r\n\x87\n"


def _jxl(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JXL")
    return buf.getvalue()


def _dng(preview: bytes, width: int, height: int, orientation: int) -> bytes:
    """A big-endian DNG laid out like an iPhone ProRAW one: IFD0 holds a full-size JPEG preview (YCbCr, one strip)
    and the file's orientation, and its SubIFD the raw image, JPEG XL compressed (52546) in tiles."""
    entries0: list[tuple[int, int, int, bytes]] = []   # (tag, type, count, value bytes)

    def short(v: int) -> bytes:
        return struct.pack(">HH", v, 0)

    def long(v: int) -> bytes:
        return struct.pack(">I", v)

    raw_tile = b"\xff\x0a" + b"\x00" * 62
    # layout: header 8 | IFD0 | SubIFD | data area
    n0, n1 = 15, 11
    ifd0_at = 8
    sub_at = ifd0_at + 2 + 12 * n0 + 4
    data_at = sub_at + 2 + 12 * n1 + 4
    blobs = bytearray()

    def blob(b: bytes) -> int:
        off = data_at + len(blobs)
        blobs.extend(b)
        if len(blobs) % 2:
            blobs.extend(b"\x00")
        return off

    bps = blob(struct.pack(">HHH", 8, 8, 8))
    make = blob(b"Apple\x00")
    model = blob(b"iPhone 18 Pro\x00")
    jpeg_at = blob(preview)
    tile_at = blob(raw_tile)
    bps_raw = blob(struct.pack(">HHH", 10, 10, 10))
    entries0 = [
        (254, 4, 1, long(1)), (256, 4, 1, long(width)), (257, 4, 1, long(height)), (258, 3, 3, long(bps)),
        (259, 3, 1, short(7)), (262, 3, 1, short(6)), (271, 2, 6, long(make)), (272, 2, 14, long(model)),
        (273, 4, 1, long(jpeg_at)), (274, 3, 1, short(orientation)), (277, 3, 1, short(3)),
        (279, 4, 1, long(len(preview))), (284, 3, 1, short(1)), (330, 4, 1, long(sub_at)),
        (50706, 1, 4, bytes([1, 7, 0, 0])),
    ]
    entries1 = [
        (254, 4, 1, long(0)), (256, 4, 1, long(width * 2)), (257, 4, 1, long(height * 2)), (258, 3, 3, long(bps_raw)),
        (259, 3, 1, short(52546)), (262, 3, 1, short(34892)), (277, 3, 1, short(3)), (322, 4, 1, long(256)),
        (323, 4, 1, long(256)), (324, 4, 1, long(tile_at)), (325, 4, 1, long(len(raw_tile))),
    ]
    assert len(entries0) == n0 and len(entries1) == n1

    def ifd(entries) -> bytes:
        out = struct.pack(">H", len(entries))
        for tag, typ, cnt, val in entries:
            out += struct.pack(">HHI", tag, typ, cnt) + val
        return out + struct.pack(">I", 0)

    return b"MM\x00\x2a" + struct.pack(">I", ifd0_at) + ifd(entries0) + ifd(entries1) + bytes(blobs)


def _libraw_cannot(monkeypatch) -> None:
    """LibRaw as built without JPEG XL: it cannot open a ProRAW DNG at all."""
    import rawpy

    def imread(path):
        raise rawpy.LibRawFileUnsupportedError(b"Unsupported file format or not RAW file")

    monkeypatch.setattr(rawpy, "imread", imread)


# ------------------------------------------------------------------ JPEG XL
def test_jpeg_xl_is_recognised_both_ways(tmp_path):
    codestream = _jxl(noise_image(64, 48))
    assert codestream[:2] == b"\xff\x0a"
    p = tmp_path / "a.bin"
    p.write_bytes(codestream)
    assert probe_mod.sniff(p) == ("image", "image/jxl", ".jxl")
    p.write_bytes(JXL_CONTAINER_HEAD + b"\x00\x00\x00\x14ftypjxl \x00\x00\x00\x00jxl " + codestream)
    assert probe_mod.sniff(p) == ("image", "image/jxl", ".jxl")
    assert ".jxl" in probe_mod.ACCEPT_ATTR


def test_a_jpeg_xl_photo_is_probed_decoded_and_uploaded(client, conn, tmp_path):
    data = _jxl(noise_image(320, 240, seed=5))
    p = tmp_path / "IMG_5785.JXL"
    p.write_bytes(data)
    info = probe_mod.probe(p, "IMG_5785.JXL")
    assert (info.kind, info.mime, info.ext, info.width, info.height) == ("image", "image/jxl", ".jxl", 320, 240)
    assert images_mod.load_oriented(p).size == (320, 240)
    up = upload_bytes(client, data, "IMG_5785.JXL")
    for _ in range(3):
        run_jobs()
    st = upload_status(client, up["id"])
    item = client.get(f"/api/items/{st['item_uid']}").json()
    assert (item["kind"], item["mime"], item["width"], item["height"]) == ("image", "image/jxl", 320, 240)
    assert client.get(item["urls"]["thumb"]).status_code == 200
    assert client.get(item["urls"]["preview"]).status_code == 200


def test_a_stored_jpeg_xl_kept_as_a_plain_file_becomes_a_photo(conn):
    from app import ingest

    item = make_item(conn, kind="other", ext=".jxl", mime="application/octet-stream", data=_jxl(noise_image(200, 150)),
                     filename="IMG_5785.JXL", width=None, height=None, analysis_status="done")
    assert ingest.plain_media_count(conn) == 1
    assert ingest.reprobe_plain(conn) == {"checked": 1, "fixed": 1, "complete": 1}
    row = conn.execute("SELECT kind, mime, width, height FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert tuple(row) == ("image", "image/jxl", 200, 150)


# ------------------------------------------------------------------ DNG LibRaw cannot open
def test_a_prores_dng_reads_from_its_embedded_preview_upright(tmp_path, monkeypatch):
    _libraw_cannot(monkeypatch)
    preview = jpeg_bytes(noise_image(800, 600, seed=9))       # stored sideways, as a camera does
    p = tmp_path / "IMG_0012.dng"
    p.write_bytes(_dng(preview, 800, 600, orientation=6))
    assert raw_mod.sniff_raw(p.read_bytes()[:4096], "IMG_0012.dng") == ("image", "image/x-adobe-dng", ".dng")
    width, height, exif = raw_mod.probe_raw(p, ("image", "image/x-adobe-dng", ".dng"))
    assert (width, height) == (600, 800) and exif.get(271) == "Apple"   # turned upright once
    img = raw_mod.load_raw(p, max_side=200)
    assert img.size == (150, 200) and img.mode == "RGB"
    info = probe_mod.probe(p, "IMG_0012.dng")
    assert (info.kind, info.width, info.height, info.raw) == ("image", 600, 800, True)


def test_a_dng_without_a_usable_preview_still_fails_cleanly(tmp_path, monkeypatch):
    _libraw_cannot(monkeypatch)
    tiny = jpeg_bytes(noise_image(160, 120, seed=3))            # a thumbnail, not a picture
    p = tmp_path / "IMG_0099.dng"
    p.write_bytes(_dng(tiny, 160, 120, orientation=1))
    assert raw_mod.embedded_preview(p) is None
    with pytest.raises(ValueError):
        raw_mod.probe_raw(p, ("image", "image/x-adobe-dng", ".dng"))
    with pytest.raises(ValueError):
        raw_mod.load_raw(p)


def test_a_stored_dng_kept_as_a_plain_file_becomes_a_photo(conn, monkeypatch):
    from app import ingest

    _libraw_cannot(monkeypatch)
    data = _dng(jpeg_bytes(noise_image(800, 600, seed=4)), 800, 600, orientation=1)
    item = make_item(conn, kind="other", ext=".dng", mime="image/x-adobe-dng", data=data, filename="IMG_0010.dng",
                     width=None, height=None, analysis_status="done")
    assert ingest.reprobe_plain(conn) == {"checked": 1, "fixed": 1, "complete": 1}
    row = conn.execute("SELECT kind, width, height, analysis_status FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert (row["kind"], row["width"], row["height"]) == ("image", 800, 600) and row["analysis_status"] == "pending"
    assert np.asarray(Image.open(io.BytesIO(data))).shape[:2] in ((600, 800), (800, 600))   # Pillow reads it too
