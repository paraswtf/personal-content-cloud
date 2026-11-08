"""Videos with a placeholder index before the real one (docs/ARCHITECTURE.md section 7): some Apple
encoders write a 'hoov' atom holding an 'mvhd' before the media data. ffmpeg's lenient default reads it,
finds nothing to decode and the clip used to be kept as a plain file. With probe.input_flags every ffmpeg
and ffprobe call reads the real index; plain files kept before are probed again once (PROBE_VERSION)."""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from tests.helpers import jpeg_bytes, make_item, noise_image, run_jobs, upload_bytes


def hoov_clip(dest: Path, seconds: float = 2.0) -> bytes:
    """A small H.264 QuickTime clip laid out [ftyp qt][wide][mdat][hoov: mvhd][moov]: the placeholder
    comes before the real index, as in the clips that failed (which sniff as video/quicktime, .mov).
    Inserting it after the mdat keeps the chunk offsets valid."""
    from app.media.probe import _top_atoms

    plain = dest.with_suffix(".plain.mov")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=25", "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(plain)], check=True, timeout=120)
    data = plain.read_bytes()
    moov = next(a for a in _top_atoms(plain) if a[0] == b"moov")
    body = data[moov[1]: moov[1] + moov[2]]
    assert body[12:16] == b"mvhd"
    mvhd = body[8: 8 + struct.unpack(">I", body[8:12])[0]]
    out = data[:moov[1]] + struct.pack(">I4s", 8 + len(mvhd), b"hoov") + mvhd + data[moov[1]:]
    dest.write_bytes(out)
    plain.unlink()
    return out


def test_input_flags_only_for_a_placeholder_index(tmp_path):
    from app.media import probe

    bad = tmp_path / "bad.mov"
    hoov_clip(bad)
    assert probe.input_flags(bad) == ["-strict", "strict"]
    good = tmp_path / "good.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120",
                    "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(good)], check=True, timeout=120)
    assert probe.input_flags(good) == []
    other = tmp_path / "x.jpg"
    other.write_bytes(jpeg_bytes(noise_image(64, 48)))
    assert probe.input_flags(other) == [] and probe.input_flags(tmp_path / "missing.mov") == []
    lenient = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(bad), "-map", "0:V:0",
                              "-frames:v", "1", "-f", "framemd5", "-"], capture_output=True, timeout=60)
    assert lenient.returncode != 0  # the failure this works around
    info = probe.probe(bad, "IMG_0001.MOV")
    assert info.kind == "video" and info.width == 320 and info.duration_s == pytest.approx(2.0, abs=0.1)
    assert info.strict_input is True
    from app.media import video

    assert video.needs_web_preview(info, info.ext) is True
    assert video.extract_frame_image(bad, 0.5, 160) is not None


def test_a_resource_error_is_a_tool_failure(tmp_path, monkeypatch):
    from app.media import probe

    clip = tmp_path / "c.mov"
    hoov_clip(clip, seconds=1.0)

    class Res:
        returncode = 1
        stdout = b""
        stderr = b"[h264 @ 0x1] Cannot allocate memory\n"

    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: Res())
    with pytest.raises(probe.ToolFailure):
        probe._decodes_one_frame(clip)
    Res.stderr = b"Invalid data found when processing input\n"
    assert probe._decodes_one_frame(clip) is False


def test_an_upload_with_a_placeholder_index_is_stored_as_a_video(client, conn, tmp_path):
    from app import storage

    data = hoov_clip(tmp_path / "IMG_0002.MOV")
    up = upload_bytes(client, data, "IMG_0002.MOV")
    assert up["status"] == "queued", up
    run_jobs()
    row = conn.execute("SELECT * FROM items").fetchone()
    assert row["kind"] == "video" and row["width"] == 320 and row["preview_status"] == "done", dict(row)
    assert storage.thumb_path(row["uid"], 1).is_file()
    assert storage.video_preview_path(row["uid"], 1).is_file()  # a web copy, never the original as it is
    assert storage.original_path(row["uid"], 1, row["ext"]).read_bytes() == data  # the original is untouched


def test_reprobe_rebuilds_a_plain_clip_and_pairs_it_with_its_still(conn, tmp_path):
    from app import ingest, storage

    still = make_item(conn, filename="IMG_0003.JPG")
    data = hoov_clip(tmp_path / "IMG_0003.MOV")
    clip = make_item(conn, kind="other", ext=".mov", mime="video/quicktime", data=data, filename="IMG_0003.MOV",
                     width=None, height=None, analysis_status="done")
    junk = make_item(conn, kind="other", ext=".jpg", mime="image/jpeg", data=b"\xff\xd8\xff\xe0 not a photo",
                     filename="broken.jpg", width=None, height=None, analysis_status="done")
    assert ingest.plain_media_count(conn) == 2

    out = ingest.reprobe_plain(conn)
    assert out == {"checked": 2, "fixed": 1, "complete": 1}
    row = conn.execute("SELECT * FROM items WHERE id = ?", (clip["id"],)).fetchone()
    assert row["kind"] == "video" and row["width"] == 320 and row["duration_s"] == pytest.approx(2.0, abs=0.1)
    assert row["preview_status"] == "pending" and row["phash"]
    assert storage.thumb_path(row["uid"], 1).is_file()
    assert row["companion_of"] == still["id"]  # the Live Photo pair, found by the file stem
    s = conn.execute("SELECT live, live_video_item_id FROM items WHERE id = ?", (still["id"],)).fetchone()
    assert s["live"] == 1 and s["live_video_item_id"] == clip["id"]
    types = {r["type"] for r in conn.execute("SELECT type FROM jobs WHERE json_extract(payload, '$.item_id') = ?",
                                              (clip["id"],))}
    assert "video_preview" in types
    assert conn.execute("SELECT kind FROM items WHERE id = ?", (junk["id"],)).fetchone()["kind"] == "other"
    assert ingest.plain_media_count(conn) == 1
    assert ingest.reprobe_plain(conn) == {"checked": 1, "fixed": 0, "complete": 1}


def test_startup_reprobes_once_per_probe_version(conn, tmp_path):
    from app import ingest, worker
    from app.db import kv_get
    from app.media.probe import PROBE_VERSION

    data = hoov_clip(tmp_path / "IMG_0004.MOV")
    clip = make_item(conn, kind="other", ext=".mov", mime="video/quicktime", data=data, filename="IMG_0004.MOV",
                     width=None, height=None, analysis_status="done")
    ingest.recover_on_startup(conn)
    worker.wait_background(60)
    assert conn.execute("SELECT kind FROM items WHERE id = ?", (clip["id"],)).fetchone()["kind"] == "video"
    assert kv_get(conn, ingest.PROBE_VERSION_KEY) == str(PROBE_VERSION)
