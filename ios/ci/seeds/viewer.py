"""The viewer area's items for its screen tests (ios/UITests/ViewerScreens.swift), all dated April 2026 and named
after the area, so they sit at the end of the library and touch nothing the other areas use:

- a two-page boarding pass PDF ("Viewer boarding pass.pdf", 20 Apr): its pages rendered and its text read as the
  pipeline does, then its virtual document, a boarding pass with one hard-to-read field and a barcode;
- a photo received on WhatsApp and kept in the library ("VIEWER_RECEIVED.JPG", 18 Apr), with a place, category and
  quality scores, labels, and the tag "Viewer trip" (added in `api`, as the owner would);
- a short playable video ("VIEWER_CLIP.MP4", 12 Apr, made with ffmpeg when it is there, else with OpenCV) with a
  face of "Viewer Guest" at one second;
- a second short video with nobody on it ("VIEWER_DRAW.MP4", 11 Apr), where the test draws a person box on the
  paused frame;
- a file the server keeps without a preview ("VIEWER_NOTES.BIN", 10 Apr).

A video whose streaming copy is still being made is not seeded: the server queues the copy again when it starts,
so the "preparing" state would not last until the tests.

Every item is marked as already checked for forwards (forward_checked), so the server's backfill does not decide
again how it was received (it would take "WhatsApp" off the photo, whose name is not WhatsApp's).

Everything is generated; no real people, no real photos.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

GUEST = "Viewer Guest"


def _previews(kit, item, img) -> None:
    """The item's thumb and previews from its picture, in folders the item helper only makes for photos."""
    for path, side in ((kit.storage.thumb_path(item["uid"], 1), 480), (kit.storage.preview_path(item["uid"], 1), 2048),
                       (kit.storage.preview_small_path(item["uid"], 1), 1280)):
        path.parent.mkdir(parents=True, exist_ok=True)
        copy = img.copy()
        copy.thumbnail((side, side))
        copy.save(path, "WEBP", quality=80)


def _clip(frame, tmp: Path):
    """Three seconds of `frame` as an MP4: (bytes, codec), or None when nothing here can write one."""
    clip = tmp / "clip.mp4"
    if shutil.which("ffmpeg"):
        still = tmp / "frame.png"
        frame.save(still)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "25",
                        "-i", str(still), "-t", "3", "-vf", "format=yuv420p", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(clip)], check=True, timeout=120)
        return clip.read_bytes(), "h264"
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    bgr = cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
    for fourcc, codec in (("avc1", "h264"), ("mp4v", "mpeg4")):
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*fourcc), 25, (frame.width, frame.height))
        if not writer.isOpened():
            continue
        for _ in range(75):
            writer.write(bgr)
        writer.release()
        if clip.is_file() and clip.stat().st_size > 1000:
            return clip.read_bytes(), codec
    return None


def library(kit) -> dict:
    from app.documents.extract import write_document
    from app.ingest import _regenerate_derivatives, enqueue_text
    from app.worker import drain
    from tests.helpers import pdf_bytes

    conn, out = kit.conn, {}

    # the boarding pass: pages and text as the pipeline makes them, then the parsed document
    data = pdf_bytes([
        ["BOARDING PASS", "Passenger VIEWER TRAVELLER", "Flight 6E 2041 Mumbai BOM to Goa GOI",
         "Date 20 Apr 2026   Seat 14C   Gate 22", "PNR VWR123"],
        ["Viewer test, page two", "Please be at the gate 25 minutes before departure."],
    ])
    pdf = kit.make_item(conn, kind="pdf", data=data, ext=".pdf", mime="application/pdf", width=612, height=792,
                        filename="Viewer boarding pass.pdf", taken_at="2026-04-20T08:30:00.000Z",
                        category="documents", page_count=2, forward_checked=1)
    _regenerate_derivatives(conn.execute("SELECT * FROM items WHERE id = ?", (pdf["id"],)).fetchone(), conn)
    enqueue_text(conn, int(pdf["id"]), 1)
    drain("text")
    fields = {
        "passenger": {"value": "Viewer Traveller", "label": "Passenger", "confidence": 0.95},
        "flight": {"value": "6E 2041", "label": "Flight", "confidence": 0.93},
        "from": {"value": "BOM", "label": "From", "confidence": 0.9},
        "to": {"value": "GOI", "label": "To", "confidence": 0.9},
        "seat": {"value": "14C", "label": "Seat", "confidence": 0.45},
        "pnr": {"value": "VWR123", "label": "PNR", "confidence": 0.97},
    }
    with kit.tx(conn):
        write_document(conn, int(pdf["id"]), {
            "doc_type": "boarding_pass", "doc_group": "pass", "confidence": 0.91, "fields": json.dumps(fields),
            "title": "Viewer flight to Goa", "doc_date": "2026-04-20", "issuer": "Viewer Air",
            "person_name": "Viewer Traveller", "barcode": "M1TRAVELLER/VIEWER EVWR123 BOMGOI6E 2041 110Y014C",
            "barcode_format": "PDF417", "source": "auto"})
        conn.execute("UPDATE items SET page_count = 2 WHERE id = ?", (pdf["id"],))
    out["pdf"] = pdf["uid"]

    # a photo received on WhatsApp, kept in the library
    img = kit.scene(8100, [])
    received = kit.make_item(
        conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename="VIEWER_RECEIVED.JPG",
        taken_at="2026-04-18T17:05:00.000Z", camera="CI Camera", category="places", received_via="whatsapp",
        gps_lat=15.4909, gps_lon=73.8278, forward_checked=1,
        category_scores=json.dumps({"places": 0.62, "misc": 0.21, "objects": 0.09, "food": 0.04}),
        quality_scores=json.dumps({"blur_var": 182.4, "mean_lum": 141.0, "p95_lum": 236.0, "clipped_frac": 0.01,
                                   "accident_p": 0.04}))
    _previews(kit, received, img)
    out["received"] = received["uid"]

    # a short playable video with a named face at one second (ffmpeg makes it when it is here, else OpenCV, which
    # the backend installs)
    drawn = kit.scene(8300, [((0.40, 0.25, 0.16, 0.24), (210, 170, 140), (40, 30, 30))])   # kit.W x kit.H
    frame = drawn.resize((640, 480))
    with tempfile.TemporaryDirectory() as tmp:
        made = _clip(frame, Path(tmp))
        if made is not None:
            data, codec = made
            video = kit.make_item(
                conn, kind="video", data=data, ext=".mp4", mime="video/mp4", width=640, height=480,
                filename="VIEWER_CLIP.MP4", taken_at="2026-04-12T11:00:00.000Z", category="personal",
                duration_s=3.0, video_codec=codec, bitrate=300_000, preview_status="none", forward_checked=1)
            _previews(kit, video, frame)
            guest = kit.make_person(conn, GUEST)
            face_id = kit.make_face(conn, int(video["id"]), guest, box=(0.40, 0.25, 0.16, 0.24), size_px=102,
                                    assigned_by="user", frame_time_s=1.0)
            # the crop helper works in kit.W x kit.H, the size the frame was drawn at
            kit.crop(drawn, (0.40, 0.25, 0.16, 0.24)).save(kit.storage.face_crop_path(face_id), "WEBP", quality=85)
            out["clip"] = video["uid"]
            out["clip_codec"] = codec

    # a second short video with nobody on it, to draw a person box on its paused frame
    empty = kit.scene(8310, []).resize((640, 480))
    with tempfile.TemporaryDirectory() as tmp:
        made = _clip(empty, Path(tmp))
        if made is not None:
            data, codec = made
            draw = kit.make_item(
                conn, kind="video", data=data, ext=".mp4", mime="video/mp4", width=640, height=480,
                filename="VIEWER_DRAW.MP4", taken_at="2026-04-11T11:00:00.000Z", category="misc",
                duration_s=3.0, video_codec=codec, bitrate=300_000, preview_status="none", forward_checked=1)
            _previews(kit, draw, empty)
            out["draw_clip"] = draw["uid"]

    # a file without a preview
    other = kit.make_item(conn, kind="other", data=b"viewer notes, not a picture\n" * 8, ext=".bin",
                          mime="application/octet-stream", width=None, height=None, filename="VIEWER_NOTES.BIN",
                          taken_at="2026-04-10T10:00:00.000Z", category="misc", forward_checked=1)
    out["other"] = other["uid"]
    return out


def api(kit) -> dict:
    """The owner tags the received photo, as they would on the website."""
    items = kit.client.call("GET", "/api/items?limit=200&from=2026-04-01&to=2026-04-30")["items"]
    received = next((it for it in items if it["filename"] == "VIEWER_RECEIVED.JPG"), None)
    if received is None:
        raise SystemExit("the viewer seed's received photo is not in the library")
    kit.client.call("POST", f"/api/items/{received['uid']}/user-tags", {"name": "Viewer trip"})
    return {"received": received["uid"], "items": len(items)}
