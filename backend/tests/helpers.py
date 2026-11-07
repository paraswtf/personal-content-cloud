"""Test helpers: drive the upload protocol, run jobs, make synthetic files, insert rows directly."""
from __future__ import annotations

import hashlib
import io
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app import storage
from app.db import next_item_id, tx
from app.util import new_uid, now_iso

H = {"X-Requested-With": "pcc"}


# ------------------------------------------------------------------ upload protocol
def upload_bytes(client, data: bytes, filename: str, client_id: str = "test-client", send_hash: bool = False,
                 source: str = "web") -> dict:
    """create -> chunks -> complete. Returns the final UploadOut JSON (status queued/duplicate/...)."""
    body = {"filename": filename, "size": len(data), "client_id": client_id, "source": source}
    if send_hash:
        body["sha256"] = hashlib.sha256(data).hexdigest()
    r = client.post("/api/uploads", json=body, headers=H)
    assert r.status_code in (200, 201), r.text
    up = r.json()
    if up["status"] != "uploading":
        return up
    chunk = up["chunk_size"]
    offset = 0
    while offset < len(data):
        part = data[offset: offset + chunk]
        r = client.put(f"/api/uploads/{up['id']}/chunk", params={"offset": offset}, content=part,
                       headers={**H, "Content-Type": "application/octet-stream"})
        assert r.status_code == 200, r.text
        offset = r.json()["received_bytes"]
    r = client.post(f"/api/uploads/{up['id']}/complete", headers=H)
    assert r.status_code == 200, r.text
    return r.json()


def upload_file(client, path: Path, filename: str | None = None, **kw) -> dict:
    return upload_bytes(client, Path(path).read_bytes(), filename or Path(path).name, **kw)


def run_jobs(queue: str | None = None) -> int:
    from app.worker import drain

    return drain(queue)


def upload_status(client, upload_id: str) -> dict:
    r = client.get(f"/api/uploads/{upload_id}")
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ makers
def jpeg_bytes(img: Image.Image, quality: int = 90, exif: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    kw = {"quality": quality}
    if exif:
        kw["exif"] = exif
    img.convert("RGB").save(buf, "JPEG", **kw)
    return buf.getvalue()


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def noise_image(w: int = 640, h: int = 480, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


def flat_image(w: int = 640, h: int = 480, color=(200, 160, 120)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def text_image(lines: list[str], w: int = 1000, h: int = 640, size: int = 28) -> Image.Image:
    """A white page with black text lines (for OCR and document parser tests)."""
    from PIL import ImageFont

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default(size)
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += int(size * 1.6)
    return img


DEVANAGARI_FONT = Path(__file__).resolve().parents[2] / "models" / "fonts" / "NotoSansDevanagari-Regular.ttf"
_LATIN_FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf")


def devanagari_font_available() -> bool:
    return DEVANAGARI_FONT.is_file()


def devanagari_text_image(lines: list[str], w: int = 1100, h: int | None = None, size: int = 30) -> Image.Image:
    """A white page with lines that mix Devanagari (Hindi, Marathi) and Latin text. Devanagari runs are
    drawn with Noto Sans Devanagari (shaped with raqm when Pillow has it, so vowel signs and conjuncts
    join), other runs with DejaVu Sans. Raises FileNotFoundError when the Noto font is missing; tests
    check devanagari_font_available() first and skip."""
    import re

    from PIL import ImageFont, features

    if not devanagari_font_available():
        raise FileNotFoundError(DEVANAGARI_FONT)
    engine = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    dev = ImageFont.truetype(str(DEVANAGARI_FONT), size, layout_engine=engine)
    latin = None
    for candidate in _LATIN_FONTS:
        try:
            latin = ImageFont.truetype(candidate, size, layout_engine=engine)
            break
        except OSError:
            continue
    latin = latin or dev
    h = h or 60 + int(size * 1.7) * len(lines)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    y = 30
    for line in lines:
        x = 40.0
        # runs of Devanagari (with the spaces between its words) and of everything else
        for run in re.findall(r"[\u0900-\u097F\u200c\u200d][\u0900-\u097F\u200c\u200d ]*|[^\u0900-\u097F\u200c\u200d]+",
                              line):
            font = dev if re.match(r"[\u0900-\u097F]", run) else latin
            draw.text((x, y), run, fill="black", font=font)
            x += font.getlength(run)
        y += int(size * 1.7)
    return img


def pdf_bytes(pages: list[list[str]]) -> bytes:
    """A small PDF with a real text layer: one page per list of lines (Helvetica, no images)."""
    objs: list[bytes] = []

    def add(obj: bytes) -> int:
        objs.append(obj)
        return len(objs)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    pages_id_placeholder = len(objs) + 1 + 2 * len(pages)  # pages object comes after all page+content objects
    for lines in pages:
        content = b"BT /F1 14 Tf 50 750 Td 18 TL\n"
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content += f"({safe}) Tj T*\n".encode("latin-1", "replace")
        content += b"ET"
        c_id = add(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
        p_id = add(
            f"<< /Type /Page /Parent {pages_id_placeholder} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {c_id} 0 R >>".encode()
        )
        page_ids.append(p_id)
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    pages_id = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    assert pages_id == pages_id_placeholder
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_video(path: Path, *, source: str = "testsrc2", size: str = "640x360", seconds: float = 4,
               fps: int = 25, crf: int = 23, codec: str = "libx264", image: Path | None = None) -> Path:
    """Small synthetic video via ffmpeg (lavfi source, or a still image looped)."""
    if image is not None:
        inp = ["-loop", "1", "-framerate", str(fps), "-i", str(image)]
        vf = ["-vf", f"scale={size.replace('x', ':')}:force_original_aspect_ratio=decrease,"
              f"pad={size.replace('x', ':')}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"]
    else:
        inp = ["-f", "lavfi", "-i", f"{source}=size={size}:rate={fps}"]
        vf = ["-vf", "format=yuv420p"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inp, "-t", str(seconds), *vf,
           "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


# ------------------------------------------------------------------ row factories
def make_item(conn: sqlite3.Connection, *, kind: str = "image", data: bytes | None = None, ext: str = ".jpg",
              mime: str = "image/jpeg", width: int = 640, height: int = 480, taken_at: str | None = None,
              category: str = "misc", nsfw: int = 0, misclick: int = 0, filename: str | None = None,
              analysis_status: str = "done", write_files: bool = True, **extra) -> sqlite3.Row:
    """Insert an item row (and, by default, its original + a thumb/preview) without the pipeline."""
    uid = new_uid()
    if data is None and write_files:
        data = jpeg_bytes(noise_image(width, height, seed=hash(uid) & 0xFFFF)) if kind == "image" else b"x" * 100
    sha = hashlib.sha256(data if data is not None else uid.encode()).hexdigest()
    now = now_iso()
    cols = dict(
        uid=uid, kind=kind, version=1, original_filename=filename or f"{uid}{ext}", ext=ext, mime=mime,
        sha256=sha, size_bytes=len(data) if data is not None else 0, width=width, height=height,
        taken_at=taken_at, category=category, nsfw=nsfw, misclick=misclick, analysis_status=analysis_status,
        created_at=now, updated_at=now,
    )
    cols.update(extra)
    with tx(conn):
        cols["id"] = next_item_id(conn)
        keys = ", ".join(cols)
        conn.execute(f"INSERT INTO items({keys}) VALUES ({', '.join('?' * len(cols))})", tuple(cols.values()))
    if write_files and data is not None:
        storage.write_bytes_atomic(storage.original_path(uid, 1, ext), data)
        if kind == "image":
            img = Image.open(io.BytesIO(data)).convert("RGB")
            for p in (storage.thumb_path(uid, 1), storage.preview_path(uid, 1), storage.preview_small_path(uid, 1)):
                p.parent.mkdir(parents=True, exist_ok=True)
                img.save(p, "WEBP", quality=70)
    return conn.execute("SELECT * FROM items WHERE id = ?", (cols["id"],)).fetchone()


def make_person(conn: sqlite3.Connection, name: str | None = None) -> int:
    from app.util import name_key

    now = now_iso()
    with tx(conn):
        cur = conn.execute("INSERT INTO persons(name, name_key, created_at, updated_at) VALUES (?, ?, ?, ?)",
                           (name, name_key(name) if name else None, now, now))
        return int(cur.lastrowid)


def make_face(conn: sqlite3.Connection, item_id: int, person_id: int | None = None, *, embedding=None,
              box=(0.3, 0.2, 0.2, 0.3), det_score: float = 0.9, size_px: int = 120, assigned_by: str = "auto",
              torso=None, write_crop: bool = True, **extra) -> int:
    rng = np.random.default_rng(item_id * 7919 + (person_id or 0))
    emb = embedding if embedding is not None else rng.standard_normal(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    cols = dict(item_id=item_id, person_id=person_id, assigned_by=assigned_by, x=box[0], y=box[1], w=box[2], h=box[3],
                det_score=det_score, size_px=size_px, quality=det_score * min(size_px, 160) / 160,
                embedding=emb.astype(np.float32).tobytes(), created_at=now_iso())
    if torso is not None:
        t = np.asarray(torso, dtype=np.float32)
        cols["torso_embedding"] = (t / np.linalg.norm(t)).astype(np.float32).tobytes()
    cols.update(extra)
    with tx(conn):
        cur = conn.execute(f"INSERT INTO faces({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                           tuple(cols.values()))
        face_id = int(cur.lastrowid)
        conn.execute("UPDATE items SET face_count = (SELECT COUNT(*) FROM faces WHERE item_id = ?) WHERE id = ?",
                     (item_id, item_id))
        from app.db import bump_faces_version

        bump_faces_version(conn)
    if write_crop:
        p = storage.face_crop_path(face_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (192, 192), (120, 120, 120)).save(p, "WEBP")
    return face_id
