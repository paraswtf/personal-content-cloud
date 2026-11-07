"""Model-backed text stage tests: OCR of a rendered PAN card and the ID photo crop (skipped without the
models or with PCC_NO_MODELS=1). Run in one pytest process at a time, ORT_THREADS=2."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from app import storage
from app.documents import extract
from app.util import json_loads
from tests.conftest import skip_without_models
from tests.helpers import make_item

FIX = Path(__file__).parent / "fixtures"


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_ocr_pan_card(conn):
    skip_without_models("ocr")
    data = (FIX / "documents" / "pan_card.png").read_bytes()
    item = make_item(conn, kind="image", data=data, ext=".png", mime="image/png", width=1000, height=640,
                     text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    page = conn.execute("SELECT * FROM pages WHERE item_id = ?", (item["id"],)).fetchone()
    assert page["text_source"] == "ocr"
    boxes = json_loads(page["ocr_boxes"])
    assert boxes and all(0 <= b[0] <= b[2] <= 1 and 0 <= b[1] <= b[3] <= 1 for b in boxes)
    assert "ABCPS1234K" in page["text"].replace(" ", "")
    assert conn.execute("SELECT source FROM item_text WHERE item_id = ?", (item["id"],)).fetchone()[0] == "ocr"
    doc = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item["id"],)).fetchone()
    assert doc["doc_type"] == "pan"
    f = json_loads(doc["fields"])
    assert f["number"]["value"] == "ABCPS1234K"
    assert f["dob"]["value"] == "1988-08-15"
    assert doc["person_name"] == "Rahul Sharma"


def test_id_photo_crop(conn):
    skip_without_models("ocr", "faces")
    card = Image.open(FIX / "documents" / "pan_card.png").convert("RGB")
    face = Image.open(FIX / "armstrong.jpg").convert("RGB")
    face.thumbnail((220, 280))
    card.paste(face, (card.width - face.width - 30, 140))
    item = make_item(conn, kind="image", data=_png(card), ext=".png", mime="image/png", width=card.width,
                     height=card.height, text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    doc = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item["id"],)).fetchone()
    assert doc["doc_type"] == "pan" and doc["has_photo"] == 1
    p = storage.doc_photo_path(item["uid"], 1)
    assert p.exists()
    with Image.open(p) as im:
        assert im.size == (320, 320)


def test_ocr_boarding_pass_text_and_barcode(conn):
    skip_without_models("ocr")
    data = (FIX / "documents" / "boarding_pass.png").read_bytes()
    item = make_item(conn, kind="image", data=data, ext=".png", mime="image/png", width=900, height=520,
                     text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    page = conn.execute("SELECT * FROM pages WHERE item_id = ?", (item["id"],)).fetchone()
    assert page["text_source"] == "ocr" and "BOARDING PASS" in page["text"].upper()
    doc = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item["id"],)).fetchone()
    assert doc["doc_type"] == "boarding_pass" and doc["barcode_format"] == "PDF417"
    f = json_loads(doc["fields"])
    assert f["pnr"]["value"] == "ABC123" and f["gate"]["value"] == "23" and f["seat"]["value"] == "12A"
    assert f["passenger"]["value"] == "Rahul Kumar Sharma"
