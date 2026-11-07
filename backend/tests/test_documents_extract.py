"""Text stage tests: extract_item on PDFs with a text layer, plain text and images (no OCR model needed),
the FTS row, user edits surviving a re-run, barcodes and the ID photo crop geometry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.documents import extract
from app.util import json_loads
from tests.helpers import make_face, make_item, make_person, pdf_bytes

FIX = Path(__file__).parent / "fixtures" / "documents"

INVOICE_LINES = [
    "ACME RETAIL PRIVATE LIMITED", "TAX INVOICE", "Invoice No: INV-2024-0042", "Invoice Date: 15-Mar-2024",
    "GSTIN: 27ABCDE1234F1Z0", "Bill To: Rahul Sharma", "Taxable Value: 1,500.00", "IGST 18%: 270.00",
    "Grand Total: Rs. 1,770.00",
]


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    monkeypatch.setenv("PCC_NO_MODELS", "1")


def _doc(conn, item_id):
    return conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()


def _match(conn, q):
    expr = extract.match_query(q)
    return [r[0] for r in conn.execute("SELECT rowid FROM items_fts WHERE items_fts MATCH ?", (expr,))]


def test_pdf_text_layer(conn):
    pytest.importorskip("app.media.documents")
    data = pdf_bytes([INVOICE_LINES, ["Terms: payment within 30 days", "Thank you for your business"]])
    item = make_item(conn, kind="pdf", data=data, ext=".pdf", mime="application/pdf", filename="acme-invoice.pdf",
                     page_count=2, text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert row["text_status"] == "done"
    pages = conn.execute("SELECT * FROM pages WHERE item_id = ? ORDER BY page_no", (item["id"],)).fetchall()
    assert [p["text_source"] for p in pages] == ["layer", "layer"]
    assert "INV-2024-0042" in pages[0]["text"]
    it = conn.execute("SELECT * FROM item_text WHERE item_id = ?", (item["id"],)).fetchone()
    assert it["source"] == "pdf" and "\f" in it["text"] and it["char_count"] == len(it["text"])
    doc = _doc(conn, item["id"])
    assert doc["doc_type"] == "invoice" and doc["doc_group"] == "money"
    fields = json_loads(doc["fields"])
    assert fields["invoice_number"]["value"] == "INV-2024-0042"
    assert fields["total"]["value"] == "\u20b91,770"
    assert doc["doc_date"] == "2024-03-15"
    assert list(fields)[0] == "invoice_number"  # type order
    assert _match(conn, "INV-2024-0042") == [item["id"]]
    assert _match(conn, "thank business") == [item["id"]]
    assert _match(conn, "acme invoice") == [item["id"]]


def test_plain_text_document(conn):
    text = "\n".join(INVOICE_LINES).encode()
    item = make_item(conn, kind="document", data=text, ext=".txt", mime="text/plain", filename="bill.txt",
                     page_count=1, text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    assert conn.execute("SELECT text_status FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == "done"
    it = conn.execute("SELECT * FROM item_text WHERE item_id = ?", (item["id"],)).fetchone()
    assert it["source"] == "plain"
    assert _doc(conn, item["id"])["doc_type"] == "invoice"
    page = conn.execute("SELECT * FROM pages WHERE item_id = ?", (item["id"],)).fetchone()
    assert page["text_source"] == "layer"


def test_short_text_gets_no_document(conn):
    item = make_item(conn, kind="document", data=b"hello there", ext=".txt", mime="text/plain", page_count=1,
                     text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    assert conn.execute("SELECT text_status FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == "done"
    assert _doc(conn, item["id"]) is None
    assert _match(conn, "hello") == [item["id"]]


def test_image_without_ocr_still_completes(conn):
    item = make_item(conn, kind="image", text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    row = conn.execute("SELECT text_status FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert row[0] == "done"
    page = conn.execute("SELECT * FROM pages WHERE item_id = ?", (item["id"],)).fetchone()
    assert page["text_source"] == "none" and page["width"] == 640
    assert _doc(conn, item["id"]) is None


def test_boarding_pass_image_from_barcode_alone(conn):
    data = (FIX / "boarding_pass.png").read_bytes()
    item = make_item(conn, kind="image", data=data, ext=".png", mime="image/png", width=900, height=520,
                     write_files=True, text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    doc = _doc(conn, item["id"])
    assert doc is not None and doc["doc_type"] == "boarding_pass" and doc["doc_group"] == "pass"
    assert doc["barcode_format"] == "PDF417" and doc["barcode"].startswith("M1SHARMA/RAHUL")
    f = json_loads(doc["fields"])
    assert f["pnr"]["value"] == "ABC123" and f["flight"]["value"] == "AI 865" and f["seat"]["value"] == "12A"
    assert doc["person_name"] == "Rahul Kumar Sharma"


def test_reparse_keeps_barcode_only_document(conn):
    data = (FIX / "boarding_pass.png").read_bytes()
    item = make_item(conn, kind="image", data=data, ext=".png", mime="image/png", width=900, height=520,
                     text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    assert extract.reparse(conn, item["id"])
    doc = _doc(conn, item["id"])
    assert doc is not None and doc["doc_type"] == "boarding_pass"
    assert json_loads(doc["fields"])["pnr"]["value"] == "ABC123"
    fields = json_loads(doc["fields"])
    fields["gate"] = {"value": "23", "label": "Gate", "confidence": 1.0, "source": "user"}
    conn.execute("UPDATE documents SET fields = ?, source = 'user' WHERE item_id = ?", (json.dumps(fields), item["id"]))
    assert extract.reparse(conn, item["id"])
    f = json_loads(_doc(conn, item["id"])["fields"])
    assert f["flight"]["value"] == "AI 865" and f["gate"]["value"] == "23" and f["gate"]["source"] == "user"


def test_stale_version_and_deleted_are_skipped(conn):
    item = make_item(conn, kind="document", data="\n".join(INVOICE_LINES).encode(), ext=".txt", mime="text/plain",
                     page_count=1, text_status="pending")
    extract.extract_item(conn, item["id"], 2)
    assert conn.execute("SELECT text_status FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == "pending"
    conn.execute("UPDATE items SET deleted_at = '2026-01-01T00:00:00.000Z' WHERE id = ?", (item["id"],))
    extract.extract_item(conn, item["id"], 1)
    assert _doc(conn, item["id"]) is None


def test_error_sets_status(conn, monkeypatch):
    item = make_item(conn, kind="document", data=b"x" * 50, ext=".txt", mime="text/plain", text_status="pending")

    def boom(_item):
        raise RuntimeError("broken file")

    monkeypatch.setattr(extract, "_collect", boom)
    extract.extract_item(conn, item["id"], 1)
    assert conn.execute("SELECT text_status FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == "error"


def test_user_fields_survive_rerun_and_reparse(conn):
    item = make_item(conn, kind="document", data="\n".join(INVOICE_LINES).encode(), ext=".txt", mime="text/plain",
                     page_count=1, text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    doc = _doc(conn, item["id"])
    fields = json_loads(doc["fields"])
    fields["total"] = {"value": "\u20b9999", "label": "Total", "confidence": 1.0, "source": "user"}
    conn.execute("UPDATE documents SET fields = ?, source = 'user', title = 'My bill' WHERE item_id = ?",
                 (json.dumps(fields), item["id"]))
    extract.extract_item(conn, item["id"], 1)
    doc = _doc(conn, item["id"])
    f = json_loads(doc["fields"])
    assert f["total"]["value"] == "\u20b9999" and f["total"]["source"] == "user"
    assert f["invoice_number"]["value"] == "INV-2024-0042"
    assert doc["title"] == "My bill" and doc["source"] == "user"
    assert extract.reparse(conn, item["id"])
    assert json_loads(_doc(conn, item["id"])["fields"])["total"]["value"] == "\u20b9999"
    # a forced type change regroups and keeps the user's field
    assert extract.reparse(conn, item["id"], forced_type="receipt")
    doc = _doc(conn, item["id"])
    assert doc["doc_type"] == "receipt" and doc["doc_group"] == "money" and doc["source"] == "user"
    f = json_loads(doc["fields"])
    assert f["total"]["source"] == "user" and "invoice_number" not in f


def test_user_row_kept_when_text_disappears(conn):
    item = make_item(conn, kind="document", data=b"tiny", ext=".txt", mime="text/plain", text_status="pending")
    now = "2026-01-01T00:00:00.000Z"
    conn.execute("INSERT INTO documents(item_id, doc_type, doc_group, confidence, fields, source, created_at, updated_at) "
                 "VALUES (?, 'pan', 'id', 1.0, '{}', 'user', ?, ?)", (item["id"], now, now))
    extract.extract_item(conn, item["id"], 1)
    assert _doc(conn, item["id"])["doc_type"] == "pan"


def test_refresh_fts_people_labels_and_remove(conn):
    item = make_item(conn, kind="image", filename="beach_day.jpg")
    pid = make_person(conn, "Asha Rao")
    make_face(conn, item["id"], pid)
    make_person(conn, None)
    conn.execute("INSERT INTO item_labels(item_id, label, score) VALUES (?, 'sunset', 0.9)", (item["id"],))
    extract.refresh_fts(conn, item["id"])
    assert _match(conn, "asha") == [item["id"]]
    assert _match(conn, "sunset") == [item["id"]]
    assert _match(conn, "beach") == [item["id"]]
    extract.refresh_fts(conn, item["id"])
    assert conn.execute("SELECT COUNT(*) FROM items_fts WHERE rowid = ?", (item["id"],)).fetchone()[0] == 1
    extract.remove_fts(conn, item["id"])
    assert _match(conn, "asha") == []


@pytest.mark.parametrize("q", ['S/O "', "-x AND OR NOT", "a*b:c^d", "NEAR(", '""', "col:val", "\u20b9 1,770"])
def test_match_query_is_safe(conn, q):
    expr = extract.match_query(q)
    if expr is None:
        return
    conn.execute("SELECT rowid FROM items_fts WHERE items_fts MATCH ?", (expr,)).fetchall()


def test_match_query_empty():
    assert extract.match_query("  ,,, ") is None
    assert extract.match_query('say "hi"') == '"say"* "hi"*'


def test_barcode_decode_fixtures():
    import cv2

    from app.documents import barcodes

    img = cv2.imread(str(FIX / "event_ticket.png"))
    assert barcodes.decode(img) == [("QRCode", '{"booking": "TKT8X2K91", "seat": "K-23", "gate": "5"}')]
    assert barcodes.decode(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0][0] == "QRCode"
    assert barcodes.decode(np.full((50, 50, 3), 255, np.uint8)) == []
    assert barcodes.decode(None) == []


def test_square_crop_geometry():
    from app.documents.photo import square_crop

    img = np.zeros((200, 300, 3), np.uint8)
    img[50:150, 100:180] = (0, 0, 255)  # red in BGR
    crop = square_crop(img, (100, 50, 180, 150))
    assert crop.size == (150, 150)  # 100 * 1.5
    # near the edge the square is padded with white
    edge = square_crop(img, (0, 0, 40, 40))
    assert edge.size == (60, 60) and edge.getpixel((0, 0)) == (255, 255, 255)


def test_boxes_text_column_gap():
    from app.documents.fields import Box

    boxes = [Box(1, 0.05, 0.10, 0.20, 0.13, "Name:"), Box(1, 0.21, 0.10, 0.40, 0.13, "Rahul Sharma"),
             Box(1, 0.70, 0.10, 0.85, 0.13, "Seat 12"), Box(1, 0.05, 0.20, 0.30, 0.23, "Gate 5")]
    assert extract.boxes_text(boxes) == "Name: Rahul Sharma   Seat 12\nGate 5"
