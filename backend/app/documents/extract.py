"""The 'text' stage (docs/ARCHITECTURE.md sections 7 and 11): text extraction, the full-text index and
the parsed "virtual document" of an item.

Contract:
- extract_item(conn, item_id, version) -> None. Idempotent; a stale version or a deleted item is skipped.
  Sources, per page: PDF text layer (app.media.documents.text_layer) -> OCR (app.ml.ocr) on page renders
  without a layer and on images (the oriented original, at most OCR_IMAGE_SIDE px) -> office logical pages
  -> plain text. Writes `pages` (text, text_source, ocr_boxes normalised 0..1), `item_text` (pages joined
  by form feeds), the FTS row, then detection + parsing into `documents` (barcodes from page renders and
  images, the ID photo for the 'id' group). Sets text_status 'done', or 'error' after logging an
  exception that is a verdict on the file (the job itself does not fail). Database errors, MemoryError
  and a full disk (also a failure to record 'error') are raised, so the worker retries or defers the job
  and its final failure marks text_status 'error'. The boarding pass parser gets the item's capture day
  (taken_at, else created_at) for the BCBP year. Respects settings.ocr_enabled and settings.ocr_max_pages.
  Without the OCR model the pages get text_source 'none' and the item still completes.
- reparse(conn, item_id, forced_type=None) -> bool: detection + parsing again on the stored text and OCR
  boxes (no OCR, no image work); keeps fields the user edited. False when the item has no stored text.
- refresh_fts(conn, item_id) / remove_fts(conn, item_id): rewrite / drop the items_fts row (filename, text,
  doc = title + type label + field values, people = named persons on faces, bodies and tags, labels =
  item_labels). Works for any item, with or without text or a document.
- fold_for_index(text) -> str: the fold applied to every items_fts column (zero-width joiners dropped,
  Devanagari digits as ASCII). Queries must get the same fold (match_query does; search.service too).
- match_query(q) -> str | None: a safe FTS5 MATCH expression (every token quoted, prefix match) over
  fold_for_index(q). Tokens keep combining marks (Devanagari vowel signs, virama) inside words.
- snippet_originals(conn, ids) -> {id: (text, doc)} and restore_snippet(raw, original, open, close): map an
  FTS5 snippet() of the folded text or doc column back onto the original text, so the user sees '१५' and
  joiners as written. The snippet markers must still be in `raw` (restore before escaping).
- analyze_text adds a "language" field ('Hindi', 'Marathi', 'Hindi or Marathi', each with ', English'
  when the text also has English words) when the text has Devanagari letters.
User edits survive: fields edited through the API carry "source": "user" in the fields JSON; when the
documents row has source 'user', its doc_type and non-empty title, doc_date and person_name are kept.
"""
from __future__ import annotations

import errno
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from app import storage
from app.config import settings
from app.db import tx
from app.documents.fields import (
    Box, FieldValue, PageText, ParsedDocument, has_devanagari, iso_valid, languages_of, word_tokens,
)
from app.util import json_dumps, json_loads, now_iso

log = logging.getLogger(__name__)

OCR_IMAGE_SIDE = 2048
MIN_TEXT_CHARS = 20          # below this (and no detected type) an item gets no documents row
LAYER_MIN_CHARS = 3          # a PDF page layer shorter than this counts as missing (OCR takes over)
SCAN_PAGES = 3               # pages scanned for barcodes and the ID photo
GAP_FACTOR = 2.0             # OCR boxes on one line further apart than this many heights get a column gap
PLAIN_EXTS = (".txt", ".md", ".csv", ".text", ".log", ".markdown")


@dataclass
class Extraction:
    pages: list[PageText] = field(default_factory=list)
    images: list[tuple[int, np.ndarray]] = field(default_factory=list)   # (page_no, BGR) for barcodes / photo
    source: str | None = None                                             # item_text.source


# ------------------------------------------------------------------ availability

def _models_allowed() -> bool:
    """PCC_NO_MODELS=1 (tests run in parallel) keeps every model unloaded."""
    return os.environ.get("PCC_NO_MODELS") != "1"


def _ocr_ready() -> bool:
    if not settings.ocr_enabled or not _models_allowed():
        return False
    try:
        from app.ml import ocr as ml_ocr

        return bool(ml_ocr.available())
    except Exception:
        log.warning("OCR engine unavailable", exc_info=True)
        return False


def _faces_ready() -> bool:
    if not _models_allowed():
        return False
    try:
        from app.ml import faces as ml_faces

        return bool(ml_faces.available())
    except Exception:
        return False


# ------------------------------------------------------------------ OCR helpers

def _ocr_page(page_no: int, bgr: np.ndarray) -> tuple[str, list[Box]]:
    from app.ml import ocr as ml_ocr

    h, w = bgr.shape[:2]
    raw = ml_ocr.ocr(bgr)
    boxes = [Box(page_no, b.x0 / w, b.y0 / h, b.x1 / w, b.y1 / h, str(b.text), float(b.score)) for b in raw]
    return boxes_text(raw), boxes


def boxes_text(raw: list[Any]) -> str:
    """Join OCR boxes (pixel or normalised coordinates, reading order) into lines. Boxes on the same line
    far apart (a column gap) are joined with three spaces so label values end there."""
    if not raw:
        return ""
    lines: list[list[Any]] = [[raw[0]]]
    for b in raw[1:]:
        prev = lines[-1][-1]
        ph, bh = prev.y1 - prev.y0, b.y1 - b.y0
        same = b.x0 >= prev.x0 and abs((b.y0 + b.y1) / 2 - (prev.y0 + prev.y1) / 2) <= max(ph, bh) / 2
        if same:
            lines[-1].append(b)
        else:
            lines.append([b])
    out = []
    for ln in lines:
        parts = [ln[0].text]
        for prev, b in zip(ln, ln[1:]):
            gap = b.x0 - prev.x1
            parts.append(("   " if gap > GAP_FACTOR * max(prev.y1 - prev.y0, 1e-6) else " ") + b.text)
        out.append("".join(parts))
    return "\n".join(out)


def _load_render(uid: str, version: int, page_no: int) -> np.ndarray | None:
    p = storage.page_path(uid, version, page_no)
    if not p.exists():
        return None
    try:
        from PIL import Image

        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])
    except Exception:
        log.warning("cannot read page render %s", storage.rel(p), exc_info=True)
        return None


def _render(item: sqlite3.Row, orig: Path, page_no: int) -> np.ndarray | None:
    """The stored render of a page, rendered now when missing (pages past the ingest limit)."""
    img = _load_render(item["uid"], int(item["version"]), page_no)
    if img is not None:
        return img
    try:
        from app.media import documents as media_docs

        media_docs.render_page(orig, item["kind"], item["uid"], int(item["version"]), page_no,
                               max_side=settings.page_render_size, ext=item["ext"])
    except Exception:
        log.warning("cannot render page %s of %s", page_no, item["uid"], exc_info=True)
        return None
    return _load_render(item["uid"], int(item["version"]), page_no)


def _size_of(img: np.ndarray | None) -> tuple[int | None, int | None]:
    if img is None:
        return None, None
    return int(img.shape[1]), int(img.shape[0])


def _render_size(uid: str, version: int, page_no: int) -> tuple[int | None, int | None]:
    p = storage.page_path(uid, version, page_no)
    if not p.exists():
        return None, None
    try:
        from PIL import Image

        with Image.open(p) as im:
            return im.size
    except Exception:
        return None, None


# ------------------------------------------------------------------ collection (heavy, no transaction)

def _read_plain(orig: Path) -> list[str]:
    data = orig.read_bytes()[: 16 * 1024 * 1024]
    text = data.decode("utf-8", errors="replace")
    pages, cur = [], []
    size = 0
    for line in text.splitlines():
        cur.append(line)
        size += len(line) + 1
        if size >= 3000:
            pages.append("\n".join(cur))
            cur, size = [], 0
    if cur or not pages:
        pages.append("\n".join(cur))
    return pages


def _collect(item: sqlite3.Row) -> Extraction:
    kind, uid, version, ext = item["kind"], item["uid"], int(item["version"]), item["ext"]
    orig = storage.original_path(uid, version, ext)
    ex = Extraction()
    ocr_ok = _ocr_ready()
    max_ocr = max(0, int(settings.ocr_max_pages))

    if kind == "pdf":
        from app.media import documents as media_docs

        try:
            layer = media_docs.text_layer(orig, "pdf", ext)
        except Exception:
            log.warning("no text layer for %s", uid, exc_info=True)
            layer = [None] * int(item["page_count"] or 1)
        ocr_done = 0
        sources = set()
        for page_no, txt in enumerate(layer, 1):
            w, h = _render_size(uid, version, page_no)
            if txt and len(txt.strip()) >= LAYER_MIN_CHARS:
                ex.pages.append(PageText(page_no, txt, "layer", width=w, height=h))
                sources.add("pdf")
                continue
            if ocr_ok and ocr_done < max_ocr:
                img = _render(item, orig, page_no)
                if img is not None:
                    text, boxes = _ocr_page(page_no, img)
                    ocr_done += 1
                    w, h = _size_of(img)
                    ex.pages.append(PageText(page_no, text, "ocr", boxes, width=w, height=h))
                    sources.add("ocr")
                    if page_no <= SCAN_PAGES:
                        ex.images.append((page_no, img))
                    continue
            ex.pages.append(PageText(page_no, "", "none", width=w, height=h))
        for page_no in range(1, min(SCAN_PAGES, len(layer)) + 1):
            if not any(p == page_no for p, _ in ex.images):
                img = _load_render(uid, version, page_no)
                if img is not None:
                    ex.images.append((page_no, img))
        ex.source = "mixed" if len(sources) > 1 else (sources.pop() if sources else None)
    elif kind == "document":
        texts: list[str | None] | None = None
        try:
            from app.media import documents as media_docs

            texts = media_docs.text_layer(orig, "document", ext)
        except Exception:
            log.warning("no logical pages for %s", uid, exc_info=True)
        if texts is None and (ext or "").lower() in PLAIN_EXTS:
            texts = list(_read_plain(orig))
        for page_no, txt in enumerate(texts or [], 1):
            w, h = _render_size(uid, version, page_no)
            ex.pages.append(PageText(page_no, txt or "", "layer" if txt else "none", width=w, height=h))
        ex.source = "plain" if (ext or "").lower() in PLAIN_EXTS else "office"
    elif kind == "image":
        from app.media import images as media_images

        pil = media_images.load_oriented(orig, max_side=OCR_IMAGE_SIDE)
        bgr = media_images.to_bgr(pil)
        w, h = _size_of(bgr)
        ex.images.append((1, bgr))
        if ocr_ok:
            text, boxes = _ocr_page(1, bgr)
            ex.pages.append(PageText(1, text, "ocr", boxes, width=w, height=h))
            ex.source = "ocr"
        else:
            ex.pages.append(PageText(1, "", "none", width=w, height=h))
    return ex


def _scan_barcodes(ex: Extraction) -> None:
    from app.documents import barcodes

    by_page = {p.page_no: p for p in ex.pages}
    for page_no, img in ex.images:
        found = barcodes.decode(img)
        if found and page_no in by_page:
            by_page[page_no].barcodes = found


def _id_photo(item: sqlite3.Row, ex: Extraction) -> bool:
    if not ex.images or not _faces_ready():
        return False
    from app.documents import photo

    dest = storage.doc_photo_path(item["uid"], int(item["version"]))
    for _page_no, img in ex.images[:2]:
        if photo.crop_id_photo(img, dest):
            return True
    return False


# ------------------------------------------------------------------ detection + parsing (pure)

def joined_text(pages: list[PageText]) -> str:
    return "\f".join(p.text or "" for p in pages)


ANALYZE_MAX_CHARS = 400_000  # about 130 logical pages; detection and parsing read at most this much
ANALYZE_MAX_PAGES = 150


def learned_doc_type(text: str) -> tuple[str, float] | None:
    """(doc_type, similarity) from the documents the owner typed by hand (section 18), or None."""
    try:
        from app.learning import doc_type as learned

        return learned.predict(text)
    except Exception:  # noqa: BLE001 - learning never blocks the text stage
        log.exception("learning: document type learner failed")
        return None


def analyze_text(text: str, pages: list[PageText], item: Any = None, *, existing: sqlite3.Row | None = None,
                 forced_type: str | None = None) -> tuple[ParsedDocument, float] | None:
    """(parsed document, detection score) or None when the item gets no documents row."""
    from app.documents import parsers
    from app.documents.detect import detect

    # Detection and the parsers are regex-heavy and documents that have a type are short. A 16 MB log
    # file still gets its whole text layer and FTS entry, but only this much of it is typed.
    if len(text) > ANALYZE_MAX_CHARS or len(pages) > ANALYZE_MAX_PAGES:
        text = text[:ANALYZE_MAX_CHARS]
        pages = pages[:ANALYZE_MAX_PAGES]
    boxes = [b for p in pages for b in p.boxes]
    if forced_type is None and existing is not None and existing["source"] == "user":
        forced_type = existing["doc_type"]
    if forced_type:
        det_score = 1.0
        doc_type = forced_type
    else:
        det = detect(text, pages, item)
        doc_type, det_score = det.doc_type, det.score
        hint = learned_doc_type(text)
        if hint is not None:
            label, sim = hint
            from app.learning.doc_type import MIN_SIM, STRONG_SIM

            if sim >= STRONG_SIM or (sim >= MIN_SIM and doc_type == "other"):
                doc_type = label
                det_score = max(float(det.scores.get(label, 0.0) or 0.0), float(sim))
        if doc_type == "other" and len(text.strip()) < MIN_TEXT_CHARS:
            return None
    parsed = parsers.parse(doc_type, text, pages, boxes, today=_capture_date(item))
    if doc_type != "other" and not forced_type:
        parsed.confidence = round(min(1.0, parsed.confidence * (0.7 + 0.3 * min(1.0, det_score))), 2)
    _add_language(parsed, text)
    return parsed, det_score


def _add_language(parsed: ParsedDocument, text: str) -> None:
    """A 'language' field for documents written (partly) in Devanagari; English-only documents get none."""
    if "language" in parsed.fields or not has_devanagari(text):
        return
    langs = languages_of(text)
    if langs and langs != ["English"]:
        parsed.fields["language"] = FieldValue(", ".join(langs), "Language", 0.9)


def _capture_date(item: Any) -> date | None:
    """The item's capture day (taken_at, else created_at): a boarding pass barcode has no year, and the
    year is chosen around the day the pass was photographed or saved, not around today."""
    if item is None:
        return None
    for col in ("taken_at", "created_at"):
        try:
            val = item[col]
        except (IndexError, KeyError, TypeError):
            continue
        if val:
            try:
                return date.fromisoformat(str(val)[:10])
            except ValueError:
                continue
    return None


def merged_values(parsed: ParsedDocument, existing: sqlite3.Row | None) -> dict[str, Any]:
    """Column values for the documents row: the parse result with the user's edits laid over it."""
    from app.documents import parsers

    spec = parsers.get(parsed.doc_type)
    fields = parsed.fields_json()
    vals: dict[str, Any] = {
        "doc_type": spec.type, "doc_group": spec.group, "confidence": float(parsed.confidence),
        "title": parsed.title, "doc_date": parsed.doc_date if iso_valid(parsed.doc_date) else None,
        "issuer": parsed.issuer, "person_name": parsed.person_name, "barcode": parsed.barcode,
        "barcode_format": parsed.barcode_format, "source": "auto",
    }
    if existing is not None:
        old = json_loads(existing["fields"], {}) or {}
        for key, f in old.items():
            if isinstance(f, dict) and f.get("source") == "user":
                fields[key] = f
        if existing["source"] == "user":
            vals["source"] = "user"
            for col in ("title", "doc_date", "person_name"):
                if existing[col]:
                    vals[col] = existing[col]
        if not vals["barcode"] and existing["barcode"]:
            vals["barcode"], vals["barcode_format"] = existing["barcode"], existing["barcode_format"]
    order = {k: i for i, k in enumerate(spec.field_order)}
    vals["fields"] = json_dumps(dict(sorted(fields.items(), key=lambda kv: order.get(kv[0], len(order)))))
    return vals


def write_document(conn: sqlite3.Connection, item_id: int, vals: dict[str, Any] | None, *,
                   has_photo: bool | None = None) -> None:
    """Insert, update or (vals None) delete the documents row of an item. Call inside a transaction.
    A row the user edited is never deleted automatically."""
    existing = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()
    now = now_iso()
    if vals is None:
        if existing is not None and existing["source"] != "user":
            conn.execute("DELETE FROM documents WHERE item_id = ?", (item_id,))
        return
    photo = int(has_photo) if has_photo is not None else (int(existing["has_photo"]) if existing else 0)
    cols = ("doc_type", "doc_group", "confidence", "fields", "title", "doc_date", "issuer", "person_name", "barcode",
            "barcode_format", "source")
    if existing is None:
        conn.execute(
            f"INSERT INTO documents(item_id, {', '.join(cols)}, has_photo, created_at, updated_at) "
            f"VALUES (?, {', '.join('?' * len(cols))}, ?, ?, ?)",
            (item_id, *(vals[c] for c in cols), photo, now, now),
        )
    else:
        conn.execute(
            f"UPDATE documents SET {', '.join(f'{c} = ?' for c in cols)}, has_photo = ?, updated_at = ? WHERE item_id = ?",
            (*(vals[c] for c in cols), photo, now, item_id),
        )


# ------------------------------------------------------------------ the job

def _source_of(ex: Extraction) -> str:
    srcs = {p.source for p in ex.pages if p.text and p.text.strip()}
    if ex.source in ("office", "plain"):
        return ex.source
    if srcs == {"layer"}:
        return "pdf"
    if srcs == {"ocr"}:
        return "ocr"
    if len(srcs) > 1:
        return "mixed"
    return ex.source or "plain"


def _set_status(conn: sqlite3.Connection, item_id: int, version: int, status: str) -> None:
    with tx(conn):
        conn.execute("UPDATE items SET text_status = ?, updated_at = ? WHERE id = ? AND version = ?",
                     (status, now_iso(), item_id, version))


def extract_item(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None or int(item["version"]) != int(version) or item["deleted_at"]:
        log.info("text job for item %s v%s skipped (gone, replaced or in the trash)", item_id, version)
        return
    try:
        ex = _collect(item)
        try:
            _scan_barcodes(ex)
        except Exception:
            log.warning("barcode scan failed for %s", item["uid"], exc_info=True)
        text = joined_text(ex.pages)
        existing = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()
        result = analyze_text(text, ex.pages, item, existing=existing)
        has_photo = False
        if result is not None:
            from app.documents import parsers

            if parsers.get(result[0].doc_type).group == "id":
                try:
                    has_photo = _id_photo(item, ex)
                except Exception:
                    log.warning("ID photo crop failed for %s", item["uid"], exc_info=True)
        if not has_photo:
            storage.safe_unlink(storage.doc_photo_path(item["uid"], int(item["version"])))
        with tx(conn):
            cur = conn.execute("SELECT version, deleted_at FROM items WHERE id = ?", (item_id,)).fetchone()
            if cur is None or int(cur["version"]) != int(version) or cur["deleted_at"]:
                log.info("item %s changed during text extraction, result dropped", item_id)
                return
            _write_pages(conn, item_id, ex.pages)
            if text.strip():
                conn.execute(
                    "INSERT INTO item_text(item_id, text, source, lang, char_count, created_at) VALUES (?, ?, ?, 'en', ?, ?) "
                    "ON CONFLICT(item_id) DO UPDATE SET text = excluded.text, source = excluded.source, "
                    "lang = excluded.lang, char_count = excluded.char_count, created_at = excluded.created_at",
                    (item_id, text, _source_of(ex), len(text), now_iso()),
                )
            else:
                conn.execute("DELETE FROM item_text WHERE item_id = ?", (item_id,))
            existing = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()
            vals = merged_values(result[0], existing) if result is not None else None
            write_document(conn, item_id, vals, has_photo=has_photo)
            try:
                from app.learning import tags as learned_tags

                learned_tags.apply_item(conn, item_id, emb=None, text=text, kind=item["kind"])
            except Exception:  # noqa: BLE001 - learning never blocks the text stage
                log.exception("learning: tag suggestions failed for item %s", item_id)
            refresh_fts(conn, item_id)
            if vals is not None and vals.get("doc_type") not in (None, "", "other") and item["kind"] == "image":
                # A typed document photographed with the camera belongs with the documents (section 9 rule 4),
                # unless the owner chose a category or the item is in the vault.
                conn.execute(
                    "UPDATE items SET category = 'documents', updated_at = ? WHERE id = ? AND category_source = 'auto' "
                    "AND nsfw = 0 AND category != 'documents'",
                    (now_iso(), item_id),
                )
            conn.execute("UPDATE items SET text_status = 'done', updated_at = ? WHERE id = ?", (now_iso(), item_id))
    except Exception as exc:
        if _environment_error(exc):
            # not a verdict on the file: the worker retries with backoff (or waits for disk space) and its
            # final failure marks text_status 'error'
            raise
        log.exception("text extraction failed for item %s", item_id)
        # an error here (database locked, disk full) reaches the worker too, so the status is not left 'pending'
        _set_status(conn, item_id, version, "error")


def _environment_error(exc: BaseException) -> bool:
    """True when the failure comes from the database, memory or a full disk anywhere in the exception chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (sqlite3.Error, MemoryError)):
            return True
        if isinstance(cur, OSError) and cur.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        cur = cur.__cause__ or (None if cur.__suppress_context__ else cur.__context__)
    return False


def _write_pages(conn: sqlite3.Connection, item_id: int, pages: list[PageText]) -> None:
    conn.execute("DELETE FROM pages WHERE item_id = ?", (item_id,))
    for p in pages:
        boxes = [[round(b.x0, 4), round(b.y0, 4), round(b.x1, 4), round(b.y1, 4), b.text, round(b.score, 3)]
                 for b in p.boxes]
        conn.execute(
            "INSERT INTO pages(item_id, page_no, width, height, text, text_source, ocr_boxes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (item_id, p.page_no, p.width, p.height, p.text or None, p.source, json_dumps(boxes) if boxes else None),
        )


# ------------------------------------------------------------------ reparse (API)

def stored_pages(conn: sqlite3.Connection, item_id: int) -> list[PageText]:
    out = []
    for r in conn.execute("SELECT * FROM pages WHERE item_id = ? ORDER BY page_no", (item_id,)):
        boxes = []
        for b in json_loads(r["ocr_boxes"], []) or []:
            try:
                boxes.append(Box(int(r["page_no"]), float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4]),
                                 float(b[5]) if len(b) > 5 else 1.0))
            except (TypeError, ValueError, IndexError):
                continue
        out.append(PageText(int(r["page_no"]), r["text"] or "", r["text_source"], boxes, width=r["width"],
                            height=r["height"]))
    return out


def reparse(conn: sqlite3.Connection, item_id: int, forced_type: str | None = None) -> bool:
    """Detection and parsing again on the stored text (no OCR). Keeps the user's field edits; a forced
    type (the user picked one) wins over detection."""
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None:
        return False
    pages = stored_pages(conn, item_id)
    text = joined_text(pages) if pages else ""
    if not text.strip():
        row = conn.execute("SELECT text FROM item_text WHERE item_id = ?", (item_id,)).fetchone()
        if row and row["text"]:
            text = row["text"]
            pages = [PageText(i, t, "layer") for i, t in enumerate(text.split("\f"), 1)]
    if not pages:
        pages = [PageText(1, "", "none")]
    existing = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()
    if existing is not None and existing["barcode"]:
        # the decoded barcode is part of the document (a boarding pass may have no readable text)
        pages[0].barcodes = [(existing["barcode_format"] or "QRCode", existing["barcode"])]
    if not text.strip() and existing is None and not forced_type:
        return False
    result = analyze_text(text, pages, item, existing=existing, forced_type=forced_type)
    with tx(conn):
        existing = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()
        # a type change drops the old type's automatic fields; user-edited ones stay (merged_values)
        vals = merged_values(result[0], existing) if result is not None else None
        if vals is not None and forced_type:
            vals["source"] = "user"
        write_document(conn, item_id, vals)
        refresh_fts(conn, item_id)
    return True


# ------------------------------------------------------------------ full-text index

def _doc_text(row: sqlite3.Row | None) -> str:
    if row is None:
        return ""
    from app.documents import parsers

    parts = [row["title"], parsers.get(row["doc_type"]).label, row["issuer"], row["person_name"]]
    for f in (json_loads(row["fields"], {}) or {}).values():
        if isinstance(f, dict):
            parts.append(str(f.get("value", "")))
        elif f is not None:
            parts.append(str(f))
    seen: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    return " ".join(seen)


_INDEX_DIGITS = str.maketrans({chr(0x0966 + i): str(i) for i in range(10)})
_INDEX_JOINERS = re.compile("[\u200c\u200d]")


def fold_for_index(text: str | None) -> str:
    """Text as the FTS index stores it: zero-width joiners dropped (they only shape glyphs, and the
    query side drops them too) and Devanagari digits written as ASCII digits, so a search for 15
    finds a receipt that says १५. item_text keeps the original text."""
    return _INDEX_JOINERS.sub("", text or "").translate(_INDEX_DIGITS)


def _fold_positions(original: str) -> list[int]:
    """For each character of fold_for_index(original), its index in original. The fold only drops
    joiners and maps digits one to one, so this map is exact."""
    return [i for i, ch in enumerate(original) if ch not in "\u200c\u200d"]


def restore_snippet(raw: str | None, original: str | None, open_mark: str, close_mark: str,
                    ellipsis: str = "...") -> str | None:
    """An FTS5 snippet() of a folded column, rewritten over the original text: the same fragment and
    the same highlighted words, with Devanagari digits and zero-width joiners as written. `raw` still
    holds the open/close markers. Returns `raw` unchanged when the fragment is not found in the folded
    original (the column was indexed from other text, or the markers are unbalanced)."""
    if not raw or not original or open_mark not in raw:
        return raw
    plain_parts: list[str] = []
    marks: list[tuple[int, str]] = []  # (offset in the plain fragment, marker)
    n = 0
    for ch in raw:
        if ch == open_mark or ch == close_mark:
            marks.append((n, ch))
        else:
            plain_parts.append(ch)
            n += 1
    plain = "".join(plain_parts)
    folded = fold_for_index(original)
    lead = plain.startswith(ellipsis)
    trail = plain.endswith(ellipsis)
    # the fragment as written first (the column may itself start or end with '...'), then without
    # the ellipses snippet() adds when the fragment does not reach a column edge
    for cut_lead, cut_trail in ((False, False), (lead, False), (False, trail), (lead, trail)):
        a = len(ellipsis) if cut_lead else 0
        b = len(plain) - (len(ellipsis) if cut_trail else 0)
        if b <= a:
            continue
        frag = plain[a:b]
        start = folded.find(frag)
        if start >= 0:
            break
    else:
        return raw
    if any(off < a or off > b for off, _m in marks):
        return raw  # a marker inside an ellipsis: not a fragment of this column
    pos = _fold_positions(original)
    end = start + len(frag)
    o_start, o_end = pos[start], pos[end - 1] + 1

    def o_index(off: int, mark: str) -> int:
        f = start + off - a
        if mark == open_mark:  # before the folded character at f
            return pos[f] if f < end else o_end
        return pos[f - 1] + 1 if f > start else o_start  # after the folded character at f - 1

    out: list[str] = [plain[:a]]
    cursor = o_start
    for off, mark in marks:
        i = o_index(off, mark)
        out.append(original[cursor:i])
        out.append(mark)
        cursor = i
    out.append(original[cursor:o_end])
    out.append(plain[b:])
    return "".join(out)


def snippet_originals(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[str, str]]:
    """{item id: (original text column, original doc column)} as refresh_fts builds them before
    folding, for restore_snippet on snippet(items_fts, 1, ...) and snippet(items_fts, 2, ...)."""
    out: dict[int, tuple[str, str]] = {}
    for i in range(0, len(ids), 500):
        chunk = [int(x) for x in ids[i:i + 500]]
        if not chunk:
            continue
        ph = ",".join("?" * len(chunk))
        texts = {int(r["item_id"]): r["text"] or "" for r in conn.execute(
            f"SELECT item_id, text FROM item_text WHERE item_id IN ({ph})", chunk)}
        docs = {int(r["item_id"]): _doc_text(r) for r in conn.execute(
            f"SELECT * FROM documents WHERE item_id IN ({ph})", chunk)}
        for item_id in chunk:
            out[item_id] = (texts.get(item_id, ""), docs.get(item_id, ""))
    return out


def refresh_fts(conn: sqlite3.Connection, item_id: int) -> None:
    item = conn.execute("SELECT original_filename, companion_of, place FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None or item["companion_of"]:
        remove_fts(conn, item_id)
        return
    text_row = conn.execute("SELECT text FROM item_text WHERE item_id = ?", (item_id,)).fetchone()
    doc_row = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item_id,)).fetchone()
    people = [r["name"] for r in conn.execute(
        "SELECT DISTINCT p.name FROM persons p WHERE p.name IS NOT NULL AND p.id IN ("
        "SELECT person_id FROM faces WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM bodies WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM animals WHERE item_id = ? AND person_id IS NOT NULL UNION "
        "SELECT person_id FROM item_tags WHERE item_id = ?) ORDER BY p.name",
        (item_id, item_id, item_id, item_id),
    )]
    labels = [r["label"] for r in conn.execute(
        "SELECT label FROM item_labels WHERE item_id = ? ORDER BY score DESC", (item_id,))]
    labels += [r["name"] for r in conn.execute(
        "SELECT t.name FROM item_user_tags ut JOIN tags t ON t.id = ut.tag_id WHERE ut.item_id = ? ORDER BY t.name",
        (item_id,))]
    if item["place"]:  # the place name (section 25), so "goa" finds the photos taken there
        labels.append(item["place"])
    filename = item["original_filename"] or ""
    stem = re.sub(r"[_\-.]+", " ", filename)
    with tx(conn):
        conn.execute("DELETE FROM items_fts WHERE rowid = ?", (item_id,))
        conn.execute(
            "INSERT INTO items_fts(rowid, filename, text, doc, people, labels) VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, fold_for_index(f"{filename} {stem}".strip()), fold_for_index(text_row["text"] if text_row else ""),
             fold_for_index(_doc_text(doc_row)), fold_for_index(" ".join(people)), fold_for_index(" ".join(labels))),
        )


def remove_fts(conn: sqlite3.Connection, item_id: int) -> None:
    with tx(conn):
        conn.execute("DELETE FROM items_fts WHERE rowid = ?", (item_id,))


def match_query(q: str | None, *, max_tokens: int = 12) -> str | None:
    """FTS5 MATCH expression for free text: each word quoted (so operators, quotes and slashes in the
    input are plain text) with prefix matching, all words required. None when q has no words. A word
    is a run of word characters, combining marks or Indic letters (fields.word_tokens), so a Hindi word
    with vowel signs stays one word. The query gets the index fold (fold_for_index: zero-width joiners
    dropped, Devanagari digits as ASCII), so '१९८८' and '1988' both find a receipt that says either."""
    tokens = word_tokens(fold_for_index(q))[:max_tokens]
    if not tokens:
        return None
    return " ".join('"' + t.replace('"', '""') + '"*' for t in tokens)
