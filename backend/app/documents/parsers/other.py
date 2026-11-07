"""Fallback parser for text documents that match no known type.

Contract: parse(text, pages, boxes) -> ParsedDocument with a title (the largest OCR line or the first
meaningful line) and the first date found.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import FieldSet, ParsedDocument, find_dates, largest_text, normalize_text, split_lines
from app.documents.parsers._base import finish

log = logging.getLogger(__name__)

DOC_TYPE = "other"
FIELDS = [("title", "Title"), ("date", "Date")]
KEY_FIELDS = ["title", "date"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    title = largest_text(boxes or [], min_len=4) if boxes else None
    if not title:
        title = next((ln for ln in split_lines(t)[:10] if len(re.findall(r"[A-Za-z]", ln)) >= 4 and len(ln) <= 100),
                     None)
    fs.set("title", title, 0.5)
    ds = find_dates(t)
    fs.set("date", ds[0][0] if ds else None, 0.5)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=fs.get("title"), doc_date=fs.get("date"), base=0.1, pages=pages)
