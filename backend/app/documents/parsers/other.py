"""Fallback parser for text documents that match no known type.

Contract: parse(text, pages, boxes) -> ParsedDocument with a title (the largest OCR line or the first
meaningful line; a Devanagari line only when no line has Latin words), the first date found and, for a
rent agreement (English, Hindi or Marathi heading), the kind 'Rent agreement'.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    DEV, FieldSet, ParsedDocument, find_dates, largest_text, normalize_text, split_lines, word_regex,
)
from app.documents.parsers._base import finish

log = logging.getLogger(__name__)

DOC_TYPE = "other"
FIELDS = [("title", "Title"), ("date", "Date"), ("kind", "Kind")]
KEY_FIELDS = ["title", "date"]
_KINDS = [(re.compile("|".join(word_regex(w) for w in ("rent agreement", "rental agreement", "किराया अनुबंध",
                                                       "किरायानामा", "किराया करार", "भाडे करार", "भाडेकरार",
                                                       "भाडे करारनामा")), re.I), "Rent agreement")]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    title = largest_text(boxes or [], min_len=4) if boxes else None
    if not title:
        lines = split_lines(t)[:10]
        # a Latin line first; a mostly Devanagari line that only carries an ID ('पैन: ABCPS1234K') is not one
        title = next((ln for ln in lines if len(ln) <= 100
                      and len(re.findall(r"[A-Za-z]", ln)) >= max(4, len(re.findall("[" + DEV + "]", ln)))), None)
        if not title:
            title = next((ln for ln in lines if len(re.findall("[" + DEV + "]", ln)) >= 4 and len(ln) <= 100), None)
    fs.set("title", title, 0.5)
    ds = find_dates(t)
    fs.set("date", ds[0][0] if ds else None, 0.5)
    fs.set("kind", next((kind for rx, kind in _KINDS if rx.search(t)), None), 0.6)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=fs.get("title"), doc_date=fs.get("date"), base=0.1, pages=pages)
