"""Generic certificate parser (completion, participation, achievement, course certificates).

Contract: parse(text, pages, boxes) -> ParsedDocument.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, date_after, find_dates, largest_text, normalize_text, split_lines,
)
from app.documents.parsers._base import finish, join_title
from app.documents.parsers.degree import recipient_of
from app.documents.parsers.marksheet import institution_line

log = logging.getLogger(__name__)

DOC_TYPE = "certificate"
FIELDS = [
    ("title", "Certificate"), ("recipient", "Awarded to"), ("course", "For"), ("issuer", "Issued by"),
    ("date", "Date"), ("certificate_id", "Certificate ID"), ("grade", "Grade or score"), ("duration", "Duration"),
]
KEY_FIELDS = ["title", "recipient", "issuer", "date"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    m = (re.search(r"(certificate[ \t]+of[ \t]+[a-z]+(?:[ \t]+(?:and|in)[ \t]+[a-z]+)?)", t, re.I) or
         re.search(r"([a-z]+[ \t]+certificate)", t, re.I))
    title = m.group(1).strip().title() if m else None
    if boxes:
        big = largest_text(boxes, skip=("certify",))
        if big and "certificate" in big.lower():
            title = big.title()
    fs.set("title", title, 0.7)
    fs.set("recipient", recipient_of(t), 0.8)
    m = re.search(r"(?:for (?:successfully )?(?:completing|completion of|participating in|participation in|"
                  r"securing|winning|the course|attending)|has (?:successfully )?completed(?: the)?|"
                  r"in recognition of)\s+(?:the\s+)?([^\n.]{3,100})", t, re.I)
    if m:
        fs.set("course", m.group(1).strip(" ,."), 0.7)
    issuer = after_label(t, ["issued by", "organised by", "organized by", "offered by", "presented by"])
    issuer = issuer or institution_line(t)
    if not issuer:
        issuer = next((ln for ln in lines[:4] if "certif" not in ln.lower() and re.search(r"[A-Za-z]{3}", ln)
                       and len(ln) < 80), None)
    fs.set("issuer", issuer, 0.6)
    d = date_after(t, ["date of issue", "issued on", "date", "dated", "awarded on"])
    if not d:
        ds = find_dates(t)
        d = ds[-1][0] if ds else None
    fs.set("date", d, 0.65)
    cid = after_label(t, ["certificate id", "certificate no", "certificate number", "credential id", "cert id",
                          "serial no", "verification code"])
    if cid:
        mm = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-_]{3,40}", cid)
        fs.set("certificate_id", mm.group(0) if mm else None, 0.75)
    g = after_label(t, ["grade", "score", "marks", "result"])
    fs.set("grade", g[:30] if g else None, 0.55)
    dur = after_label(t, ["duration", "course duration"])
    fs.set("duration", dur, 0.55)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title(fs.get("title") or "Certificate", fs.get("course")),
                  doc_date=fs.get("date"), issuer=fs.get("issuer"), person_name=fs.get("recipient"), pages=pages)
