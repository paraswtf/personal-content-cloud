"""Degree certificate parser (convocation certificates, provisional degree certificates).

Contract: parse(text, pages, boxes) -> ParsedDocument.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, date_after, find_dates, nice_name, normalize_text, plausible_name,
)
from app.documents.parsers._base import finish, join_title
from app.documents.parsers.marksheet import institution_line, year_of

log = logging.getLogger(__name__)

DOC_TYPE = "degree"
FIELDS = [
    ("recipient", "Awarded to"), ("degree", "Degree"), ("specialisation", "Specialisation"),
    ("institution", "Institution"), ("class", "Class or grade"), ("year", "Year"),
    ("registration_number", "Registration number"), ("certificate_number", "Certificate number"),
    ("date", "Date"),
]
KEY_FIELDS = ["recipient", "degree", "institution", "year"]
_DEGREE_RE = re.compile(
    r"\b((?:Bachelor|Master|Doctor)\s+of\s+(?-i:[A-Z][A-Za-z]+(?:[ \t]+(?:and|&)[ \t]+[A-Z][A-Za-z]+|[ \t]+[A-Z][A-Za-z]+){0,3})|"
    r"B\.?\s?Tech|M\.?\s?Tech|B\.?\s?E\.|M\.?\s?B\.?\s?A|B\.?\s?C\.?\s?A|M\.?\s?C\.?\s?A|B\.?\s?Sc|M\.?\s?Sc|B\.?\s?Com|"
    r"M\.?\s?Com|B\.?\s?A\.|M\.?\s?A\.|Ph\.?\s?D|LL\.?B|LL\.?M|MBBS|B\.?\s?Pharm|M\.?\s?Pharm|B\.?\s?Arch|PGDM)",
    re.I)


def recipient_of(t: str) -> str | None:
    for rx in (r"(?:certify that|conferred (?:up)?on|awarded to|presented to|admitted to the degree[^\n]*?\n?)\s*"
               r"(?:Mr\.?|Ms\.?|Mrs\.?|Shri|Smt\.?|Kumari|Dr\.?)?\s*([A-Z][A-Za-z.' ]{2,60}?)"
               r"(?=\s+(?:has|who|son|daughter|s/o|d/o|w/o|for|in|the|of|is|was)\b|\s*[,\n])",):
        m = re.search(rx, t, re.I)
        if m:
            name = m.group(1).strip()
            if plausible_name(name):
                return nice_name(name)
    return None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    fs.set("recipient", recipient_of(t), 0.8)
    m = _DEGREE_RE.search(t)
    if m:
        fs.set("degree", re.sub(r"\s+", " ", m.group(1)).strip(), 0.8)
        after = t[m.end():m.end() + 80]
        sm = re.match(r"\s*(?:\(|in\s+|with specialisation in\s+|with specialization in\s+)([A-Z][A-Za-z &]{3,50})", after)
        if sm:
            fs.set("specialisation", sm.group(1).strip(" )"), 0.65)
    fs.set("institution", institution_line(t), 0.8)
    m = re.search(r"(first class with distinction|first division|second division|first class|second class|"
                  r"distinction|summa cum laude|magna cum laude|cum laude|cgpa\s*[:\-]?\s*\d{1,2}(?:\.\d{1,2})?)", t, re.I)
    if m:
        fs.set("class", m.group(1).title() if "cgpa" not in m.group(1).lower() else m.group(1).upper(), 0.7)
    fs.set("year", year_of(t), 0.6)
    reg = after_label(t, ["registration no", "reg. no", "enrollment no", "enrolment no", "prn", "roll no"])
    if reg:
        mm = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,24}", reg)
        fs.set("registration_number", mm.group(0) if mm else None, 0.75)
    cert = after_label(t, ["certificate no", "serial no", "sr. no", "s. no", "certificate number"])
    if cert:
        mm = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,24}", cert)
        fs.set("certificate_number", mm.group(0) if mm else None, 0.7)
    d = date_after(t, ["date", "dated", "convocation held on", "given on"], past=True)
    if not d:
        ds = find_dates(t)
        d = ds[-1][0] if ds else None
    fs.set("date", d, 0.65)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title(fs.get("degree") or "Degree", fs.get("institution")),
                  doc_date=fs.get("date"), issuer=fs.get("institution"), person_name=fs.get("recipient"), pages=pages)
