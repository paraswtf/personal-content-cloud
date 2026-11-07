"""Aadhaar card parser (front and back, e-Aadhaar letters, masked copies).

Contract: parse(text, pages, boxes) -> ParsedDocument with fields in FIELDS order. The number must
pass the Verhoeff check (masked 'XXXX XXXX 1234' copies are kept with a lower confidence). Cards print
the name and labels in Hindi (or another regional language) and English: the Latin spelling of the
name goes to `name`, the Devanagari one to `name_local`.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    AADHAAR_MASKED_RE, ADDRESS_LABELS_LOCAL, ISSUE_LABELS_LOCAL, VID_RE, FieldSet, ParsedDocument, address_after,
    date_after, find_aadhaar, find_gender, find_pincode, local_name, nice_name, normalize_text, parse_date,
    plausible_name, relation_name, split_lines,
)
from app.documents.parsers._base import finish, is_name_line, join_title, local_name_line_before, name_line_before

log = logging.getLogger(__name__)

DOC_TYPE = "aadhaar"
FIELDS = [
    ("name", "Name"), ("name_local", "Name (local script)"),
    ("number", "Aadhaar number"), ("dob", "Date of birth"), ("yob", "Year of birth"),
    ("gender", "Gender"), ("relation", "Relation"), ("relative_name", "Father or spouse"),
    ("address", "Address"), ("pincode", "PIN code"), ("vid", "VID"), ("enrolment_no", "Enrolment number"),
    ("issue_date", "Issue date"),
]
KEY_FIELDS = ["name", "number", "dob", "gender"]
_YOB_LOCAL = re.compile(r"जन्म\s*(?:का\s*)?वर्ष(?:\s*/\s*[A-Za-z ]{0,20}?)?\s*[:\-]?\s*((?:19|20)\d{2})")
_NAME_LOCAL = re.compile(r"(?m)^\s*(?:नाम|नाव)\s*[:\-]\s*([\u0900-\u0963\u0971-\u097F][\u0900-\u0963\u0971-\u097F .']{2,60})$")
_REL_PREFIX_LOCAL = re.compile(r"^(?:(?:पुत्र|पुत्री|पत्नी|पति)\s*[:\-]?\s*[\u0900-\u0963\u0971-\u097F .']+,\s*)")


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    bonus = 0.0

    number = find_aadhaar(t, grouped_only=True) or find_aadhaar(t)
    if number:
        fs.set("number", number, 0.98)
        bonus = 0.1
    else:
        m = AADHAAR_MASKED_RE.search(t)
        if m:
            fs.set("number", f"XXXX XXXX {m.group(3)}", 0.6)

    m = re.search(r"\bVID\s*[:\-]?\s*(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})", t, re.I)
    if m:
        v = VID_RE.search(m.group(1))
        if v:
            fs.set("vid", " ".join(v.groups()), 0.9)

    dob_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"(?:D\.?O\.?B|date of birth|birth|जन्म)", ln, re.I):
            d = parse_date(ln, past=True)
            if d:
                fs.set("dob", d, 0.95)
                dob_idx = i
                break
            m = (re.search(r"(?:year of birth|yob)\s*[:\-]?\s*((?:19|20)\d{2})", ln, re.I)
                 or _YOB_LOCAL.search(ln))
            if m:
                fs.set("yob", m.group(1), 0.9)
                dob_idx = i
                break
    if not fs.has("dob") and not fs.has("yob"):
        m = re.search(r"(?:year of birth|yob)\s*[:\-]?\s*((?:19|20)\d{2})", t, re.I) or _YOB_LOCAL.search(t)
        if m:
            fs.set("yob", m.group(1), 0.85)

    gender = find_gender(t)
    if gender:
        fs.set("gender", gender, 0.9)

    name = local = None
    m = re.search(r"(?m)^\s*name\s*[:\-]\s*([A-Za-z][A-Za-z .']{2,60})$", t, re.I)
    if m and plausible_name(m.group(1).strip()):
        name = m.group(1).strip()
    m = _NAME_LOCAL.search(t)
    if m:
        local = local_name(m.group(1))
    if not name:
        to_line = next((i for i, ln in enumerate(lines) if re.match(r"^\s*(?:to|प्रति)\s*,?\s*$", ln, re.I)), None)
        if to_line is not None:
            # e-Aadhaar letters: 'To' and the name, often in Hindi first and then in English
            cands = [ln for ln in lines[to_line + 1:to_line + 3] if is_name_line(ln)]
            name = next((c for c in cands if re.search(r"[A-Za-z]", c)), None) or (cands[0] if cands else None)
            local = local or next((local_name(c) for c in cands if local_name(c)), None)
    if not name and dob_idx is not None:
        name = name_line_before(lines, dob_idx)
    if dob_idx is not None and not local:
        local = local_name(local_name_line_before(lines, dob_idx))
    if name:
        fs.set("name", nice_name(name), 0.8)
    if local and local != fs.get("name"):
        fs.set("name_local", local, 0.75)

    rel = relation_name(t)
    if rel:
        fs.set("relation", rel[0], 0.85)
        fs.set("relative_name", rel[1], 0.85)

    addr = address_after(t, ["address", *ADDRESS_LABELS_LOCAL])
    if addr:
        addr = re.sub(r"^(?:[SDWC]\s*/\s*O\s*[:.\-]?\s*[A-Za-z .']+,\s*)", "", addr, flags=re.I)
        addr = _REL_PREFIX_LOCAL.sub("", addr)
        fs.set("address", addr, 0.75)
        pin = find_pincode(addr)
        if pin:
            fs.set("pincode", pin, 0.85)

    m = re.search(r"(?:enrol(?:l)?ment\s*no|नामांकन\s*(?:क्रम|संख्या|क्रमांक))\.?(?:\s*/\s*[A-Za-z .]{0,20}?)?\s*[:\-]?"
                  r"\s*([\d/ ]{14,30})", t, re.I)
    if m:
        fs.set("enrolment_no", re.sub(r"\s+", " ", m.group(1)).strip(), 0.8)
    issue = date_after(t, ["issue date", "date of issue", "download date", *ISSUE_LABELS_LOCAL, "डाउनलोड तिथि",
                           "डाउनलोड दिनांक"], past=True)
    if issue:
        fs.set("issue_date", issue, 0.8)

    person = fs.get("name")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Aadhaar card", person), doc_date=fs.get("issue_date"),
                  issuer="UIDAI", person_name=person, bonus=bonus, pages=pages)
