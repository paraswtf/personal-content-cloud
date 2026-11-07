"""Voter ID (EPIC, Election Commission of India) parser.

Contract: parse(text, pages, boxes) -> ParsedDocument. The EPIC number is three letters and seven digits.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    EPIC_RE, FieldSet, ParsedDocument, address_after, after_label, date_after, find_gender, name_after,
    normalize_text,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "voter_id"
FIELDS = [
    ("name", "Elector's name"), ("number", "EPIC number"), ("relation", "Relation"),
    ("relative_name", "Father, mother or husband"), ("gender", "Gender"), ("dob", "Date of birth"), ("age", "Age"),
    ("address", "Address"), ("constituency", "Assembly constituency"), ("part_no", "Part number"),
    ("issue_date", "Date of issue"),
]
KEY_FIELDS = ["name", "number", "relative_name", "gender"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    m = EPIC_RE.search(t.upper())
    if m:
        fs.set("number", m.group(1), 0.9)
    fs.set("name", name_after(t, ["elector's name", "electors name", "name of elector"]) or
           name_after(t, ["name"], anchored=True), 0.85)
    for label, rel in (("father's name", "Father"), ("husband's name", "Husband"), ("mother's name", "Mother"),
                       ("relation's name", "Relative")):
        v = name_after(t, [label, label.replace("'s", "s")])
        if v:
            fs.set("relation", rel, 0.8)
            fs.set("relative_name", v, 0.8)
            break
    fs.set("gender", find_gender(t), 0.85)
    fs.set("dob", date_after(t, ["date of birth", "dob", "date of birth/age"], past=True), 0.85)
    m = re.search(r"\bage(?:\s+as\s+on\s+[\d./\-]+)?\s*[:\-]?\s*(\d{2,3})\b", t, re.I)
    if m:
        fs.set("age", m.group(1), 0.7)
    fs.set("address", address_after(t, ["address"]), 0.7)
    ac = after_label(t, ["assembly constituency no. and name", "assembly constituency", "constituency"])
    fs.set("constituency", ac, 0.7)
    pn = after_label(t, ["part no. and name", "part no", "part number"])
    fs.set("part_no", pn, 0.6)
    fs.set("issue_date", date_after(t, ["date of issue", "date", "issue date"]), 0.7)
    person = fs.get("name")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Voter ID", person), doc_date=fs.get("issue_date"),
                  issuer="Election Commission of India", person_name=person, pages=pages)
