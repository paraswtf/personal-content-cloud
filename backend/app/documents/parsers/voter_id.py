"""Voter ID (EPIC, Election Commission of India) parser.

Contract: parse(text, pages, boxes) -> ParsedDocument. The EPIC number is three letters and seven digits.
Cards print the fields in a regional language and English; the Latin spelling of a name wins and the
Devanagari spelling of the elector's name goes to `name_local`.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    ADDRESS_LABELS_LOCAL, DATE_LABELS_LOCAL, DOB_LABELS_LOCAL, EPIC_RE, FATHER_LABELS_LOCAL, HUSBAND_LABELS_LOCAL,
    ISSUE_LABELS_LOCAL, MOTHER_LABELS_LOCAL, NAME_LABELS_LOCAL, FieldSet, ParsedDocument, address_after, after_label,
    date_after, find_gender, local_name_after, name_after, normalize_text,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "voter_id"
FIELDS = [
    ("name", "Elector's name"), ("name_local", "Name (local script)"),
    ("number", "EPIC number"), ("relation", "Relation"),
    ("relative_name", "Father, mother or husband"), ("gender", "Gender"), ("dob", "Date of birth"), ("age", "Age"),
    ("address", "Address"), ("constituency", "Assembly constituency"), ("part_no", "Part number"),
    ("issue_date", "Date of issue"),
]
KEY_FIELDS = ["name", "number", "relative_name", "gender"]
_ELECTOR_LOCAL = ["निर्वाचक का नाम", "मतदाता का नाम", "मतदाराचे नाव", "मतदाराचे नांव", "मतदाराचे पूर्ण नाव"]
_RELATIVES = [
    (["father's name", "fathers name"], "Father"), (["husband's name", "husbands name"], "Husband"),
    (["mother's name", "mothers name"], "Mother"), (["relation's name", "relations name"], "Relative"),
    (FATHER_LABELS_LOCAL[:4], "Father"), (HUSBAND_LABELS_LOCAL[:2], "Husband"), (MOTHER_LABELS_LOCAL[:3], "Mother"),
]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    m = EPIC_RE.search(t.upper())
    if m:
        fs.set("number", m.group(1), 0.9)
    elector = ["elector's name", "electors name", "name of elector", *_ELECTOR_LOCAL]
    fs.set("name", name_after(t, elector) or name_after(t, ["name", *NAME_LABELS_LOCAL], anchored=True), 0.85)
    local = local_name_after(t, elector) or local_name_after(t, ["name", *NAME_LABELS_LOCAL], anchored=True)
    if local and local != fs.get("name"):
        fs.set("name_local", local, 0.75)
    for labels, rel in _RELATIVES:
        v = name_after(t, labels)
        if v:
            fs.set("relation", rel, 0.8)
            fs.set("relative_name", v, 0.8)
            break
    fs.set("gender", find_gender(t), 0.85)
    fs.set("dob", date_after(t, ["date of birth", "dob", "date of birth/age", *DOB_LABELS_LOCAL], past=True), 0.85)
    m = (re.search(r"\bage(?:\s+as\s+on\s+[\d./\-]+)?\s*[:\-]?\s*(\d{2,3})\b", t, re.I)
         or re.search(r"(?<![\u0900-\u097F])(?:आयु|उम्र|वय)(?![\u0900-\u097F])[^\d\n]{0,25}?(\d{2,3})(?!\d)", t))
    if m:
        fs.set("age", m.group(1), 0.7)
    fs.set("address", address_after(t, ["address", *ADDRESS_LABELS_LOCAL]), 0.7)
    ac = after_label(t, ["assembly constituency no. and name", "assembly constituency", "constituency",
                         "विधानसभा निर्वाचन क्षेत्र", "विधानसभा मतदारसंघ", "विधानसभा क्षेत्र"])
    fs.set("constituency", ac, 0.7)
    pn = after_label(t, ["part no. and name", "part no", "part number", "भाग संख्या", "भाग क्रमांक", "भाग सं"])
    fs.set("part_no", pn, 0.6)
    # 'Date' must not read the 'Date of Birth' line of a bilingual card
    fs.set("issue_date", date_after(t, ["date of issue", "date", "issue date", *ISSUE_LABELS_LOCAL, *DATE_LABELS_LOCAL],
                                    past=True, accept=lambda v: not re.match(r"of\s+birth", v, re.I)), 0.7)
    person = fs.get("name")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Voter ID", person), doc_date=fs.get("issue_date"),
                  issuer="Election Commission of India", person_name=person, pages=pages)
