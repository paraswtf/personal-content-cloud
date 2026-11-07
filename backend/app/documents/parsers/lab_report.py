"""Laboratory report parser (blood tests, pathology and diagnostics reports).

Contract: parse(text, pages, boxes) -> ParsedDocument with lab, patient, dates and the test lines
('Haemoglobin: 13.5 g/dL (13.0 - 17.0)'), joined with semicolons.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, date_after, find_gender, first_line_with, name_after, normalize_text,
    split_lines,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "lab_report"
FIELDS = [
    ("lab", "Laboratory"), ("patient", "Patient"), ("age", "Age"), ("gender", "Gender"), ("patient_id", "Patient ID"),
    ("referred_by", "Referred by"), ("sample_date", "Sample collected"), ("report_date", "Reported"),
    ("tests", "Results"), ("abnormal", "Out of range"),
]
KEY_FIELDS = ["lab", "patient", "report_date", "tests"]
LAB_WORDS = ("laboratory", "laboratories", "labs", "lab", "diagnostics", "diagnostic", "pathology", "pathlabs",
             "path lab", "imaging", "healthcare")
_TEST_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 ().,/%\-]{2,50}?)\s+(\d+(?:\.\d+)?)\s*(H|L|High|Low)?\s*"
    r"([a-zA-Z%/\u00b5\d.^]+(?:/[a-zA-Z\u00b5.]+)?)?\s*(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?|[<>]\s*\d+(?:\.\d+)?)?\s*$")
_NOT_TEST = ("age", "date", "page", "phone", "mobile", "sample", "patient", "ref", "reg", "lab", "id", "pin", "time",
             "report", "collected", "received", "uhid", "visit")


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    fs.set("lab", first_line_with(t, LAB_WORDS), 0.7)
    fs.set("patient", name_after(t, ["patient name", "patient's name", "name of patient", "pt. name", "patient",
                                     "name"], anchored=True) or name_after(t, ["patient name"]), 0.8)
    m = re.search(r"\bage\s*(?:/\s*(?:sex|gender))?\s*[:\-]?\s*(\d{1,3})\s*(?:y|yrs|years)?", t, re.I)
    fs.set("age", m.group(1) if m else None, 0.7)
    fs.set("gender", find_gender(t), 0.7)
    pid = after_label(t, ["patient id", "uhid", "lab no", "lab id", "sample id", "reg. no", "registration no",
                          "accession no", "sid"])
    if pid:
        mm = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,24}", pid)
        fs.set("patient_id", mm.group(0) if mm else None, 0.7)
    fs.set("referred_by", after_label(t, ["referred by", "ref. by", "ref by", "referring doctor", "ref. doctor"]), 0.65)
    fs.set("sample_date", date_after(t, ["sample collected on", "collected on", "collection date", "sample date",
                                         "collected", "sample collection", "registered on"]), 0.75)
    fs.set("report_date", date_after(t, ["reported on", "report date", "reported", "date of report", "report released on",
                                         "released on"]), 0.75)
    tests, abnormal = [], []
    for ln in lines:
        m = _TEST_RE.match(ln)
        if not m:
            continue
        name = m.group(1).strip(" .:")
        if any(name.lower().startswith(w) for w in _NOT_TEST) or len(name) < 3:
            continue
        entry = f"{name}: {m.group(2)}"
        if m.group(4) and not re.fullmatch(r"\d+(?:\.\d+)?", m.group(4)):
            entry += f" {m.group(4)}"
        if m.group(5):
            entry += f" ({re.sub(r'\s+', ' ', m.group(5))})"
            rng = re.findall(r"\d+(?:\.\d+)?", m.group(5))
            try:
                val = float(m.group(2))
                if len(rng) == 2 and not float(rng[0]) <= val <= float(rng[1]):
                    abnormal.append(name)
            except ValueError:
                pass
        elif m.group(3):
            abnormal.append(name)
        tests.append(entry)
    fs.set("tests", "; ".join(tests[:40]) if tests else None, 0.7)
    fs.set("abnormal", ", ".join(abnormal[:20]) if abnormal else None, 0.6)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Lab report", fs.get("lab"), fs.get("report_date")),
                  doc_date=fs.get("report_date") or fs.get("sample_date"), issuer=fs.get("lab"),
                  person_name=fs.get("patient"), pages=pages)
