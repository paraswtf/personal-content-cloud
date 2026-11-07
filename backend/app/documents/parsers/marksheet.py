"""Marksheet / grade card / transcript parser.

Contract: parse(text, pages, boxes) -> ParsedDocument; `institution_line(text)` is shared with the
degree and certificate parsers.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, date_after, find_dates, name_after, normalize_text, split_lines,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "marksheet"
FIELDS = [
    ("student_name", "Student"), ("institution", "Institution"), ("exam", "Examination"),
    ("roll_number", "Roll or seat number"), ("enrollment_number", "Enrolment number"), ("year", "Year"),
    ("total_marks", "Total marks"), ("percentage", "Percentage"), ("cgpa", "CGPA"), ("sgpa", "SGPA"),
    ("grade", "Grade"), ("result", "Result"), ("father_name", "Father's name"), ("mother_name", "Mother's name"),
    ("issue_date", "Date of issue"),
]
KEY_FIELDS = ["student_name", "institution", "roll_number", "result"]
INSTITUTION_WORDS = ("university", "board of", "institute", "college", "school", "vidyalaya", "vidyapeeth",
                     "academy", "council", "iit", "nit", "polytechnic", "deemed", "secondary education")


def institution_line(text: str) -> str | None:
    for ln in split_lines(text)[:15]:
        low = ln.lower()
        if any(re.search(r"(?<![a-z])" + w + r"(?![a-z])", low) for w in INSTITUTION_WORDS):
            v = re.sub(r"\s{2,}.*$", "", ln).strip(" ,.:")
            if 6 <= len(v) <= 120 and not re.search(r"\d{4,}", v):
                return v
    return None


def year_of(text: str) -> str | None:
    m = re.search(r"(?:examination|exam|session|year|held in|academic year)[^\n\d]{0,30}((?:19|20)\d{2}(?:\s*-\s*\d{2,4})?)",
                  text, re.I)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return m.group(1) if m else None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    fs.set("student_name", name_after(t, ["name of the candidate", "candidate's name", "candidate name",
                                          "name of student", "student's name", "student name", "student"]) or
           name_after(t, ["name"], anchored=True), 0.85)
    fs.set("institution", institution_line(t), 0.75)
    fs.set("exam", after_label(t, ["examination", "exam", "programme", "program", "course", "class", "semester"]), 0.65)
    roll = after_label(t, ["roll no", "roll number", "seat no", "seat number", "prn", "exam seat no", "hall ticket no"])
    if roll:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,20}", roll)
        fs.set("roll_number", m.group(0) if m else None, 0.85)
    enr = after_label(t, ["enrollment no", "enrolment no", "enrollment number", "registration no", "reg. no"])
    if enr:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,20}", enr)
        fs.set("enrollment_number", m.group(0) if m else None, 0.75)
    fs.set("year", year_of(t), 0.6)
    m = re.search(r"(?:grand\s+)?total(?:\s+marks)?(?:\s+obtained)?\s*[:\-]?\s*(\d{2,4}\s*/\s*\d{2,4}|\d{2,4}\s+out of\s+\d{2,4})",
                  t, re.I)
    if m:
        fs.set("total_marks", re.sub(r"\s+", " ", m.group(1)), 0.75)
    m = re.search(r"percentage\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*%?|(\d{1,3}\.\d{1,2})\s*%", t, re.I)
    if m:
        fs.set("percentage", f"{m.group(1) or m.group(2)}%", 0.8)
    elif fs.get("total_marks"):
        a, b = re.findall(r"\d+", fs.get("total_marks") or "")[:2]
        if int(b):
            fs.set("percentage", f"{100 * int(a) / int(b):.2f}%", 0.6)
    m = re.search(r"\bc\.?g\.?p\.?a\.?\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)", t, re.I)
    if m:
        fs.set("cgpa", m.group(1), 0.8)
    m = re.search(r"\bs\.?g\.?p\.?a\.?\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)", t, re.I)
    if m:
        fs.set("sgpa", m.group(1), 0.8)
    g = after_label(t, ["overall grade", "final grade", "grade"])
    if g:
        m = re.match(r"([A-F][+]?|O|PASS|FIRST CLASS|DISTINCTION)\b", g.upper())
        fs.set("grade", m.group(1) if m else None, 0.6)
    res = after_label(t, ["result", "final result"])
    if res:
        m = re.search(r"(first class with distinction|distinction|first class|second class|pass class|pass|fail|"
                      r"passed|promoted|atkt|absent)", res, re.I)
        fs.set("result", m.group(1).title() if m else res[:40], 0.8)
    else:
        m = re.search(r"\b(PASSED|PASS|FAIL|DISTINCTION|FIRST CLASS)\b", t)
        fs.set("result", m.group(1).title() if m else None, 0.5)
    fs.set("father_name", name_after(t, ["father's name", "fathers name", "father name"]), 0.7)
    fs.set("mother_name", name_after(t, ["mother's name", "mothers name", "mother name"]), 0.7)
    issue = date_after(t, ["date of issue", "issue date", "date of declaration", "date"], past=True)
    if not issue:
        ds = find_dates(t, past=True)
        issue = ds[-1][0] if ds else None
    fs.set("issue_date", issue, 0.6)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Marksheet", fs.get("exam"), fs.get("year")),
                  doc_date=fs.get("issue_date"), issuer=fs.get("institution"), person_name=fs.get("student_name"),
                  pages=pages)
