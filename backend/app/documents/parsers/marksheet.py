"""Marksheet / grade card / transcript parser.

Contract: parse(text, pages, boxes) -> ParsedDocument; `institution_line(text)` is shared with the
degree and certificate parsers. Hindi and Marathi marksheets (अंक सूची, गुणपत्रक) are read with their own
labels; result words (उत्तीर्ण, प्रथम श्रेणी ...) are given in English.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    DATE_LABELS_LOCAL, FATHER_LABELS_LOCAL, ISSUE_LABELS_LOCAL, MOTHER_LABELS_LOCAL, NAME_LABELS_LOCAL, FieldSet,
    ParsedDocument, after_label, date_after, find_dates, first_line_with, local_name_after, name_after, normalize_text,
    split_lines, word_regex,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "marksheet"
FIELDS = [
    ("student_name", "Student"), ("name_local", "Name (local script)"),
    ("institution", "Institution"), ("exam", "Examination"),
    ("roll_number", "Roll or seat number"), ("enrollment_number", "Enrolment number"), ("year", "Year"),
    ("total_marks", "Total marks"), ("percentage", "Percentage"), ("cgpa", "CGPA"), ("sgpa", "SGPA"),
    ("grade", "Grade"), ("result", "Result"), ("father_name", "Father's name"), ("mother_name", "Mother's name"),
    ("issue_date", "Date of issue"),
]
KEY_FIELDS = ["student_name", "institution", "roll_number", "result"]
INSTITUTION_WORDS = ("university", "board of", "institute", "college", "school", "vidyalaya", "vidyapeeth",
                     "academy", "council", "iit", "nit", "polytechnic", "deemed", "secondary education",
                     "विश्वविद्यालय", "विद्यापीठ", "महाविद्यालय", "विद्यालय", "मंडल", "मंडळ", "परिषद", "संस्थान",
                     "शाला", "शाळा", "बोर्ड")
_INSTITUTION_RXS = [re.compile(word_regex(w), re.I) for w in INSTITUTION_WORDS]
_STUDENT_LOCAL = ["विद्यार्थी का नाम", "छात्र का नाम", "परीक्षार्थी का नाम", "विद्यार्थ्याचे नाव", "विद्यार्थ्याचे नांव",
                  "उमेदवाराचे नाव", "परीक्षार्थीचे नाव"]
# result words -> the English wording used for English marksheets (longer phrases first)
_RESULT_LOCAL = [("प्रथम श्रेणी विशेष योग्यता", "First Class With Distinction"),
                 ("प्रथम श्रेणी विशेष प्राविण्य", "First Class With Distinction"), ("विशेष योग्यता", "Distinction"),
                 ("विशेष प्राविण्य", "Distinction"), ("प्राविण्य", "Distinction"), ("प्रावीण्य", "Distinction"),
                 ("प्रथम श्रेणी", "First Class"), ("द्वितीय श्रेणी", "Second Class"), ("तृतीय श्रेणी", "Pass Class"),
                 ("अनुत्तीर्ण", "Fail"), ("उत्तीर्ण", "Pass"), ("पास", "Pass"), ("फेल", "Fail"), ("अनुपस्थित", "Absent"),
                 ("गैरहजर", "Absent")]
_RESULT_LOCAL_RXS = [(re.compile(word_regex(w)), v) for w, v in _RESULT_LOCAL]
_TOTAL_LOCAL = re.compile(r"(?<![\u0900-\u097F])(?:कुल\s*प्राप्तांक|कुल\s*अंक|एकूण\s*गुण|प्राप्तांक|एकूण)(?![\u0900-\u097F])"
                          r"\s*[:\-]?\s*(\d{2,4}\s*/\s*\d{2,4})")
_PERCENT_LOCAL = re.compile(r"(?<![\u0900-\u097F])(?:प्रतिशत|टक्केवारी|टक्के)(?![\u0900-\u097F])\s*[:\-]?\s*"
                            r"(\d{1,3}(?:\.\d{1,2})?)")


def _local_result(text: str) -> str | None:
    for rx, v in _RESULT_LOCAL_RXS:
        if rx.search(text):
            return v
    return None


def institution_line(text: str) -> str | None:
    for ln in split_lines(text)[:15]:
        if any(rx.search(ln) for rx in _INSTITUTION_RXS):
            v = re.sub(r"\s{2,}.*$", "", ln).strip(" ,.:")
            if 6 <= len(v) <= 120 and not re.search(r"\d{4,}", v):
                return v
    return None


def year_of(text: str) -> str | None:
    m = re.search(r"(?:examination|exam|session|year|held in|academic year|परीक्षा|सत्र|वर्ष)[^\n\d]{0,30}"
                  r"((?:19|20)\d{2}(?:\s*-\s*\d{2,4})?)", normalize_text(text), re.I)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return m.group(1) if m else None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    student = ["name of the candidate", "candidate's name", "candidate name", "name of student", "student's name",
               "student name", "student", *_STUDENT_LOCAL]
    fs.set("student_name", name_after(t, student) or name_after(t, ["name", *NAME_LABELS_LOCAL], anchored=True), 0.85)
    local = local_name_after(t, student) or local_name_after(t, ["name", *NAME_LABELS_LOCAL], anchored=True)
    if local and local != fs.get("student_name"):
        fs.set("name_local", local, 0.75)
    fs.set("institution", institution_line(t), 0.75)
    fs.set("exam", after_label(t, ["examination", "exam", "programme", "program", "course", "class", "semester",
                                   "परीक्षा का नाम", "परीक्षेचे नाव", "कक्षा", "इयत्ता"]), 0.65)
    if not fs.has("exam"):
        line = first_line_with(t, ["परीक्षा", "परीक्षेचा"])
        if line:
            # 'माध्यमिक शालांत प्रमाणपत्र परीक्षा - मार्च 2023' -> the name, without the session
            fs.set("exam", re.split(r"\s+[-,]\s+|\s{2,}|\s+\d", line)[0], 0.55)
    roll = after_label(t, ["roll no", "roll number", "seat no", "seat number", "prn", "exam seat no", "hall ticket no",
                           "अनुक्रमांक", "रोल नं", "आसन क्रमांक", "बैठक क्रमांक", "परीक्षा क्रमांक"])
    if roll:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,20}", roll)
        fs.set("roll_number", m.group(0) if m else None, 0.85)
    enr = after_label(t, ["enrollment no", "enrolment no", "enrollment number", "registration no", "reg. no",
                          "नामांकन संख्या", "नामांकन क्रमांक", "नोंदणी क्रमांक", "पंजीयन क्रमांक"])
    if enr:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,20}", enr)
        fs.set("enrollment_number", m.group(0) if m else None, 0.75)
    fs.set("year", year_of(t), 0.6)
    m = re.search(r"(?:grand\s+)?total(?:\s+marks)?(?:\s+obtained)?\s*[:\-]?\s*(\d{2,4}\s*/\s*\d{2,4}|\d{2,4}\s+out of\s+\d{2,4})",
                  t, re.I) or _TOTAL_LOCAL.search(t)
    if m:
        fs.set("total_marks", re.sub(r"\s+", " ", m.group(1)), 0.75)
    m = re.search(r"percentage\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*%?|(\d{1,3}\.\d{1,2})\s*%", t, re.I)
    if m:
        fs.set("percentage", f"{m.group(1) or m.group(2)}%", 0.8)
    elif (m := _PERCENT_LOCAL.search(t)):
        fs.set("percentage", f"{m.group(1)}%", 0.8)
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
    g = after_label(t, ["overall grade", "final grade", "grade", "ग्रेड", "श्रेणी"])
    if g:
        m = re.match(r"([A-F][+]?|O|PASS|FIRST CLASS|DISTINCTION)\b", g.upper())
        fs.set("grade", m.group(1) if m else None, 0.6)
    res = after_label(t, ["result", "final result", "परिणाम", "परीक्षा परिणाम", "निकाल", "निष्कर्ष"])
    if res:
        m = re.search(r"(first class with distinction|distinction|first class|second class|pass class|pass|fail|"
                      r"passed|promoted|atkt|absent)", res, re.I)
        fs.set("result", m.group(1).title() if m else _local_result(res) or res[:40], 0.8)
    else:
        m = re.search(r"\b(PASSED|PASS|FAIL|DISTINCTION|FIRST CLASS)\b", t)
        fs.set("result", m.group(1).title() if m else _local_result(t), 0.5)
    fs.set("father_name", name_after(t, ["father's name", "fathers name", "father name", *FATHER_LABELS_LOCAL[:4]]),
           0.7)
    fs.set("mother_name", name_after(t, ["mother's name", "mothers name", "mother name", *MOTHER_LABELS_LOCAL[:3]]),
           0.7)
    issue = date_after(t, ["date of issue", "issue date", "date of declaration", "date", *ISSUE_LABELS_LOCAL,
                           "निकाल दिनांक", "परिणाम घोषणा तिथि", *DATE_LABELS_LOCAL], past=True)
    if not issue:
        ds = find_dates(t, past=True)
        issue = ds[-1][0] if ds else None
    fs.set("issue_date", issue, 0.6)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Marksheet", fs.get("exam"), fs.get("year")),
                  doc_date=fs.get("issue_date"), issuer=fs.get("institution"), person_name=fs.get("student_name"),
                  pages=pages)
