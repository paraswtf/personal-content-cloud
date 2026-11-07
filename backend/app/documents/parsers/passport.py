"""Passport parser: the machine readable zone (ICAO 9303 TD3, two lines of 44 characters) first,
the printed labels (Indian passport data and back pages, bilingual Hindi and English labels) second.

Contract: parse(text, pages, boxes) -> ParsedDocument; `read_mrz(text)` -> dict | None with every
check digit verified (`checks_ok` counts the valid ones out of 5).
"""
from __future__ import annotations

import logging
import re
from datetime import date

from app.documents.fields import (
    ADDRESS_LABELS_LOCAL, LOCAL_GENDER, FieldSet, ParsedDocument, address_after, after_label, date_after, mrz_valid,
    name_after, nice_name, normalize_text, split_lines,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "passport"
FIELDS = [
    ("surname", "Surname"), ("given_names", "Given names"), ("number", "Passport number"),
    ("nationality", "Nationality"), ("dob", "Date of birth"), ("gender", "Sex"), ("place_of_birth", "Place of birth"),
    ("issue_date", "Date of issue"), ("expiry_date", "Date of expiry"), ("place_of_issue", "Place of issue"),
    ("issuing_country", "Issuing country"), ("file_number", "File number"), ("father_name", "Father or legal guardian"),
    ("mother_name", "Mother"), ("spouse_name", "Spouse"), ("address", "Address"), ("mrz_valid", "MRZ check digits"),
]
KEY_FIELDS = ["surname", "given_names", "number", "dob", "expiry_date", "gender"]

_L1 = re.compile(r"^P[A-Z<][A-Z<]{3}[A-Z<]{30,40}$")
_L2 = re.compile(r"^[A-Z0-9<]{9}[0-9<][A-Z<]{3}\d{6}\d[MFX<]\d{6}\d[A-Z0-9<]{14}[0-9<]\d$")
_SEX = {"M": "Male", "F": "Female", "X": "Unspecified", "<": "Unspecified"}


def _mrz_clean(line: str) -> str:
    s = re.sub(r"\s+", "", line.upper()).replace("\u00ab", "<").replace("\u2039", "<")
    return s


def _yymmdd(s: str, *, future: bool) -> str | None:
    if not re.fullmatch(r"\d{6}", s):
        return None
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:])
    this = date.today().year % 100
    if future:
        year = 2000 + yy
    else:
        year = 2000 + yy if yy <= this else 1900 + yy
    try:
        return date(year, mm, dd).isoformat()
    except ValueError:
        return None


def read_mrz(text: str) -> dict | None:
    lines = [_mrz_clean(ln) for ln in normalize_text(text).split("\n")]
    lines = [ln for ln in lines if ln]
    for i in range(len(lines) - 1):
        l1, l2 = lines[i], lines[i + 1]
        if not l1.startswith("P") or "<<" not in l1:
            continue
        l1 = (l1 + "<" * 44)[:44]
        if len(l2) < 42:
            continue
        l2 = (l2 + "<" * 44)[:44]
        if not _L1.match(l1) or not _L2.match(l2):
            continue
        names = l1[5:].split("<<", 1)
        surname = names[0].replace("<", " ").strip()
        given = names[1].replace("<", " ").strip() if len(names) > 1 else ""
        checks = [
            mrz_valid(l2[0:9], l2[9]),
            mrz_valid(l2[13:19], l2[19]),
            mrz_valid(l2[21:27], l2[27]),
            mrz_valid(l2[28:42], l2[42]),
            mrz_valid(l2[0:10] + l2[13:20] + l2[21:43], l2[43]),
        ]
        return {
            "issuing_country": l1[2:5].replace("<", ""),
            "surname": surname,
            "given_names": re.sub(r"\s+", " ", given),
            "number": l2[0:9].replace("<", ""),
            "nationality": l2[10:13].replace("<", ""),
            "dob": _yymmdd(l2[13:19], future=False),
            "gender": _SEX.get(l2[20], "Unspecified"),
            "expiry_date": _yymmdd(l2[21:27], future=True),
            "personal_number": l2[28:42].replace("<", ""),
            "checks_ok": sum(checks),
            "lines": (l1, l2),
        }
    return None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    bonus = 0.0
    mrz = read_mrz(t)
    if mrz:
        conf = 0.98 if mrz["checks_ok"] == 5 else 0.7
        fs.set("surname", nice_name(mrz["surname"]), conf)
        fs.set("given_names", nice_name(mrz["given_names"]), conf)
        fs.set("number", mrz["number"], conf)
        fs.set("nationality", "Indian" if mrz["nationality"] == "IND" else mrz["nationality"], conf)
        fs.set("dob", mrz["dob"], conf)
        fs.set("gender", mrz["gender"], conf)
        fs.set("expiry_date", mrz["expiry_date"], conf)
        fs.set("issuing_country", mrz["issuing_country"], conf)
        fs.set("mrz_valid", f"{mrz['checks_ok']} of 5", 1.0)
        bonus = 0.1 if mrz["checks_ok"] == 5 else 0.0

    fs.set("surname", name_after(t, ["surname", "उपनाम"]), 0.75)
    fs.set("given_names", name_after(t, ["given name(s)", "given names", "given name", "दिया गया नाम",
                                         "दिए गए नाम"]), 0.75)
    num = after_label(t, ["passport no", "passport number", "पासपोर्ट सं", "पासपोर्ट संख्या", "पासपोर्ट नं"])
    if num:
        m = re.search(r"[A-Z]\s?\d{7}", num.upper())
        if m:
            fs.set("number", m.group(0).replace(" ", ""), 0.8)
    if not fs.has("number"):
        m = re.search(r"(?<![A-Z0-9])([A-Z][0-9]{7})(?![A-Z0-9])", t)
        if m:
            fs.set("number", m.group(1), 0.6)
    fs.set("nationality", after_label(t, ["nationality", "राष्ट्रीयता"]), 0.7)
    fs.set("dob", date_after(t, ["date of birth", "जन्मतिथि", "जन्म तिथि", "जन्म की तारीख"], past=True), 0.8)
    sex = after_label(t, ["sex", "लिंग"])
    if sex and sex[:1].upper() in "MF":
        fs.set("gender", "Male" if sex[:1].upper() == "M" else "Female", 0.7)
    elif sex and re.split(r"[\s/]+", sex)[0] in LOCAL_GENDER:
        fs.set("gender", LOCAL_GENDER[re.split(r"[\s/]+", sex)[0]], 0.7)
    fs.set("place_of_birth", after_label(t, ["place of birth", "जन्म स्थान"]), 0.7)
    fs.set("place_of_issue", after_label(t, ["place of issue", "जारी करने का स्थान"]), 0.75)
    fs.set("issue_date", date_after(t, ["date of issue", "जारी करने की तिथि"], past=True), 0.8)
    fs.set("expiry_date", date_after(t, ["date of expiry", "समाप्ति की तिथि"]), 0.8)
    fno = after_label(t, ["file no", "file number", "फ़ाइल सं", "फाइल सं", "फ़ाइल संख्या"])
    if fno:
        fs.set("file_number", re.split(r"\s", fno)[0], 0.7)
    fs.set("father_name", name_after(t, ["name of father / legal guardian", "name of father", "father's name",
                                         "पिता/कानूनी अभिभावक का नाम", "पिता का नाम"]), 0.75)
    fs.set("mother_name", name_after(t, ["name of mother", "mother's name", "माता का नाम"]), 0.75)
    fs.set("spouse_name", name_after(t, ["name of spouse", "spouse's name", "पति/पत्नी का नाम"]), 0.75)
    fs.set("address", address_after(t, ["address", *ADDRESS_LABELS_LOCAL]), 0.7)

    full = " ".join(p for p in (fs.get("given_names"), fs.get("surname")) if p) or None
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Passport", full), doc_date=fs.get("issue_date"),
                  issuer=fs.get("issuing_country") and ("Republic of India" if fs.get("issuing_country") == "IND"
                                                        else fs.get("issuing_country")),
                  person_name=full, bonus=bonus, pages=pages)
