"""ID card parser (docs/ARCHITECTURE.md section 11): the identity cards that are not a government
document with a type of their own, such as a college, school, employee or membership card.

Contract: parse(text, pages, boxes) -> ParsedDocument with the fields of FIELDS in that order. The
organisation is read from the heading lines (an acronym line plus the line with the most organisation
words), the holder's name from a name line that is not an organisation, the number from a label, the
barcode or the longest digit run, the course or role from a label or a short degree token.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, address_after, after_label, all_barcodes, date_after, name_after, nice_name,
    normalize_text, split_lines,
)
from app.documents.parsers._base import finish, is_name_line

log = logging.getLogger(__name__)

DOC_TYPE = "id_card"
FIELDS = [
    ("name", "Name"), ("organisation", "Organisation"), ("id_number", "ID number"), ("course", "Course"),
    ("designation", "Designation"), ("department", "Department"), ("valid_until", "Valid until"),
    ("blood_group", "Blood group"), ("dob", "Date of birth"), ("phone", "Phone"), ("address", "Address"),
]
KEY_FIELDS = ["name", "organisation", "id_number"]
ORG_WORDS = ("institute", "university", "college", "school", "academy", "company", "pvt", "ltd", "limited",
             "corporation", "technologies", "technology", "services", "solutions", "systems", "hospital", "bank",
             "foundation", "trust", "society", "international", "association", "council", "board", "research",
             "studies", "institution", "polytechnic", "vidyalaya", "mahavidyalaya", "industries", "enterprises",
             "organisation", "organization", "group", "club", "library", "department", "ministry")
NOT_A_NAME = ORG_WORDS + ("constituent", "accredited", "awarded", "category", "building", "leaders", "industry",
                          "worldwide", "signature", "director", "principal", "authorised", "authorized", "signatory",
                          "holder", "card", "identity", "valid", "issued", "student", "employee", "member",
                          "grade", "naac", "ugc", "future", "sign")
_ORG_RE = re.compile(r"(?i)\b(" + "|".join(ORG_WORDS) + r")\b")
_NOT_NAME_RE = re.compile(r"(?i)\b(" + "|".join(NOT_A_NAME) + r")\b")
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z&.]{2,9}$")
_DEGREE_RE = re.compile(r"(?i)^(?:b\.?tech|m\.?tech|b\.?sc|m\.?sc|bca|mca|mba|bba|b\.?com|m\.?com|b\.?a|m\.?a|llb|llm|mbbs|"
                        r"b\.?e|m\.?e|ph\.?d|bds|b\.?pharm|m\.?pharm|bfa|mfa|pgdm|diploma|hsc|ssc)"
                        r"(?:\s*[-(]\s*[A-Za-z &.]{1,30}\)?)?(?:\s+(?:[iv]+|\d)(?:st|nd|rd|th)?\s*(?:year|yr|sem(?:ester)?))?$")
_BLOOD_RE = re.compile(r"(?i)\bblood\s*(?:group|grp)?\s*[:\-]?\s*((?:AB|A|B|O)\s*[+-]\s*(?:ve)?)")
_DIGITS_RE = re.compile(r"(?<!\d)(\d{6,16})(?!\d)")


def _organisation(lines: list[str]) -> tuple[str | None, str | None]:
    """(short name for the title, the full organisation line) from the first lines."""
    head = [ln.strip() for ln in lines[:10] if ln.strip()]
    acronym = next((ln for ln in head[:4] if _ACRONYM_RE.match(ln)), None)
    best, best_n = None, 0
    for ln in head:
        n = len(_ORG_RE.findall(ln))
        if n > best_n and len(ln) <= 90 and not re.search(r"(?i)\b(constituent of|accredited|awarded)\b", ln):
            best, best_n = ln, n
    full = ", ".join(x for x in (acronym, best) if x and x != best or (x == best and not acronym)) if best else acronym
    if acronym and best:
        full = f"{acronym}, {best}"
    short = acronym or (best.split(",")[0][:40] if best else None)
    return short, full


def _name(t: str, lines: list[str]) -> str | None:
    v = name_after(t, ["student name", "employee name", "member name", "name of the student", "name"])
    if v:
        return v
    for ln in lines:
        s = ln.strip()
        if is_name_line(s) and not _NOT_NAME_RE.search(s) and not _ACRONYM_RE.match(s):
            return nice_name(s)
    return None


def _id_number(t: str, pages) -> str | None:
    v = after_label(t, ["id no", "id number", "card no", "emp no", "employee no", "employee id", "emp id",
                        "enrolment no", "enrollment no", "roll no", "prn", "reg no", "registration no", "member no",
                        "membership no", "student id", "gr no", "admission no"])
    if v:
        m = re.search(r"[A-Z0-9][A-Z0-9/\-]{3,24}", v.upper())
        if m:
            return m.group(0)
    codes = all_barcodes(pages) if pages else []
    for _fmt, payload in codes:
        if payload and re.fullmatch(r"[A-Za-z0-9/\-]{4,30}", payload.strip()):
            return payload.strip()
    runs = _DIGITS_RE.findall(t)
    return max(runs, key=len) if runs else None


def _course(t: str, lines: list[str]) -> str | None:
    v = after_label(t, ["course", "programme", "program", "class", "branch", "stream", "year"])
    if v and len(v) <= 40:
        return v
    for ln in lines:
        s = ln.strip()
        if 2 <= len(s) <= 40 and _DEGREE_RE.match(s):
            return s
    return None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    name = _name(t, lines)
    fs.set("name", name, 0.75)
    short, org = _organisation(lines)
    fs.set("organisation", org, 0.7)
    fs.set("id_number", _id_number(t, pages), 0.7)
    fs.set("course", _course(t, lines), 0.6)
    fs.set("designation", after_label(t, ["designation", "post", "position", "role"]), 0.7)
    fs.set("department", after_label(t, ["department", "dept"]), 0.7)
    fs.set("valid_until", date_after(t, ["valid upto", "valid up to", "valid till", "valid until", "validity", "expiry",
                                         "expires on", "valid thru"]), 0.7)
    m = _BLOOD_RE.search(t)
    fs.set("blood_group", re.sub(r"\s+", "", m.group(1)).upper().replace("VE", "") if m else None, 0.8)
    fs.set("dob", date_after(t, ["date of birth", "dob", "birth date", "born"], past=True), 0.7)
    phone = after_label(t, ["phone", "mobile", "contact", "tel", "mob"])
    fs.set("phone", phone if phone and len(re.sub(r"\D", "", phone)) >= 10 else None, 0.6)
    fs.set("address", address_after(t, ["address", "add", "residence"], max_lines=3), 0.6)
    title = f"{short} ID card" if short else "ID card"
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=title, issuer=org, person_name=name, pages=pages)
