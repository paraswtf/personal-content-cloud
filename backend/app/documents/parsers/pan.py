"""PAN card parser (old layout without labels and the newer labelled layout).

Contract: parse(text, pages, boxes) -> ParsedDocument. The number matches [A-Z]{5}[0-9]{4}[A-Z]; the
4th letter gives the holder type (P = individual) and, for individuals, the 5th letter is usually
the first letter of the surname (a match raises confidence).
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    PAN_HOLDER, FieldSet, ParsedDocument, find_pans, name_after, nice_name, normalize_text, parse_date, split_lines,
)
from app.documents.parsers._base import finish, is_name_line, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "pan"
FIELDS = [
    ("name", "Name"), ("number", "PAN"), ("father_name", "Father's name"), ("dob", "Date of birth"),
    ("holder_type", "Holder type"),
]
KEY_FIELDS = ["name", "number", "dob", "father_name"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    bonus = 0.0

    pans = find_pans(t)
    number = pans[0] if pans else None
    if number:
        fs.set("number", number, 0.95)
        fs.set("holder_type", PAN_HOLDER.get(number[3], "Unknown"), 0.9)

    name = name_after(t, ["name"], anchored=True)
    father = name_after(t, ["father's name", "fathers name", "father name"])
    for i, ln in enumerate(lines):
        d = parse_date(ln, past=True)
        if d and not fs.has("dob"):
            fs.set("dob", d, 0.9)
            if not name or not father:
                # old layout: NAME, FATHER'S NAME, DATE OF BIRTH on consecutive lines
                above = [lines[j] for j in range(max(0, i - 3), i) if is_name_line(lines[j])]
                if len(above) >= 2:
                    name = name or nice_name(above[-2])
                    father = father or nice_name(above[-1])
                elif len(above) == 1 and not name:
                    name = nice_name(above[0])
    if name:
        fs.set("name", name, 0.85)
    if father:
        fs.set("father_name", father, 0.8)

    if number and name and number[3] == "P":
        surname = name.split()[-1]
        if surname[:1].upper() == number[4]:
            bonus = 0.05
    person = fs.get("name")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("PAN card", person), issuer="Income Tax Department",
                  person_name=person, bonus=bonus, pages=pages)
