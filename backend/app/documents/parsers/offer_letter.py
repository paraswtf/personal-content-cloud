"""Offer / appointment letter parser.

Contract: parse(text, pages, boxes) -> ParsedDocument with employer, candidate, designation, CTC
(lakh, crore and LPA figures normalised to rupees), joining date and location.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, date_after, find_dates, format_inr, nice_name, normalize_text,
    parse_amount, plausible_name, split_lines,
)
from app.documents.parsers._base import finish, join_title
from app.documents.parsers.invoice import seller_line

log = logging.getLogger(__name__)

DOC_TYPE = "offer_letter"
FIELDS = [
    ("employer", "Employer"), ("candidate", "Candidate"), ("designation", "Designation"), ("ctc", "Annual CTC"),
    ("joining_date", "Joining date"), ("location", "Location"), ("department", "Department"),
    ("reporting_to", "Reporting to"), ("probation", "Probation"), ("letter_date", "Letter date"),
    ("reference", "Reference"), ("acceptance_by", "Accept by"),
]
KEY_FIELDS = ["employer", "candidate", "designation", "ctc", "joining_date"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    m = re.search(r"(?:join|offer you employment (?:with|at)|employment with|welcome you to|on behalf of)\s+"
                  r"([A-Z][A-Za-z0-9&.,' ]{2,60}?(?:Pvt\.? Ltd\.?|Private Limited|Limited|Ltd\.?|LLP|Inc\.?|"
                  r"Technologies|Solutions|Services|Corporation|Corp\.?))", t)
    employer = m.group(1).strip(" ,.") if m else None
    fs.set("employer", employer or seller_line([ln for ln in lines if not re.search(r"offer|appointment|letter|ref",
                                                                                    ln, re.I)]), 0.7)
    m = re.search(r"(?m)^\s*dear\s+(?:mr\.?|ms\.?|mrs\.?|dr\.?)?\s*([A-Za-z][A-Za-z .']{1,60}?)\s*[,:]?\s*$", t, re.I)
    cand = m.group(1).strip() if m and plausible_name(m.group(1).strip()) else None
    if not cand:
        v = after_label(t, ["candidate name", "name of the candidate", "name", "to"], anchored=True)
        cand = v if plausible_name(v) else None
    fs.set("candidate", nice_name(cand), 0.8)
    des = after_label(t, ["designation", "position", "job title", "role"], colon=True)
    if not des:
        m = re.search(r"(?:position|role|post|designation) of\s+([A-Z][A-Za-z\- ]{2,50}?)(?=\s+(?:at|with|in|on|"
                      r"and|reporting)\b|[,.\n])", t)
        des = m.group(1).strip() if m else None
    fs.set("designation", des, 0.8)
    ctc = None
    for rx in (r"(?:ctc|cost to company|annual (?:gross )?compensation|annual salary|gross annual salary|"
               r"total compensation|package)[^\n\d\u20b9]{0,40}([^\n]{0,60})",):
        m = re.search(rx, t, re.I)
        if m:
            amt = parse_amount(m.group(1))
            if amt is not None and amt >= 1000:
                ctc = format_inr(amt)
    fs.set("ctc", ctc, 0.8)
    jd = date_after(t, ["date of joining", "joining date", "start date", "date of commencement"])
    if not jd:
        m = re.search(r"(?:join(?:ing)? (?:us )?on|report(?:ing)? on|commence on|start on|effective from)\s+([^\n]{0,40})",
                      t, re.I)
        if m:
            ds = find_dates(m.group(1))
            jd = ds[0][0] if ds else None
    fs.set("joining_date", jd, 0.8)
    loc = after_label(t, ["location", "work location", "place of posting", "posting"])
    if not loc:
        m = re.search(r"(?:based (?:at|in|out of)|location will be)\s+([A-Z][A-Za-z ,]{2,40}?)(?=[.\n]|\s+office)", t)
        loc = m.group(1).strip(" ,") if m else None
    fs.set("location", loc, 0.7)
    fs.set("department", after_label(t, ["department", "team", "business unit"]), 0.6)
    fs.set("reporting_to", after_label(t, ["reporting to", "reporting manager", "you will report to"]), 0.6)
    m = re.search(r"probation(?:ary)? period of\s+([^\n.,]{2,30})", t, re.I)
    fs.set("probation", m.group(1).strip() if m else None, 0.6)
    ld = date_after(t, ["date", "dated"], anchored=True)
    if not ld:
        ds = find_dates("\n".join(lines[:8]))
        ld = ds[0][0] if ds else None
    fs.set("letter_date", ld, 0.65)
    ref = after_label(t, ["ref no", "ref. no", "reference no", "ref", "our ref"])
    if ref:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-_.]{2,40}", ref)
        fs.set("reference", m.group(0) if m else None, 0.6)
    m = re.search(r"(?:accept(?:ance)?[^\n]{0,60}?(?:by|before|on or before))\s+([^\n]{0,30})", t, re.I)
    if m:
        ds = find_dates(m.group(1))
        fs.set("acceptance_by", ds[0][0] if ds else None, 0.6)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Offer letter", fs.get("employer"), fs.get("designation")),
                  doc_date=fs.get("letter_date"), issuer=fs.get("employer"), person_name=fs.get("candidate"),
                  pages=pages)
