"""Prescription parser (clinic and hospital prescriptions, OPD slips).

Contract: parse(text, pages, boxes) -> ParsedDocument with doctor, patient, date, diagnosis and the
medicine lines (Tab, Cap, Syp, Inj, drops) joined with semicolons.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, date_after, find_dates, find_gender, first_line_with, name_after,
    nice_name, normalize_text, split_lines,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "prescription"
FIELDS = [
    ("doctor", "Doctor"), ("qualification", "Qualification"), ("registration_number", "Registration number"),
    ("clinic", "Clinic or hospital"), ("patient", "Patient"), ("age", "Age"), ("gender", "Gender"),
    ("date", "Date"), ("diagnosis", "Diagnosis"), ("medicines", "Medicines"), ("advice", "Advice"),
    ("follow_up", "Follow up"),
]
KEY_FIELDS = ["doctor", "patient", "date", "medicines"]
_MED_RE = re.compile(r"^\s*(?:\d+[.)]\s*)?(tab|tablet|cap|capsule|syp|syrup|inj|injection|oint|ointment|drops?|"
                     r"gel|cream|susp|lotion|inh|sachet)\b\.?\s*(.+)$", re.I)
_QUAL_RE = re.compile(r"\b(MBBS|MD|MS|DNB|BDS|MDS|BAMS|BHMS|DM|MCh|FRCS|MRCP|DCH|DGO|DNB|D\.?Ortho)\b")
CLINIC_WORDS = ("clinic", "hospital", "nursing home", "medical centre", "medical center", "health centre",
                "polyclinic", "healthcare", "multispeciality", "multi speciality", "dispensary")


def doctor_name(t: str) -> str | None:
    m = re.search(r"\bDr\.?\s+([A-Z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*){0,3})", t)
    if m:
        name = re.split(r"\s+(?:MBBS|MD|MS|BDS|BAMS|BHMS|DNB)\b", m.group(1))[0]
        return f"Dr. {nice_name(name) or name}"
    v = name_after(t, ["consultant", "doctor", "consulting doctor", "ref. by", "referred by"])
    return v


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    fs.set("doctor", doctor_name(t), 0.8)
    quals = []
    for q in _QUAL_RE.findall(t):
        if q not in quals:
            quals.append(q)
    fs.set("qualification", ", ".join(quals) if quals else None, 0.7)
    reg = after_label(t, ["reg. no", "regd. no", "registration no", "reg no", "mci reg", "kmc reg", "mmc reg"])
    if reg:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{2,20}", reg)
        fs.set("registration_number", m.group(0) if m else None, 0.7)
    fs.set("clinic", first_line_with(t, CLINIC_WORDS), 0.65)
    fs.set("patient", name_after(t, ["patient name", "patient's name", "pt. name", "name of patient", "patient",
                                     "name"], anchored=True) or name_after(t, ["patient name", "patient"]), 0.8)
    m = re.search(r"\bage\s*[:/\-]?\s*(\d{1,3})\s*(?:y|yrs|years)?", t, re.I)
    fs.set("age", m.group(1) if m else None, 0.7)
    m = re.search(r"\b(\d{1,3})\s*(?:y|yrs|years)\s*/\s*([MF])\b", t, re.I)
    if m:
        fs.set("age", m.group(1), 0.7)
        fs.set("gender", "Male" if m.group(2).upper() == "M" else "Female", 0.7)
    fs.set("gender", find_gender(t), 0.65)
    d = date_after(t, ["date", "dated", "visit date"])
    if not d:
        ds = find_dates(t)
        d = ds[0][0] if ds else None
    fs.set("date", d, 0.75)
    fs.set("diagnosis", after_label(t, ["diagnosis", "provisional diagnosis", "dx", "impression", "c/o",
                                        "chief complaints", "complaints"]), 0.65)
    meds = []
    for ln in lines:
        m = _MED_RE.match(ln)
        if m:
            meds.append(f"{m.group(1).capitalize()} {m.group(2).strip()}"[:100])
    fs.set("medicines", "; ".join(meds[:15]) if meds else None, 0.75)
    fs.set("advice", after_label(t, ["advice", "advised", "instructions"]), 0.6)
    fu = after_label(t, ["follow up", "follow-up", "review after", "review on", "next visit"])
    fs.set("follow_up", fu, 0.6)
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Prescription", fs.get("doctor"), fs.get("date")),
                  doc_date=fs.get("date"), issuer=fs.get("clinic") or fs.get("doctor"), person_name=fs.get("patient"),
                  pages=pages)
