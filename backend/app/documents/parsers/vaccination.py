"""Vaccination certificate parser (CoWIN COVID-19 certificates and other immunisation records).

Contract: parse(text, pages, boxes) -> ParsedDocument with beneficiary, beneficiary id, vaccine,
dose dates and the number of doses.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, date_after, find_dates, find_gender, name_after, normalize_text,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "vaccination"
FIELDS = [
    ("beneficiary_name", "Beneficiary"), ("beneficiary_id", "Beneficiary reference ID"), ("age", "Age"),
    ("gender", "Gender"), ("id_verified", "ID verified"), ("vaccine", "Vaccine"), ("doses", "Doses"),
    ("dose1_date", "First dose"), ("dose2_date", "Second dose"), ("booster_date", "Precaution dose"),
    ("vaccinated_by", "Vaccinated by"), ("centre", "Vaccination centre"), ("certificate_id", "Certificate ID"),
]
KEY_FIELDS = ["beneficiary_name", "beneficiary_id", "vaccine", "dose1_date"]
VACCINES = ["COVISHIELD", "COVAXIN", "Sputnik V", "Corbevax", "Covovax", "ZyCoV-D", "Pfizer", "Moderna",
            "Hepatitis B", "Typhoid", "Influenza", "HPV", "Rabies", "Tetanus", "MMR", "BCG", "Polio"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    fs.set("beneficiary_name", name_after(t, ["beneficiary name", "name of beneficiary", "beneficiary", "name"],
                                          anchored=True) or name_after(t, ["beneficiary name"]), 0.85)
    bid = after_label(t, ["beneficiary reference id", "beneficiary ref id", "beneficiary id", "reference id"])
    if bid:
        m = re.search(r"\d{10,16}|[A-Z0-9]{8,20}", bid.replace(" ", ""))
        fs.set("beneficiary_id", m.group(0) if m else None, 0.85)
    m = re.search(r"\bage\s*[:\-]?\s*(\d{1,3})", t, re.I)
    fs.set("age", m.group(1) if m else None, 0.7)
    fs.set("gender", find_gender(t), 0.75)
    fs.set("id_verified", after_label(t, ["id verified", "photo id", "id used"]), 0.6)
    vac = after_label(t, ["vaccine name", "vaccine", "name of vaccine"])
    hit = next((v for v in VACCINES if re.search(r"(?<![A-Za-z])" + re.escape(v) + r"(?![A-Za-z])", vac or t, re.I)),
               None)
    fs.set("vaccine", hit or vac, 0.8)
    d1 = date_after(t, ["date of 1st dose", "date of first dose", "1st dose", "dose 1", "first dose"])
    d2 = date_after(t, ["date of 2nd dose", "date of second dose", "2nd dose", "dose 2", "second dose"])
    d3 = date_after(t, ["date of precaution dose", "precaution dose", "booster dose", "date of booster dose"])
    if not d1:
        dd = date_after(t, ["date of dose", "date of vaccination", "vaccination date"])
        d1 = dd
    fs.set("dose1_date", d1, 0.8)
    fs.set("dose2_date", d2, 0.8)
    fs.set("booster_date", d3, 0.8)
    m = re.search(r"dose\s*(?:number)?\s*[:\-]?\s*(\d)\s*(?:/|of)\s*(\d)", t, re.I)
    if m:
        fs.set("doses", f"{m.group(1)} of {m.group(2)}", 0.8)
    else:
        n = sum(1 for d in (d1, d2, d3) if d)
        fs.set("doses", str(n) if n else None, 0.6)
    fs.set("vaccinated_by", after_label(t, ["vaccinated by", "vaccinator"]), 0.6)
    fs.set("centre", after_label(t, ["vaccination at", "vaccination centre", "vaccination center", "centre", "center"]),
           0.6)
    cid = after_label(t, ["certificate id", "certificate no", "certificate number"])
    if cid:
        m = re.search(r"[A-Za-z0-9]{6,30}", cid.replace(" ", ""))
        fs.set("certificate_id", m.group(0) if m else None, 0.75)
    last = d3 or d2 or d1
    if not last:
        ds = find_dates(t)
        last = ds[-1][0] if ds else None
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Vaccination certificate", fs.get("vaccine"),
                                                              fs.get("beneficiary_name")),
                  doc_date=last, issuer="CoWIN" if re.search(r"cowin|co-win", t, re.I) else None,
                  person_name=fs.get("beneficiary_name"), pages=pages)
