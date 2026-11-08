"""Traffic challan parser (docs/ARCHITECTURE.md section 30): e-challan pages and their printouts from the
state traffic police portals, the national Parivahan portal and city police challans.

Contract: parse(text, pages, boxes) -> ParsedDocument with the fields of FIELDS in that order. Labels are
read from "Label   Value" rows and from the table rows the OCR flattens. `pay_url` is the portal that
takes the challan number: portal_for(issuer, challan_number, location) gives Maharashtra's portal
(settings.challan_portal_url) when the issuer, the number or the location says Maharashtra, else the
Parivahan portal. Nothing is fetched from either.
"""
from __future__ import annotations

import logging
import re

from app.config import settings
from app.documents.fields import (
    FieldSet, ParsedDocument, address_after, after_label, amount_after, find_amounts, find_dates,
    find_vehicle_number, format_inr, name_after, normalize_text, split_lines,
)
from app.documents.parsers._base import finish

log = logging.getLogger(__name__)

DOC_TYPE = "traffic_challan"
FIELDS = [
    ("challan_number", "Challan number"), ("date", "Date and time"), ("vehicle_number", "Vehicle number"),
    ("vehicle_owner", "Vehicle owner"), ("owner_address", "Owner address"), ("violations", "Violations"),
    ("amount", "Amount"), ("payment_status", "Payment status"), ("location", "Location"), ("officer", "Officer"),
    ("mobile", "Mobile"), ("evidences", "Evidences"), ("pay_url", "Pay at"),
]
KEY_FIELDS = ["challan_number", "date", "vehicle_number", "amount"]
PARIVAHAN_URL = "https://echallan.parivahan.gov.in/index/accused-challan"
MAHARASHTRA_URL = "https://mahatrafficechallan.gov.in/payechallan/PaymentService.htm"
MAHARASHTRA_WORDS = ("maharashtra", "mumbai", "pune", "thane", "nagpur", "nashik", "navi mumbai", "pimpri",
                     "chinchwad", "aurangabad", "sambhajinagar", "solapur", "kolhapur", "raigad", "raigarh")
CHALLAN_NO_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,6}\d{11,20})(?![A-Z0-9])")
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
_STOP_RE = re.compile(r"(?i)^(driver name|mobile|payment status|comp\.? ?fee|challan location|evidences|violation|mva|"
                      r"section|print|raise grievance|license)")
_LABEL_BITS = re.compile(r"(?i)^(?:vehicle owner|address\.?|name\.?|owner)\s*")
_ISSUER_RE = re.compile(r"(?i)\b(traffic police|police|parivahan|transport department|rto|traffic branch)\b")


def challan_number(t: str) -> str | None:
    v = after_label(t, ["challan no", "challan number", "challan id", "e-challan no", "echallan no", "notice no"])
    for src in (v or "", t):
        m = CHALLAN_NO_RE.search(src.upper())
        if m:
            return m.group(1)
    return None


def _date_and_time(t: str) -> tuple[str | None, str | None]:
    """(ISO date, display 'YYYY-MM-DD HH:MM') from the challan date row."""
    v = after_label(t, ["challan date & time", "challan date and time", "challan date &", "challan date",
                        "date & time", "date and time", "date of challan", "violation date", "offence date",
                        "date of offence", "date"], stop=False)
    ds = find_dates(v) if v else []
    if not ds:
        ds = find_dates(t)
    if not ds:
        return None, None
    iso = ds[0][0]
    m = _TIME_RE.search(v or "")
    return iso, f"{iso} {int(m.group(1)):02d}:{m.group(2)}" if m else iso


def _owner_address(t: str, lines: list[str]) -> str | None:
    v = address_after(t, ["vehicle owner address", "owner address", "owner's address"], max_lines=4)
    if v and len(v) >= 8:
        return v
    # the portal's table: "Vehicle Owner" alone, the name, then address lines with the labels broken over them
    start = next((i for i, ln in enumerate(lines) if re.fullmatch(r"(?i)vehicle owner(?: name)?\.?", ln.strip())), None)
    if start is None:
        return None
    parts: list[str] = []
    for ln in lines[start + 2:start + 9]:
        s = ln.strip()
        if _STOP_RE.match(s):
            break
        s = _LABEL_BITS.sub("", s).strip()
        if s and s.lower() not in ("name.", "name", "address.", "address"):
            parts.append(s)
    out = " ".join(parts).strip(" ,")
    return out if len(out) >= 8 else None


_HEADER_RE = re.compile(r"(?i)\b(?:mva\s+description\s+fine|description|fine|violation\(s\)|violations?|section\(s\)|"
                        r"sections?|offences?)(?=\W|$)")
_SEC_TOKEN = r"\d{2,3}(?:\([a-z0-9]+\))?"


def _violations(lines: list[str]) -> tuple[str | None, str | None]:
    """(violations text, fine) from the table at the end: the sections, the description and the fine. The
    OCR flattens the table's columns into each other, so the sections are picked up wherever they sit
    (slash-joined numbers, a number after 'Sec' or before 'MVA') and the rest of the words are the description."""
    start = next((i for i, ln in enumerate(lines) if re.search(r"(?i)\b(violation|mva|section\(s\)|offence)\b", ln)), None)
    if start is None:
        return None, None
    end = next((i for i in range(start, len(lines)) if re.match(r"(?i)^(print|raise grievance|pay now|note)", lines[i].strip())),
               len(lines))
    block = " ".join(ln.strip() for ln in lines[start:end])
    secs: list[str] = []
    secs += re.findall(rf"(?<![\d/])({_SEC_TOKEN}(?:/{_SEC_TOKEN})+)(?![\d/])", block)
    secs += re.findall(rf"(?i)\bsec(?:tion)?\.?\s*({_SEC_TOKEN})(?![\d/])", block)
    secs += re.findall(rf"(?i)(?<![\d/])({_SEC_TOKEN})\s*(?:of\s+)?(?:the\s+)?mva\b", block)
    amounts = [a for a in find_amounts(block) if a[0] > 0]
    fine = format_inr(max(a[0] for a in amounts)) if amounts else None
    fine_digits = {str(int(a[0])) for a in amounts}   # a fine printed before "MVA" is not a section
    secs = [x for x in dict.fromkeys(secs) if x not in fine_digits]
    desc = _HEADER_RE.sub(" ", block)
    desc = re.sub(r"(?i)\bsec(?:tion)?\.?(?=\s|$)", " ", desc)
    desc = re.sub(r"(?i)\bmva\.?(?=\W|$)|\bmv act\b|\bmotor vehicles? act\b", " ", desc)
    desc = re.sub(r"(?<![A-Za-z])\d[\d/()a-z]*(?![A-Za-z])", " ", desc)
    desc = re.sub(r"\(\s*\)", " ", desc)
    desc = re.sub(r"\s*/\s*", " / ", desc)
    desc = re.sub(r"\s+", " ", desc).strip(" .,:-/")
    desc = re.sub(r"\s+([.,])", r"\1", desc)
    parts = []
    if secs:
        parts.append("Sec " + ", ".join(secs) + " MVA")
    if desc:
        parts.append(desc[0].upper() + desc[1:] if len(desc) > 1 else desc)
    text = ": ".join(parts) if len(parts) == 2 else "".join(parts)
    if fine:
        text = f"{text} ({fine})" if text else fine
    return (text[:300] or None), fine


def _payment_status(t: str) -> str | None:
    v = after_label(t, ["payment status", "challan status", "status"])
    if not v:
        return None
    low = v.lower()
    if re.search(r"\b(unpaid|not paid|pending|due|payable)\b", low):
        return "Unpaid"
    if re.search(r"\b(disposed|closed|cancelled|withdrawn)\b", low):
        return "Disposed"
    if re.search(r"\bpaid\b", low):
        return "Paid"
    return v[:40]


def _location(t: str, lines: list[str]) -> str | None:
    i = next((k for k, ln in enumerate(lines) if re.search(r"(?i)\bchallan location\b|\bplace of (?:violation|offence)\b|\boffence place\b", ln)), None)
    if i is not None:
        parts: list[str] = []
        prev = lines[i - 1].strip() if i > 0 else ""
        if "," in prev and ":" not in prev and not _STOP_RE.match(prev) and not re.search(r"(?i)\b(fee|status|mobile|evidence)", prev):
            parts.append(prev)
        head = re.sub(r"(?i)^.*?(challan location|place of (?:violation|offence)|offence place)\.?:?\s*", "", lines[i]).strip()
        if head:
            parts.append(head)
        for ln in lines[i + 1:i + 4]:
            s = ln.strip()
            if not s or _STOP_RE.match(s) or re.match(r"(?i)^(evidences|mva|violation|section)", s):
                break
            parts.append(s)
        out = " ".join(parts).strip(" ,")
        if len(out) >= 6:
            return out[:200]
    return address_after(t, ["challan location", "location of violation", "location", "place"], max_lines=3)


def _issuer(lines: list[str]) -> str | None:
    for ln in lines[:12]:
        s = ln.strip()
        if _ISSUER_RE.search(s) and not re.search(r"(?i)\bchallan\b", s) and len(s) <= 60 and re.search(r"[A-Za-z]", s):
            return s
    return None


def portal_for(issuer: str | None, challan_number: str | None, location: str | None, text: str = "") -> str:
    hay = " ".join(x for x in (issuer, location) if x).lower()
    first_lines = " ".join(split_lines(normalize_text(text))[:6]).lower() if text else ""
    number = (challan_number or "").upper()
    if any(w in hay or w in first_lines for w in MAHARASHTRA_WORDS) or number.startswith("MH"):
        return settings.challan_portal_url or MAHARASHTRA_URL
    return PARIVAHAN_URL


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    number = challan_number(t)
    fs.set("challan_number", number, 0.95 if number else 0.0)
    iso, shown = _date_and_time(t)
    fs.set("date", shown, 0.8)
    vehicle = find_vehicle_number(after_label(t, ["vehicle no", "vehicle number", "registration no", "regn no",
                                                  "vehicle reg no"]) or "") or find_vehicle_number(t)
    fs.set("vehicle_number", vehicle, 0.9)
    owner = name_after(t, ["vehicle owner name", "owner name", "name of owner", "vehicle owner", "owner"])
    fs.set("vehicle_owner", owner, 0.8)
    fs.set("owner_address", _owner_address(t, lines), 0.6)
    violations, fine = _violations(lines)
    fs.set("violations", violations, 0.7)
    amount = amount_after(t, ["comp.fee (rs)", "comp.fee", "comp. fee", "comp fee", "compounding fee",
                              "compounding amount", "total fine", "fine amount", "total amount", "penalty amount",
                              "amount"]) or fine
    fs.set("amount", amount, 0.85 if amount else 0.0)
    fs.set("payment_status", _payment_status(t), 0.85)
    fs.set("location", _location(t, lines), 0.6)
    officer = name_after(t, ["officer name", "officer", "issued by"])
    designation = after_label(t, ["designation", "rank"])
    if officer and designation and len(designation) <= 20:
        officer = f"{officer} ({designation})"
    fs.set("officer", officer, 0.7)
    mobile = after_label(t, ["mobile no", "mobile number", "mobile", "phone no", "phone"])
    fs.set("mobile", mobile[:60] if mobile else None, 0.6)
    ev = len(re.findall(r"(?i)\bimage\s*\d+", t))
    fs.set("evidences", str(ev) if ev else None, 0.7)
    issuer = _issuer(lines)
    fs.set("pay_url", portal_for(issuer, number, fs.get("location"), t), 0.9)
    title = f"Challan {number}" if number else "Traffic challan"
    if vehicle:
        title = f"{title} {vehicle}"
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=title, doc_date=iso, issuer=issuer, person_name=owner, pages=pages)
