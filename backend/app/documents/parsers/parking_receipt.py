"""Parking receipt parser (docs/ARCHITECTURE.md section 30): thermal and printed parking tickets.

Contract: parse(text, pages, boxes) -> ParsedDocument with the fields of FIELDS in that order. The OCR of a
thermal print is rough ("U.No" for "V.No", "PUT. LTD" for "PVT. LTD", "OUNERS RISK"), so labels are matched
loosely and a few known slips are corrected in the operator's name. In and out times combine the date and
time rows into one 'YYYY-MM-DD HH:MM' value; a vehicle number may be only its last digits.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, amount_after, find_dates, find_vehicle_number, normalize_text, split_lines,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "parking_receipt"
FIELDS = [
    ("receipt_number", "Receipt number"), ("vehicle_number", "Vehicle number"), ("vehicle_type", "Vehicle type"),
    ("in_time", "In"), ("out_time", "Out"), ("duration", "Duration"), ("amount", "Amount"), ("operator", "Operator"),
    ("location", "Location"), ("tariff", "Tariff"),
]
KEY_FIELDS = ["receipt_number", "in_time", "amount", "operator"]
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)(?!\d)")
_COMPANY_RE = re.compile(r"(?i)\b(pvt|put|ltd|llp|limited|managed by|infra|solutions?|services?|association|society|"
                         r"parking|enterprises?|corporation|municipal|mall|hospital|developers?|estates?|towers?|"
                         r"plaza|complex|trust|co-op|cooperative)\b")
_TARIFF_RE = re.compile(r"(?i)(\d\s*(?:to\s*\d+\s*)?(?:hours?|hrs?)\.?\s*rs|rs\.?\s*\d+\s*/-|per\s*(?:hour|hr)|hrs?\.?\s*rs)")
_OCR_FIXES = (("PUT. LTD", "PVT. LTD"), ("PUT LTD", "PVT LTD"), ("MANACEMENT", "MANAGEMENT"), ("SOLUTIOM", "SOLUTION"))


def _receipt_number(t: str) -> str | None:
    v = after_label(t, ["bill no", "ticket no", "receipt no", "slip no", "token no", "bill number", "ticket number",
                        "receipt number", "sr no", "sl no"])
    if not v:
        return None
    keep: list[str] = []
    for tok in v.split():
        if re.fullmatch(r"[A-Z0-9/\-]{1,10}", tok) and not (tok.isalpha() and len(tok) >= 5):
            keep.append(tok)
        else:
            break
    return " ".join(keep) or None


def _vehicle(t: str) -> str | None:
    v = after_label(t, ["vehicle no", "vehicle number", "veh no", "veh. no", "v.no", "v. no", "u.no", "u. no",
                        "reg no", "regn no", "car no", "bike no", "vehicle"])
    if not v:
        return find_vehicle_number(t)
    full = find_vehicle_number(v)
    if full:
        return full
    m = re.search(r"\b([A-Z]{0,3}\d{3,4})\b", v.upper())
    return m.group(1) if m else None


def _vehicle_type(t: str) -> str | None:
    v = after_label(t, ["vehicle type", "veh type", "veh. type", "type of vehicle", "category"])
    if not v:
        m = re.search(r"(?i)\b([24])\s*-?\s*w(?:heeler)?\b", t)
        return f"{m.group(1)} wheeler" if m else None
    m = re.search(r"(?i)\b([24])\s*-?\s*w(?:heeler)?\b|\b(two|four)\s*wheeler\b", v)
    if m:
        n = m.group(1) or {"two": "2", "four": "4"}[m.group(2).lower()]
        return f"{n} wheeler"
    low = v.lower()
    for word, label in (("car", "4 wheeler"), ("bike", "2 wheeler"), ("scooter", "2 wheeler"), ("bus", "Bus"), ("truck", "Truck")):
        if word in low:
            return label
    return v[:30]


def _stamp(t: str, date_labels: list[str], time_labels: list[str]) -> tuple[str | None, str | None]:
    """(ISO date, 'YYYY-MM-DD HH:MM') from a date row and a time row, or from one row that holds both."""
    dv = after_label(t, date_labels, stop=False)
    ds = find_dates(dv, past=True) if dv else []
    tv = after_label(t, time_labels, stop=False)
    tm = _TIME_RE.search(tv or "") or _TIME_RE.search(dv or "")
    if not ds:
        return None, None
    iso = ds[0][0]
    return iso, f"{iso} {int(tm.group(1)):02d}:{tm.group(2)}" if tm else iso


def _duration(a: str | None, b: str | None) -> str | None:
    try:
        if not a or not b or len(a) < 16 or len(b) < 16:
            return None
        delta = datetime.strptime(b, "%Y-%m-%d %H:%M") - datetime.strptime(a, "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    mins = int(delta.total_seconds() // 60)
    if mins <= 0:
        return None
    h, m = divmod(mins, 60)
    return f"{h} h {m} min" if h else f"{m} min"


def _operator(lines: list[str]) -> str | None:
    cut = next((i for i, ln in enumerate(lines) if re.search(r"(?i)\b(bill|ticket|receipt|slip|token)\s*(no|number)\b", ln)), len(lines))
    picked: list[str] = []
    for ln in lines[:max(cut, 1) + 1][:8]:
        s = ln.strip()
        if re.search(r"(?i)\b(bill|ticket|receipt|slip|token)\s*(no|number)\b", s):
            s = re.sub(r"(?i)^.*?\b(bill|ticket|receipt|slip|token)\s*(no|number)\.?\s*:?\s*[A-Z0-9/\- ]*?(?=[A-Z]{5,}|$)", "", s).strip()
        if s and _COMPANY_RE.search(s) and not re.fullmatch(r"(?i)(parking )?(charges|bill|receipt|ticket)", s):
            for wrong, right in _OCR_FIXES:
                s = s.replace(wrong, right)
            picked.append(s)
    out = ", ".join(dict.fromkeys(picked))
    return out[:120] or None


def _tariff(lines: list[str]) -> str | None:
    hits = [ln.strip() for ln in lines if _TARIFF_RE.search(ln)]
    return "; ".join(hits)[:300] or None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    fs.set("receipt_number", _receipt_number(t), 0.85)
    fs.set("vehicle_number", _vehicle(t), 0.7)
    fs.set("vehicle_type", _vehicle_type(t), 0.75)
    in_iso, in_shown = _stamp(t, ["in dt", "in date", "entry date", "date in", "in day", "entry", "date"],
                              ["in tm", "in time", "entry time", "time in", "in"])
    out_iso, out_shown = _stamp(t, ["out dt", "out date", "exit date", "date out"], ["out tm", "out time", "exit time", "time out"])
    fs.set("in_time", in_shown, 0.85)
    fs.set("out_time", out_shown, 0.8)
    fs.set("duration", _duration(in_shown, out_shown), 0.7)
    amount = amount_after(t, ["total", "amount", "paid", "charges", "fee", "fees", "rs"])
    fs.set("amount", amount, 0.85 if amount else 0.0)
    operator = _operator(lines)
    fs.set("operator", operator, 0.7)
    fs.set("location", after_label(t, ["location", "site", "place", "parking at", "lot", "address"]), 0.6)
    fs.set("tariff", _tariff(lines), 0.6)
    short = (operator or "").split(",")[0].strip() or fs.get("location")
    title = join_title("Parking", short, in_iso) or "Parking receipt"
    title = title.replace("Parking, ", "Parking ")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=title, doc_date=in_iso, issuer=operator, pages=pages)
