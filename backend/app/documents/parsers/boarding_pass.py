"""Boarding pass parser: the IATA BCBP barcode string ('M1' + fixed-width mandatory items) first,
the printed text second.

Contract: parse(text, pages, boxes) -> ParsedDocument; `read_bcbp(payload, today=None)` -> dict | None
(passenger, pnr, from, to, carrier, flight, date ISO, seat, sequence, cabin).
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, all_barcodes, date_after, find_dates, nice_name, normalize_text,
    parse_time,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "boarding_pass"
FIELDS = [
    ("passenger", "Passenger"), ("from", "From"), ("to", "To"), ("flight", "Flight"), ("date", "Date"),
    ("departure_time", "Departure"), ("boarding_time", "Boarding"), ("gate", "Gate"), ("seat", "Seat"),
    ("cabin", "Class"), ("pnr", "PNR"), ("sequence", "Sequence"), ("zone", "Zone"), ("airline", "Airline"),
]
KEY_FIELDS = ["passenger", "from", "to", "flight", "seat", "pnr"]

_BCBP_RE = re.compile(
    r"M([1-4])(.{20})([E ])([A-Z0-9 ]{7})([A-Z]{3})([A-Z]{3})([A-Z0-9 ]{3})([0-9 ]{4}[A-Z ])(\d{3})([A-Z])"
    r"([0-9A-Z ]{4})([0-9A-Z ]{5})(.)")
_TITLE_RE = re.compile(r"\s*\b(MR|MRS|MS|MISS|MSTR|DR)$", re.I)
_CABINS = {"F": "First", "A": "First", "P": "First", "J": "Business", "C": "Business", "D": "Business",
           "I": "Business", "Z": "Business", "R": "Business", "W": "Premium economy", "E": "Premium economy"}
AIRLINES = {"AI": "Air India", "6E": "IndiGo", "UK": "Vistara", "SG": "SpiceJet", "QP": "Akasa Air",
            "IX": "Air India Express", "I5": "AIX Connect", "G8": "Go First", "9I": "Alliance Air", "EK": "Emirates",
            "QR": "Qatar Airways", "EY": "Etihad Airways", "SQ": "Singapore Airlines", "LH": "Lufthansa",
            "BA": "British Airways", "AF": "Air France", "KL": "KLM", "UA": "United Airlines", "AA": "American Airlines",
            "DL": "Delta Air Lines", "TG": "Thai Airways", "MH": "Malaysia Airlines", "CX": "Cathay Pacific",
            "FZ": "flydubai", "G9": "Air Arabia", "WY": "Oman Air", "SV": "Saudia", "TK": "Turkish Airlines"}


def _julian(day: int, today: date | None) -> str | None:
    today = today or date.today()
    if not 1 <= day <= 366:
        return None
    for year in (today.year, today.year - 1, today.year + 1):
        try:
            d = date(year, 1, 1) + timedelta(days=day - 1)
        except OverflowError:
            continue
        if d.year != year:
            continue
        if today - timedelta(days=330) <= d <= today + timedelta(days=200):
            return d.isoformat()
    return (date(today.year, 1, 1) + timedelta(days=day - 1)).isoformat()


def read_bcbp(payload: str, today: date | None = None) -> dict | None:
    m = _BCBP_RE.search(payload or "")
    if not m:
        return None
    g = m.groups()
    name = g[1].strip()
    if "/" in name:
        last, first = name.split("/", 1)
        first = _TITLE_RE.sub("", first.strip())
        name = f"{first} {last}".strip()
    carrier = g[6].strip()
    flight_no = g[7].strip().lstrip("0") or "0"
    seat = g[10].strip().lstrip("0")
    seq = g[11].strip().lstrip("0")
    return {
        "legs": int(g[0]), "passenger": nice_name(name) or name, "pnr": g[3].strip(), "from": g[4], "to": g[5],
        "carrier": carrier, "flight": f"{carrier} {flight_no}", "date": _julian(int(g[8]), today),
        "cabin": _CABINS.get(g[9], "Economy"), "seat": seat or None, "sequence": seq or None,
        "raw": m.group(0),
    }


def parse(text: str, pages=None, boxes=None, *, today: date | None = None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    bcbp = None
    bcbp_code = None
    for fmt, payload in all_barcodes(pages or []):
        bcbp = read_bcbp(payload, today)
        if bcbp:
            bcbp_code = (fmt, payload)
            break
    if not bcbp:
        bcbp = read_bcbp(t.replace("\n", " "), today)
    text_dates = find_dates(t)
    if bcbp:
        for k in ("passenger", "from", "to", "flight", "seat", "pnr", "sequence", "cabin"):
            fs.set(k, bcbp.get(k), 0.95)
        if text_dates and bcbp["date"]:
            # the printed date carries the year the barcode leaves out
            jd = date.fromisoformat(bcbp["date"])
            for iso, _s, _e in text_dates:
                d = date.fromisoformat(iso)
                if (d.month, d.day) == (jd.month, jd.day):
                    bcbp["date"] = iso
                    break
        fs.set("date", bcbp["date"], 0.85)
        fs.set("airline", AIRLINES.get(bcbp["carrier"]), 0.8)

    name = after_label(t, ["name of passenger", "passenger name", "passenger", "name"])
    if name:
        if "/" in name:
            last, first = name.split("/", 1)
            name = f"{_TITLE_RE.sub('', first.strip())} {last.strip()}"
        fs.set("passenger", nice_name(name), 0.75)
    m = re.search(r"\b(?:flight|flt)\.?\s*(?:no\.?)?\s*[:\-]?\s*([A-Z0-9]{2})\s?-?\s?(\d{1,4})\b", t, re.I)
    if m:
        fs.set("flight", f"{m.group(1).upper()} {m.group(2).lstrip('0')}", 0.8)
        fs.set("airline", AIRLINES.get(m.group(1).upper()), 0.7)
    m = re.search(r"\b(?i:from)\s*[:\-]?\s*([A-Z]{3})\b[^\n]{0,40}?\b(?i:to)\s*[:\-]?\s*([A-Z]{3})\b", t)
    if not m:
        m = re.search(r"\b([A-Z]{3})\s*(?:->|\u2192|to|-|\u2708)\s*([A-Z]{3})\b", t)
    if m and m.group(1) not in ("PNR", "THE"):
        fs.set("from", m.group(1), 0.7)
        fs.set("to", m.group(2), 0.7)
    pnr = after_label(t, ["pnr", "booking reference", "booking ref", "pnr no", "confirmation"])
    if pnr:
        mm = re.search(r"\b[A-Z0-9]{5,8}\b", pnr.upper())
        fs.set("pnr", mm.group(0) if mm else None, 0.75)
    seat = after_label(t, ["seat no", "seat number", "seat"],
                       accept=lambda v: bool(re.match(r"\s*\d{1,3}[A-K]\b", v.upper())))
    if seat:
        mm = re.match(r"\s*(\d{1,3}[A-K])\b", seat.upper())
        fs.set("seat", mm.group(1) if mm else None, 0.75)
    gate = after_label(t, ["gate", "boarding gate"])
    if gate:
        mm = re.match(r"\s*([A-Z]?\d{1,3}[A-Z]?)\b", gate.upper())
        fs.set("gate", mm.group(1) if mm else None, 0.75)
    fs.set("boarding_time", parse_time(after_label(t, ["boarding time", "boards at", "boarding"], stop=False,
                                                   accept=lambda v: parse_time(v) is not None)), 0.75)
    fs.set("departure_time", parse_time(after_label(t, ["departure time", "departure", "dep time", "departs",
                                                        "std"], stop=False)), 0.7)
    zone = after_label(t, ["zone", "group", "boarding group", "seq"])
    if zone:
        mm = re.match(r"\s*([A-Z0-9]{1,3})\b", zone.upper())
        fs.set("zone", mm.group(1) if mm else None, 0.6)
    fs.set("date", date_after(t, ["date", "flight date", "date of journey", "departure date"]) or
           (text_dates[0][0] if text_dates else None), 0.7)
    cabin = after_label(t, ["class", "cabin"])
    if cabin:
        fs.set("cabin", cabin.split()[0].capitalize(), 0.5)

    route = f"{fs.get('from')} to {fs.get('to')}" if fs.get("from") and fs.get("to") else None
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title(f"Boarding pass {route}" if route else "Boarding pass",
                                                              fs.get("flight")),
                  doc_date=fs.get("date"), issuer=fs.get("airline"), person_name=fs.get("passenger"),
                  bonus=0.1 if bcbp else 0.0, pages=pages, barcode=bcbp_code)
