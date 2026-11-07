"""Event pass / ticket parser (concerts, shows, movies, matches, conferences).

Contract: parse(text, pages, boxes) -> ParsedDocument. Uses the decoded barcode payload too: JSON or
key=value payloads fill booking id, seat and gate; a plain short id becomes the booking id.
"""
from __future__ import annotations

import json
import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, all_barcodes, amount_after, date_after, find_dates, largest_text,
    name_after, normalize_text, parse_time, split_lines,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "event_pass"
FIELDS = [
    ("event", "Event"), ("venue", "Venue"), ("date", "Date"), ("time", "Time"), ("section", "Section"),
    ("row", "Row"), ("seat", "Seat"), ("gate", "Gate"), ("booking_id", "Booking ID"), ("attendee", "Name"),
    ("quantity", "Tickets"), ("category", "Ticket type"), ("price", "Price"), ("organiser", "Organiser"),
]
KEY_FIELDS = ["event", "venue", "date", "booking_id"]
_SKIP_TITLE = ("ticket", "e-ticket", "admit", "booking", "entry pass", "this is your", "order", "scan", "gate",
               "seat", "venue", "date", "time", "powered by", "bookmyshow", "paytm insider", "district", "ticketmaster",
               "pass", "row", "name")
_PAYLOAD_KEYS = {"booking_id": ("booking", "bookingid", "booking_id", "order", "orderid", "order_id", "ticket",
                                "ticketid", "ticket_id", "id", "ref", "reference", "confirmation"),
                 "seat": ("seat", "seatno", "seat_no", "seats"), "gate": ("gate", "entry", "entrygate"),
                 "section": ("section", "block", "stand", "zone"), "row": ("row",),
                 "event": ("event", "eventname", "event_name", "show"), "attendee": ("name", "attendee", "guest")}
ORGANISERS = ["BookMyShow", "Paytm Insider", "District", "Ticketmaster", "Eventbrite", "Skillbox", "Zomato Live",
              "Townscript", "Konfhub", "Luma", "PVR", "INOX", "Cinepolis", "Ticketnew"]


def payload_fields(payload: str) -> dict[str, str]:
    """Fields from a barcode payload: JSON objects, key=value or key:value lists, or a bare id."""
    out: dict[str, str] = {}
    data: dict | None = None
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            data = {str(k).lower().replace(" ", ""): str(v) for k, v in obj.items() if not isinstance(v, (dict, list))}
    except (ValueError, TypeError):
        data = None
    if data is None:
        pairs = re.findall(r"([A-Za-z_ ]{2,20})\s*[=:]\s*([^;&|\n,]{1,60})", payload)
        if len(pairs) >= 1 and ("=" in payload or ";" in payload or "|" in payload or "\n" in payload):
            data = {k.strip().lower().replace(" ", ""): v.strip() for k, v in pairs}
    if data:
        for field, keys in _PAYLOAD_KEYS.items():
            for k in keys:
                if k in data and data[k]:
                    out[field] = data[k]
                    break
    elif re.fullmatch(r"[A-Za-z0-9\-]{6,40}", payload.strip()):
        out["booking_id"] = payload.strip()
    return out


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    event = after_label(t, ["event name", "event", "show", "movie", "match", "concert"], colon=True)
    if not event and boxes:
        event = largest_text(boxes, skip=_SKIP_TITLE)
    if not event:
        event = next((ln for ln in lines[:6] if not any(s in ln.lower() for s in _SKIP_TITLE)
                      and re.search(r"[A-Za-z]{3}", ln) and len(ln) <= 80 and not find_dates(ln)), None)
    fs.set("event", event, 0.7)
    fs.set("venue", after_label(t, ["venue", "location", "address", "theatre", "cinema", "stadium", "at"],
                                anchored=True) or after_label(t, ["venue"]), 0.75)
    d = date_after(t, ["event date", "show date", "date", "when", "date & time", "date and time"])
    if not d:
        ds = find_dates(t)
        d = ds[0][0] if ds else None
    fs.set("date", d, 0.75)
    tm = None
    for labels in (["show time", "showtime", "event time", "time", "starts at", "start time", "doors open",
                    "gates open", "date & time", "date and time", "when"],):
        tm = parse_time(after_label(t, labels, stop=False))
    if not tm:
        m = re.search(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)\b", t, re.I)
        tm = parse_time(m.group(0)) if m else None
    fs.set("time", tm, 0.7)
    for key, labels in (("section", ["section", "block", "stand", "zone", "screen", "audi", "level", "tier"]),
                        ("row", ["row"]), ("seat", ["seat no", "seat number", "seats", "seat"]),
                        ("gate", ["entry gate", "gate no", "gate"])):
        v = after_label(t, labels)
        if v:
            fs.set(key, v.split("  ")[0][:30], 0.75)
    bid = after_label(t, ["booking id", "booking ref", "booking reference", "booking no", "order id", "order no",
                          "ticket id", "ticket no", "confirmation no", "confirmation number", "reference no",
                          "registration id"])
    if bid:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9\-]{3,40}", bid)
        fs.set("booking_id", m.group(0) if m else None, 0.8)
    fs.set("attendee", name_after(t, ["attendee name", "attendee", "guest name", "guest", "ticket holder", "name"],
                                  anchored=True), 0.7)
    q = after_label(t, ["quantity", "qty", "no. of tickets", "tickets", "admit"])
    if q:
        m = re.match(r"\s*(\d{1,2})\b", q)
        fs.set("quantity", m.group(1) if m else None, 0.65)
    else:
        m = re.search(r"\badmit\s+(one|two|three|four|\d{1,2})\b", t, re.I)
        if m:
            words = {"one": "1", "two": "2", "three": "3", "four": "4"}
            fs.set("quantity", words.get(m.group(1).lower(), m.group(1)), 0.65)
    fs.set("category", after_label(t, ["ticket type", "ticket category", "category", "pass type", "tier"]), 0.6)
    fs.set("price", amount_after(t, ["total amount", "amount paid", "ticket price", "price", "amount", "total"]), 0.6)
    org = next((o for o in ORGANISERS if o.lower() in t.lower()), None)
    fs.set("organiser", org or after_label(t, ["organised by", "organized by", "presented by", "organiser"]), 0.6)

    for _fmt, payload in all_barcodes(pages or []):
        for key, val in payload_fields(payload).items():
            fs.set(key, val, 0.9)
        break

    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title(fs.get("event") or "Event pass", fs.get("date")),
                  doc_date=fs.get("date"), issuer=fs.get("organiser"), person_name=fs.get("attendee"), pages=pages)
