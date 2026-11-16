"""Seed for the map and documents screens (ios/UITests/MapDocsScreens.swift).

Three photos with a place name (two in Panaji, one in Shimla), so the map's Places list has rows (the core's
photos have coordinates but no place names: the CI server has no gazetteer), and three parsed documents of
three groups (an identity card, a boarding pass and an electricity bill) with their search index rows, so the
Documents screen has cards, type chips and something to search. Everything is made up; dates are before
1 May 2026 and every name starts with "MapDocs".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

PLACES = [
    # filename, latitude, longitude, place, city, admin, country, taken
    ("MAPDOCS_1001.JPG", 15.4989, 73.8278, "Panaji, Goa, India", "Panaji", "Goa", "India", "2026-03-14T10:20:00.000Z"),
    ("MAPDOCS_1002.JPG", 15.5003, 73.8301, "Panaji, Goa, India", "Panaji", "Goa", "India", "2026-03-15T08:05:00.000Z"),
    ("MAPDOCS_1003.JPG", 31.1048, 77.1734, "Shimla, Himachal Pradesh, India", "Shimla", "Himachal Pradesh", "India",
     "2026-02-02T12:40:00.000Z"),
]

DOCUMENTS = [
    {
        "filename": "MAPDOCS_PAN.JPG",
        "taken": "2026-01-20T09:00:00.000Z",
        "lines": ["INCOME TAX DEPARTMENT", "Permanent Account Number Card", "ABCDE1234F",
                  "Name: MAPDOCS HOLDER", "Date of Birth: 01/01/1990"],
        "doc_type": "pan", "doc_group": "id", "title": "MapDocs PAN card", "doc_date": None,
        "issuer": "Income Tax Department", "person_name": "MapDocs Holder",
        "fields": {"number": ("ABCDE1234F", "PAN"), "name": ("MapDocs Holder", "Name"),
                   "dob": ("1990-01-01", "Date of birth")},
    },
    {
        "filename": "MAPDOCS_BOARDING.JPG",
        "taken": "2026-03-13T06:30:00.000Z",
        "lines": ["BOARDING PASS", "MAPDOCS AIR", "From BOM  To GOI", "Flight MD 2134  Seat 14C", "PNR MD7Q2X",
                  "Passenger MAPDOCS HOLDER"],
        "doc_type": "boarding_pass", "doc_group": "pass", "title": "MapDocs flight to Goa", "doc_date": "2026-03-13",
        "issuer": "MapDocs Air", "person_name": "MapDocs Holder",
        "fields": {"pnr": ("MD7Q2X", "PNR"), "flight": ("MD 2134", "Flight"), "from": ("BOM", "From"),
                   "to": ("GOI", "To"), "seat": ("14C", "Seat")},
    },
    {
        "filename": "MAPDOCS_BILL.JPG",
        "taken": "2026-02-28T18:00:00.000Z",
        "lines": ["MAPDOCS POWER", "Electricity bill", "Invoice INV-2026-0042", "Amount due Rs 1,240.00",
                  "Due 15/03/2026"],
        "doc_type": "invoice", "doc_group": "money", "title": "MapDocs electricity bill", "doc_date": "2026-02-28",
        "issuer": "MapDocs Power", "person_name": None,
        "fields": {"invoice_number": ("INV-2026-0042", "Invoice number"), "total": ("Rs 1,240.00", "Total"),
                   "due_date": ("2026-03-15", "Due date")},
    },
]


def library(kit) -> dict:
    from app.documents.extract import refresh_fts
    from tests.helpers import text_image

    conn = kit.conn
    for k, (filename, lat, lon, place, city, admin, country, taken) in enumerate(PLACES):
        img = kit.scene(4100 + k, [])
        item = kit.make_item(conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename=filename, taken_at=taken,
                             camera="CI Camera", category="places", gps_lat=lat, gps_lon=lon, place=place,
                             place_city=city, place_admin=admin, place_country=country, place_source="geonames",
                             gps_checked=1)
        kit.previews(item, img)
        refresh_fts(conn, int(item["id"]))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    for doc in DOCUMENTS:
        img = text_image(doc["lines"], w=1000, h=640, size=34)
        item = kit.make_item(conn, data=kit.jpeg(img), width=img.width, height=img.height, filename=doc["filename"],
                             taken_at=doc["taken"], camera="CI Camera", category="documents")
        kit.previews(item, img)
        fields = {key: {"value": value, "label": label, "confidence": 0.95}
                  for key, (value, label) in doc["fields"].items()}
        with kit.tx(conn):
            conn.execute(
                "INSERT INTO documents(item_id, doc_type, doc_group, confidence, fields, title, doc_date, issuer, "
                "person_name, created_at, updated_at) VALUES (?, ?, ?, 0.93, ?, ?, ?, ?, ?, ?, ?)",
                (item["id"], doc["doc_type"], doc["doc_group"], json.dumps(fields), doc["title"], doc["doc_date"],
                 doc["issuer"], doc["person_name"], now, now))
        refresh_fts(conn, int(item["id"]))
    return {"places": len(PLACES), "documents": len(DOCUMENTS)}
