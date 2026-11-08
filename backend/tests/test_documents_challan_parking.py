"""Traffic challans and parking receipts (docs/ARCHITECTURE.md section 30): detection, the two parsers on
anonymised copies of the OCR texts measured on 2026-11-08, the pay portal choice and the startup reparse."""
from __future__ import annotations

import json

import pytest

from app.documents import parsers
from app.documents.detect import detect
from app.documents.parsers import traffic_challan as tc
from tests.test_documents_parsers import TEXTS

CHALLAN = """महाराप्ट्र पोलीस
Maharashtra Traffic Police
Challan Details
Challan No.   PNCCC26009876543
Challan Date &   02-Oct-2026 13:09
Time
Officer Name   Rekha Raut
Designation   WPC
License No.   NA
License Holder   NA
Name.
License Holder NA
Address.
Vehicle No.   MH12AB1234
Vehicle Owner
ANITA S KULKARNI
Name.
FLAT 4, SUNRISE APARTMENTS,
Vehicle Owner   BANER ROAD,
Address.   PUNE,
Maharashtra-411045
Driver Name   NA
Mobile No.   98XXXXXX11, 97XXXXXX22
Payment Status Unpaid
Comp.Fee (Rs)   500
GRJH+PJP, Shivaji Co operative Housing
Challan Location Society, Bhageerath, Gokhalenagar, Pune,
Maharashtra 411016, India
Evidences   image1 image2 image3
MVA   Description   Fine
Section(s)
Causing danger /
obstructions or undue
inconvenience to other
Violation(s)   Sec   users of public place or
122/177 to the passengers by 500
MVA.   allowing the vehicle to
be abandoned or to
remain at rest on any
public place.
Print
Raise Grievance
"""

PARKING = """BILL
IATION CHARGES
P ONE INFRA PUT. LTD
MANAGED BY BALAJI
BILL NO.: 0010 AAA 0000123 MANACEMENT SOLUTION
VEHICLE TYPE: 4 WHEELER 60
U.No:   1234
IN DT: 13/08/26
IN TM: 15:35
TOTAL:E 60.00
2W 0 TO 3 HOURS RS 20/-
AFTER 3 HOURS RS 30/-
4W O TO 4 HOURS RS 60/-
AFTER 4 HOURS RS 20/-
OLY PASS NOT ALLOWED Hrs.Rs.80/-
Ars. Rs.50/- OUNERS RISK
"""

DELHI_CHALLAN = """Delhi Traffic Police
E-Challan
Challan No. DL123456789012345
Challan Date & Time 12-Jan-2026 09:15
Vehicle No. DL8CAF1234
Payment Status Paid
Comp. Fee (Rs) 1000
Violation(s) Sec 184 MVA Dangerous driving 1000
"""


def test_detects_the_challan_and_the_parking_ticket():
    c = detect(CHALLAN)
    assert c.doc_type == "traffic_challan" and c.score >= 0.5
    assert c.scores["parking_receipt"] < 0.5
    p = detect(PARKING)
    assert p.doc_type == "parking_receipt" and p.score >= 0.5
    assert p.scores["traffic_challan"] < 0.5


@pytest.mark.parametrize("doc_type", ["receipt", "invoice", "vehicle_rc", "boarding_pass"])
def test_other_money_and_vehicle_texts_keep_their_type(doc_type):
    assert detect(TEXTS[doc_type]).doc_type == doc_type


def test_challan_fields():
    d = parsers.parse("traffic_challan", CHALLAN)
    f = {k: v.value for k, v in d.fields.items()}
    assert f["challan_number"] == "PNCCC26009876543"
    assert f["date"].startswith("2026-10-02") and "13:09" in f["date"]
    assert f["vehicle_number"].replace(" ", "") == "MH12AB1234"   # the library writes plates with spaces
    assert f["vehicle_owner"].lower() == "anita s kulkarni"
    assert "SUNRISE APARTMENTS" in f["owner_address"] and "411045" in f["owner_address"]
    assert "122/177" in f["violations"] and "500" in f["violations"] and "abandoned" in f["violations"].lower()
    assert "500" in f["amount"]
    assert f["payment_status"] == "Unpaid"
    assert "Gokhalenagar" in f["location"] and "Shivaji" in f["location"]
    assert "Rekha Raut" in f["officer"] and "WPC" in f["officer"]
    assert f["mobile"].startswith("98XXXXXX11")
    assert f["evidences"] == "3"
    assert f["pay_url"] == tc.MAHARASHTRA_URL
    assert d.doc_date == "2026-10-02"
    assert d.title.startswith("Challan PNCCC26009876543") and "MH12AB1234" in d.title.replace(" ", "")
    assert d.issuer == "Maharashtra Traffic Police"
    assert d.person_name and d.person_name.lower() == "anita s kulkarni"
    assert d.confidence >= 0.8


def test_parking_fields():
    d = parsers.parse("parking_receipt", PARKING)
    f = {k: v.value for k, v in d.fields.items()}
    assert f["receipt_number"] == "0010 AAA 0000123"
    assert f["vehicle_number"].endswith("1234")
    assert f["vehicle_type"] == "4 wheeler"
    assert f["in_time"] == "2026-08-13 15:35"
    assert "out_time" not in f and "duration" not in f
    assert "60" in f["amount"]
    assert "P ONE INFRA PVT. LTD" in f["operator"] and "BALAJI" in f["operator"]
    assert "HOURS RS 20" in f["tariff"]
    assert d.doc_date == "2026-08-13"
    assert d.title.startswith("Parking P ONE INFRA")
    assert d.issuer and "P ONE INFRA" in d.issuer


def test_parking_out_time_and_duration():
    text = PARKING + "OUT DT: 13/08/26\nOUT TM: 18:05\n"
    f = {k: v.value for k, v in parsers.parse("parking_receipt", text).fields.items()}
    assert f["out_time"] == "2026-08-13 18:05" and f["duration"] == "2 h 30 min"


def test_portal_choice(monkeypatch):
    from app.config import settings

    d = parsers.parse("traffic_challan", DELHI_CHALLAN)
    f = {k: v.value for k, v in d.fields.items()}
    assert f["pay_url"] == tc.PARIVAHAN_URL and f["payment_status"] == "Paid" and f["challan_number"] == "DL123456789012345"
    assert tc.portal_for("Maharashtra Traffic Police", None, None) == tc.MAHARASHTRA_URL
    assert tc.portal_for(None, "MH46CV7441X", None) == tc.MAHARASHTRA_URL
    assert tc.portal_for("Pune City Police", None, None) == tc.MAHARASHTRA_URL
    assert tc.portal_for("Karnataka Traffic Police", "KA01AB1234", "Bengaluru") == tc.PARIVAHAN_URL
    monkeypatch.setattr(settings, "challan_portal_url", "https://example.test/pay")
    assert tc.portal_for("Maharashtra Traffic Police", None, None) == "https://example.test/pay"


def _document(conn, text: str, *, source: str = "auto") -> int:
    from app.db import tx
    from app.util import now_iso
    from tests.helpers import make_item

    item = make_item(conn, kind="document", ext=".txt", write_files=False, data=None)
    now = now_iso()
    with tx(conn):
        conn.execute("UPDATE items SET text_status = 'done' WHERE id = ?", (item["id"],))
        conn.execute("INSERT INTO item_text(item_id, text, source, lang, char_count, created_at) VALUES (?, ?, 'plain', 'en', ?, ?)",
                     (item["id"], text, len(text), now))
        conn.execute("INSERT INTO documents(item_id, doc_type, doc_group, confidence, fields, title, source, created_at, updated_at) "
                     "VALUES (?, 'other', 'other', 0.3, '{}', 'Something', ?, ?, ?)", (item["id"], source, now, now))
    return int(item["id"])


def test_startup_queues_a_reparse_for_old_other_documents(conn):
    from app import ingest

    challan = _document(conn, CHALLAN)
    kept = _document(conn, CHALLAN, source="user")      # the owner's own type stays
    plain = _document(conn, "Minutes of the society meeting held on 3 March 2026. Agenda: water tank.")
    ingest.recover_on_startup(conn)
    queued = {json.loads(r["payload"])["item_id"] for r in conn.execute("SELECT payload FROM jobs WHERE type = 'text' AND status = 'queued'")}
    assert challan in queued and kept not in queued and plain not in queued
    assert conn.execute("SELECT value FROM kv WHERE key = 'doc_rules_version'").fetchone()[0] == str(ingest.DOC_RULES_VERSION)
    # the second start has nothing more to queue
    before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    ingest.recover_on_startup(conn)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == before


def test_reparse_gives_the_old_document_its_type(conn):
    from app.documents import extract

    item_id = _document(conn, PARKING)
    assert extract.reparse(conn, item_id)
    row = conn.execute("SELECT doc_type, doc_group, title, source FROM documents WHERE item_id = ?", (item_id,)).fetchone()
    assert row["doc_type"] == "parking_receipt" and row["doc_group"] == "money" and row["source"] == "auto"
    assert row["title"].startswith("Parking P ONE INFRA")


# ---------------------------------------------------------------- id cards (section 11)
COLLEGE_ID = """SICSR
Symbiosis Institute of Computer Studies & Research
Building Future Leaders for the IT Industry Worldwide
A Constituent of Symbiosis International University
Re-accredited by NAAC with 'A++' grade
Awarded Category- I by UGC
Rohan Kulkarni
BCA
Director's Signature
23030199999
"""

EMPLOYEE_ID = """Bright Tech Solutions Pvt. Ltd.
IDENTITY CARD
Name: Meera Nair
Designation: Software Engineer
Department: Platform
Emp No: BTS-04512
Blood Group: B+
Valid upto: 31/12/2027
If found please return to Bright Tech Solutions, Hinjewadi, Pune
Authorised Signatory
"""


def test_detects_id_cards_and_keeps_them_apart_from_other_types():
    c = detect(COLLEGE_ID)
    assert c.doc_type == "id_card" and c.score >= 0.5
    e = detect(EMPLOYEE_ID)
    assert e.doc_type == "id_card" and e.score >= 1.0
    for doc_type in ("aadhaar", "pan", "marksheet", "certificate", "degree", "offer_letter"):
        assert detect(TEXTS[doc_type]).doc_type == doc_type, doc_type


def test_college_id_card_fields():
    d = parsers.parse("id_card", COLLEGE_ID)
    f = {k: v.value for k, v in d.fields.items()}
    assert f["name"] == "Rohan Kulkarni" and f["course"] == "BCA" and f["id_number"] == "23030199999"
    assert f["organisation"].startswith("SICSR, Symbiosis Institute of Computer Studies")
    assert d.title == "SICSR ID card" and d.person_name == "Rohan Kulkarni" and d.issuer == f["organisation"]


def test_employee_id_card_fields():
    d = parsers.parse("id_card", EMPLOYEE_ID)
    f = {k: v.value for k, v in d.fields.items()}
    assert f["name"] == "Meera Nair" and f["designation"] == "Software Engineer" and f["department"] == "Platform"
    assert f["id_number"] == "BTS-04512" and f["blood_group"] == "B+" and f["valid_until"] == "2027-12-31"
    assert "Bright Tech Solutions" in f["organisation"]
    assert d.title.endswith("ID card")


def test_startup_reparse_covers_id_cards(conn):
    from app import ingest

    assert "id_card" in ingest.DOC_RULES_NEW_TYPES and ingest.DOC_RULES_VERSION >= 3
    card = _document(conn, COLLEGE_ID)
    ingest.recover_on_startup(conn)
    queued = {json.loads(r["payload"])["item_id"] for r in conn.execute("SELECT payload FROM jobs WHERE type = 'text' AND status = 'queued'")}
    assert card in queued
