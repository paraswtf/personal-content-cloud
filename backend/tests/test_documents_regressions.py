"""Regression tests for reviewed documents findings: amounts (Cr/Dr, dates, percentages), month and
date words, two-digit years, Aadhaar inside a VID, boarding pass labels, barcode choice and year,
driving licence validity, salary slip month, ticket payload nulls, fallback parser and text job errors."""
from __future__ import annotations

import errno
import sqlite3
from datetime import date

import pytest
from PIL import UnidentifiedImageError

from app.documents import extract
from app.documents.fields import PageText, amount_after, date_after, find_aadhaar, find_dates, parse_amount, parse_month
from app.documents.parsers import parse
from app.documents.parsers.event_pass import payload_fields
from tests.helpers import make_item


def _vals(res):
    return {k: f.value for k, f in res.fields.items()}


BCBP = "M1" + "SHARMA/RAHUL MR".ljust(20) + "E" + "ABC123 " + "DEL" + "BOM" + "6E " + "2134 " + "340" + "Y" + "014C" \
       + "00001" + "1"


# ------------------------------------------------------------------ amounts

def test_cr_dr_balance_markers_are_not_crore():
    assert amount_after("Closing Balance: 12,345.00 Cr", ["closing balance"]) == "₹12,345"
    assert amount_after("Opening Balance : 5,000.00 CR", ["opening balance"]) == "₹5,000"
    assert amount_after("Closing Balance: 12,345.00 Dr", ["closing balance"]) == "-₹12,345"
    res = parse("bank_statement", "State Bank of India\nAccount Statement\nOpening Balance   1,000.00 Cr\n"
                                  "Closing Balance   2,500.50 Cr")
    v = _vals(res)
    assert v["opening_balance"] == "₹1,000" and v["closing_balance"] == "₹2,500.50"
    # crore written as a multiplier still works
    assert parse_amount("Sum insured Rs 1.25 Cr") == 12500000
    assert parse_amount("2.5 Cr") == 25000000
    assert amount_after("Fee 500 Dr. Sharma", ["fee"]) == "₹500"


@pytest.mark.parametrize("text,label,want", [
    ("Opening Balance as on 01-04-2024: 12,345.67", "opening balance", "₹12,345.67"),
    ("Opening Balance as on 01.04.2024: 12,345.67", "opening balance", "₹12,345.67"),
    ("Opening Balance as on 1 Apr 2024 : 12,345.00", "opening balance", "₹12,345"),
    ("Total Tax (18%): 180.00", "total tax", "₹180"),
    ("Total Premium (incl. 18% GST): 12,345.00", "total premium", "₹12,345"),
    ("Net Pay for March 2024: 55,000.00", "net pay", "₹55,000"),
])
def test_amount_skips_dates_years_and_percentages(text, label, want):
    assert amount_after(text, [label]) == want


def test_rupee_sign_offsets_match_after_normalising():
    assert parse_amount("₹500 on 01/04/2024") == 500
    assert parse_amount("Rs.500") == 500


# ------------------------------------------------------------------ dates and months

def test_words_starting_with_a_month_are_not_months():
    assert find_dates("Subject 17 Marathi 85") == []
    assert find_dates("Paper 12 Marks 25") == []
    assert find_dates("Seat 12 Separate 24") == []
    assert find_dates("Junior 5 2020") == []
    assert [d for d, _s, _e in find_dates("15 Sept 2024, 3 September 2023, Mar 4, 2022, 1 jun 2021")] == [
        "2024-09-15", "2023-09-03", "2022-03-04", "2021-06-01"]


def test_long_s_does_not_crash_dates_or_parsers():
    assert find_dates("12 ſep 2024") == []
    for doc_type in ("other", "invoice"):
        res = parse(doc_type, "Invoice\n12 ſep 2024\nTotal 100")
        assert res.doc_type == doc_type


def test_fallback_parser_failure_gives_empty_document(monkeypatch):
    from app.documents import parsers

    def boom(*_a, **_k):
        raise KeyError("x")

    monkeypatch.setattr(parsers, "REGISTRY", {**parsers.REGISTRY,
                                              "invoice": parsers.REGISTRY["invoice"].__class__(
                                                  **{**parsers.REGISTRY["invoice"].__dict__, "parser": boom}),
                                              "other": parsers.REGISTRY["other"].__class__(
                                                  **{**parsers.REGISTRY["other"].__dict__, "parser": boom})})
    res = parsers.parse("invoice", "anything")
    assert res.doc_type == "invoice" and res.fields == {}


def test_two_digit_years():
    assert date_after("Date of Issue: 15/06/98", ["date of issue"]) == "1998-06-15"
    assert date_after("Valid till: 15/06/35", ["valid till"]) == "2035-06-15"
    res = parse("passport", "REPUBLIC OF INDIA\nPassport No: K1234567\nDate of Issue: 15/06/98")
    assert _vals(res).get("issue_date") == "1998-06-15"


def test_parse_month_two_digit_year():
    assert parse_month("Payslip for the month of Mar-24") == "2024-03"
    assert parse_month("Mar '24") == "2024-03"
    assert parse_month("Mar 15, 2024") is None
    assert parse_month("March 2024") == "2024-03"


def test_salary_slip_month_from_pay_period_not_joining_date():
    res = parse("salary_slip", "Acme Pvt Ltd\nPay Period: Mar-24\nEmployee Name: Rahul Sharma\n"
                               "Date of Joining: 15 Jan 2019\nNet Pay: 55,000.00")
    assert _vals(res)["month"] == "March 2024" and res.doc_date == "2024-03-01"
    res = parse("salary_slip", "Acme Pvt Ltd\nPay Period: 01/03/2024 - 31/03/2024\nEmployee Name: Rahul Sharma\n"
                               "Date of Joining: 15 Jan 2019\nNet Pay: 55,000.00")
    assert _vals(res)["month"] == "March 2024"


# ------------------------------------------------------------------ identifiers

def test_aadhaar_not_taken_from_vid_tail():
    t = "Your Aadhaar No. XXXX XXXX 4321\nVID : 1234 2008 9012 3456"
    assert find_aadhaar(t) is None
    res = parse("aadhaar", "Rahul Sharma\nDOB: 01/01/1990\nMale\n" + t)
    assert _vals(res)["number"] == "XXXX XXXX 4321"


def test_driving_licence_transport_validity_only():
    v = _vals(parse("driving_licence", "DRIVING LICENCE\nDL No: MH12 20110012345\nName: Rahul Sharma\n"
                                       "DOB: 01-01-1990\nValidity(TR): 01-01-2030\nCOV: LMV"))
    assert v["valid_till_transport"] == "2030-01-01" and "valid_till" not in v
    v = _vals(parse("driving_licence", "DRIVING LICENCE\nName: Rahul Sharma\n"
                                       "Validity (NT): 01-01-2040   Validity (TR): 01-01-2030\nCOV: LMV"))
    assert v["valid_till"] == "2040-01-01" and v["valid_till_transport"] == "2030-01-01"
    v = _vals(parse("driving_licence", "DRIVING LICENCE\nName: Rahul Sharma\nValid till: 01-01-2040\nCOV: LMV"))
    assert v["valid_till"] == "2040-01-01"


# ------------------------------------------------------------------ passes

def test_boarding_pass_seat_and_boarding_time_after_longer_labels():
    v = _vals(parse("boarding_pass", "BOARDING PASS\nIndiGo\nName: SHARMA/RAHUL MR\nFlight 6E 2134   Date 12 Mar 2025\n"
                                     "From DEL To BOM\nBoarding 06:15   Departure 06:55\nSeat No. 14C   Gate 22"))
    assert v["seat"] == "14C" and v["boarding_time"] == "06:15"
    v = _vals(parse("boarding_pass", "BOARDING PASS\nAir India\nFlight AI 865\nFrom DEL To BOM\nSeat No: 12A\n"
                                     "Boards at 06:45"))
    assert v["seat"] == "12A" and v["boarding_time"] == "06:45"


def test_boarding_pass_barcode_is_the_bcbp_one():
    pg = PageText(1, "", "none")
    pg.barcodes = [("QRCode", "https://promo.example/x"), ("PDF417", BCBP)]
    res = parse("boarding_pass", "", [pg])
    assert res.barcode_format == "PDF417" and res.barcode == BCBP


def test_boarding_pass_year_from_capture_date(conn):
    item = make_item(conn, taken_at="2025-12-07T10:00:00.000Z")
    pg = PageText(1, "", "none")
    pg.barcodes = [("PDF417", BCBP)]
    parsed, _score = extract.analyze_text("", [pg], item, forced_type="boarding_pass")
    assert parsed.fields["date"].value == "2025-12-06"
    assert parse("boarding_pass", "", [pg], today=date(2025, 12, 7)).fields["date"].value == "2025-12-06"


def test_ticket_payload_nulls_are_skipped():
    assert payload_fields('{"booking_id":"BMS12345","seat":null,"gate":null,"name":null,"vip":false}') == {
        "booking_id": "BMS12345"}


def test_event_pass_barcode_is_the_ticket_one():
    pg = PageText(1, "", "none")
    pg.barcodes = [("QRCode", "https://promo.example/app download now"), ("QRCode", '{"booking_id":"BMS12345"}')]
    res = parse("event_pass", "Coldplay Live\nVenue: DY Patil Stadium\nDate: 18 Jan 2025", [pg])
    assert res.barcode == '{"booking_id":"BMS12345"}' and _vals(res)["booking_id"] == "BMS12345"


# ------------------------------------------------------------------ the text job

def _text_item(conn):
    return make_item(conn, kind="document", data=b"x" * 50, ext=".txt", mime="text/plain", text_status="pending")


def _status(conn, item_id):
    return conn.execute("SELECT text_status FROM items WHERE id = ?", (item_id,)).fetchone()[0]


@pytest.mark.parametrize("exc", [
    sqlite3.OperationalError("database is locked"),
    sqlite3.OperationalError("database or disk is full"),
    OSError(errno.ENOSPC, "No space left on device"),
])
def test_environment_errors_reach_the_worker(conn, monkeypatch, exc):
    item = _text_item(conn)

    def boom(_item):
        raise exc

    monkeypatch.setattr(extract, "_collect", boom)
    with pytest.raises(type(exc)):
        extract.extract_item(conn, item["id"], 1)
    assert _status(conn, item["id"]) == "pending"  # the worker retries or defers, then marks 'error'


def test_bad_file_is_a_verdict(conn, monkeypatch):
    item = _text_item(conn)

    def boom(_item):
        raise UnidentifiedImageError("cannot identify image file")

    monkeypatch.setattr(extract, "_collect", boom)
    extract.extract_item(conn, item["id"], 1)
    assert _status(conn, item["id"]) == "error"


def test_status_write_failure_reaches_the_worker(conn, monkeypatch):
    item = _text_item(conn)

    def boom(_item):
        raise RuntimeError("broken file")

    def locked(*_a):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(extract, "_collect", boom)
    monkeypatch.setattr(extract, "_set_status", locked)
    with pytest.raises(sqlite3.OperationalError):
        extract.extract_item(conn, item["id"], 1)
