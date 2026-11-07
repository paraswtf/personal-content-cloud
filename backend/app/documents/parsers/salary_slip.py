"""Salary slip (payslip) parser.

Contract: parse(text, pages, boxes) -> ParsedDocument with employer, employee, month (YYYY-MM shown as
'March 2024'), gross earnings, deductions and net pay.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, amount_after, find_dates, find_pans, month_label, name_after, normalize_text,
    parse_month, split_lines, words_to_number, format_inr,
)
from app.documents.parsers._base import finish, join_title
from app.documents.parsers.invoice import seller_line

log = logging.getLogger(__name__)

DOC_TYPE = "salary_slip"
FIELDS = [
    ("employer", "Employer"), ("employee_name", "Employee"), ("employee_id", "Employee ID"),
    ("designation", "Designation"), ("department", "Department"), ("month", "Pay period"),
    ("gross_earnings", "Gross earnings"), ("total_deductions", "Total deductions"), ("net_pay", "Net pay"),
    ("basic", "Basic"), ("hra", "HRA"), ("pan", "PAN"), ("uan", "UAN"), ("bank_account", "Bank account"),
    ("days_paid", "Days paid"),
]
KEY_FIELDS = ["employer", "employee_name", "month", "net_pay"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)
    employer = after_label(t, ["company name", "employer", "company"])
    fs.set("employer", employer or seller_line([ln for ln in lines if not re.search(r"pay\s?slip|salary slip", ln, re.I)]), 0.7)
    fs.set("employee_name", name_after(t, ["employee name", "name of employee", "emp name", "name"], anchored=True)
           or name_after(t, ["employee name", "emp name"]), 0.85)
    eid = after_label(t, ["employee id", "employee code", "emp id", "emp code", "employee no", "emp no", "staff id"])
    if eid:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{1,20}", eid)
        fs.set("employee_id", m.group(0) if m else None, 0.85)
    fs.set("designation", after_label(t, ["designation", "position", "role"]), 0.75)
    fs.set("department", after_label(t, ["department", "dept"]), 0.7)
    month = None
    labelled = False  # a pay-period line exists: never take another date's month from the rest of the text
    for ln in lines:
        if re.search(r"pay\s?slip|salary slip|month|pay period|salary for", ln, re.I):
            explicit = bool(re.search(r"month|pay period|salary for", ln, re.I))
            labelled = labelled or explicit
            month = parse_month(ln)
            if not month and explicit:
                ds = find_dates(ln)
                month = ds[0][0][:7] if ds else None
            if month:
                break
    if not month and not labelled:
        month = parse_month(t)
    fs.set("month", month_label(month), 0.85)
    fs.set("gross_earnings", amount_after(t, ["gross earnings", "total earnings", "gross salary", "gross pay",
                                              "gross"]), 0.8)
    fs.set("total_deductions", amount_after(t, ["total deductions", "total deduction", "deductions"]), 0.8)
    net = amount_after(t, ["net pay", "net salary", "net amount payable", "take home", "net payable", "net paid"])
    if not net:
        m = re.search(r"(rupees\s+[a-z ]+only)", t, re.I)
        n = words_to_number(m.group(1)) if m else None
        net = format_inr(n) if n else None
    fs.set("net_pay", net, 0.85)
    fs.set("basic", amount_after(t, ["basic salary", "basic pay", "basic"]), 0.75)
    fs.set("hra", amount_after(t, ["house rent allowance", "hra"]), 0.75)
    pans = find_pans(t)
    fs.set("pan", pans[0] if pans else None, 0.7)
    uan = after_label(t, ["uan", "uan no", "pf uan"])
    if uan:
        m = re.search(r"\d{12}", uan.replace(" ", ""))
        fs.set("uan", m.group(0) if m else None, 0.8)
    acc = after_label(t, ["bank a/c no", "bank account no", "bank account", "account no", "a/c no"], stop=False)
    if acc:
        m = re.search(r"[X*\d][X*\d ]{5,20}\d", acc)
        fs.set("bank_account", m.group(0).replace(" ", "") if m else None, 0.7)
    days = after_label(t, ["days paid", "paid days", "payable days", "working days"])
    if days:
        m = re.search(r"\d{1,2}(?:\.\d)?", days)
        fs.set("days_paid", m.group(0) if m else None, 0.7)
    doc_date = f"{month}-01" if month else None
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Salary slip", fs.get("month"), fs.get("employer")),
                  doc_date=doc_date, issuer=fs.get("employer"), person_name=fs.get("employee_name"), pages=pages)
