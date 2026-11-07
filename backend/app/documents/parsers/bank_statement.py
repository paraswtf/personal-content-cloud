"""Bank statement parser.

Contract: parse(text, pages, boxes) -> ParsedDocument with account number, IFSC, period and the
opening and closing balances (amounts formatted with Indian grouping).
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    IFSC_RE, FieldSet, ParsedDocument, after_label, amount_after, find_dates, format_inr, known_name, name_after,
    normalize_text, parse_amount, split_lines,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "bank_statement"
FIELDS = [
    ("bank", "Bank"), ("account_holder", "Account holder"), ("account_number", "Account number"),
    ("account_type", "Account type"), ("ifsc", "IFSC"), ("branch", "Branch"), ("customer_id", "Customer ID"),
    ("period_from", "Period from"), ("period_to", "Period to"), ("opening_balance", "Opening balance"),
    ("closing_balance", "Closing balance"), ("total_credits", "Total credits"), ("total_debits", "Total debits"),
]
KEY_FIELDS = ["bank", "account_number", "period_from", "period_to", "opening_balance", "closing_balance"]

BANKS = [
    "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "Punjab National Bank",
    "Bank of Baroda", "Canara Bank", "Union Bank of India", "IDFC First Bank", "IDFC FIRST Bank", "Yes Bank",
    "IndusInd Bank", "Federal Bank", "Bank of India", "Indian Bank", "Central Bank of India", "Bank of Maharashtra",
    "Indian Overseas Bank", "UCO Bank", "Punjab & Sind Bank", "RBL Bank", "South Indian Bank", "Karur Vysya Bank",
    "AU Small Finance Bank", "Equitas Small Finance Bank", "Bandhan Bank", "DBS Bank", "Standard Chartered",
    "HSBC", "Citibank", "Paytm Payments Bank", "Airtel Payments Bank", "Jana Small Finance Bank", "IDBI Bank",
    "Saraswat Bank", "Cosmos Bank", "Kotak Bank",
]


def _balance(text: str, labels: list[str]) -> str | None:
    v = amount_after(text, labels)
    if v:
        return v
    # tables: label row followed by a row of numbers
    lines = split_lines(text)
    for i, ln in enumerate(lines):
        if any(lab in ln.lower() for lab in labels):
            for nxt in lines[i:i + 2]:
                amt = parse_amount(re.sub(r"^[^\d\u20b9]*", "", nxt))
                if amt is not None:
                    return format_inr(amt)
    return None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    fs.set("bank", known_name(t, BANKS), 0.9)
    fs.set("account_holder", name_after(t, ["account holder name", "account holder", "account name", "customer name",
                                            "name"], anchored=True), 0.75)
    acc = after_label(t, ["account number", "account no", "a/c no", "a/c number", "acct no", "account #"], stop=False)
    if acc:
        m = re.search(r"[X*\d][X*\d \-]{5,22}\d", acc)
        if m:
            fs.set("account_number", re.sub(r"[ \-]", "", m.group(0)), 0.85)
    fs.set("account_type", after_label(t, ["account type", "a/c type", "product"]), 0.7)
    m = IFSC_RE.search(t.upper())
    if m:
        fs.set("ifsc", m.group(1), 0.9)
    fs.set("branch", after_label(t, ["branch name", "branch"]), 0.7)
    cid = after_label(t, ["customer id", "cust id", "cif no", "cif number", "customer no"])
    if cid:
        m = re.search(r"[A-Z0-9]{4,20}", cid.upper())
        fs.set("customer_id", m.group(0) if m else None, 0.7)

    m = re.search(r"(?:statement\s+(?:of\s+account\s+)?(?:for\s+the\s+period\s+)?(?:from|for)|period|from)\s*[:\-]?"
                  r"([^\n]{0,80})", t, re.I)
    period = []
    if m:
        period = find_dates(m.group(0))
    if len(period) < 2:
        m = re.search(r"([^\n]{0,30})\s+(?:to|till|-)\s+([^\n]{0,30})", t, re.I)
        for mm in re.finditer(r"[^\n]*\b(?:to|till)\b[^\n]*", t, re.I):
            ds = find_dates(mm.group(0))
            if len(ds) >= 2:
                period = ds
                break
    if len(period) >= 2:
        fs.set("period_from", period[0][0], 0.85)
        fs.set("period_to", period[1][0], 0.85)
    fs.set("opening_balance", _balance(t, ["opening balance", "balance b/f", "balance brought forward", "opening bal"]), 0.8)
    fs.set("closing_balance", _balance(t, ["closing balance", "balance c/f", "closing bal", "available balance"]), 0.8)
    fs.set("total_credits", amount_after(t, ["total credits", "total credit", "credits"]), 0.7)
    fs.set("total_debits", amount_after(t, ["total debits", "total debit", "debits"]), 0.7)

    bank = fs.get("bank")
    acc = fs.get("account_number")
    period_label = None
    if fs.get("period_from") and fs.get("period_to"):
        period_label = f"{fs.get('period_from')} to {fs.get('period_to')}"
    return finish(DOC_TYPE, fs, KEY_FIELDS,
                  title=join_title(f"{bank} statement" if bank else "Bank statement",
                                   f"account ending {acc[-4:]}" if acc else None, period_label),
                  doc_date=fs.get("period_to"), issuer=bank, person_name=fs.get("account_holder"), pages=pages)
