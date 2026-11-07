"""Receipt parser (payment receipts, fee receipts, UPI and card payment confirmations).

Contract: parse(text, pages, boxes) -> ParsedDocument; shares extraction with parsers.invoice.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import ParsedDocument, after_label, name_after
from app.documents.parsers._base import finish, join_title
from app.documents.parsers.invoice import parse_invoice_like

log = logging.getLogger(__name__)

DOC_TYPE = "receipt"
FIELDS = [
    ("receipt_number", "Receipt number"), ("date", "Date"), ("merchant", "Paid to"), ("payer", "Received from"),
    ("amount", "Amount"), ("payment_mode", "Payment mode"), ("transaction_id", "Transaction ID"),
    ("purpose", "Towards"), ("seller_gstin", "GSTIN"), ("amount_in_words", "Amount in words"),
]
KEY_FIELDS = ["receipt_number", "date", "merchant", "amount"]
_MODES = ["UPI", "Cash", "Cheque", "Credit Card", "Debit Card", "Card", "Net Banking", "NEFT", "RTGS", "IMPS",
          "Wallet", "Demand Draft"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    fs, t = parse_invoice_like(text, DOC_TYPE, FIELDS, KEY_FIELDS, pages)
    fs.set("payer", name_after(t, ["received from", "received with thanks from", "payer", "paid by", "student name",
                                   "customer name"]), 0.75)
    mode = after_label(t, ["payment mode", "mode of payment", "paid via", "payment method", "paid by"])
    if mode:
        hit = next((m for m in _MODES if m.lower() in mode.lower()), None)
        fs.set("payment_mode", hit or mode, 0.75)
    else:
        hit = next((m for m in _MODES if re.search(r"(?<![A-Za-z])" + re.escape(m) + r"(?![A-Za-z])", t, re.I)), None)
        fs.set("payment_mode", hit, 0.5)
    tx = after_label(t, ["transaction id", "txn id", "utr", "upi ref no", "upi reference", "reference no", "ref no",
                         "rrn"])
    if tx:
        m = re.search(r"[A-Za-z0-9]{6,30}", tx)
        fs.set("transaction_id", m.group(0) if m else None, 0.75)
    fs.set("purpose", after_label(t, ["towards", "being", "for", "purpose", "description"]), 0.6)
    merchant = fs.get("merchant")
    amount = fs.get("amount")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Receipt", merchant, amount), doc_date=fs.get("date"),
                  issuer=merchant, person_name=fs.get("payer"), pages=pages)
