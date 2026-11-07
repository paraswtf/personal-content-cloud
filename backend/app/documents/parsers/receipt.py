"""Receipt parser (payment receipts, fee receipts, UPI and card payment confirmations).

Contract: parse(text, pages, boxes) -> ParsedDocument; shares extraction with parsers.invoice. Hindi and
Marathi receipts (रसीद, पावती) are read with their own labels; payment modes are given in English.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import NAME_LABELS_LOCAL, ParsedDocument, after_label, local_name, name_after, word_regex
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
_MODES_LOCAL = [("नकद", "Cash"), ("नगद", "Cash"), ("रोख", "Cash"), ("रोखीने", "Cash"), ("चेक", "Cheque"),
                ("धनादेश", "Cheque"), ("यूपीआई", "UPI"), ("नेट बैंकिंग", "Net Banking"), ("नेट बँकिंग", "Net Banking"),
                ("डिमांड ड्राफ्ट", "Demand Draft"), ("कार्ड", "Card")]
_MODES_LOCAL_RXS = [(re.compile(word_regex(w)), v) for w, v in _MODES_LOCAL]


# 'श्री राहुल शर्मा से धन्यवाद सहित प्राप्त किए', 'श्री. राहुल शर्मा यांच्याकडून ... मिळाले'
_PAYER_LOCAL = re.compile(r"(?m)(?:(?:श्रीमती|श्री|सुश्री|कु)\.?[ ]+|^)([\u0900-\u0963\u0971-\u097F]{2,}(?:[ ]+[\u0900-\u0963"
                          r"\u0971-\u097F]{2,}){0,3})[ ]+(?:से|यांच्याकडून|कडून)(?![\u0900-\u097F])")


def _local_mode(text: str) -> str | None:
    return next((v for rx, v in _MODES_LOCAL_RXS if rx.search(text)), None)


def _local_payer(text: str) -> str | None:
    m = _PAYER_LOCAL.search(text)
    return local_name(m.group(1)) if m else None


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    fs, t = parse_invoice_like(text, DOC_TYPE, FIELDS, KEY_FIELDS, pages)
    fs.set("payer", name_after(t, ["received from", "received with thanks from", "payer", "paid by", "student name",
                                   "customer name", "भुगतानकर्ता का नाम", "भुगतानकर्ता", "जमाकर्ता", "विद्यार्थी का नाम",
                                   "ग्राहक का नाम", "विद्यार्थ्याचे नाव", "ग्राहकाचे नाव", "भरणाऱ्याचे नाव"]) or
           name_after(t, NAME_LABELS_LOCAL, anchored=True) or _local_payer(t), 0.75)
    mode = after_label(t, ["payment mode", "mode of payment", "paid via", "payment method", "paid by",
                           "भुगतान का माध्यम", "भुगतान माध्यम", "भुगतान का प्रकार", "भरणा पद्धत", "पेमेंट मोड"])
    if mode:
        hit = next((m for m in _MODES if m.lower() in mode.lower()), None) or _local_mode(mode)
        fs.set("payment_mode", hit or mode, 0.75)
    else:
        hit = next((m for m in _MODES if re.search(r"(?<![A-Za-z])" + re.escape(m) + r"(?![A-Za-z])", t, re.I)), None)
        fs.set("payment_mode", hit or _local_mode(t), 0.5)
    tx = after_label(t, ["transaction id", "txn id", "utr", "upi ref no", "upi reference", "reference no", "ref no",
                         "rrn", "लेनदेन आईडी", "लेनदेन संख्या", "संदर्भ संख्या", "संदर्भ क्रमांक", "व्यवहार क्रमांक"])
    if tx:
        m = re.search(r"[A-Za-z0-9]{6,30}", tx)
        fs.set("transaction_id", m.group(0) if m else None, 0.75)
    fs.set("purpose", after_label(t, ["towards", "being", "for", "purpose", "description", "हेतु", "उद्देश्य", "विवरण",
                                      "बाबत", "तपशील"]), 0.6)
    merchant = fs.get("merchant")
    amount = fs.get("amount")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Receipt", merchant, amount), doc_date=fs.get("date"),
                  issuer=merchant, person_name=fs.get("payer"), pages=pages)
