"""Invoice parser (GST tax invoices, bills, online order invoices).

Contract: parse(text, pages, boxes) -> ParsedDocument; `parse_invoice_like(text, doc_type)` is shared
with the receipt parser. GSTINs are checksum-validated (fields.gstin_valid); the first is the seller's.
Hindi and Marathi bills (बिल, बीजक, चालान, रसीद, पावती) are read with their own labels.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    AMOUNT_LABELS_LOCAL, DATE_LABELS_LOCAL, DEV, FieldSet, ParsedDocument, after_label, amount_after, date_after,
    find_amounts, find_dates, find_gstins, find_pans, format_inr, gstin_valid, normalize_text, parse_amount,
    split_lines, word_regex, words_to_number,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "invoice"
FIELDS = [
    ("invoice_number", "Invoice number"), ("invoice_date", "Invoice date"), ("seller", "Seller"),
    ("seller_gstin", "Seller GSTIN"), ("buyer", "Billed to"), ("buyer_gstin", "Buyer GSTIN"), ("pan", "PAN"),
    ("order_number", "Order number"), ("place_of_supply", "Place of supply"), ("subtotal", "Taxable amount"),
    ("tax", "Tax"), ("total", "Total"), ("amount_in_words", "Amount in words"), ("due_date", "Due date"),
]
KEY_FIELDS = ["invoice_number", "invoice_date", "seller", "total"]
_SKIP_HEAD = ("tax invoice", "invoice", "bill of supply", "original for recipient", "duplicate", "triplicate",
              "receipt", "cash memo", "retail invoice", "page", "payment receipt", "e-receipt")
# headings of Hindi and Marathi bills and receipts, and of the money and salary documents that share
# seller_line
_SKIP_HEAD_LOCAL = ("कर बीजक", "बीजक", "बिल", "चालान", "रसीद", "पावती", "शुल्क रसीद", "फीस रसीद", "नकद ज्ञापन",
                    "वेतन पर्ची", "पगार पावती", "वेतन पत्रक", "पगार पत्रक", "मूल प्रति", "दिनांक", "तारीख")
_SKIP_HEAD_LOCAL_RXS = [re.compile(r"^" + word_regex(s)) for s in _SKIP_HEAD_LOCAL]
_TOTAL_LABELS = (["grand total", "invoice total", "total amount payable", "amount payable", "net payable",
                  "total amount", "net amount", "amount due", "total payable", "balance due", "कुल देय राशि",
                  "कुल राशि", "कुल रकम", "देय राशि", "एकूण देय रक्कम", "एकूण रक्कम", "निव्वळ रक्कम"],
                 ["total", "कुल", "एकूण"])
_TOTAL_RXS = [[re.compile(word_regex(lab), re.I) for lab in group] for group in _TOTAL_LABELS]


def seller_line(lines: list[str]) -> str | None:
    """The seller's name from the head of the document: the first line that is not a heading and has no
    long number. A Latin line wins; a Devanagari line is used only when no Latin line qualifies."""
    for script in (r"[A-Za-z]{3}", "[" + DEV + "]{3}"):
        for ln in lines[:6]:
            low = ln.lower().strip(" :")
            if not low or any(low.startswith(s) for s in _SKIP_HEAD) or re.search(r"\d{3}", ln):
                continue
            if any(rx.search(low) for rx in _SKIP_HEAD_LOCAL_RXS):
                continue
            if re.search(script, ln) and len(ln) <= 80:
                return ln.strip(" :")
    return None


def total_amount(t: str) -> str | None:
    for i, rxs in enumerate(_TOTAL_RXS):
        lines = split_lines(t)
        for ln in reversed(lines):
            low = ln.lower()
            if any(rx.search(low) for rx in rxs):
                if "sub" in low and i == 1:
                    continue
                if re.search(r"total\s+(?:tax|gst|cgst|sgst|igst|discount|qty|quantity|items)", low):
                    continue
                if re.search(r"(?:कुल|एकूण)\s+(?:कर|जीएसटी|मात्रा|नग|वस्तु)(?![" + DEV + "])", low):
                    continue
                amts = [a for a in find_amounts(ln) if a[0] > 0]
                if amts:
                    return format_inr(amts[-1][0])
    return None


def parse_invoice_like(text: str, doc_type: str, fields_spec: list[tuple[str, str]], key_fields: list[str],
                       pages=None) -> tuple[FieldSet, str]:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(fields_spec)
    num_labels = ["invoice no", "invoice number", "invoice #", "inv no", "bill no", "bill number", "invoice id",
                  "बीजक संख्या", "बीजक क्रमांक", "बिल संख्या", "बिल क्रमांक", "बिल नं", "चालान संख्या", "चालान नं"]
    if doc_type == "receipt":
        num_labels = ["receipt no", "receipt number", "receipt #", "receipt id", "रसीद संख्या", "रसीद क्रमांक",
                      "रसीद नं", "पावती क्रमांक", "पावती नं", *num_labels]
    num = after_label(t, num_labels)
    if num:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-_.]{1,30}", num)
        if m:
            fs.set("receipt_number" if doc_type == "receipt" and "receipt_number" in fs.labels else "invoice_number",
                   m.group(0), 0.85)
    date_key = "date" if "date" in fs.labels else "invoice_date"
    d = date_after(t, ["invoice date", "inv date", "bill date", "receipt date", "date of issue", "dated", "date",
                       "बीजक दिनांक", "बिल दिनांक", "रसीद दिनांक", "पावती दिनांक", *DATE_LABELS_LOCAL])
    if not d:
        ds = find_dates(t)
        d = ds[0][0] if ds else None
    fs.set(date_key, d, 0.8)

    gst = [g for g in find_gstins(t) if gstin_valid(g)] or find_gstins(t)
    if gst and "seller_gstin" in fs.labels:
        fs.set("seller_gstin", gst[0], 0.9 if gstin_valid(gst[0]) else 0.5)
        if len(gst) > 1 and "buyer_gstin" in fs.labels:
            fs.set("buyer_gstin", gst[1], 0.9 if gstin_valid(gst[1]) else 0.5)
    seller = after_label(t, ["sold by", "seller", "supplier", "from", "merchant", "billed by", "विक्रेता",
                             "आपूर्तिकर्ता", "पुरवठादार"])
    if seller and not re.search(r"\d{4}", seller):
        fs.set("seller" if "seller" in fs.labels else "merchant", seller, 0.75)
    else:
        fs.set("seller" if "seller" in fs.labels else "merchant", seller_line(lines), 0.6)
    if "buyer" in fs.labels:
        buyer = after_label(t, ["bill to", "billed to", "billing address", "buyer", "customer name", "customer",
                                "ship to", "ग्राहक का नाम", "ग्राहकाचे नाव", "ग्राहक", "खरीदार", "क्रेता"])
        if buyer and not re.search(r"\d{5}", buyer):
            fs.set("buyer", buyer, 0.7)
    if "pan" in fs.labels:
        pans = find_pans(t)
        fs.set("pan", pans[0] if pans else None, 0.7)
    if "order_number" in fs.labels:
        o = after_label(t, ["order no", "order number", "order id", "po no", "purchase order"])
        if o:
            m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-_]{2,30}", o)
            fs.set("order_number", m.group(0) if m else None, 0.75)
    if "place_of_supply" in fs.labels:
        fs.set("place_of_supply", after_label(t, ["place of supply"]), 0.7)
    if "subtotal" in fs.labels:
        fs.set("subtotal", amount_after(t, ["taxable value", "taxable amount", "sub total", "subtotal",
                                            "amount before tax", "कर योग्य राशि", "करयोग्य मूल्य", "करपात्र रक्कम"]),
               0.75)
    if "tax" in fs.labels:
        tax = amount_after(t, ["total tax", "total gst", "gst amount", "tax amount", "कुल कर", "कर राशि", "एकूण कर",
                               "कर रक्कम"])
        if not tax:
            parts = []
            for lab in ("cgst", "sgst", "utgst", "igst"):
                v = after_label(t, [lab], stop=False)
                if v:
                    amts = find_amounts(v)
                    if amts:
                        parts.append(amts[-1][0])
            if parts:
                tax = format_inr(sum(parts))
        fs.set("tax", tax, 0.7)
    total_key = "total" if "total" in fs.labels else "amount"
    total = total_amount(t)
    if doc_type == "receipt":
        total = amount_after(t, ["amount paid", "amount received", "paid amount", "total paid", "amount",
                                 "प्राप्त राशि", "भुगतान राशि", "जमा राशि", "भरलेली रक्कम", "मिळालेली रक्कम",
                                 *AMOUNT_LABELS_LOCAL]) or total
    words = after_label(t, ["amount in words", "rupees in words", "in words", "amount chargeable (in words)",
                            "राशि शब्दों में", "शब्दों में", "अक्षरी रक्कम", "अक्षरी"], stop=False)
    if not words:
        m = re.search(r"(rupees\s+[a-z ]+only)", t, re.I)
        words = m.group(1) if m else None
    if words and "amount_in_words" in fs.labels:
        fs.set("amount_in_words", words, 0.7)
    if not total and words:
        n = words_to_number(words)
        total = format_inr(n) if n else None
    if not total:
        amt = parse_amount(after_label(t, ["total", "कुल", "एकूण"], stop=False) or "")
        total = format_inr(amt) if amt is not None else None
    fs.set(total_key, total, 0.8)
    if "due_date" in fs.labels:
        fs.set("due_date", date_after(t, ["due date", "payment due", "देय तिथि", "देय दिनांक",
                                          "भरण्याची अंतिम तारीख"]), 0.7)
    return fs, t


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    fs, _t = parse_invoice_like(text, DOC_TYPE, FIELDS, KEY_FIELDS, pages)
    seller = fs.get("seller")
    num = fs.get("invoice_number")
    title = join_title(f"Invoice {num}" if num else "Invoice", f"from {seller}" if seller else None)
    bonus = 0.05 if fs.get("seller_gstin") and gstin_valid(fs.get("seller_gstin") or "") else 0.0
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=title.replace(", from", " from") if title else None,
                  doc_date=fs.get("invoice_date"), issuer=seller, person_name=fs.get("buyer"), bonus=bonus, pages=pages)
