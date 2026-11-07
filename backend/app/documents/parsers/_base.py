"""Shared bits of the parsers: the result builder and the person-line heuristics.

Contract: `finish(doc_type, fs, key_fields, ...) -> ParsedDocument` turns a FieldSet into the parser
result (fields in the type's order, confidence from the share of key fields found plus checksum bonus).
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, PageText, ParsedDocument, all_barcodes, confidence_from, has_devanagari, is_letter, word_regex,
)

log = logging.getLogger(__name__)


def finish(doc_type: str, fs: FieldSet, key_fields: list[str], *, title: str | None = None,
           doc_date: str | None = None, issuer: str | None = None, person_name: str | None = None,
           bonus: float = 0.0, base: float = 0.25, pages: list[PageText] | None = None,
           barcode: tuple[str, str] | None = None) -> ParsedDocument:
    """barcode=(format, payload) names the code the parser read its fields from (the BCBP of a boarding
    pass); otherwise the document's barcode is the first one found on the pages."""
    code = barcode
    if code is None and pages:
        codes = all_barcodes(pages)
        code = codes[0] if codes else None
    barcode_format, payload = code if code else (None, None)
    return ParsedDocument(
        doc_type=doc_type,
        confidence=confidence_from(fs, key_fields, base=base, bonus=bonus),
        fields=fs.ordered(),
        title=title,
        doc_date=doc_date,
        issuer=issuer,
        person_name=person_name,
        barcode=payload,
        barcode_format=barcode_format,
    )


def join_title(*parts: str | None) -> str | None:
    vals = [p for p in parts if p]
    return ", ".join(vals)[:200] if vals else None


HEADER_WORDS = ("government", "india", "department", "authority", "card", "income tax", "republic", "election",
                "commission", "unique", "identification", "signature", "issue", "valid", "dob", "birth", "male",
                "female", "address", "aadhaar", "aadhar", "permanent", "account", "number", "father", "help",
                "www", "enrol", "vid", "photo", "elector", "identity", "licence", "license", "transport", "union",
                "state", "govt", "name", "year")


# the same for Hindi and Marathi card headers and labels ('भारत' is left out: it is also a given name)
HEADER_WORDS_LOCAL = ("सरकार", "शासन", "आधार", "जन्म", "तिथि", "तारीख", "दिनांक", "पुरुष", "पुरूष", "महिला",
                      "स्त्री", "लिंग", "पता", "पत्ता", "नाम", "नाव", "पिता", "वडिलांचे", "आयकर", "विभाग", "निर्वाचन",
                      "निवडणूक", "आयोग", "पहचान", "प्राधिकरण", "विशिष्ट", "भारतीय", "वर्ष", "मेरा", "मेरी", "कार्ड",
                      "संख्या", "लेखा", "स्थायी", "हस्ताक्षर", "स्वाक्षरी", "जारी", "वैध", "परिवहन", "अनुज्ञप्ति",
                      "परवाना", "चालक", "राज्य", "संघ", "गणराज्य", "पत्र", "प्रमाण", "प्रमाणपत्र", "मंडल", "मंडळ",
                      "परीक्षा", "क्रमांक", "पुत्र", "पुत्री", "पत्नी", "पति")
_HEADER_RXS = [re.compile(word_regex(w), re.I) for w in HEADER_WORDS + HEADER_WORDS_LOCAL]


def name_line_before(lines: list[str], index: int, *, lookback: int = 3) -> str | None:
    """The closest line above `index` that looks like a person's name (letters only, 2 to 5 words,
    no header words). Used for cards that print the name without a label (Aadhaar front). A Latin
    spelling in the window wins over a closer Devanagari one (cards print the two in either order)."""
    local = None
    for j in range(index - 1, max(-1, index - 1 - lookback), -1):
        cand = lines[j].strip()
        if is_name_line(cand):
            if not has_devanagari(cand):
                return cand
            local = local or cand
    return local


def local_name_line_before(lines: list[str], index: int, *, lookback: int = 3) -> str | None:
    """The closest Devanagari name line above `index` (see name_line_before)."""
    for j in range(index - 1, max(-1, index - 1 - lookback), -1):
        cand = lines[j].strip()
        if has_devanagari(cand) and is_name_line(cand):
            return cand
    return None


def is_name_line(cand: str) -> bool:
    if not cand or re.search(r"[\d:/@]", cand):
        return False
    words = cand.split()
    if not 2 <= len(words) <= 5:
        return False
    if any(rx.search(cand) for rx in _HEADER_RXS):
        return False
    return sum(is_letter(c) for c in cand) >= 0.85 * len(cand.replace(" ", "").replace(".", ""))
