"""Parser registry: doc_type -> DocType(group, label, field order, parser).

Contract:
- REGISTRY: dict[str, DocType] in display order (ids, money, education and work, health, passes, other).
- get(doc_type) -> DocType (falls back to 'other'); parse(doc_type, text, pages, boxes) -> ParsedDocument.
Labels and groups match app.api.common.DOC_TYPE_LABELS / DOC_GROUP_OF (a test checks this).
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from datetime import date
from types import ModuleType
from typing import Callable

from app.documents.fields import Box, PageText, ParsedDocument
from app.documents.parsers import (
    aadhaar, bank_statement, boarding_pass, certificate, degree, driving_licence, event_pass, id_card,
    insurance_policy, invoice, lab_report, marksheet, offer_letter, other, pan, parking_receipt, passport,
    prescription, receipt, salary_slip, traffic_challan, vaccination, vehicle_rc, voter_id,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocType:
    type: str
    group: str
    label: str
    field_order: tuple[str, ...]
    field_labels: dict[str, str]
    key_fields: tuple[str, ...]
    parser: Callable[..., ParsedDocument]


_TABLE: list[tuple[ModuleType, str, str]] = [
    (aadhaar, "id", "Aadhaar card"), (pan, "id", "PAN card"), (passport, "id", "Passport"),
    (driving_licence, "id", "Driving licence"), (voter_id, "id", "Voter ID"),
    (vehicle_rc, "id", "Vehicle registration"), (id_card, "id", "ID card"),
    (bank_statement, "money", "Bank statement"), (invoice, "money", "Invoice"), (receipt, "money", "Receipt"),
    (salary_slip, "money", "Salary slip"), (traffic_challan, "money", "Traffic challan"),
    (parking_receipt, "money", "Parking receipt"),
    (marksheet, "education_work", "Marksheet"), (degree, "education_work", "Degree"),
    (certificate, "education_work", "Certificate"), (offer_letter, "education_work", "Offer letter"),
    (insurance_policy, "health", "Insurance policy"), (prescription, "health", "Prescription"),
    (lab_report, "health", "Lab report"), (vaccination, "health", "Vaccination certificate"),
    (event_pass, "pass", "Event pass"), (boarding_pass, "pass", "Boarding pass"),
    (other, "other", "Document"),
]

REGISTRY: dict[str, DocType] = {
    mod.DOC_TYPE: DocType(
        type=mod.DOC_TYPE, group=group, label=label, field_order=tuple(k for k, _ in mod.FIELDS),
        field_labels=dict(mod.FIELDS), key_fields=tuple(mod.KEY_FIELDS), parser=mod.parse,
    )
    for mod, group, label in _TABLE
}
GROUPS = ("id", "money", "education_work", "health", "pass", "other")


def get(doc_type: str | None) -> DocType:
    return REGISTRY.get(doc_type or "other", REGISTRY["other"])


def parse(doc_type: str, text: str, pages: list[PageText] | None = None,
          boxes: list[Box] | None = None, *, today: date | None = None) -> ParsedDocument:
    """Run one type's parser; a parser that raises falls back to the 'other' parser (logged), and when
    that raises too the result is an empty document of the type, so the item's text is still indexed.
    today (the item's capture date) goes to parsers that take it (the boarding pass year)."""
    spec = get(doc_type)
    kw = {"today": today} if today is not None and "today" in inspect.signature(spec.parser).parameters else {}
    try:
        return spec.parser(text or "", pages or [], boxes or [], **kw)
    except Exception:
        log.exception("parser %s failed", spec.type)
    try:
        res = REGISTRY["other"].parser(text or "", pages or [], boxes or [])
    except Exception:
        log.exception("fallback parser failed for %s", spec.type)
        res = ParsedDocument(doc_type=spec.type, confidence=0.05, fields={})
    res.doc_type = spec.type
    return res
