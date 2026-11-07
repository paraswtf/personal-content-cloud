"""Driving licence parser (Indian smart cards and Sarathi printouts of several states).

Contract: parse(text, pages, boxes) -> ParsedDocument. Licence numbers follow fields.DL_RES (the
national SS RR YYYY NNNNNNN format and older state formats); the state comes from the prefix.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, address_after, after_label, date_after, find_dates, find_dl_number, name_after,
    normalize_text, relation_name, split_lines,
)
from app.documents.parsers._base import finish, join_title, name_line_before

log = logging.getLogger(__name__)

DOC_TYPE = "driving_licence"
FIELDS = [
    ("name", "Name"), ("number", "Licence number"), ("dob", "Date of birth"), ("relation", "Relation"),
    ("relative_name", "Father or spouse"), ("blood_group", "Blood group"), ("address", "Address"),
    ("issue_date", "Date of issue"), ("valid_till", "Valid till (non-transport)"),
    ("valid_till_transport", "Valid till (transport)"), ("vehicle_classes", "Vehicle classes"),
    ("issuing_authority", "Issuing authority"), ("state", "State"),
]
KEY_FIELDS = ["name", "number", "dob", "valid_till", "vehicle_classes"]
_COV = ["MCWOG", "MCWG", "M/CYCL.WG", "MCYCL", "LMV-NT", "LMV-TR", "LMV", "HMV", "HGMV", "HPMV", "HTV", "MGV", "TRANS",
        "LMV-TAXI", "E-RICKSHAW", "ERIKSHAW", "INVCRG", "3W-NT", "3W-T", "TRCTR", "ADPVEH", "FVG", "LDRXCV"]


# the value after a generic 'Validity' / 'Valid till' label that is tagged as the transport one
_TR_TAG = re.compile(r"^[\s(\[:\-]*(?:till|upto|up to)?[\s(\[:\-]*(?:tr|transport)\b", re.I)


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    lines = split_lines(t)
    fs = FieldSet(FIELDS)

    dl = None
    num_label = after_label(t, ["dl no", "d.l. no", "licence no", "license no", "licence number", "license number",
                                "dl number"], stop=False)
    if num_label:
        dl = find_dl_number(num_label)
    dl = dl or find_dl_number(t)
    if dl:
        fs.set("number", dl[0], 0.9)
        fs.set("state", dl[1], 0.8)

    name = name_after(t, ["name"], anchored=True)
    fs.set("dob", date_after(t, ["date of birth", "dob", "d.o.b"], past=True), 0.9)
    if not name:
        idx = next((i for i, ln in enumerate(lines) if re.search(r"birth|d\.?o\.?b", ln, re.I)), None)
        if idx is not None:
            name = name_line_before(lines, idx)
    fs.set("name", name, 0.8)

    rel = relation_name(t)
    if rel:
        fs.set("relation", rel[0], 0.8)
        fs.set("relative_name", rel[1], 0.8)
    else:
        fs.set("relative_name", name_after(t, ["son/daughter/wife of", "s/d/w of", "father's name",
                                               "husband's name"]), 0.75)
    bg = after_label(t, ["blood group", "b.g."])
    if bg:
        m = re.search(r"\b(AB|A|B|O)\s*([+-]|ve|pos|neg)", bg, re.I)
        if m:
            sign = "+" if m.group(2).lower() in ("+", "ve", "pos") else "-"
            fs.set("blood_group", f"{m.group(1).upper()}{sign}", 0.8)
    fs.set("address", address_after(t, ["address", "permanent address", "present address"]), 0.7)
    fs.set("issue_date", date_after(t, ["date of issue", "issue date", "doi", "issued on", "date of first issue"],
                                    past=True), 0.85)

    nt = date_after(t, ["validity (nt)", "valid till (nt)", "validity(nt)", "non-transport", "nt validity",
                        "valid till", "validity", "valid upto", "valid up to"],
                    accept=lambda v: not _TR_TAG.match(v))
    tr = date_after(t, ["validity (tr)", "valid till (tr)", "validity(tr)", "transport validity"])
    if not nt:
        for m in re.finditer(r"valid(?:ity)?([^\n]{0,30})", t, re.I):
            if _TR_TAG.match(m.group(1)):
                continue  # 'Validity(TR)' is the transport validity
            ds = find_dates(t[m.start():m.start() + 60])
            if ds:
                nt = ds[0][0]
            break
    fs.set("valid_till", nt, 0.85)
    fs.set("valid_till_transport", tr, 0.8)

    covs: list[str] = []
    up = t.upper()
    for c in _COV:
        if re.search(r"(?<![A-Z])" + re.escape(c) + r"(?![A-Z])", up) and not any(c in x for x in covs):
            covs.append(c)
    if covs:
        fs.set("vehicle_classes", ", ".join(covs), 0.8)

    auth = after_label(t, ["issuing authority", "licensing authority", "issued by"])
    if not auth:
        m = re.search(r"\b(RTO|R\.T\.O\.?|regional transport office)\s*[,:\-]?\s*([A-Za-z ]{3,30})", t, re.I)
        if m:
            auth = f"RTO {m.group(2).strip()}"
    fs.set("issuing_authority", auth, 0.7)

    person = fs.get("name")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Driving licence", person), doc_date=fs.get("issue_date"),
                  issuer=fs.get("issuing_authority") or (f"Transport Department, {fs.get('state')}" if fs.get("state") else None),
                  person_name=person, pages=pages)
