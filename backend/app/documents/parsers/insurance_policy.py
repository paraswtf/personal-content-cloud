"""Insurance policy parser (health, motor, life, travel policy schedules and certificates).

Contract: parse(text, pages, boxes) -> ParsedDocument with insurer, policy number, insured, sum
insured, premium and the validity period.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    FieldSet, ParsedDocument, after_label, amount_after, date_after, find_dates, find_vehicle_number, known_name,
    name_after, normalize_text,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "insurance_policy"
FIELDS = [
    ("insurer", "Insurer"), ("policy_type", "Policy type"), ("policy_number", "Policy number"),
    ("insured_name", "Insured"), ("sum_insured", "Sum insured"), ("premium", "Premium"),
    ("valid_from", "Valid from"), ("valid_to", "Valid to"), ("nominee", "Nominee"), ("plan", "Plan"),
    ("vehicle_number", "Vehicle number"), ("issue_date", "Issue date"),
]
KEY_FIELDS = ["insurer", "policy_number", "insured_name", "sum_insured", "valid_to"]
INSURERS = [
    "Life Insurance Corporation of India", "LIC of India", "HDFC ERGO", "HDFC Life", "ICICI Lombard",
    "ICICI Prudential", "Star Health", "Bajaj Allianz", "Bajaj General", "Tata AIG", "Tata AIA", "New India Assurance",
    "Niva Bupa", "Max Bupa", "Care Health", "Max Life", "Axis Max Life", "SBI Life", "SBI General",
    "Reliance General", "United India Insurance", "Oriental Insurance", "National Insurance", "Go Digit", "Digit",
    "Acko", "Aditya Birla Health", "Aditya Birla Sun Life", "ManipalCigna", "Kotak Mahindra General",
    "Kotak Life", "Future Generali", "Royal Sundaram", "Cholamandalam MS", "IFFCO Tokio", "Liberty General",
    "Universal Sompo", "PNB MetLife", "Canara HSBC Life", "Bharti AXA", "Edelweiss", "Zuno", "Galaxy Health",
]
_TYPES = [("health", ("health", "mediclaim", "hospitalisation", "hospitalization")),
          ("motor", ("motor", "vehicle", "two wheeler", "private car", "idv")),
          ("life", ("life insurance", "term plan", "sum assured", "maturity")),
          ("travel", ("travel insurance", "trip")), ("home", ("home insurance", "householder", "fire"))]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    low = t.lower()
    fs = FieldSet(FIELDS)
    fs.set("insurer", known_name(t, INSURERS) or after_label(t, ["insurer", "insurance company", "issued by"]), 0.85)
    ptype = next((name for name, words in _TYPES if any(w in low for w in words)), None)
    fs.set("policy_type", ptype.capitalize() if ptype else None, 0.6)
    pn = after_label(t, ["policy number", "policy no", "certificate no", "policy #", "policy id"], stop=False)
    if pn:
        m = re.search(r"[A-Za-z0-9][A-Za-z0-9/\-]{4,40}", pn)
        fs.set("policy_number", m.group(0) if m else None, 0.85)
    fs.set("insured_name", name_after(t, ["name of the insured", "insured name", "name of insured", "policyholder name",
                                          "policy holder name", "policyholder", "policy holder", "proposer name",
                                          "proposer", "insured", "life assured"]), 0.8)
    fs.set("sum_insured", amount_after(t, ["sum insured", "sum assured", "basic sum insured", "insured declared value",
                                           "idv", "cover amount", "coverage"]), 0.85)
    fs.set("premium", amount_after(t, ["total premium", "gross premium", "premium amount", "premium payable",
                                       "net premium", "premium"]), 0.75)
    m = re.search(r"(?:period of insurance|policy period|period of cover|policy term|coverage period|valid)"
                  r"[^\n]{0,120}", t, re.I)
    ds = find_dates(m.group(0)) if m else []
    if len(ds) >= 2:
        fs.set("valid_from", ds[0][0], 0.8)
        fs.set("valid_to", ds[1][0], 0.8)
    fs.set("valid_from", date_after(t, ["policy start date", "start date", "risk start date", "valid from",
                                        "inception date", "from"]), 0.75)
    fs.set("valid_to", date_after(t, ["policy end date", "end date", "expiry date", "valid till", "valid upto",
                                      "valid to", "risk end date", "maturity date"]), 0.75)
    fs.set("nominee", name_after(t, ["nominee name", "nominee"]), 0.7)
    fs.set("plan", after_label(t, ["plan name", "product name", "plan", "product"]), 0.6)
    if ptype == "motor":
        fs.set("vehicle_number", find_vehicle_number(t), 0.75)
    fs.set("issue_date", date_after(t, ["date of issue", "issue date", "issued on"], past=True), 0.7)
    insurer = fs.get("insurer")
    return finish(DOC_TYPE, fs, KEY_FIELDS,
                  title=join_title(f"{fs.get('policy_type')} insurance" if fs.get("policy_type") else "Insurance policy",
                                   insurer),
                  doc_date=fs.get("valid_from") or fs.get("issue_date"), issuer=insurer,
                  person_name=fs.get("insured_name"), pages=pages)
