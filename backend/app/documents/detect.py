"""Document type detection (docs/ARCHITECTURE.md section 11).

Contract: detect(text, pages=None, item=None) -> Detection(doc_type, score, scores).
Each type has weighted keyword phrases, identifier checks (with checksums where they exist) and
negative phrases. Score = sum of the weights of the phrases found (each counted once) + identifier
weights - negatives, capped at 1.5. The best type wins when its score >= MIN_SCORE (0.5); otherwise
doc_type is 'other' (score 0). Ties go to the earlier type in RULES. The item's file name adds a
small hint (FILENAME_HINT) when it names a type. Hindi and Marathi phrases (LOCAL_RULES) are merged
into RULES with weights like their English counterparts; a phrase matches whole words of its script.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.documents import fields as F
from app.documents.fields import PageText, normalize_text, word_guards

log = logging.getLogger(__name__)

MIN_SCORE = 0.5
MAX_SCORE = 1.5
FILENAME_HINT = 0.2


@dataclass
class Rule:
    keywords: list[tuple[str, float]]
    ids: list[tuple[Callable[[str, list[PageText]], bool], float]] = field(default_factory=list)
    negative: list[tuple[str, float]] = field(default_factory=list)
    filename: tuple[str, ...] = ()


@dataclass
class Detection:
    doc_type: str
    score: float
    scores: dict[str, float]


def _bcbp(t: str, pages: list[PageText]) -> bool:
    from app.documents.parsers.boarding_pass import read_bcbp

    if read_bcbp(t.replace("\n", " ")):
        return True
    return any(read_bcbp(p) for _f, p in F.all_barcodes(pages))


def _mrz(min_checks: int) -> Callable[[str, list[PageText]], bool]:
    def check(t: str, pages: list[PageText]) -> bool:
        from app.documents.parsers.passport import read_mrz

        m = read_mrz(t)
        return bool(m and m["checks_ok"] >= min_checks)
    return check


def _dl(t: str, pages: list[PageText]) -> bool:
    hit = F.find_dl_number(t)
    return bool(hit)


def _has_barcode(t: str, pages: list[PageText]) -> bool:
    return bool(F.all_barcodes(pages))


def _valid_gstin(t: str, pages: list[PageText]) -> bool:
    return any(F.gstin_valid(g) for g in F.find_gstins(t))


RULES: dict[str, Rule] = {
    "aadhaar": Rule(
        keywords=[("aadhaar", 0.4), ("aadhar", 0.4), ("\u0906\u0927\u093e\u0930", 0.4),
                  ("unique identification authority of india", 0.45), ("uidai", 0.35), ("enrolment no", 0.2),
                  ("enrollment no", 0.2), ("mera aadhaar", 0.3), ("meri pehchan", 0.2), ("government of india", 0.1),
                  ("vid", 0.1), ("help@uidai.gov.in", 0.3), ("year of birth", 0.1)],
        ids=[(lambda t, p: bool(F.find_aadhaar(t, grouped_only=True)), 0.45),
             (lambda t, p: bool(F.AADHAAR_MASKED_RE.search(t)), 0.25)],
        negative=[("income tax department", 0.4), ("permanent account number", 0.4), ("election commission", 0.4),
                  ("statement of account", 0.4), ("tax invoice", 0.4), ("passport", 0.2), ("payslip", 0.3),
                  ("pay slip", 0.3), ("driving licence", 0.3), ("beneficiary reference id", 0.4)],
        filename=("aadhaar", "aadhar", "uid"),
    ),
    "pan": Rule(
        keywords=[("income tax department", 0.45), ("permanent account number", 0.5), ("pan card", 0.3),
                  ("govt. of india", 0.1), ("father's name", 0.05)],
        ids=[(lambda t, p: bool(F.find_pans(t)), 0.35)],
        negative=[("invoice", 0.4), ("statement", 0.3), ("salary", 0.4), ("payslip", 0.4), ("receipt", 0.3),
                  ("policy", 0.3), ("gstin", 0.3), ("form 16", 0.2)],
        filename=("pan",),
    ),
    "passport": Rule(
        keywords=[("passport", 0.35), ("republic of india", 0.2), ("place of issue", 0.15), ("date of expiry", 0.15),
                  ("nationality", 0.1), ("given name", 0.15), ("surname", 0.1), ("file no", 0.1),
                  ("name of father / legal guardian", 0.3), ("legal guardian", 0.2), ("old passport no", 0.2)],
        ids=[(_mrz(4), 0.9), (_mrz(0), 0.3)],
        negative=[("boarding pass", 0.5), ("visa application", 0.2)],
        filename=("passport",),
    ),
    "driving_licence": Rule(
        keywords=[("driving licence", 0.5), ("driving license", 0.5), ("transport department", 0.2),
                  ("union of india", 0.1), ("dl no", 0.3), ("licence no", 0.2), ("license no", 0.2),
                  ("class of vehicle", 0.2), ("cov", 0.1), ("validity (nt)", 0.25), ("validity(nt)", 0.25),
                  ("lmv", 0.1), ("mcwg", 0.1), ("licensing authority", 0.2)],
        ids=[(_dl, 0.3)],
        negative=[("registration certificate", 0.3), ("certificate of registration", 0.3), ("chassis", 0.2),
                  ("insurance", 0.3)],
        filename=("licence", "license", "dl"),
    ),
    "voter_id": Rule(
        keywords=[("election commission of india", 0.5), ("election commission", 0.3), ("elector", 0.3),
                  ("electors photo identity card", 0.4), ("elector's name", 0.3), ("epic", 0.2),
                  ("identity card", 0.1), ("assembly constituency", 0.2)],
        ids=[(lambda t, p: bool(F.EPIC_RE.search(t.upper())), 0.25)],
        filename=("voter", "epic"),
    ),
    "vehicle_rc": Rule(
        keywords=[("certificate of registration", 0.45), ("registration certificate", 0.45), ("chassis no", 0.3),
                  ("chassis number", 0.3), ("engine no", 0.2), ("engine number", 0.2), ("regn. no", 0.15),
                  ("date of regn", 0.2), ("maker's name", 0.15), ("fuel", 0.05), ("owner", 0.05),
                  ("seating capacity", 0.1), ("registering authority", 0.2), ("vehicle class", 0.1)],
        ids=[(lambda t, p: bool(F.find_vehicle_number(t)), 0.15)],
        negative=[("insurance", 0.5), ("policy", 0.3), ("premium", 0.3), ("invoice", 0.3)],
        filename=("rc", "registration"),
    ),
    "bank_statement": Rule(
        keywords=[("statement of account", 0.5), ("account statement", 0.5), ("bank statement", 0.5),
                  ("opening balance", 0.25), ("closing balance", 0.25), ("ifsc", 0.1), ("withdrawal", 0.1),
                  ("deposit", 0.05), ("transaction date", 0.1), ("value date", 0.1), ("cheque", 0.05),
                  ("narration", 0.15), ("balance", 0.05)],
        ids=[(lambda t, p: bool(F.IFSC_RE.search(t.upper())), 0.1)],
        # A letter asking the bank for a statement names the statement without being one.
        negative=[("tax invoice", 0.3), ("payslip", 0.4), ("salary slip", 0.4), ("dear sir", 0.6),
                  ("dear madam", 0.6), ("respected sir", 0.6), ("subject", 0.2)],
        filename=("statement",),
    ),
    "invoice": Rule(
        keywords=[("tax invoice", 0.6), ("invoice", 0.35), ("invoice no", 0.2), ("invoice number", 0.2),
                  ("invoice date", 0.15), ("gstin", 0.15), ("hsn", 0.1), ("sac", 0.05), ("bill to", 0.1),
                  ("billed to", 0.1), ("ship to", 0.1), ("cgst", 0.1), ("sgst", 0.1), ("igst", 0.1),
                  ("grand total", 0.1), ("place of supply", 0.15), ("bill of supply", 0.4)],
        ids=[(_valid_gstin, 0.15)],
        negative=[("payslip", 0.4), ("salary slip", 0.4), ("statement of account", 0.4), ("receipt no", 0.2),
                  ("policy schedule", 0.4), ("boarding pass", 0.4)],
        filename=("invoice", "bill"),
    ),
    "receipt": Rule(
        keywords=[("receipt", 0.4), ("payment receipt", 0.3), ("receipt no", 0.2), ("received with thanks", 0.3),
                  ("received from", 0.2), ("payment received", 0.25), ("amount paid", 0.15), ("paid", 0.05),
                  ("transaction id", 0.1), ("payment successful", 0.2), ("utr", 0.1), ("fee receipt", 0.3)],
        negative=[("tax invoice", 0.25), ("statement of account", 0.4), ("policy schedule", 0.3), ("payslip", 0.4),
                  ("boarding pass", 0.4), ("ticket", 0.2)],
        filename=("receipt",),
    ),
    "salary_slip": Rule(
        keywords=[("payslip", 0.6), ("pay slip", 0.6), ("salary slip", 0.6), ("net pay", 0.3), ("net salary", 0.3),
                  ("gross earnings", 0.2), ("total earnings", 0.15), ("total deductions", 0.15), ("basic", 0.05),
                  ("house rent allowance", 0.1), ("hra", 0.05), ("provident fund", 0.1), ("professional tax", 0.1),
                  ("employee id", 0.1), ("employee code", 0.1), ("uan", 0.1), ("days paid", 0.1)],
        filename=("payslip", "salary"),
    ),
    "marksheet": Rule(
        keywords=[("marksheet", 0.5), ("mark sheet", 0.5), ("statement of marks", 0.5), ("grade card", 0.5),
                  ("memorandum of marks", 0.5), ("transcript", 0.4), ("statement of grades", 0.5),
                  ("marks obtained", 0.3), ("roll no", 0.15), ("seat no", 0.15), ("sgpa", 0.15), ("cgpa", 0.1),
                  ("max marks", 0.2), ("maximum marks", 0.2), ("subject", 0.05), ("result", 0.05),
                  ("examination", 0.1)],
        negative=[("conferred", 0.3), ("invoice", 0.3)],
        filename=("marksheet", "marks", "grade", "transcript", "result"),
    ),
    "degree": Rule(
        keywords=[("degree of", 0.4), ("conferred", 0.35), ("is hereby conferred", 0.3), ("admitted to the degree", 0.5),
                  ("bachelor of", 0.25), ("master of", 0.25), ("doctor of philosophy", 0.3), ("convocation", 0.25),
                  ("degree certificate", 0.4), ("university", 0.1), ("provisional degree", 0.4)],
        negative=[("marks obtained", 0.3), ("statement of marks", 0.4), ("offer", 0.3)],
        filename=("degree", "convocation"),
    ),
    "certificate": Rule(
        keywords=[("certificate", 0.3), ("this is to certify", 0.3), ("certify that", 0.2), ("awarded to", 0.2),
                  ("presented to", 0.2), ("in recognition of", 0.2), ("successfully completed", 0.2),
                  ("certificate of completion", 0.3), ("certificate of participation", 0.3),
                  ("certificate of achievement", 0.3), ("credential id", 0.2), ("participation", 0.1),
                  ("completion", 0.1)],
        negative=[("vaccin", 0.5), ("registration", 0.3), ("insurance", 0.4), ("marks obtained", 0.3),
                  ("degree of", 0.3), ("conferred", 0.2), ("chassis", 0.4), ("beneficiary", 0.4),
                  ("statement of marks", 0.4)],
        filename=("certificate", "cert"),
    ),
    "offer_letter": Rule(
        keywords=[("offer letter", 0.5), ("letter of offer", 0.5), ("offer of employment", 0.5),
                  ("appointment letter", 0.5), ("letter of appointment", 0.5), ("pleased to offer", 0.35),
                  ("pleased to appoint", 0.35), ("cost to company", 0.25), ("ctc", 0.15), ("date of joining", 0.15),
                  ("designation", 0.1), ("probation", 0.1), ("annual compensation", 0.2), ("joining", 0.05)],
        filename=("offer", "appointment"),
    ),
    "insurance_policy": Rule(
        keywords=[("policy schedule", 0.45), ("insurance policy", 0.4), ("certificate of insurance", 0.4),
                  ("insurance", 0.2), ("policy number", 0.2), ("policy no", 0.2), ("sum insured", 0.3),
                  ("sum assured", 0.3), ("premium", 0.15), ("insured", 0.1), ("policyholder", 0.15),
                  ("policy holder", 0.15), ("nominee", 0.05), ("period of insurance", 0.2), ("idv", 0.1)],
        filename=("policy", "insurance"),
    ),
    "prescription": Rule(
        keywords=[("prescription", 0.3), ("rx", 0.25), ("\u211e", 0.3), ("mbbs", 0.2), ("dr.", 0.1),
                  ("tab", 0.1), ("tab.", 0.1), ("cap.", 0.1), ("syp", 0.1), ("once daily", 0.1),
                  ("twice daily", 0.1), ("after food", 0.1), ("before food", 0.1), ("diagnosis", 0.1),
                  ("follow up", 0.1), ("review after", 0.1), ("c/o", 0.05), ("clinic", 0.1), ("bd", 0.05),
                  ("od", 0.05), ("tds", 0.05)],
        negative=[("reference range", 0.4), ("biological reference", 0.4), ("invoice", 0.2), ("policy", 0.3),
                  ("vaccin", 0.3)],
        filename=("prescription", "rx"),
    ),
    "lab_report": Rule(
        keywords=[("laboratory", 0.2), ("test report", 0.3), ("lab report", 0.3), ("investigation", 0.15),
                  ("reference range", 0.3), ("biological reference interval", 0.35), ("normal range", 0.25),
                  ("specimen", 0.15), ("sample collected", 0.2), ("sample type", 0.15), ("pathology", 0.2),
                  ("diagnostics", 0.15), ("haemoglobin", 0.1), ("hemoglobin", 0.1), ("glucose", 0.1),
                  ("cholesterol", 0.1), ("units", 0.05), ("reported", 0.05)],
        filename=("report", "lab", "blood"),
    ),
    "vaccination": Rule(
        keywords=[("vaccination certificate", 0.5), ("certificate for covid-19 vaccination", 0.6),
                  ("immunization", 0.3), ("immunisation", 0.3), ("vaccine", 0.2), ("vaccinated", 0.2),
                  ("dose", 0.1), ("beneficiary", 0.2), ("beneficiary reference id", 0.3), ("covishield", 0.25),
                  ("covaxin", 0.25), ("cowin", 0.25), ("precaution dose", 0.2)],
        filename=("vaccin", "cowin"),
    ),
    "event_pass": Rule(
        keywords=[("ticket", 0.2), ("e-ticket", 0.2), ("admit", 0.2), ("admit one", 0.2), ("entry pass", 0.3),
                  ("event pass", 0.3), ("booking id", 0.15), ("order id", 0.1), ("seat", 0.1), ("row", 0.05),
                  ("section", 0.05), ("gate", 0.05), ("venue", 0.2), ("show time", 0.2), ("showtime", 0.2),
                  ("doors open", 0.2), ("event date", 0.15), ("screen", 0.05), ("bookmyshow", 0.3),
                  ("paytm insider", 0.3), ("ticketmaster", 0.3), ("eventbrite", 0.3), ("audi", 0.05)],
        ids=[(_has_barcode, 0.1)],
        negative=[("boarding pass", 0.6), ("flight", 0.3), ("pnr", 0.2), ("invoice", 0.2), ("payslip", 0.4),
                  ("boarding time", 0.3)],
        filename=("ticket", "pass", "event"),
    ),
    "boarding_pass": Rule(
        keywords=[("boarding pass", 0.6), ("flight", 0.2), ("boarding time", 0.25), ("gate", 0.05), ("seat", 0.05),
                  ("pnr", 0.1), ("departure", 0.05), ("arrival", 0.05), ("boarding", 0.1), ("e-boarding", 0.3)],
        ids=[(_bcbp, 1.0)],
        filename=("boarding", "bp_"),
    ),
}

# Hindi and Marathi phrases: (keywords, negatives) per type, merged into RULES. A rent agreement
# (किराया अनुबंध, भाडे करार) has no type of its own; its phrases keep it out of the money types and, since
# it prints both parties' Aadhaar and PAN numbers, out of those two as well.
_RENT = [("किराया अनुबंध", 0.4), ("किरायानामा", 0.4), ("भाडे करार", 0.4), ("भाडेकरार", 0.4)]
_MARKSHEET_LOCAL = [("अंक सूची", 0.4), ("अंकसूची", 0.4), ("अंकपत्र", 0.4), ("अंक पत्र", 0.4), ("गुणपत्रक", 0.4),
                    ("गुणपत्रिका", 0.4), ("प्राप्तांक", 0.3)]
_SALARY_LOCAL = [("वेतन पर्ची", 0.4), ("पगार पावती", 0.4), ("वेतन पत्रक", 0.4), ("पगार पत्रक", 0.4)]
LOCAL_RULES: dict[str, tuple[list[tuple[str, float]], list[tuple[str, float]]]] = {
    "aadhaar": (
        [("भारत सरकार", 0.1), ("मेरा आधार", 0.3), ("मेरी पहचान", 0.2), ("माझे आधार", 0.3), ("माझी ओळख", 0.2),
         ("भारतीय विशिष्ट पहचान प्राधिकरण", 0.45), ("भारतीय विशिष्ट ओळख प्राधिकरण", 0.45), ("नामांकन क्रम", 0.2),
         ("जन्म वर्ष", 0.1)],
        [("आयकर विभाग", 0.4), ("स्थायी लेखा संख्या", 0.4), ("निर्वाचन आयोग", 0.4), ("निवडणूक आयोग", 0.4),
         ("खाता विवरण", 0.4), ("कर बीजक", 0.4), ("पासपोर्ट", 0.2), ("वेतन पर्ची", 0.3), ("पगार पावती", 0.3),
         ("ड्राइविंग लाइसेंस", 0.3), ("चालक अनुज्ञप्ति", 0.3), *[(w, 0.6) for w, _ in _RENT]],
    ),
    "pan": (
        [("आयकर विभाग", 0.45), ("स्थायी लेखा संख्या", 0.5), ("भारत सरकार", 0.1), ("पिता का नाम", 0.05)],
        [("बीजक", 0.4), ("चालान", 0.3), ("खाता विवरण", 0.3), ("वेतन", 0.4), ("पगार", 0.4), ("रसीद", 0.3),
         ("पावती", 0.3), ("बीमा", 0.3), ("पॉलिसी", 0.3), *[(w, 0.6) for w, _ in _RENT]],
    ),
    "passport": (
        [("पासपोर्ट", 0.35), ("भारत गणराज्य", 0.2), ("जारी करने का स्थान", 0.15), ("समाप्ति की तिथि", 0.15),
         ("राष्ट्रीयता", 0.1), ("दिया गया नाम", 0.15), ("फ़ाइल सं", 0.1), ("फाइल सं", 0.1)],
        [("बोर्डिंग पास", 0.5)],
    ),
    "driving_licence": (
        [("ड्राइविंग लाइसेंस", 0.5), ("ड्रायव्हिंग लायसन्स", 0.5), ("चालक अनुज्ञप्ति", 0.5), ("वाहन चालक परवाना", 0.5),
         ("चालक परवाना", 0.3), ("परिवहन विभाग", 0.2), ("अनुज्ञप्ति संख्या", 0.2), ("अनुज्ञापन प्राधिकारी", 0.2)],
        [("पंजीकरण प्रमाण पत्र", 0.3), ("पंजीयन प्रमाण पत्र", 0.3), ("नोंदणी प्रमाणपत्र", 0.3), ("चेसिस", 0.2),
         ("बीमा", 0.3), ("विमा", 0.3)],
    ),
    "voter_id": (
        [("भारत निर्वाचन आयोग", 0.5), ("भारत निवडणूक आयोग", 0.5), ("निर्वाचन आयोग", 0.3), ("निवडणूक आयोग", 0.3),
         ("निर्वाचक फोटो पहचान पत्र", 0.4), ("मतदार ओळखपत्र", 0.4), ("मतदाता", 0.3), ("मतदार", 0.3),
         ("निर्वाचक", 0.3), ("निर्वाचक का नाम", 0.3), ("मतदाराचे नाव", 0.3), ("विधानसभा", 0.2)],
        [],
    ),
    "vehicle_rc": (
        [("पंजीकरण प्रमाण पत्र", 0.45), ("पंजीयन प्रमाण पत्र", 0.45), ("नोंदणी प्रमाणपत्र", 0.45),
         ("नोंदणी प्रमाण पत्र", 0.45), ("चेसिस", 0.3), ("इंजन", 0.2), ("इंजिन", 0.2), ("पंजीकरण संख्या", 0.15),
         ("नोंदणी क्रमांक", 0.15), ("ईंधन", 0.05), ("इंधन", 0.05), ("मालक", 0.05), ("स्वामी", 0.05)],
        [("बीमा", 0.5), ("विमा", 0.5), ("पॉलिसी", 0.3), ("प्रीमियम", 0.3), ("बीजक", 0.3)],
    ),
    "bank_statement": (
        [("खाता विवरण", 0.5), ("खाते उतारा", 0.5), ("खाते विवरण", 0.5), ("बँक स्टेटमेंट", 0.5), ("बैंक स्टेटमेंट", 0.5),
         ("प्रारंभिक शेष", 0.25), ("अंतिम शेष", 0.25), ("आरंभीची शिल्लक", 0.25), ("अखेरची शिल्लक", 0.25),
         ("शेष", 0.05), ("शिल्लक", 0.05), ("जमा", 0.05), ("नामे", 0.05)],
        # A letter asking the bank for a statement names the statement without being one.
        [("कर बीजक", 0.3), ("सेवा में", 0.4), ("महोदय", 0.3), ("महोदया", 0.3), ("अनुरोध", 0.2), ("विनंती", 0.2),
         *_SALARY_LOCAL],
    ),
    "invoice": (
        [("कर बीजक", 0.6), ("बीजक", 0.35), ("बिल", 0.35), ("चालान", 0.35), ("बिल संख्या", 0.2), ("बिल क्रमांक", 0.2),
         ("बिल नं", 0.2), ("बीजक संख्या", 0.2), ("बीजक क्रमांक", 0.2), ("बिल दिनांक", 0.15), ("जीएसटी", 0.1),
         ("कुल राशि", 0.1), ("एकूण रक्कम", 0.1), ("प्राप्तकर्ता", 0.1)],
        [*_SALARY_LOCAL, ("खाता विवरण", 0.4), ("रसीद संख्या", 0.2), ("रसीद क्रमांक", 0.2), ("पावती क्रमांक", 0.2),
         *_MARKSHEET_LOCAL, *_RENT],
    ),
    "receipt": (
        [("रसीद", 0.4), ("पावती", 0.4), ("रसीद संख्या", 0.2), ("रसीद क्रमांक", 0.2), ("रसीद नं", 0.2),
         ("पावती क्रमांक", 0.2), ("पावती नं", 0.2), ("धन्यवाद सहित प्राप्त", 0.3), ("प्राप्त किए", 0.2),
         ("प्राप्त किया", 0.2), ("प्राप्त झाले", 0.2), ("भुगतान प्राप्त", 0.25), ("शुल्क रसीद", 0.3),
         ("फीस रसीद", 0.3), ("शुल्क पावती", 0.3), ("भुगतान", 0.05), ("सशुल्क", 0.05)],
        [("कर बीजक", 0.25), ("खाता विवरण", 0.4), ("पगार पावती", 0.5), ("वेतन पर्ची", 0.4), ("वेतन पत्रक", 0.4),
         ("पगार पत्रक", 0.4), *_MARKSHEET_LOCAL, *_RENT],
    ),
    "salary_slip": (
        [("वेतन पर्ची", 0.6), ("पगार पावती", 0.6), ("वेतन पत्रक", 0.6), ("पगार पत्रक", 0.6), ("पगार स्लिप", 0.6),
         ("शुद्ध वेतन", 0.3), ("निव्वळ पगार", 0.3), ("निव्वळ वेतन", 0.3), ("सकल वेतन", 0.2), ("एकूण उत्पन्न", 0.15),
         ("कुल कटौती", 0.15), ("एकूण कपात", 0.15), ("मूल वेतन", 0.05), ("मूळ वेतन", 0.05),
         ("मकान किराया भत्ता", 0.1), ("घरभाडे भत्ता", 0.1), ("भविष्य निधि", 0.1), ("भविष्य निर्वाह निधी", 0.1),
         ("व्यवसाय कर", 0.1), ("कर्मचारी कोड", 0.1), ("कर्मचारी क्रमांक", 0.1)],
        [*_RENT],
    ),
    "marksheet": (
        [("अंक सूची", 0.5), ("अंकसूची", 0.5), ("अंकपत्र", 0.5), ("अंक पत्र", 0.5), ("गुणपत्रक", 0.5),
         ("गुणपत्रिका", 0.5), ("माध्यमिक शिक्षा मंडल", 0.3), ("माध्यमिक शिक्षा मंडळ", 0.3),
         ("महाराष्ट्र राज्य माध्यमिक", 0.3), ("प्राप्तांक", 0.3), ("मिळालेले गुण", 0.3), ("प्राप्त गुण", 0.3),
         ("अनुक्रमांक", 0.15), ("आसन क्रमांक", 0.15), ("बैठक क्रमांक", 0.15), ("पूर्णांक", 0.2), ("कमाल गुण", 0.2),
         ("एकूण गुण", 0.1), ("परीक्षा", 0.1), ("परिणाम", 0.05), ("निकाल", 0.05), ("विषय", 0.05)],
        [("बीजक", 0.3), ("बिल", 0.3)],
    ),
}
for _t, (_kw, _neg) in LOCAL_RULES.items():
    RULES[_t].keywords.extend(_kw)
    RULES[_t].negative.extend(_neg)

_KW_CACHE: dict[str, re.Pattern[str]] = {}


def _phrase(p: str) -> re.Pattern[str]:
    rx = _KW_CACHE.get(p)
    if rx is None:
        body = re.escape(F.dev_fold(p)).replace(r"\ ", r"\s+")
        lead, trail = word_guards(p)
        if lead == r"(?<![A-Za-z])":
            lead, trail = r"(?<![a-z0-9])", r"(?![a-z0-9])"
        rx = re.compile(lead + body + trail)
        _KW_CACHE[p] = rx
    return rx


def _filename(item: Any) -> str:
    if item is None:
        return ""
    try:
        return str(item["original_filename"] or "").lower()
    except (KeyError, IndexError, TypeError):
        return str(getattr(item, "original_filename", "") or "").lower()


def score_types(text: str, pages: list[PageText] | None = None, item: Any = None) -> dict[str, float]:
    t = normalize_text(text or "")
    low = re.sub(r"[ \t]+", " ", t.lower())
    pages = pages or []
    fname = _filename(item)
    stem = re.sub(r"[^a-z0-9]+", " ", fname)
    out: dict[str, float] = {}
    for doc_type, rule in RULES.items():
        s = 0.0
        for phrase, w in rule.keywords:
            if _phrase(phrase).search(low):
                s += w
        for check, w in rule.ids:
            try:
                if check(t, pages):
                    s += w
            except Exception:
                log.debug("id check for %s failed", doc_type, exc_info=True)
        for phrase, w in rule.negative:
            if _phrase(phrase).search(low):
                s -= w
        if stem and any(_phrase(h.strip("_")).search(stem) for h in rule.filename):
            s += FILENAME_HINT
        out[doc_type] = round(max(0.0, min(MAX_SCORE, s)), 3)
    return out


def detect(text: str, pages: list[PageText] | None = None, item: Any = None) -> Detection:
    scores = score_types(text, pages, item)
    best, best_score = "other", 0.0
    for doc_type in RULES:
        if scores[doc_type] > best_score:
            best, best_score = doc_type, scores[doc_type]
    if best_score < MIN_SCORE:
        return Detection("other", 0.0, scores)
    return Detection(best, best_score, scores)
