"""Vehicle registration certificate (RC) parser.

Contract: parse(text, pages, boxes) -> ParsedDocument. Registration numbers use fields.VEHICLE_RE
(state code, RTO number, series, number) or the BH series. Hindi and Marathi labels are read too.
"""
from __future__ import annotations

import logging
import re

from app.documents.fields import (
    ADDRESS_LABELS_LOCAL, NAME_LABELS_LOCAL, VALID_LABELS_LOCAL, FieldSet, ParsedDocument, STATE_CODES,
    address_after, after_label, date_after, find_vehicle_number, local_name_after, name_after, normalize_text,
    relation_name,
)
from app.documents.parsers._base import finish, join_title

log = logging.getLogger(__name__)

DOC_TYPE = "vehicle_rc"
FIELDS = [
    ("registration_number", "Registration number"), ("owner_name", "Owner"), ("name_local", "Name (local script)"),
    ("relation", "Relation"),
    ("relative_name", "Father or spouse"), ("address", "Address"), ("maker", "Maker"), ("model", "Model"),
    ("vehicle_class", "Vehicle class"), ("fuel", "Fuel"), ("colour", "Colour"), ("chassis_number", "Chassis number"),
    ("engine_number", "Engine number"), ("registration_date", "Registration date"), ("valid_till", "Valid till"),
    ("seating", "Seating capacity"), ("registering_authority", "Registering authority"), ("state", "State"),
]
KEY_FIELDS = ["registration_number", "owner_name", "chassis_number", "engine_number", "registration_date"]
_OWNER_LOCAL = ["स्वामी का नाम", "मालिक का नाम", "मालकाचे नाव", "मालकाचे नांव", "स्वामी", "मालक"]


def parse(text: str, pages=None, boxes=None) -> ParsedDocument:
    t = normalize_text(text)
    fs = FieldSet(FIELDS)
    reg_label = after_label(t, ["regn. number", "regn. no", "registration no", "registration number", "reg. no",
                                "vehicle no", "vehicle number", "पंजीकरण संख्या", "पंजीकरण सं", "पंजीयन क्रमांक",
                                "नोंदणी क्रमांक", "वाहन क्रमांक", "वाहन संख्या"], stop=False)
    reg = find_vehicle_number(reg_label or "") or find_vehicle_number(t)
    if reg:
        fs.set("registration_number", reg, 0.9)
        fs.set("state", STATE_CODES.get(reg[:2]), 0.7)
    owner = ["owner name", "owner's name", "name of owner", "registered owner", "owner", *_OWNER_LOCAL]
    fs.set("owner_name", name_after(t, owner) or name_after(t, ["name", *NAME_LABELS_LOCAL], anchored=True), 0.8)
    local = local_name_after(t, owner) or local_name_after(t, ["name", *NAME_LABELS_LOCAL], anchored=True)
    if local and local != fs.get("owner_name"):
        fs.set("name_local", local, 0.7)
    rel = relation_name(t)
    if rel:
        fs.set("relation", rel[0], 0.75)
        fs.set("relative_name", rel[1], 0.75)
    fs.set("address", address_after(t, ["address", "permanent address", "present address", *ADDRESS_LABELS_LOCAL]),
           0.65)
    fs.set("maker", after_label(t, ["maker's name", "makers name", "maker name", "make", "manufacturer",
                                    "निर्माता का नाम", "निर्माता", "उत्पादक"]), 0.75)
    fs.set("model", after_label(t, ["model name", "model", "maker's classification", "makers classification", "मॉडल",
                                    "मॉडेल"]), 0.7)
    fs.set("vehicle_class", after_label(t, ["vehicle class", "class of vehicle", "veh. class", "vehicle description",
                                            "वाहन वर्ग", "वाहनाचा वर्ग", "वाहन श्रेणी"]), 0.7)
    fs.set("fuel", after_label(t, ["fuel used", "fuel type", "fuel", "ईंधन", "इंधन"]), 0.75)
    fs.set("colour", after_label(t, ["colour", "color", "रंग"]), 0.7)
    ch = after_label(t, ["chassis no", "chassis number", "ch. no", "chasis no", "चेसिस नं", "चेसिस संख्या",
                         "चेसिस क्रमांक", "चेसिस"], stop=False)
    if ch:
        m = re.search(r"[A-Z0-9]{10,20}", ch.upper().replace(" ", ""))
        fs.set("chassis_number", m.group(0) if m else None, 0.8)
    en = after_label(t, ["engine no", "engine number", "eng. no", "engine/motor no", "इंजन नं", "इंजन संख्या",
                         "इंजिन क्रमांक", "इंजन", "इंजिन"], stop=False)
    if en:
        m = re.search(r"[A-Z0-9]{6,20}", en.upper().replace(" ", ""))
        fs.set("engine_number", m.group(0) if m else None, 0.8)
    fs.set("registration_date", date_after(t, ["date of regn", "date of registration", "regn. date", "reg. date",
                                               "registration date", "date of reg", "पंजीकरण की तिथि", "पंजीकरण तिथि",
                                               "पंजीयन दिनांक", "नोंदणी दिनांक", "नोंदणी तारीख"]), 0.85)
    fs.set("valid_till", date_after(t, ["regn. validity", "registration validity", "valid upto", "valid up to",
                                        "valid till", "fitness upto", "fitness valid upto", *VALID_LABELS_LOCAL]), 0.8)
    seat = after_label(t, ["seating capacity", "seat cap", "seating", "बैठक क्षमता", "आसन क्षमता"])
    if seat:
        m = re.search(r"\d{1,2}", seat)
        fs.set("seating", m.group(0) if m else None, 0.7)
    fs.set("registering_authority", after_label(t, ["registering authority", "registration authority",
                                                    "पंजीकरण प्राधिकारी", "पंजीयन प्राधिकारी", "नोंदणी प्राधिकारी"]), 0.7)

    owner = fs.get("owner_name")
    return finish(DOC_TYPE, fs, KEY_FIELDS, title=join_title("Vehicle registration", fs.get("registration_number")),
                  doc_date=fs.get("registration_date"), issuer=fs.get("registering_authority"), person_name=owner,
                  pages=pages)
