"""Hindi and Marathi (Devanagari) documents: text helpers, detection, parsers, the language field and the
search tokens. Fixtures are rendered with Noto Sans Devanagari (tests/helpers.devanagari_text_image) and
their TEXT goes through detection and the parsers; OCR itself is covered by the OCR tests."""
from __future__ import annotations

import numpy as np
import pytest

from app.documents import fields as F
from app.documents.detect import detect, score_types
from app.documents.extract import analyze_text, match_query
from app.documents.fields import PageText
from app.documents.parsers import REGISTRY, parse
from tests.helpers import devanagari_font_available, devanagari_text_image

AADHAAR_VALID = "4991 8877 6540"  # Verhoeff-valid (see test_documents_parsers)

AADHAAR_FRONT = [
    "भारत सरकार",
    "Government of India",
    "राहुल कुमार शर्मा",
    "Rahul Kumar Sharma",
    "जन्म तिथि/DOB: १५/०८/१९८८",
    "पुरुष / MALE",
    AADHAAR_VALID,
    "मेरा आधार, मेरी पहचान",
]

MARATHI_MARKSHEET = [
    "महाराष्ट्र राज्य माध्यमिक व उच्च माध्यमिक शिक्षण मंडळ, पुणे",
    "माध्यमिक शालांत प्रमाणपत्र परीक्षा - मार्च २०२३",
    "गुणपत्रक",
    "विद्यार्थ्याचे नाव: प्रिया सुनील नायर",
    "आईचे नाव: सुनीता",
    "आसन क्रमांक: M123456",
    "विषय   कमाल गुण   मिळालेले गुण",
    "मराठी   100   88",
    "इंग्रजी   100   91",
    "एकूण गुण: 452/500",
    "टक्केवारी: 90.40",
    "निकाल: उत्तीर्ण - प्रथम श्रेणी विशेष प्राविण्य",
    "दिनांक: ०२/०६/२०२३",
]

HINDI_RECEIPT = [
    "शर्मा किराना भंडार",
    "रसीद",
    "रसीद संख्या: R-१०२४",
    "दिनांक: १२ मार्च २०२४",
    "श्री राहुल शर्मा से धन्यवाद सहित प्राप्त किए",
    "राशि: रु. १,२५०.००",
    "भुगतान का माध्यम: नकद",
    "विवरण: मासिक राशन",
]

PAN_BILINGUAL = [
    "आयकर विभाग   INCOME TAX DEPARTMENT",
    "भारत सरकार   GOVT. OF INDIA",
    "स्थायी लेखा संख्या कार्ड",
    "Permanent Account Number Card",
    "ABCPS1234K",
    "नाम / Name",
    "RAHUL SHARMA",
    "पिता का नाम / Father's Name",
    "SURESH SHARMA",
    "जन्म की तारीख / Date of Birth",
    "15/08/1988",
]


def _text(lines: list[str]) -> str:
    return "\n".join(lines)


def _rendered(lines: list[str]) -> str:
    """Render the fixture (checks the font draws it) and return its text."""
    if not devanagari_font_available():
        pytest.skip("Noto Sans Devanagari font not available")
    img = np.asarray(devanagari_text_image(lines).convert("L"))
    assert (img < 128).sum() > 500 * len(lines) // 4, "the fixture rendered almost no ink"
    return _text(lines)


def _vals(doc) -> dict[str, str]:
    return {k: v.value for k, v in doc.fields.items()}


# ------------------------------------------------------------------ normalize_text

def test_normalize_devanagari_digits_danda_and_joiners():
    t = F.normalize_text("दिनांक: १२/०३/२०२४। राशि ₹ १,२५०‍\nमुं‌बई   पुणे")
    assert "12/03/2024." in t
    assert "1,250" in t
    assert "‌" not in t and "‍" not in t
    assert "मुंबई   पुणे" in t  # a column gap of 3 spaces is kept
    assert F.normalize_text(t) == t  # idempotent


def test_normalize_keeps_english_behaviour():
    assert F.normalize_text("Name:  Rahul–Kumar     Pune") == "Name: Rahul-Kumar   Pune"


def test_nukta_letters_compare_equal():
    pre = "फ़रवरी"            # फ़रवरी with the precomposed letter
    dec = "फ़रवरी"      # फ + nukta
    assert F.normalize_text(pre) == F.normalize_text(dec)


# ------------------------------------------------------------------ dates and months

@pytest.mark.parametrize("text,iso", [
    ("15 अगस्त 1988", "1988-08-15"),
    ("१५ अगस्त १९८८", "1988-08-15"),
    ("3 फ़रवरी 2020", "2020-02-03"),
    ("3 फ़रवरी 2020", "2020-02-03"),
    ("3 फरवरी 2020", "2020-02-03"),
    ("3 फेब्रुवारी 2020", "2020-02-03"),
    ("1 जानेवारी 2021", "2021-01-01"),
    ("12 मे 2023", "2023-05-12"),
    ("12 मई 2023", "2023-05-12"),
    ("4 जुलै 2022", "2022-07-04"),
    ("9 सप्टेंबर 2019", "2019-09-09"),
    ("9 सितम्बर 2019", "2019-09-09"),
    ("2 अक्तूबर 2018", "2018-10-02"),
    ("2 ऑक्टोबर 2018", "2018-10-02"),
    ("11 नोव्हेंबर 2017", "2017-11-11"),
    ("25 डिसेंबर 2016", "2016-12-25"),
    ("25 दिसम्बर 2016", "2016-12-25"),
    ("जन्म तिथि: १५/०८/१९८८", "1988-08-15"),
    ("दिनांक ०१-०४-२०२४", "2024-04-01"),
])
def test_hindi_and_marathi_dates(text, iso):
    assert F.parse_date(text) == iso


def test_me_inside_mein_is_not_a_month():
    assert F.find_dates("2023 में 5 लोग") == []


@pytest.mark.parametrize("text,ym", [
    ("मार्च 2024", "2024-03"), ("वेतन पर्ची - मार्च २०२४", "2024-03"), ("पगार - ऑक्टोबर २०२३", "2023-10"),
    ("March 2024", "2024-03"), ("Mar-24", "2024-03"),
])
def test_parse_month_local(text, ym):
    assert F.parse_month(text) == ym


def test_rupee_words_are_currency():
    assert str(F.parse_amount("एकूण रक्कम रु. 25,000")) == "25000"
    assert str(F.parse_amount("राशि: रुपये ५००")) == "500"


# ------------------------------------------------------------------ labels, names, gender, relations

def test_bilingual_label_tails_are_skipped():
    assert F.after_label("जन्म तिथि/DOB: 15/08/1988", ["जन्म तिथि"]) == "15/08/1988"
    assert F.after_label("Name / नाम : Rahul Sharma", ["name"]) == "Rahul Sharma"
    assert F.after_label("नाम / Name : राहुल शर्मा", ["name"], anchored=True) == "राहुल शर्मा"


def test_value_after_slash_is_not_a_label():
    assert F.after_label("Name / राहुल शर्मा\nDOB: 01/01/1990", ["name"], anchored=True) == "राहुल शर्मा"
    assert F.after_label("नाम / Rahul Sharma", ["नाम"]) == "Rahul Sharma"
    assert F.after_label("दिया गया नाम / Given Name(s)\nRAHUL", ["दिया गया नाम"]) == "RAHUL"


def test_bharat_is_a_name_not_a_header():
    from app.documents.parsers._base import is_name_line

    assert is_name_line("भारत कुमार शर्मा")
    assert not is_name_line("भारत सरकार")
    assert not is_name_line("भारत निर्वाचन आयोग")


def test_local_labels_match_whole_words_only():
    # 'नाम' must not match inside 'उपनाम' (surname); 'कुल' not inside 'कुलकर्णी'
    assert F.after_label("उपनाम: SHARMA", ["नाम"]) is None
    assert F.after_label("कुलकर्णी 500", ["कुल"]) is None
    assert F.after_label("कुल: 500", ["कुल"]) == "500"


def test_generic_date_label_skips_birth_date():
    text = "जन्म तिथि: 01/01/1990\nजन्म की तारीख 02/02/1991\nजारी करने की तिथि: 05/06/2020"
    assert F.date_after(text, ["तिथि", "तारीख"]) == "2020-06-05"


def test_stop_labels_end_local_values():
    assert F.after_label("नाम: राहुल शर्मा पिता का नाम: रमेश शर्मा", ["नाम"]) == "राहुल शर्मा"


def test_devanagari_names():
    assert F.plausible_name("राहुल कुमार शर्मा")
    assert F.nice_name("राहुल कुमार शर्मा") == "राहुल कुमार शर्मा"
    assert F.nice_name("राहुल शर्मा / RAHUL SHARMA") == "Rahul Sharma"  # the Latin spelling wins
    assert F.local_name("राहुल शर्मा / RAHUL SHARMA") == "राहुल शर्मा"
    assert F.local_name("Rahul Sharma") is None
    assert not F.plausible_name("राहुल 1988")


def test_name_after_prefers_latin_spelling():
    text = "नाम: राहुल शर्मा\nName: Rahul Sharma"
    assert F.name_after(text, ["नाम", "name"]) == "Rahul Sharma"
    assert F.local_name_after(text, ["नाम", "name"]) == "राहुल शर्मा"


@pytest.mark.parametrize("text,gender", [
    ("लिंग: पुरुष", "Male"), ("लिंग / Gender: महिला", "Female"), ("लिंग: स्त्री", "Female"),
    ("पुरुष / MALE", "Male"), ("लिंग: अन्य", "Transgender"), ("पुरुषोत्तम जोशी", None),
])
def test_local_gender(text, gender):
    assert F.find_gender(text) == gender


def test_local_relations():
    assert F.relation_name("पुत्र श्री रमेश कुमार, गाँव") == ("Son of", "रमेश कुमार")
    assert F.relation_name("मुलगी: सुनीता पाटील\n") == ("Daughter of", "सुनीता पाटील")
    assert F.relation_name("पति का नाम सुरेश") is None
    assert F.relation_name("S/O Ramesh Kumar, पुत्र रमेश") == ("Son of", "Ramesh Kumar")


# ------------------------------------------------------------------ languages_of and search tokens

@pytest.mark.parametrize("text,langs", [
    ("यह मेरा घर है और मैं यहाँ रहता हूँ", ["Hindi"]),
    ("माझे नाव प्रिया नायर आहे", ["Marathi"]),
    ("गुणपत्रक विद्यार्थ्याचे नाव", ["Marathi"]),
    ("भारत सरकार Government of India राहुल शर्मा", ["Hindi or Marathi", "English"]),
    (_text(AADHAAR_FRONT), ["Hindi", "English"]),
    ("राशि रु. 500 UPI नकद भुगतान", ["Hindi"]),        # "UPI" alone does not make it English
    ("नकद भुगतान", ["Hindi or Marathi"]),
    ("Invoice for Rahul आ", ["English"]),                    # one stray glyph is not a language
    ("Plain English text", ["English"]),
    ("", []),
])
def test_languages_of(text, langs):
    assert F.languages_of(text) == langs


def test_word_tokens_keep_vowel_signs():
    assert F.word_tokens("मुंबईत‌ आहे। Rahul_1") == ["मुंबईत", "आहे", "Rahul_1"]


def test_match_query_keeps_devanagari_words_whole():
    assert match_query("मुंबईत आधार") == '"मुंबईत"* "आधार"*'
    assert match_query("मुं‌बई") == '"मुंबई"*'
    assert match_query("Rahul-Sharma") == '"Rahul"* "Sharma"*'


# ------------------------------------------------------------------ detection

def test_keyword_needs_whole_word():
    # 'आधारित' (based on) is not 'आधार' (Aadhaar)
    assert score_types("यह योजना आय पर आधारित है")["aadhaar"] == 0.0
    assert score_types("आधार")["aadhaar"] > 0


@pytest.mark.parametrize("lines,doc_type", [
    (AADHAAR_FRONT, "aadhaar"),
    (MARATHI_MARKSHEET, "marksheet"),
    (HINDI_RECEIPT, "receipt"),
    (PAN_BILINGUAL, "pan"),
    (["भारत निर्वाचन आयोग", "निर्वाचक फोटो पहचान पत्र", "ABC1234567", "मतदाता का नाम : राहुल शर्मा"], "voter_id"),
    (["महाराष्ट्र शासन", "वाहन चालक परवाना", "परवाना क्रमांक: MH12 20110012345", "नाव: राहुल शर्मा"],
     "driving_licence"),
    (["पंजीकरण प्रमाण पत्र", "पंजीकरण संख्या: MH 12 AB 1234", "चेसिस संख्या: MA3EWDE1S00123456"], "vehicle_rc"),
    (["भारत गणराज्य", "पासपोर्ट", "पासपोर्ट सं.: K1234567", "राष्ट्रीयता: भारतीय"], "passport"),
    (["कर बीजक", "सिंह इलेक्ट्रॉनिक्स", "बीजक संख्या: INV/2024/118", "कुल राशि: रु. 5,900.00"], "invoice"),
    (["एबीसी सॉफ्टवेयर प्राइवेट लिमिटेड", "वेतन पर्ची - मार्च २०२४", "शुद्ध वेतन: 72,000"], "salary_slip"),
    (["एबीसी कंपनी", "पगार पावती - एप्रिल २०२४", "निव्वळ पगार: 45,000"], "salary_slip"),
    (["बँक ऑफ महाराष्ट्र", "खाते उतारा", "आरंभीची शिल्लक 10,000"], "bank_statement"),
    (["माध्यमिक शिक्षा मंडल, मध्यप्रदेश", "अंक सूची", "अनुक्रमांक: 123456", "कुल प्राप्तांक: 420/500"], "marksheet"),
])
def test_detect_local_documents(lines, doc_type):
    assert detect(_text(lines)).doc_type == doc_type


def test_local_marksheet_is_not_money():
    scores = score_types(_text(MARATHI_MARKSHEET))
    assert scores["invoice"] < 0.5 and scores["receipt"] < 0.5


def test_rent_agreement_is_other_with_a_kind():
    text = ("किराया अनुबंध\nयह किरायानामा दिनांक 01/04/2024 को किया गया है। मासिक किराया राशि रु. 15,000 है।\n"
            "मकान मालिक: राहुल शर्मा, आधार संख्या: 4991 8877 6540, पैन: ABCPS1234K\nरसीद दी जाएगी।")
    assert detect(text).doc_type == "other"  # the parties' Aadhaar and PAN numbers do not make it an ID
    doc = parse("other", text)
    assert _vals(doc) == {"title": "किराया अनुबंध", "date": "2024-04-01", "kind": "Rent agreement"}
    assert parse("other", "भाडे करार\nदिनांक: ०१/०४/२०२४").fields["kind"].value == "Rent agreement"


# ------------------------------------------------------------------ parsers on rendered fixtures

def test_aadhaar_front_hindi_and_english():
    text = _rendered(AADHAAR_FRONT)
    det = detect(text)
    assert det.doc_type == "aadhaar"
    doc = parse("aadhaar", text)
    v = _vals(doc)
    assert v["name"] == "Rahul Kumar Sharma"
    assert v["name_local"] == "राहुल कुमार शर्मा"
    assert v["dob"] == "1988-08-15"
    assert v["gender"] == "Male"
    assert v["number"] == AADHAAR_VALID
    assert doc.person_name == "Rahul Kumar Sharma"
    assert "name_local" not in REGISTRY["aadhaar"].key_fields  # never lowers English confidence


def test_aadhaar_english_line_first_and_devanagari_digits():
    lines = ["Government of India", "Rahul Kumar Sharma", "राहुल कुमार शर्मा", "जन्म तिथि / DOB : 15/08/1988",
             "पुरुष / MALE", "४९९१ ८८७७ ६५४०"]
    v = _vals(parse("aadhaar", _rendered(lines)))
    assert v["name"] == "Rahul Kumar Sharma" and v["name_local"] == "राहुल कुमार शर्मा"
    assert v["number"] == AADHAAR_VALID


def test_aadhaar_hindi_only_name_and_address():
    lines = ["भारत सरकार", "राहुल कुमार शर्मा", "जन्म वर्ष: 1988", "पुरुष", AADHAAR_VALID,
             "पता: पुत्र रमेश शर्मा, 12, एमजी रोड,", "कोथरूड, पुणे 411038"]
    v = _vals(parse("aadhaar", _rendered(lines)))
    assert v["name"] == "राहुल कुमार शर्मा" and "name_local" not in v
    assert v["yob"] == "1988"
    assert v["relation"] == "Son of" and v["relative_name"] == "रमेश शर्मा"
    assert v["address"].startswith("12, एमजी रोड") and v["pincode"] == "411038"


def test_marathi_marksheet():
    text = _rendered(MARATHI_MARKSHEET)
    assert detect(text).doc_type == "marksheet"
    v = _vals(parse("marksheet", text))
    assert v["student_name"] == "प्रिया सुनील नायर"
    assert v["mother_name"] == "सुनीता"
    assert v["institution"].startswith("महाराष्ट्र राज्य माध्यमिक")
    assert v["exam"] == "माध्यमिक शालांत प्रमाणपत्र परीक्षा"
    assert v["roll_number"] == "M123456"
    assert v["total_marks"] == "452/500"
    assert v["percentage"] == "90.40%"
    assert v["result"] == "First Class With Distinction"
    assert v["year"] == "2023"
    assert v["issue_date"] == "2023-06-02"


def test_hindi_receipt_with_devanagari_digits_and_month():
    text = _rendered(HINDI_RECEIPT)
    assert detect(text).doc_type == "receipt"
    doc = parse("receipt", text)
    v = _vals(doc)
    assert v["receipt_number"] == "R-1024"
    assert v["date"] == "2024-03-12"
    assert v["amount"] == "₹1,250"
    assert v["payment_mode"] == "Cash"
    assert v["payer"] == "राहुल शर्मा"
    assert v["merchant"] == "शर्मा किराना भंडार"
    assert v["purpose"] == "मासिक राशन"
    assert doc.doc_date == "2024-03-12"


def test_pan_bilingual_labels():
    v = _vals(parse("pan", _rendered(PAN_BILINGUAL)))
    assert v["name"] == "Rahul Sharma"
    assert v["father_name"] == "Suresh Sharma"
    assert v["dob"] == "1988-08-15"
    assert v["number"] == "ABCPS1234K"


def test_voter_id_bilingual_and_hindi_only():
    lines = ["भारत निर्वाचन आयोग", "निर्वाचक फोटो पहचान पत्र", "ABC1234567", "निर्वाचक का नाम : राहुल शर्मा",
             "Elector's Name : Rahul Sharma", "पिता का नाम : सुरेश शर्मा", "Father's Name : Suresh Sharma",
             "लिंग / Gender : पुरुष / Male", "जन्म तिथि / Date of Birth : 15/08/1988", "पता : 12, एमजी रोड, पुणे 411038"]
    v = _vals(parse("voter_id", _rendered(lines)))
    assert v["name"] == "Rahul Sharma" and v["name_local"] == "राहुल शर्मा"
    assert v["relation"] == "Father" and v["relative_name"] == "Suresh Sharma"
    assert v["gender"] == "Male" and v["dob"] == "1988-08-15"
    assert v["address"] == "12, एमजी रोड, पुणे 411038"
    assert "issue_date" not in v  # the 'Date of Birth' line is not an issue date
    hindi = ["भारत निर्वाचन आयोग", "मतदाता का नाम : सीमा शर्मा", "पति का नाम : सुरेश शर्मा", "लिंग : महिला",
             "उम्र : 35", "ABC1234567"]
    v = _vals(parse("voter_id", _rendered(hindi)))
    assert v["name"] == "सीमा शर्मा" and v["relation"] == "Husband" and v["relative_name"] == "सुरेश शर्मा"
    assert v["gender"] == "Female" and v["age"] == "35"


def test_driving_licence_marathi():
    lines = ["महाराष्ट्र शासन", "वाहन चालक परवाना / Driving Licence", "परवाना क्रमांक: MH12 20110012345",
             "नाव / Name: Rahul Sharma", "नाव: राहुल शर्मा", "जन्म तारीख / DOB: 15-08-1988", "रक्त गट: B+",
             "पत्ता: 12, एमजी रोड, पुणे 411038", "जारी दिनांक: 01/02/2011", "वैधता (गैर परिवहन): 14/08/2038",
             "वैधता (परिवहन): 01/02/2014", "LMV MCWG"]
    v = _vals(parse("driving_licence", _rendered(lines)))
    assert v["number"] == "MH12 20110012345"
    assert v["name"] == "Rahul Sharma" and v["name_local"] == "राहुल शर्मा"
    assert v["dob"] == "1988-08-15" and v["blood_group"] == "B+"
    assert v["issue_date"] == "2011-02-01"
    assert v["valid_till"] == "2038-08-14" and v["valid_till_transport"] == "2014-02-01"


def test_vehicle_rc_hindi():
    lines = ["पंजीकरण प्रमाण पत्र", "पंजीकरण संख्या: MH 12 AB 1234", "स्वामी का नाम: राहुल शर्मा",
             "चेसिस संख्या: MA3EWDE1S00123456", "इंजन संख्या: K12MN1234567", "पंजीकरण की तिथि: 05 मार्च 2019",
             "ईंधन: पेट्रोल"]
    v = _vals(parse("vehicle_rc", _rendered(lines)))
    assert v["registration_number"] == "MH 12 AB 1234"
    assert v["owner_name"] == "राहुल शर्मा"
    assert v["chassis_number"] == "MA3EWDE1S00123456" and v["engine_number"] == "K12MN1234567"
    assert v["registration_date"] == "2019-03-05" and v["fuel"] == "पेट्रोल"


def test_passport_bilingual_labels():
    lines = ["भारत गणराज्य REPUBLIC OF INDIA", "पासपोर्ट / PASSPORT", "पासपोर्ट सं. / Passport No.", "K1234567",
             "उपनाम / Surname", "SHARMA", "दिया गया नाम / Given Name(s)", "RAHUL KUMAR",
             "राष्ट्रीयता / Nationality INDIAN", "लिंग / Sex M", "जन्मतिथि / Date of Birth 15/08/1988",
             "जारी करने का स्थान / Place of Issue MUMBAI", "जारी करने की तिथि / Date of Issue 10/01/2018",
             "समाप्ति की तिथि / Date of Expiry 09/01/2028"]
    text = _rendered(lines)
    assert detect(text).doc_type == "passport"
    v = _vals(parse("passport", text))
    assert v["surname"] == "Sharma" and v["given_names"] == "Rahul Kumar"
    assert v["number"] == "K1234567" and v["gender"] == "Male"
    assert v["dob"] == "1988-08-15" and v["issue_date"] == "2018-01-10" and v["expiry_date"] == "2028-01-09"
    hindi_only = ["पासपोर्ट", "उपनाम: शर्मा", "लिंग: पुरुष", "जन्मतिथि: 15/08/1988"]
    v = _vals(parse("passport", _text(hindi_only)))
    assert v["gender"] == "Male" and v["dob"] == "1988-08-15"


def test_invoice_hindi():
    lines = ["कर बीजक", "सिंह इलेक्ट्रॉनिक्स", "बीजक संख्या: INV/2024/118", "दिनांक: 05/04/2024",
             "ग्राहक का नाम: प्रिया नायर", "कुल कर: 900.00", "कुल राशि: रु. 5,900.00"]
    v = _vals(parse("invoice", _rendered(lines)))
    assert v["invoice_number"] == "INV/2024/118" and v["invoice_date"] == "2024-04-05"
    assert v["seller"] == "सिंह इलेक्ट्रॉनिक्स" and v["buyer"] == "प्रिया नायर"
    assert v["tax"] == "₹900" and v["total"] == "₹5,900"


def test_invoice_latin_seller_wins_over_devanagari():
    lines = ["बिल", "शर्मा स्टोर्स", "Sharma Stores", "बिल नं: 55", "कुल: 300"]
    v = _vals(parse("invoice", _text(lines)))
    assert v["seller"] == "Sharma Stores" and v["total"] == "₹300"


def test_salary_slip_hindi_and_marathi():
    hindi = ["एबीसी सॉफ्टवेयर प्राइवेट लिमिटेड", "वेतन पर्ची - मार्च २०२४", "कर्मचारी का नाम: राहुल शर्मा",
             "कर्मचारी कोड: EMP042", "पदनाम: वरिष्ठ अभियंता", "मूल वेतन: 50,000", "सकल वेतन: 80,000",
             "कुल कटौती: 8,000", "शुद्ध वेतन: 72,000"]
    v = _vals(parse("salary_slip", _rendered(hindi)))
    assert v["employer"] == "एबीसी सॉफ्टवेयर प्राइवेट लिमिटेड"
    assert v["employee_name"] == "राहुल शर्मा" and v["employee_id"] == "EMP042"
    assert v["month"] == "March 2024"
    assert v["gross_earnings"] == "₹80,000" and v["total_deductions"] == "₹8,000"
    assert v["net_pay"] == "₹72,000" and v["basic"] == "₹50,000"
    marathi = ["एबीसी कंपनी", "पगार पावती", "महिना: एप्रिल २०२४", "कर्मचाऱ्याचे नाव: प्रिया नायर",
               "निव्वळ पगार: 45,000"]
    v = _vals(parse("salary_slip", _rendered(marathi)))
    assert v["month"] == "April 2024" and v["employee_name"] == "प्रिया नायर" and v["net_pay"] == "₹45,000"


def test_other_parser_devanagari_title_only_without_latin():
    assert parse("other", "सूचना\nबैठक दिनांक १० जून २०२४ को होगी").title == "सूचना"
    assert parse("other", "सूचना\nNotice of meeting\nदिनांक १० जून २०२४").title == "Notice of meeting"


# ------------------------------------------------------------------ the language field

def _analyze(lines: list[str]):
    text = _text(lines)
    return analyze_text(text, [PageText(1, text, "ocr")])


def test_language_field_hindi_english():
    parsed, _score = _analyze(AADHAAR_FRONT)
    f = parsed.fields["language"]
    assert (f.value, f.label, f.confidence) == ("Hindi, English", "Language", 0.9)


def test_language_field_marathi_and_ambiguous():
    parsed, _ = _analyze(MARATHI_MARKSHEET)
    assert parsed.fields["language"].value == "Marathi"
    parsed, _ = _analyze(["रसीद", "रसीद संख्या: 12", "भुगतान राशि: रु. 500", "विवरण: मासिक शुल्क"])
    assert parsed.fields["language"].value == "Hindi"
    parsed, _ = _analyze(["कर बीजक", "बीजक क्रमांक: 12", "दिनांक: 05/04/2024", "कुल राशि: 500"])
    assert parsed.fields["language"].value == "Hindi"  # दिनांक and क्रमांक are not Marathi-only words


def test_no_language_field_for_english_documents():
    lines = ["INCOME TAX DEPARTMENT", "Permanent Account Number Card", "ABCPS1234K", "Name", "RAHUL SHARMA",
             "Date of Birth", "15/08/1988"]
    parsed, _ = _analyze(lines)
    assert "language" not in parsed.fields


def test_request_letter_about_a_statement_is_not_a_bank_statement():
    from app.documents.detect import detect

    letter = (
        "सेवा में,\nप्रबंधक, भारतीय स्टेट बैंक, पुणे शाखा\nविषय: बचत खाता विवरण के लिए अनुरोध\n"
        "महोदय, मैं राहुल शर्मा, खाता संख्या 1234567890 का धारक हूँ।\n"
        "कृपया अप्रैल 2026 से सितंबर 2026 तक का खाता विवरण भेजने की कृपा करें।\nधन्यवाद\nराहुल शर्मा"
    )
    assert detect(letter).doc_type == "other"
    english = "Dear Sir,\nSubject: request for account statement\nKindly send my bank statement for April 2026.\nThanks"
    assert detect(english).doc_type == "other"
    statement = "Bank statement\nAccount statement for 01/04/2026 to 30/04/2026\nOpening balance 1,000.00\nClosing balance 2,000.00\nNarration"
    assert detect(statement).doc_type == "bank_statement"
