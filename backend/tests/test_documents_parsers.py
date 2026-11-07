"""Parser and detection tests on synthetic OCR-like text for every document type (no models)."""
from __future__ import annotations

from datetime import date

import pytest

from app.documents import fields as F
from app.documents.detect import detect
from app.documents.fields import Box, PageText
from app.documents.parsers import REGISTRY, parse

# Checked by hand against the Verhoeff tables (the one-digit mutation fails).
AADHAAR_VALID = "4991 8877 6540"
AADHAAR_INVALID = "4991 8877 6541"
# ICAO 9303 specimen (every check digit valid) and an Indian passport line with computed check digits.
ICAO_L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
BCBP = "M1SHARMA/RAHUL KUMAR  EABC123 DELBOMAI 0865 123Y012A0045 100"

TEXTS: dict[str, str] = {
    "aadhaar": f"""GOVERNMENT OF INDIA
Rahul Kumar Sharma
DOB: 12/05/1990
MALE
{AADHAAR_VALID}
VID : 9123 4567 8901 2345
Mera Aadhaar, Meri Pehchan
Unique Identification Authority of India
Address: S/O Ramesh Kumar Sharma, House No 12, MG Road,
Kothrud, Pune, Maharashtra - 411038
""",
    "pan": """INCOME TAX DEPARTMENT        GOVT. OF INDIA
Permanent Account Number Card
ABCPS1234K
Name
RAHUL SHARMA
Father's Name
SURESH SHARMA
Date of Birth
15/08/1988
""",
    "passport": """REPUBLIC OF INDIA
Type P   Country Code IND   Passport No. Z1234567
Surname
SHARMA
Given Name(s)
RAHUL KUMAR
Nationality INDIAN   Sex M   Date of Birth 12/05/1990
Place of Birth PUNE, MAHARASHTRA
Place of Issue MUMBAI
Date of Issue 20/08/2021   Date of Expiry 19/08/2031
P<INDSHARMA<<RAHUL<KUMAR<<<<<<<<<<<<<<<<<<<<
Z1234567<1IND9005123M3108192<<<<<<<<<<<<<<08
""",
    "driving_licence": """Union of India
Maharashtra State Motor Driving Licence
DL No: MH12 20110012345
Issue Date: 05/03/2011
Validity (NT): 04/03/2031
Name: RAHUL SHARMA
S/O: SURESH SHARMA
DOB: 12/05/1990     Blood Group: B+
Address: Flat 4, Shanti Apartments, Kothrud, Pune 411038
COV: MCWG LMV
Licensing Authority: RTO PUNE
""",
    "voter_id": """ELECTION COMMISSION OF INDIA
ELECTOR PHOTO IDENTITY CARD
ABC1234567
Elector's Name : Rahul Sharma
Father's Name : Suresh Sharma
Sex : Male
Date of Birth : 12/05/1990
Address : 12, MG Road, Kothrud, Pune 411038
""",
    "vehicle_rc": """Government of Maharashtra
Certificate of Registration
Regn. No: MH 12 AB 1234
Date of Regn: 15/06/2019
Regn. Validity: 14/06/2034
Owner Name: RAHUL SHARMA
S/O SURESH SHARMA
Chassis No: MA3EWDE1S00123456
Engine No: K12MN1234567
Maker's Name: MARUTI SUZUKI INDIA LTD
Model: SWIFT VXI
Fuel: PETROL
Colour: WHITE
Seating Capacity: 5
Registering Authority: RTO PUNE
""",
    "bank_statement": """HDFC Bank
Statement of Account
Account Holder Name: RAHUL SHARMA
Account Number: 50100123456789
IFSC: HDFC0001234
Branch: Kothrud, Pune
Statement From 01/04/2024 To 30/06/2024
Opening Balance: Rs. 1,25,430.50
Date        Narration                    Withdrawal   Deposit    Balance
02/04/2024  UPI-SWIGGY                   450.00                  1,24,980.50
05/04/2024  SALARY APRIL                              85,000.00  2,09,980.50
Closing Balance: Rs. 1,98,765.25
""",
    "invoice": """ACME RETAIL PRIVATE LIMITED
TAX INVOICE
Invoice No: INV-2024-0042
Invoice Date: 15-Mar-2024
GSTIN: 27ABCDE1234F1Z0
Bill To: Rahul Sharma
Customer GSTIN: 29AAACR5055K1Z3
Place of Supply: Karnataka
Item              HSN     Qty   Rate      Amount
Wireless Mouse    8471    2     750.00    1,500.00
Taxable Value: 1,500.00
IGST 18%: 270.00
Grand Total: Rs. 1,770.00
Amount in words: Rupees One Thousand Seven Hundred Seventy Only
""",
    "receipt": """Sunrise Coaching Classes
FEE RECEIPT
Receipt No: R-1182
Date: 10/07/2024
Received with thanks from: Rahul Sharma
towards: Tuition fee for July 2024
Payment Mode: UPI
Transaction ID: 418922334455
Amount Paid: Rs. 5,000/-
""",
    "salary_slip": """Bright Tech Solutions Pvt Ltd
Payslip for the month of March 2024
Employee Name: Rahul Sharma
Employee ID: BT1042
Designation: Software Engineer
Department: Engineering
UAN: 100123456789
Days Paid: 31
Earnings                      Deductions
Basic: 40,000.00              Provident Fund: 4,800.00
HRA: 16,000.00                Professional Tax: 200.00
Gross Earnings: 75,000.00
Total Deductions: 5,000.00
Net Pay: Rs. 70,000.00
""",
    "marksheet": """SAVITRIBAI PHULE PUNE UNIVERSITY
STATEMENT OF MARKS
Examination: B.Sc. Computer Science Third Year, April 2012
Name of the Candidate: RAHUL SHARMA
Seat No: S123456
Mother's Name: SUNITA
Subject                 Max Marks    Marks Obtained
Data Structures         100          78
Operating Systems       100          81
Total: 456/600
SGPA: 8.20
Result: FIRST CLASS WITH DISTINCTION
""",
    "degree": """University of Mumbai
This is to certify that Rahul Sharma has been admitted to the degree of
Bachelor of Engineering
in Computer Engineering
having passed the examination held in May 2012 and was placed in First Class with Distinction.
Given under the seal of the University at the Convocation held on 15/02/2013
Registration No: 2008016400123
""",
    "certificate": """Coursera
Certificate of Completion
This is to certify that Rahul Sharma has successfully completed Machine Learning Specialization
an online non-credit course authorized by Stanford University
Date: 22/09/2023
Credential ID: ABC123XYZ789
""",
    "offer_letter": """Bright Tech Solutions Pvt Ltd
Date: 01/02/2024
Ref No: BT/HR/2024/017
Dear Rahul Sharma,
Offer of Employment
We are pleased to offer you the position of Senior Software Engineer at Bright Tech Solutions Pvt Ltd.
Your annual Cost to Company (CTC) will be Rs. 18,50,000 per annum.
Your date of joining will be 01/03/2024.
Location: Pune
You will be on a probation period of six months.
""",
    "insurance_policy": """Star Health and Allied Insurance Co. Ltd
Health Insurance Policy Schedule
Policy Number: P/171111/01/2024/012345
Name of the Insured: Rahul Sharma
Plan: Family Health Optima
Period of Insurance: From 01/04/2024 to 31/03/2025
Sum Insured: Rs. 5,00,000
Total Premium: Rs. 18,450.00
Nominee: Sunita Sharma
""",
    "prescription": """City Care Clinic
Dr. Anil Mehta  MBBS, MD
Reg. No: MMC 12345
Date: 12/06/2024
Patient Name: Rahul Sharma        Age: 34 Yrs
Diagnosis: Acute pharyngitis
Rx
1. Tab Azithromycin 500 mg  once daily x 3 days
2. Tab Paracetamol 650 mg  SOS
3. Syp Ascoril LS  10 ml twice daily
Review after 5 days
""",
    "lab_report": """Metro Diagnostics Laboratory
Patient Name: Rahul Sharma
Age/Sex: 34 Years / Male
Ref. By: Dr. Anil Mehta
Sample Collected On: 10/06/2024
Reported On: 11/06/2024
Test Report: Complete Blood Count
Test                 Result   Units     Biological Reference Interval
Haemoglobin          12.1     g/dL      13.0 - 17.0
Total WBC Count      7200     /cumm     4000 - 11000
Platelet Count       2.5      lakh/cumm 1.5 - 4.1
""",
    "vaccination": """Certificate for COVID-19 Vaccination
Issued in India by Ministry of Health & Family Welfare, Govt. of India
Beneficiary Details
Beneficiary Name: Rahul Sharma
Age: 34
Gender: Male
ID Verified: Aadhaar # XXXXXXXX6540
Beneficiary Reference ID: 12345678901234
Vaccination Details
Vaccine Name: COVISHIELD
Date of 1st Dose: 10/05/2021
Date of 2nd Dose: 02/08/2021
Vaccinated by: Priya Patil
Vaccination at: District Hospital, Pune
CoWIN
""",
    "event_pass": """BookMyShow
Coldplay Music of the Spheres World Tour
Venue: DY Patil Stadium, Navi Mumbai
Date: 19 Jan 2025
Show Time: 7:30 PM
Section: North Stand L2   Row: K   Seat: 23
Entry Gate: 5
Booking ID: TKT8X2K91
Admit One
Ticket Type: General Standing
""",
    "boarding_pass": f"""BOARDING PASS
Air India
Name of Passenger: SHARMA/RAHUL KUMAR MR
From DEL To BOM
Flight AI 865
Date 02 May 2025
Boarding Time 06:45
Gate 23   Seat 12A
PNR: ABC123
{BCBP}
""",
}


def page_of(text: str, barcodes=None) -> list[PageText]:
    return [PageText(page_no=1, text=text, source="ocr", barcodes=list(barcodes or []))]


# ------------------------------------------------------------------ helpers

def test_verhoeff():
    assert F.verhoeff_valid(AADHAAR_VALID)
    assert not F.verhoeff_valid(AADHAAR_INVALID)
    assert F.find_aadhaar(f"no {AADHAAR_INVALID} here") is None
    assert F.find_aadhaar(f"Aadhaar {AADHAAR_VALID}") == AADHAAR_VALID


def test_mrz_icao_specimen():
    l2 = ICAO_L2
    assert F.mrz_valid(l2[0:9], l2[9])
    assert F.mrz_valid(l2[13:19], l2[19])
    assert F.mrz_valid(l2[21:27], l2[27])
    assert F.mrz_valid(l2[28:42], l2[42])
    assert F.mrz_valid(l2[0:10] + l2[13:20] + l2[21:43], l2[43])
    assert not F.mrz_valid(l2[0:9], "7")


def test_gstin_checksum_and_pan_masking():
    assert F.gstin_valid("27ABCDE1234F1Z0")
    assert not F.gstin_valid("27ABCDE1234F1Z1")
    # the PAN inside a GSTIN is not reported as a PAN
    assert F.find_pans("GSTIN: 27ABCDE1234F1Z0") == []
    assert F.find_pans("PAN ABCPS1234K") == ["ABCPS1234K"]


@pytest.mark.parametrize("raw,iso", [
    ("12/05/1990", "1990-05-12"), ("12-05-1990", "1990-05-12"), ("12.05.1990", "1990-05-12"),
    ("12 May 1990", "1990-05-12"), ("12-May-1990", "1990-05-12"), ("May 12, 1990", "1990-05-12"),
    ("1990-05-12", "1990-05-12"), ("31st December 2023", "2023-12-31"), ("30/02/2020", None),
])
def test_dates(raw, iso):
    assert F.parse_date(raw) == iso


def test_two_digit_year_pivot():
    yy = (date.today().year + 5) % 100
    assert F.parse_date(f"01/01/{yy:02d}", past=True).startswith("19")
    assert F.parse_date(f"01/01/{yy:02d}").startswith("20")


@pytest.mark.parametrize("raw,val", [
    ("Rs. 1,23,456.50", "\u20b91,23,456.50"), ("INR 5000", "\u20b95,000"), ("\u20b9 75,000.00", "\u20b975,000"),
    ("12 lakh", "\u20b912,00,000"), ("1.5 crore", "\u20b91,50,00,000"), ("CTC 18 LPA", "\u20b918,00,000"),
    ("Rupees One Lakh Twenty Thousand Five Hundred Only", "\u20b91,20,500"),
])
def test_amounts(raw, val):
    assert F.format_inr(F.parse_amount(raw)) == val


def test_labels_and_relations():
    assert F.after_label("Name: Rahul Sharma  DOB: 12/05/1990", ["name"], anchored=True) == "Rahul Sharma"
    assert F.after_label("Father's Name\nSURESH SHARMA", ["father's name"]) == "SURESH SHARMA"
    assert F.relation_name("S/O Ramesh Kumar, House 12") == ("Son of", "Ramesh Kumar")
    assert F.relation_name("W/O: Anil Mehta\n") == ("Wife of", "Anil Mehta")
    assert F.relation_name("D/O Suresh Rao, Pune") == ("Daughter of", "Suresh Rao")
    assert F.find_gender("Sex: F") == "Female"


@pytest.mark.parametrize("raw,num,state", [
    ("DL No: MH12 20110012345", "MH12 20110012345", "Maharashtra"),
    ("DL-0420110149646", "DL04 20110149646", "Delhi"),
    ("TN-01-2008-0012345", "TN01 20080012345", "Tamil Nadu"),
    ("KA0119990001234", "KA01 19990001234", "Karnataka"),
    ("Licence No. MH/12/001234/2011", "MH/12/001234/2011", "Maharashtra"),
])
def test_dl_formats(raw, num, state):
    assert F.find_dl_number(raw) == (num, state)


def test_vehicle_numbers():
    assert F.find_vehicle_number("MH 12 AB 1234") == "MH 12 AB 1234"
    assert F.find_vehicle_number("ka-01-mj-0007") == "KA 01 MJ 0007"
    assert F.find_vehicle_number("22 BH 1234 AA") == "22 BH 1234 AA"


# ------------------------------------------------------------------ detection

@pytest.mark.parametrize("doc_type", list(TEXTS))
def test_detect_picks_own_type(doc_type):
    barcodes = [("QRCode", BCBP)] if doc_type == "boarding_pass" else []
    d = detect(TEXTS[doc_type], page_of(TEXTS[doc_type], barcodes))
    assert d.doc_type == doc_type, (d.doc_type, sorted(d.scores.items(), key=lambda kv: -kv[1])[:3])
    assert d.score >= 0.5


def test_detect_other_and_filename_hint():
    assert detect("Shopping list: milk, eggs, bread and some fruit for the week").doc_type == "other"
    assert detect("").doc_type == "other"
    d = detect("Name: Rahul\nTotal Premium: 100", item={"original_filename": "policy.pdf"})
    assert d.scores["insurance_policy"] > 0


# ------------------------------------------------------------------ parsers

def run(doc_type: str, **kw):
    pages = page_of(TEXTS[doc_type], kw.pop("barcodes", None))
    res = parse(doc_type, TEXTS[doc_type], pages, kw.pop("boxes", []))
    vals = {k: v.value for k, v in res.fields.items()}
    return res, vals


def test_aadhaar():
    res, v = run("aadhaar")
    assert v["number"] == AADHAAR_VALID
    assert v["name"] == "Rahul Kumar Sharma"
    assert v["dob"] == "1990-05-12"
    assert v["gender"] == "Male"
    assert v["vid"] == "9123 4567 8901 2345"
    assert v["relative_name"] == "Ramesh Kumar Sharma"
    assert v["pincode"] == "411038"
    assert "Pune" in v["address"] and not v["address"].lower().startswith("s/o")
    assert res.person_name == "Rahul Kumar Sharma"
    assert res.confidence >= 0.9


def test_aadhaar_invalid_number_rejected():
    text = TEXTS["aadhaar"].replace(AADHAAR_VALID, AADHAAR_INVALID)
    res = parse("aadhaar", text)
    assert "number" not in res.fields


def test_pan_labelled_and_old_layout():
    res, v = run("pan")
    assert v["number"] == "ABCPS1234K"
    assert v["holder_type"] == "Individual"
    assert v["name"] == "Rahul Sharma" and v["father_name"] == "Suresh Sharma"
    assert v["dob"] == "1988-08-15"
    old = "INCOME TAX DEPARTMENT GOVT. OF INDIA\nRAHUL SHARMA\nSURESH SHARMA\n15/08/1988\nPermanent Account Number\nABCPS1234K\nSignature"
    r2 = parse("pan", old)
    v2 = {k: f.value for k, f in r2.fields.items()}
    assert v2["name"] == "Rahul Sharma" and v2["father_name"] == "Suresh Sharma" and v2["dob"] == "1988-08-15"


def test_passport_mrz():
    res, v = run("passport")
    assert v["number"] == "Z1234567"
    assert v["surname"] == "Sharma" and v["given_names"] == "Rahul Kumar"
    assert v["dob"] == "1990-05-12" and v["expiry_date"] == "2031-08-19"
    assert v["gender"] == "Male" and v["nationality"] == "Indian"
    assert v["mrz_valid"] == "5 of 5"
    assert v["issue_date"] == "2021-08-20" and v["place_of_issue"].startswith("MUMBAI")
    assert res.person_name == "Rahul Kumar Sharma"
    assert res.confidence >= 0.9


def test_passport_bad_check_digit_lowers_confidence():
    text = TEXTS["passport"].replace("Z1234567<1IND", "Z1234567<7IND")
    from app.documents.parsers.passport import read_mrz

    assert read_mrz(text)["checks_ok"] == 3
    assert parse("passport", text).fields["mrz_valid"].value == "3 of 5"


def test_driving_licence():
    res, v = run("driving_licence")
    assert v["number"] == "MH12 20110012345" and v["state"] == "Maharashtra"
    assert v["name"] == "Rahul Sharma" and v["relative_name"] == "Suresh Sharma"
    assert v["dob"] == "1990-05-12" and v["issue_date"] == "2011-03-05" and v["valid_till"] == "2031-03-04"
    assert v["blood_group"] == "B+"
    assert "MCWG" in v["vehicle_classes"] and "LMV" in v["vehicle_classes"]
    assert v["issuing_authority"] == "RTO PUNE"


def test_voter_id():
    _res, v = run("voter_id")
    assert v["number"] == "ABC1234567"
    assert v["name"] == "Rahul Sharma" and v["relative_name"] == "Suresh Sharma" and v["relation"] == "Father"
    assert v["gender"] == "Male" and v["dob"] == "1990-05-12"


def test_vehicle_rc():
    _res, v = run("vehicle_rc")
    assert v["registration_number"] == "MH 12 AB 1234"
    assert v["owner_name"] == "Rahul Sharma"
    assert v["chassis_number"] == "MA3EWDE1S00123456" and v["engine_number"] == "K12MN1234567"
    assert v["registration_date"] == "2019-06-15" and v["valid_till"] == "2034-06-14"
    assert v["fuel"] == "PETROL" and v["seating"] == "5"


def test_bank_statement():
    res, v = run("bank_statement")
    assert v["bank"] == "HDFC Bank"
    assert v["account_number"] == "50100123456789" and v["ifsc"] == "HDFC0001234"
    assert v["period_from"] == "2024-04-01" and v["period_to"] == "2024-06-30"
    assert v["opening_balance"] == "\u20b91,25,430.50" and v["closing_balance"] == "\u20b91,98,765.25"
    assert res.doc_date == "2024-06-30"


def test_invoice():
    res, v = run("invoice")
    assert v["invoice_number"] == "INV-2024-0042" and v["invoice_date"] == "2024-03-15"
    assert v["seller"] == "ACME RETAIL PRIVATE LIMITED"
    assert v["seller_gstin"] == "27ABCDE1234F1Z0" and v["buyer_gstin"] == "29AAACR5055K1Z3"
    assert v["total"] == "\u20b91,770" and v["tax"] == "\u20b9270"
    assert res.title == "Invoice INV-2024-0042 from ACME RETAIL PRIVATE LIMITED"


def test_receipt():
    _res, v = run("receipt")
    assert v["receipt_number"] == "R-1182" and v["date"] == "2024-07-10"
    assert v["amount"] == "\u20b95,000" and v["payment_mode"] == "UPI"
    assert v["transaction_id"] == "418922334455" and v["payer"] == "Rahul Sharma"


def test_salary_slip():
    res, v = run("salary_slip")
    assert v["employer"] == "Bright Tech Solutions Pvt Ltd"
    assert v["employee_name"] == "Rahul Sharma" and v["employee_id"] == "BT1042"
    assert v["month"] == "March 2024" and res.doc_date == "2024-03-01"
    assert v["net_pay"] == "\u20b970,000" and v["gross_earnings"] == "\u20b975,000"
    assert v["uan"] == "100123456789"


def test_marksheet():
    _res, v = run("marksheet")
    assert v["institution"] == "SAVITRIBAI PHULE PUNE UNIVERSITY"
    assert v["student_name"] == "Rahul Sharma" and v["roll_number"] == "S123456"
    assert v["sgpa"] == "8.20" and v["total_marks"] == "456/600" and v["percentage"] == "76.00%"
    assert v["result"] == "First Class With Distinction"
    assert v["year"] == "2012"


def test_degree():
    _res, v = run("degree")
    assert v["recipient"] == "Rahul Sharma"
    assert v["degree"] == "Bachelor of Engineering"
    assert v["specialisation"] == "Computer Engineering"
    assert v["institution"] == "University of Mumbai"
    assert v["class"] == "First Class With Distinction"
    assert v["registration_number"] == "2008016400123"


def test_certificate():
    _res, v = run("certificate")
    assert v["recipient"] == "Rahul Sharma"
    assert v["title"] == "Certificate Of Completion"
    assert v["course"].startswith("Machine Learning Specialization")
    assert v["date"] == "2023-09-22" and v["certificate_id"] == "ABC123XYZ789"


def test_offer_letter():
    res, v = run("offer_letter")
    assert v["candidate"] == "Rahul Sharma"
    assert v["designation"] == "Senior Software Engineer"
    assert v["ctc"] == "\u20b918,50,000"
    assert v["joining_date"] == "2024-03-01" and v["letter_date"] == "2024-02-01"
    assert v["location"] == "Pune" and v["probation"] == "six months"
    assert "Bright Tech Solutions" in v["employer"]


def test_offer_letter_lpa():
    res = parse("offer_letter", "Offer Letter\nDear Asha Rao,\nYour CTC will be 12.5 LPA.\n")
    assert res.fields["ctc"].value == "\u20b912,50,000"


def test_insurance():
    res, v = run("insurance_policy")
    assert v["insurer"] == "Star Health" and v["policy_type"] == "Health"
    assert v["policy_number"] == "P/171111/01/2024/012345"
    assert v["insured_name"] == "Rahul Sharma"
    assert v["sum_insured"] == "\u20b95,00,000" and v["premium"] == "\u20b918,450"
    assert v["valid_from"] == "2024-04-01" and v["valid_to"] == "2025-03-31"
    assert v["nominee"] == "Sunita Sharma"


def test_prescription():
    _res, v = run("prescription")
    assert v["doctor"] == "Dr. Anil Mehta"
    assert "MBBS" in v["qualification"] and "MD" in v["qualification"]
    assert v["patient"] == "Rahul Sharma" and v["age"] == "34"
    assert v["date"] == "2024-06-12" and v["clinic"] == "City Care Clinic"
    assert v["medicines"].count(";") == 2 and v["medicines"].startswith("Tab Azithromycin")
    assert v["diagnosis"] == "Acute pharyngitis"


def test_lab_report():
    _res, v = run("lab_report")
    assert v["lab"] == "Metro Diagnostics Laboratory"
    assert v["patient"] == "Rahul Sharma" and v["gender"] == "Male" and v["age"] == "34"
    assert v["sample_date"] == "2024-06-10" and v["report_date"] == "2024-06-11"
    assert "Haemoglobin: 12.1 g/dL (13.0 - 17.0)" in v["tests"]
    assert v["abnormal"] == "Haemoglobin"


def test_vaccination():
    _res, v = run("vaccination")
    assert v["beneficiary_name"] == "Rahul Sharma"
    assert v["beneficiary_id"] == "12345678901234"
    assert v["vaccine"] == "COVISHIELD"
    assert v["dose1_date"] == "2021-05-10" and v["dose2_date"] == "2021-08-02"
    assert v["doses"] == "2"


def test_event_pass_with_boxes_and_payload():
    boxes = [Box(1, 0.1, 0.02, 0.5, 0.05, "BookMyShow"),
             Box(1, 0.1, 0.08, 0.9, 0.16, "Coldplay Music of the Spheres World Tour")]
    res, v = run("event_pass", boxes=boxes, barcodes=[("QRCode", '{"booking": "TKT8X2K91", "seat": "K-23", "gate": "5"}')])
    assert v["event"] == "Coldplay Music of the Spheres World Tour"
    assert v["venue"] == "DY Patil Stadium, Navi Mumbai"
    assert v["date"] == "2025-01-19" and v["time"] == "19:30"
    assert v["booking_id"] == "TKT8X2K91" and v["gate"] == "5"
    assert v["section"].startswith("North Stand") and v["row"] == "K"
    assert v["organiser"] == "BookMyShow"
    assert res.barcode == '{"booking": "TKT8X2K91", "seat": "K-23", "gate": "5"}' and res.barcode_format == "QRCode"


def test_event_pass_payload_forms():
    from app.documents.parsers.event_pass import payload_fields

    assert payload_fields("TKT8X2K91") == {"booking_id": "TKT8X2K91"}
    assert payload_fields("order=AB1234;seat=12;gate=3") == {"booking_id": "AB1234", "seat": "12", "gate": "3"}


def test_boarding_pass_bcbp():
    from app.documents.parsers.boarding_pass import read_bcbp

    b = read_bcbp(BCBP, today=date(2025, 4, 20))
    assert b["passenger"] == "Rahul Kumar Sharma"
    assert (b["pnr"], b["from"], b["to"], b["flight"], b["seat"], b["sequence"]) == ("ABC123", "DEL", "BOM", "AI 865", "12A", "45")
    assert b["date"] == "2025-05-03"  # day 123 of 2025
    assert b["cabin"] == "Economy"
    res, v = run("boarding_pass", barcodes=[("PDF417", BCBP)])
    assert v["passenger"] == "Rahul Kumar Sharma" and v["pnr"] == "ABC123"
    assert v["from"] == "DEL" and v["to"] == "BOM" and v["flight"] == "AI 865"
    assert v["gate"] == "23" and v["boarding_time"] == "06:45" and v["seat"] == "12A"
    assert v["airline"] == "Air India"
    assert res.barcode_format == "PDF417"
    assert res.title == "Boarding pass DEL to BOM, AI 865"


def test_boarding_pass_text_only():
    text = "BOARDING PASS\nIndiGo\nPassenger: RAO/ASHA MS\nFlight 6E 2134\nFrom BLR To DEL\nSeat 3C  Gate A7\nPNR: XY7Z9Q\nDate 05/11/2025"
    res = parse("boarding_pass", text)
    v = {k: f.value for k, f in res.fields.items()}
    assert v["passenger"] == "Asha Rao" and v["flight"] == "6E 2134" and v["airline"] == "IndiGo"
    assert v["from"] == "BLR" and v["to"] == "DEL" and v["seat"] == "3C" and v["gate"] == "A7"
    assert v["date"] == "2025-11-05"


def test_other_parser():
    res = parse("other", "Meeting notes\nDiscussed the roadmap on 03/04/2024")
    assert res.title == "Meeting notes" and res.doc_date == "2024-04-03"


def test_every_parser_tolerates_garbage():
    for doc_type in REGISTRY:
        for text in ("", "\f\f", "12345 ::: //// ----", "x" * 5000):
            res = parse(doc_type, text)
            assert res.doc_type == doc_type
            assert 0.0 <= res.confidence <= 1.0


def test_registry_matches_api_labels():
    from app.api.common import DOC_GROUP_OF, DOC_TYPE_LABELS

    assert set(REGISTRY) == set(DOC_TYPE_LABELS) == set(DOC_GROUP_OF)
    for t, spec in REGISTRY.items():
        assert spec.label == DOC_TYPE_LABELS[t]
        assert spec.group == DOC_GROUP_OF[t]
        assert set(spec.key_fields) <= set(spec.field_order)


def test_fields_in_type_order():
    res, _v = run("aadhaar")
    order = list(REGISTRY["aadhaar"].field_order)
    keys = list(res.fields)
    assert keys == sorted(keys, key=order.index)
