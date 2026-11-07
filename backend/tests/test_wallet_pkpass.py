"""Apple Wallet pass builder: configuration check, pass.json layout and a signature that verifies
against the manifest with a self-signed certificate standing in for Apple's."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import zipfile

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

from app.util import json_dumps, now_iso
from tests.helpers import make_item

OID_MESSAGE_DIGEST = bytes.fromhex("2a864886f70d010904")   # 1.2.840.113549.1.9.4
BCBP = "M1SHARMA/RAHUL KUMAR  EABC123 DELBOMAI 0865 123Y012A0045 100"


def _cert(name: str, key=None):
    key = key or rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=30)).sign(key, hashes.SHA256()))
    return cert, key


@pytest.fixture()
def wallet(data_dir):
    from app import storage
    from app.config import settings

    d = storage.wallet_dir()
    d.mkdir(parents=True, exist_ok=True)
    cert, key = _cert("Pass Type ID: pass.test.pcc")
    wwdr, _k = _cert("Test WWDR")
    (d / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (d / "wwdr.pem").write_bytes(wwdr.public_bytes(serialization.Encoding.PEM))
    (d / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
    settings.wallet_pass_type_id = "pass.test.pcc"
    settings.wallet_team_id = "TEAM123456"
    return {"cert": cert, "key": key, "wwdr": wwdr}


def _pass_item(conn, doc_type="event_pass", fields=None, barcode=None, barcode_format=None, doc_date="2025-01-19"):
    item = make_item(conn, kind="image")
    fields = fields or {
        "event": {"value": "Coldplay Music of the Spheres", "label": "Event", "confidence": 0.9},
        "venue": {"value": "DY Patil Stadium", "label": "Venue", "confidence": 0.9},
        "date": {"value": "2025-01-19", "label": "Date", "confidence": 0.9},
        "time": {"value": "19:30", "label": "Time", "confidence": 0.9},
        "seat": {"value": "K-23", "label": "Seat", "confidence": 0.9},
        "gate": {"value": "5", "label": "Gate", "confidence": 0.9},
        "booking_id": {"value": "TKT8X2K91", "label": "Booking ID", "confidence": 0.9},
    }
    group = "pass" if doc_type in ("event_pass", "boarding_pass") else "id"
    now = now_iso()
    conn.execute(
        "INSERT INTO documents(item_id, doc_type, doc_group, confidence, fields, title, doc_date, barcode, "
        "barcode_format, created_at, updated_at) VALUES (?, ?, ?, 0.9, ?, 'Coldplay', ?, ?, ?, ?, ?)",
        (item["id"], doc_type, group, json_dumps(fields), doc_date, barcode, barcode_format, now, now),
    )
    doc = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item["id"],)).fetchone()
    return item, doc


# ------------------------------------------------------------------ a tiny DER reader for the checks

def _tlv(buf: bytes, i: int) -> tuple[int, int, int, int]:
    """(tag, header length, content length, end) of the DER element at i."""
    tag = buf[i]
    ln = buf[i + 1]
    hdr = 2
    if ln & 0x80:
        n = ln & 0x7F
        ln = int.from_bytes(buf[i + 2:i + 2 + n], "big")
        hdr += n
    return tag, hdr, ln, i + hdr + ln


def _children(buf: bytes, start: int, end: int) -> list[tuple[int, int, int, int]]:
    out, i = [], start
    while i < end:
        tag, hdr, ln, nxt = _tlv(buf, i)
        out.append((tag, i, i + hdr, nxt))
        i = nxt
    return out


def _signer_info(sig: bytes) -> tuple[bytes, bytes, bytes]:
    """(signedAttrs DER re-tagged as SET, messageDigest value, signature) of the first SignerInfo."""
    _t, hdr, _l, end = _tlv(sig, 0)                       # ContentInfo SEQUENCE
    kids = _children(sig, hdr, end)                       # contentType OID, [0] content
    _tag, s, cs, e = kids[1]
    sd_tag, sd_s, sd_cs, sd_e = _children(sig, cs, e)[0]  # SignedData SEQUENCE
    sd = _children(sig, sd_cs, sd_e)
    signer_infos = [k for k in sd if k[0] == 0x31][-1]    # last SET = signerInfos
    si = _children(sig, signer_infos[2], signer_infos[3])[0]
    parts = _children(sig, si[2], si[3])
    attrs = next(p for p in parts if p[0] == 0xA0)
    signature = next(p for p in parts if p[0] == 0x04)
    signed_attrs = b"\x31" + sig[attrs[1] + 1:attrs[3]]
    digest = None
    for a in _children(sig, attrs[2], attrs[3]):
        a_kids = _children(sig, a[2], a[3])
        oid = sig[a_kids[0][2]:a_kids[0][3]]
        if oid == OID_MESSAGE_DIGEST:
            octet = _children(sig, a_kids[1][2], a_kids[1][3])[0]
            digest = sig[octet[2]:octet[3]]
    assert digest is not None
    return signed_attrs, digest, sig[signature[2]:signature[3]]


# ------------------------------------------------------------------ tests

def test_configured_reports_missing(data_dir):
    from app.wallet import pkpass

    ok, missing = pkpass.configured()
    assert not ok
    assert set(missing) == {"WALLET_PASS_TYPE_ID", "WALLET_TEAM_ID", "cert.pem", "key.pem", "wwdr.pem"}


def test_build_unconfigured_raises(conn):
    from app.wallet import pkpass

    item, doc = _pass_item(conn)
    with pytest.raises(pkpass.WalletNotConfigured):
        pkpass.build_pass(conn, item, doc)


def test_event_ticket_pass(conn, wallet):
    from cryptography.hazmat.primitives.serialization import pkcs7

    from app.wallet import pkpass

    assert pkpass.configured() == (True, [])
    item, doc = _pass_item(conn, barcode='{"booking": "TKT8X2K91"}', barcode_format="QRCode")
    data = pkpass.build_pass(conn, item, doc)
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {"pass.json", "manifest.json", "signature", "icon.png", "icon@2x.png", "logo.png", "thumbnail.png"} <= names
    assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
    pj = json.loads(z.read("pass.json"))
    assert pj["formatVersion"] == 1 and pj["passTypeIdentifier"] == "pass.test.pcc"
    assert pj["teamIdentifier"] == "TEAM123456" and pj["serialNumber"] == item["uid"]
    assert pj["organizationName"] and pj["description"] == "Coldplay"
    et = pj["eventTicket"]
    assert et["primaryFields"][0]["value"] == "Coldplay Music of the Spheres"
    assert {f["key"] for f in et["secondaryFields"]} == {"venue", "date"}
    assert {f["key"] for f in et["auxiliaryFields"]} == {"time", "seat", "gate"}
    assert et["backFields"][0]["key"] == "booking_id"
    keys = [f["key"] for part in et.values() if isinstance(part, list) for f in part]
    assert len(keys) == len(set(keys))
    assert pj["barcode"] == {"format": "PKBarcodeFormatQR", "message": '{"booking": "TKT8X2K91"}',
                             "messageEncoding": "iso-8859-1", "altText": "TKT8X2K91"}
    assert pj["barcodes"][0]["format"] == "PKBarcodeFormatQR"
    assert pj["relevantDate"] == "2025-01-19T19:30+05:30"
    assert pj["backgroundColor"].startswith("rgb(")
    # manifest covers every other file
    manifest = json.loads(z.read("manifest.json"))
    assert set(manifest) == names - {"manifest.json", "signature"}
    for n, h in manifest.items():
        assert hashlib.sha1(z.read(n)).hexdigest() == h
    # signature: SignedData with the signer and WWDR certificates, digest of manifest.json, RSA signature
    sig = z.read("signature")
    certs = pkcs7.load_der_pkcs7_certificates(sig)
    subjects = {c.subject.rfc4514_string() for c in certs}
    assert subjects == {"CN=Pass Type ID: pass.test.pcc", "CN=Test WWDR"}
    signed_attrs, digest, signature = _signer_info(sig)
    assert digest == hashlib.sha256(z.read("manifest.json")).digest()
    wallet["cert"].public_key().verify(signature, signed_attrs, padding.PKCS1v15(), hashes.SHA256())
    assert z.read("manifest.json") not in sig  # detached


def test_boarding_pass_layout_and_code128(conn, wallet):
    from app.wallet import pkpass

    fields = {
        "passenger": {"value": "Rahul Kumar Sharma", "label": "Passenger"},
        "from": {"value": "DEL", "label": "From"}, "to": {"value": "BOM", "label": "To"},
        "flight": {"value": "AI 865", "label": "Flight"}, "date": {"value": "2025-05-03", "label": "Date"},
        "boarding_time": {"value": "06:45", "label": "Boarding"}, "seat": {"value": "12A", "label": "Seat"},
        "pnr": {"value": "ABC123", "label": "PNR"},
    }
    item, doc = _pass_item(conn, "boarding_pass", fields, barcode=BCBP, barcode_format="Code128", doc_date="2025-05-03")
    z = zipfile.ZipFile(io.BytesIO(pkpass.build_pass(conn, item, doc)))
    pj = json.loads(z.read("pass.json"))
    bp = pj["boardingPass"]
    assert bp["transitType"] == "PKTransitTypeAir"
    assert [f["value"] for f in bp["primaryFields"]] == ["DEL", "BOM"]
    assert "barcode" not in pj and pj["barcodes"][0]["format"] == "PKBarcodeFormatCode128"
    assert pj["relevantDate"] == "2025-05-03T06:45+05:30"
    assert "thumbnail.png" not in z.namelist()


def test_not_a_pass(conn, wallet):
    from app.wallet import pkpass

    item, doc = _pass_item(conn, "pan", {"number": {"value": "ABCPS1234K", "label": "PAN"}})
    with pytest.raises(pkpass.NotAPass):
        pkpass.build_pass(conn, item, doc)


def test_bad_key_is_a_wallet_error(conn, wallet):
    from app import storage
    from app.wallet import pkpass

    (storage.wallet_dir() / "key.pem").write_text("not a key")
    item, doc = _pass_item(conn)
    with pytest.raises(pkpass.WalletError):
        pkpass.build_pass(conn, item, doc)


def test_dominant_colour_darkens_light_images():
    from PIL import Image

    from app.wallet.pkpass import colours_for, dominant_colour

    assert dominant_colour(Image.new("RGB", (50, 50), (200, 20, 20))) == (200, 20, 20)
    r, g, b = dominant_colour(Image.new("RGB", (50, 50), (250, 250, 250)))
    assert max(r, g, b) < 200
    assert colours_for(None)["foregroundColor"] == "rgb(255, 255, 255)"
