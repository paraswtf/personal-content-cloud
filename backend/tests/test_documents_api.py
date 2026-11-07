"""Documents API: list (filters, FTS q, cursor, vault), types, get, patch, reparse, wallet download.

Runs the documents router alone with require_user overridden, so it does not depend on the auth area."""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.documents import extract
from tests.helpers import make_item

H = {"X-Requested-With": "pcc"}
INVOICE = "\n".join([
    "ACME RETAIL PRIVATE LIMITED", "TAX INVOICE", "Invoice No: INV-2024-0042", "Invoice Date: 15-Mar-2024",
    "GSTIN: 27ABCDE1234F1Z0", "Bill To: Rahul Sharma", "Grand Total: Rs. 1,770.00",
])
PAN = "\n".join(["INCOME TAX DEPARTMENT   GOVT. OF INDIA", "Permanent Account Number Card", "ABCPS1234K", "Name",
                 "RAHUL SHARMA", "Father's Name", "SURESH SHARMA", "Date of Birth", "15/08/1988"])
TICKET = "\n".join(["BookMyShow", "Event: Coldplay Music of the Spheres", "Venue: DY Patil Stadium",
                    "Date: 19 Jan 2025", "Show Time: 7:30 PM", "Seat: K-23", "Booking ID: TKT8X2K91", "Admit One"])


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    monkeypatch.setenv("PCC_NO_MODELS", "1")


def _session(vault: bool):
    from app.auth.sessions import SessionInfo
    from app.util import now_iso

    return SessionInfo(id=1, created_at="", last_seen_at="", expires_at="",
                       vault_until=now_iso(600) if vault else None, user_agent=None, ip=None)


@pytest.fixture()
def api(data_dir):
    from app.api import documents
    from app.api.deps import require_user

    app = FastAPI()
    app.include_router(documents.router)
    state = {"vault": False}
    app.dependency_overrides[require_user] = lambda: _session(state["vault"])
    with TestClient(app) as c:
        c.headers.update(H)
        c.state = state
        yield c


def _doc_item(conn, text: str, filename: str, **kw):
    item = make_item(conn, kind="document", data=text.encode(), ext=".txt", mime="text/plain", filename=filename,
                     page_count=1, text_status="pending", **kw)
    extract.extract_item(conn, item["id"], 1)
    return item


def test_requires_login(data_dir):
    from app.api import documents

    app = FastAPI()
    app.include_router(documents.router)
    with TestClient(app) as c:
        try:
            r = c.get("/api/documents")
        except Exception as e:  # auth area missing in this run
            pytest.skip(f"auth sessions not available: {e}")
        assert r.status_code == 401


def test_list_filters_search_and_cursor(api, conn):
    inv = _doc_item(conn, INVOICE, "acme.txt")
    pan = _doc_item(conn, PAN, "pan.txt")
    tick = _doc_item(conn, TICKET, "ticket.txt")
    make_item(conn, kind="image")  # not a document
    r = api.get("/api/documents")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 3
    assert [x["item"]["uid"] for x in body["items"]] == [tick["uid"], pan["uid"], inv["uid"]]
    assert body["items"][1]["document"]["doc_type"] == "pan"
    assert body["items"][1]["item"]["doc_type"] == "pan"
    assert body["items"][1]["document"]["label"] == "PAN card"

    assert [x["item"]["uid"] for x in api.get("/api/documents", params={"type": "invoice"}).json()["items"]] == [inv["uid"]]
    assert api.get("/api/documents", params={"group": "id"}).json()["total"] == 1
    r = api.get("/api/documents", params={"q": "INV-2024-0042"})
    assert [x["item"]["uid"] for x in r.json()["items"]] == [inv["uid"]]
    assert api.get("/api/documents", params={"q": "coldpl"}).json()["total"] == 1   # prefix match
    assert api.get("/api/documents", params={"q": 'S/O "AND -('}).status_code == 200

    p1 = api.get("/api/documents", params={"limit": 2}).json()
    assert len(p1["items"]) == 2 and p1["next_cursor"]
    p2 = api.get("/api/documents", params={"limit": 2, "cursor": p1["next_cursor"]}).json()
    assert [x["item"]["uid"] for x in p2["items"]] == [inv["uid"]] and p2["next_cursor"] is None
    assert api.get("/api/documents", params={"cursor": "garbage!"}).status_code == 400


def test_nsfw_and_deleted_hidden(api, conn):
    secret = _doc_item(conn, PAN, "secret.txt", nsfw=1)
    gone = _doc_item(conn, INVOICE, "gone.txt")
    conn.execute("UPDATE items SET deleted_at = '2026-01-01T00:00:00.000Z' WHERE id = ?", (gone["id"],))
    assert api.get("/api/documents").json()["total"] == 0
    assert api.get(f"/api/documents/{secret['uid']}").status_code == 403
    assert api.get("/api/documents", params={"vault": 1}).status_code == 403
    api.state["vault"] = True
    r = api.get("/api/documents", params={"vault": 1})
    assert r.status_code == 200 and r.json()["total"] == 1
    assert api.get(f"/api/documents/{secret['uid']}").json()["doc_type"] == "pan"


def test_types(api, conn):
    _doc_item(conn, INVOICE, "a.txt")
    _doc_item(conn, INVOICE.replace("0042", "0043"), "b.txt")
    _doc_item(conn, PAN, "c.txt")
    types = api.get("/api/documents/types").json()
    by = {t["type"]: t for t in types}
    assert by["invoice"]["count"] == 2 and by["pan"]["count"] == 1 and by["passport"]["count"] == 0
    assert by["invoice"]["group"] == "money" and by["invoice"]["label"] == "Invoice"
    assert types[0]["group"] == "id"


def test_get_patch_and_reparse(api, conn):
    item = _doc_item(conn, INVOICE, "acme.txt")
    uid = item["uid"]
    d = api.get(f"/api/documents/{uid}").json()
    assert d["fields"]["invoice_number"]["value"] == "INV-2024-0042"
    assert d["field_order"][0] == "invoice_number" and d["source"] == "auto"
    assert api.get("/api/documents/nope").status_code == 404

    r = api.patch(f"/api/documents/{uid}", json={"fields": {"total": "\u20b92,000", "seller": "", "note": "paid"},
                                                  "title": "Mouse purchase", "doc_date": "2024-03-16"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["fields"]["total"]["value"] == "\u20b92,000" and d["fields"]["total"]["confidence"] == 1.0
    assert "seller" not in d["fields"] and d["fields"]["note"]["value"] == "paid"
    assert d["title"] == "Mouse purchase" and d["doc_date"] == "2024-03-16" and d["source"] == "user"
    assert api.get("/api/documents", params={"q": "mouse purchase"}).json()["total"] == 1

    assert api.patch(f"/api/documents/{uid}", json={"doc_date": "16/03/2024"}).status_code == 422
    assert api.patch(f"/api/documents/{uid}", json={"doc_type": "spaceship"}).status_code == 422
    assert api.patch(f"/api/documents/{uid}", json={"fields": {"Bad Key": "x"}}).status_code == 422

    assert api.post(f"/api/documents/{uid}/reparse").json() == {"ok": True}
    d = api.get(f"/api/documents/{uid}").json()
    assert d["fields"]["total"]["value"] == "\u20b92,000" and d["title"] == "Mouse purchase"

    # changing the type re-parses with the new type and regroups
    d = api.patch(f"/api/documents/{uid}", json={"doc_type": "receipt"}).json()
    assert d["doc_type"] == "receipt" and d["doc_group"] == "money" and d["label"] == "Receipt"
    assert d["fields"]["total"]["value"] == "\u20b92,000"


def test_patch_type_creates_document(api, conn):
    item = make_item(conn, kind="document", data=b"short", ext=".txt", mime="text/plain", page_count=1)
    extract.extract_item(conn, item["id"], 1)
    assert api.get(f"/api/documents/{item['uid']}").status_code == 404
    assert api.post(f"/api/documents/{item['uid']}/reparse").status_code in (200, 409)
    d = api.patch(f"/api/documents/{item['uid']}", json={"doc_type": "certificate", "fields": {"title": "Swimming"}})
    assert d.status_code == 200, d.text
    assert d.json()["doc_type"] == "certificate" and d.json()["fields"]["title"]["value"] == "Swimming"


def test_reparse_without_text(api, conn):
    item = make_item(conn, kind="image")
    assert api.post(f"/api/documents/{item['uid']}/reparse").status_code == 409


def test_csrf_on_writes(data_dir, conn):
    from app.api import documents
    from app.api.deps import check_csrf

    item = _doc_item(conn, INVOICE, "acme.txt")
    app = FastAPI()
    app.include_router(documents.router)
    from app.api.deps import require_user

    def fake(request: Request):
        check_csrf(request)
        return _session(False)

    app.dependency_overrides[require_user] = fake
    with TestClient(app) as c:
        assert c.post(f"/api/documents/{item['uid']}/reparse").status_code == 403
        assert c.post(f"/api/documents/{item['uid']}/reparse", headers=H).status_code == 200


def test_wallet_routes(api, conn):
    inv = _doc_item(conn, INVOICE, "acme.txt")
    tick = _doc_item(conn, TICKET, "ticket.txt")
    assert api.get(f"/api/documents/{inv['uid']}/wallet.pkpass").json()["detail"] == "not_a_pass"
    r = api.get(f"/api/documents/{tick['uid']}/wallet.pkpass")
    assert r.status_code == 409 and r.json()["detail"] == "wallet_not_configured"
    assert api.get(f"/api/documents/{tick['uid']}").json()["wallet_available"] is False

    from tests.test_wallet_pkpass import _cert
    from cryptography.hazmat.primitives import serialization

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
    settings.wallet_pass_type_id, settings.wallet_team_id = "pass.test.pcc", "TEAM123456"
    r = api.get(f"/api/documents/{tick['uid']}/wallet.pkpass")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/vnd.apple.pkpass"
    assert r.headers["content-disposition"].startswith("attachment;") and ".pkpass" in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert "pass.json" in z.namelist() and "signature" in z.namelist()
    # a pass with a booking id but no decoded barcode still exports (QR from the booking id)
    doc = api.get(f"/api/documents/{tick['uid']}").json()
    assert doc["fields"]["booking_id"]["value"] == "TKT8X2K91"
