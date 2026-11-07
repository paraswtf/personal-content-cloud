"""Review round 2 for the documents area: the Documents API query gets the index fold, and FTS5
snippets of the folded columns map back onto the original text."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.documents import extract
from tests.helpers import make_item

H = {"X-Requested-With": "pcc"}
ZWJ = "‍"
OPEN, CLOSE = "\x02", "\x03"
MARATHI = f"पावती दिनांक १५ मार्च, कर{chr(0x94d)}{ZWJ}याचा हिशोब"
HINDI_RECEIPT = "\n".join(["रसीद", "दिनांक: १५/०८/१९८८", "कुल राशि: ₹ ५००"])


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    monkeypatch.setenv("PCC_NO_MODELS", "1")


@pytest.fixture()
def api(data_dir):
    from app.api import documents
    from app.api.deps import require_user
    from app.auth.sessions import SessionInfo

    app = FastAPI()
    app.include_router(documents.router)
    app.dependency_overrides[require_user] = lambda: SessionInfo(
        id=1, created_at="", last_seen_at="", expires_at="", vault_until=None, user_agent=None, ip=None)
    with TestClient(app) as c:
        c.headers.update(H)
        yield c


def _doc_item(conn, text: str, filename: str):
    item = make_item(conn, kind="document", data=text.encode(), ext=".txt", mime="text/plain", filename=filename,
                     page_count=1, text_status="pending")
    extract.extract_item(conn, item["id"], 1)
    return item


def _snippet(conn, item_id: int, match: str, column: int = 1) -> str:
    row = conn.execute(
        f"SELECT snippet(items_fts, {column}, ?, ?, '...', 12) FROM items_fts WHERE items_fts MATCH ? AND rowid = ?",
        (OPEN, CLOSE, match, item_id)).fetchone()
    assert row is not None
    return row[0]


# ------------------------------------------------------------------ finding 1: match_query fold

def test_match_query_folds_devanagari_digits_and_joiners():
    assert extract.match_query("१९८८") == '"1988"*'
    assert extract.match_query(f"कर्{ZWJ}याचा") == '"कर्याचा"*'
    # the fold is the index fold, not dev_fold: nukta letters and danda stay as typed
    assert extract.match_query("ज़मीन।") == '"ज़मीन"*'


def test_documents_api_finds_devanagari_digits_both_ways(api, conn):
    item = _doc_item(conn, HINDI_RECEIPT, "rasid.txt")
    assert conn.execute("SELECT 1 FROM documents WHERE item_id = ?", (item["id"],)).fetchone()
    for q in ("१९८८", "1988", "१५/०८/१९८८"):
        r = api.get("/api/documents", params={"q": q})
        assert r.status_code == 200, q
        assert [x["item"]["uid"] for x in r.json()["items"]] == [item["uid"]], q


# ------------------------------------------------------------------ finding 2: snippets show the original

def test_restore_snippet_marathi_joiner_and_digits(conn):
    item = _doc_item(conn, MARATHI, "pavti.txt")
    raw = _snippet(conn, item["id"], '"मार्च"*')
    assert "15" in raw and ZWJ not in raw  # the index holds the folded text
    out = extract.restore_snippet(raw, MARATHI, OPEN, CLOSE)
    assert out == f"पावती दिनांक १५ {OPEN}मार्च{CLOSE}, कर{chr(0x94d)}{ZWJ}याचा हिशोब"
    raw = _snippet(conn, item["id"], '"कर्याचा"*')
    assert extract.restore_snippet(raw, MARATHI, OPEN, CLOSE) == \
        f"पावती दिनांक १५ मार्च, {OPEN}कर{chr(0x94d)}{ZWJ}याचा{CLOSE} हिशोब"
    raw = _snippet(conn, item["id"], '"15"*')
    assert extract.restore_snippet(raw, MARATHI, OPEN, CLOSE) == \
        f"पावती दिनांक {OPEN}१५{CLOSE} मार्च, कर{chr(0x94d)}{ZWJ}याचा हिशोब"


def test_restore_snippet_fragment_with_ellipses(conn):
    text = " ".join(f"शब्द{i}" for i in range(30)) + " दिनांक १५ मार्च " + " ".join(f"अंत{i}" for i in range(30))
    item = _doc_item(conn, text, "lamba.txt")
    raw = _snippet(conn, item["id"], '"15"*')
    assert raw.startswith("...") and raw.endswith("...")
    out = extract.restore_snippet(raw, text, OPEN, CLOSE)
    assert out.startswith("...") and out.endswith("...")
    assert f"दिनांक {OPEN}१५{CLOSE} मार्च" in out
    assert out.replace(OPEN, "").replace(CLOSE, "").strip(".") in text


def test_restore_snippet_column_that_starts_with_dots():
    original = "...१५ लिखा"
    raw = f"...{OPEN}15{CLOSE} लिखा"
    assert extract.restore_snippet(raw, original, OPEN, CLOSE) == f"...{OPEN}१५{CLOSE} लिखा"


def test_restore_snippet_falls_back_to_raw():
    raw = f"दिनांक {OPEN}15{CLOSE} मार्च"
    assert extract.restore_snippet(raw, "कुछ और पाठ", OPEN, CLOSE) == raw
    assert extract.restore_snippet(raw, "", OPEN, CLOSE) == raw
    assert extract.restore_snippet(None, "x", OPEN, CLOSE) is None
    assert extract.restore_snippet("no markers", "no markers", OPEN, CLOSE) == "no markers"


def test_snippet_originals_text_and_doc_columns(conn):
    item = _doc_item(conn, MARATHI, "pavti.txt")
    other = make_item(conn, kind="image")
    got = extract.snippet_originals(conn, [item["id"], other["id"]])
    text, doc = got[item["id"]]
    assert text == MARATHI
    row = conn.execute("SELECT * FROM documents WHERE item_id = ?", (item["id"],)).fetchone()
    assert doc == extract._doc_text(row)
    assert got[other["id"]] == ("", "")
    # the doc column: its original is _doc_text, so its snippet maps back onto that text
    raw = _snippet(conn, item["id"], '"मार्च"*', column=2)
    assert OPEN in raw
    out = extract.restore_snippet(raw, doc, OPEN, CLOSE)
    assert out.replace(OPEN, "").replace(CLOSE, "").strip(".") in doc
    assert f"{OPEN}मार्च{CLOSE}" in out
