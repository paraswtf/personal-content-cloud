"""Hindi and Marathi (Devanagari) full-text search: the items_fts tokenizer, its migration on an
existing database, query cleaning, snippets and the semantic skip for non-Latin queries.
Items are indexed through app.documents.extract.refresh_fts into the real test database.
Runs without any model (the SigLIP engine is reported missing or faked)."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from app import db
from app.db import FTS_REBUILD_KEY, FTS_TOKENCHARS, FTS_TOKENIZE, kv_get, tx
from app.search import service
from app.util import now_iso
from tests.test_search_support import fake_engine, hit_ids, item, no_engine  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_engine")

OLD_DEFINITION = (
    "CREATE VIRTUAL TABLE items_fts USING fts5(filename, text, doc, people, labels, "
    "tokenize = 'unicode61 remove_diacritics 2')"
)
NUKTA_DECOMPOSED = "बड़ा"  # "big", with da + nukta
NUKTA_PRECOMPOSED = "बड़ा"  # the same word with U+095C


def indexed(conn, text: str, *, filename: str = "scan.pdf", deleted: bool = False) -> int:
    """An item whose extracted text is `text`, indexed with refresh_fts. Returns its id."""
    from app.documents.extract import refresh_fts

    row = item(conn, kind="pdf", filename=filename, ext=".pdf", mime="application/pdf",
               deleted_at=now_iso() if deleted else None)
    with tx(conn):
        conn.execute(
            "INSERT INTO item_text(item_id, text, source, lang, char_count, created_at) VALUES (?, ?, 'ocr', NULL, ?, ?)",
            (row["id"], text, len(text), now_iso()),
        )
    refresh_fts(conn, int(row["id"]))
    return int(row["id"])


def text_terms(conn, item_id: int) -> list[str]:
    """Tokens of the text column of one row, in document order (fts5vocab 'instance')."""
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.fts_inst USING fts5vocab(main, items_fts, instance)")
    rows = conn.execute(
        "SELECT term FROM temp.fts_inst WHERE doc = ? AND col = 'text' ORDER BY offset", (item_id,)
    ).fetchall()
    return [r[0] for r in rows]


def fts_sql(conn) -> str:
    return conn.execute("SELECT sql FROM sqlite_master WHERE name = 'items_fts'").fetchone()[0]


# ------------------------------------------------------------------ tokenizer definition
def test_schema_sql_carries_the_tokenizer_literal():
    schema = (Path(db.__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    assert FTS_TOKENIZE in schema
    assert FTS_TOKENIZE.startswith("tokenize = 'unicode61 remove_diacritics 2 tokenchars ''")
    assert "'" not in FTS_TOKENCHARS and '"' not in FTS_TOKENCHARS
    for ch in FTS_TOKENCHARS:
        assert 0x0900 <= ord(ch) <= 0x0DFF and unicodedata.category(ch) in ("Mn", "Mc"), hex(ord(ch))
    devanagari_marks = {chr(c) for c in range(0x0900, 0x0980) if unicodedata.category(chr(c)) in ("Mn", "Mc")}
    assert devanagari_marks <= set(FTS_TOKENCHARS)
    # nukta, virama, candrabindu, anusvara, visarga
    assert set("़्ँंः") <= set(FTS_TOKENCHARS)
    # danda stays a separator
    assert "।" not in FTS_TOKENCHARS


def test_fresh_database_takes_the_new_tokenizer(conn):
    assert FTS_TOKENIZE in fts_sql(conn)
    assert kv_get(conn, FTS_REBUILD_KEY) is None


def test_a_marathi_sentence_indexes_as_whole_words(conn):
    a = indexed(conn, "माझे नाव प्रिया नायर आहे")
    assert text_terms(conn, a) == ["माझे", "नाव", "प्रिया", "नायर", "आहे"]
    b = indexed(conn, "मुंबईत राहुल कुमार शर्मा")
    assert text_terms(conn, b) == ["मुंबईत", "राहुल", "कुमार", "शर्मा"]


# ------------------------------------------------------------------ query cleaning
def test_fts_terms_keep_devanagari_words_whole():
    assert service.fts_terms("राहुल") == ['"राहुल"']
    assert service.fts_terms("मुंबई*") == ['"मुंबई"*']
    assert service.fts_terms("प्रिया नायर") == ['"प्रिया"', '"नायर"']
    # a danda is punctuation; a word of marks alone is not a word
    assert service.fts_terms("राहुल। ।") == ['"राहुल"']
    assert service.fts_terms("ाि") == []
    # zero-width joiner and non-joiner are removed, not turned into a word break
    assert service.fts_terms("माझे‍ नाव‌") == ['"माझे"', '"नाव"']
    # Devanagari digits are folded to ASCII on both sides (the index stores ASCII digits).
    assert service.fts_terms("१५/०८/१९८८") == ['"15/08/1988"']
    # Latin pieces are unchanged
    assert service.fts_terms("passport AND K1234567") == ['"passport"', '"K1234567"']


def test_nukta_letters_are_searched_in_both_spellings(conn):
    a = indexed(conn, f"{NUKTA_DECOMPOSED} घर")
    b = indexed(conn, f"{NUKTA_PRECOMPOSED} मकान")
    terms = service.fts_terms(NUKTA_PRECOMPOSED)
    assert terms == [f'("{NUKTA_PRECOMPOSED}" OR "{NUKTA_DECOMPOSED}")']
    for q in (NUKTA_DECOMPOSED, NUKTA_PRECOMPOSED):
        assert sorted(hit_ids(service.search(conn, q, mode="text"))) == sorted([a, b])
    # an OR group still combines with other terms (explicit AND)
    assert hit_ids(service.search(conn, f"घर {NUKTA_PRECOMPOSED}", mode="text")) == [a]
    assert hit_ids(service.search(conn, f"{NUKTA_PRECOMPOSED[:-1]}*", mode="text")) != []


# ------------------------------------------------------------------ matching
def test_whole_word_and_prefix_search(conn):
    hindi = indexed(conn, "भारत सरकार राहुल कुमार शर्मा मुंबईत रहते हैं", filename="aadhaar.jpg")
    marathi = indexed(conn, "माझे नाव प्रिया नायर आहे", filename="patra.pdf")
    other = indexed(conn, "Government of India Permanent Account Number ABCPS1234K", filename="pan.pdf")

    assert hit_ids(service.search(conn, "राहुल", mode="text")) == [hindi]
    assert hit_ids(service.search(conn, "प्रिया नायर", mode="text")) == [marathi]
    assert hit_ids(service.search(conn, "मुंबई*", mode="text")) == [hindi]
    assert hit_ids(service.search(conn, "मुंबई", mode="text")) == []  # whole word only without *
    # a single consonant no longer matches every word that starts with it
    assert hit_ids(service.search(conn, "म", mode="text")) == []
    assert hit_ids(service.search(conn, "ABCPS1234K", mode="text")) == [other]


def test_devanagari_digits(conn):
    a = indexed(conn, "जन्म तिथि: १५/०८/१९८८ पुरुष / MALE")
    b = indexed(conn, "एकूण रक्कम रु. २५,००० फक्त")
    assert hit_ids(service.search(conn, "१९८८", mode="text")) == [a]
    assert hit_ids(service.search(conn, "१५/०८/१९८८", mode="text")) == [a]
    assert hit_ids(service.search(conn, "२५,०००", mode="text")) == [b]


def test_mixed_hindi_english_query(conn):
    a = indexed(conn, "किराया अनुबंध - Rent Agreement मकान मालिक राहुल")
    b = indexed(conn, "Rent receipt for October")
    assert hit_ids(service.search(conn, "किराया agreement", mode="text")) == [a]
    # no item has both words: the relaxed OR query finds each item with either
    assert sorted(hit_ids(service.search(conn, "किराया receipt", mode="text"))) == sorted([a, b])


def test_hindi_snippet_wraps_the_whole_word(conn):
    indexed(conn, "भारत सरकार राहुल कुमार शर्मा जन्म तिथि")
    res = service.search(conn, "कुमार", mode="text")
    assert "<b>कुमार</b>" in res.hits[0]["snippet"]
    res = service.search(conn, "शर्*", mode="text")
    assert "<b>शर्मा</b>" in res.hits[0]["snippet"]


def test_snippet_shows_the_original_digits_and_joiners(conn):
    # The index stores ASCII digits and no joiners; the snippet shown maps back onto the original text.
    original = "पावती दिनांक १५ मार्च, कर्\u200dयाचा हिशोब"
    indexed(conn, original)
    res = service.search(conn, "मार्च", mode="text")
    snippet = res.hits[0]["snippet"]
    assert "<b>मार्च</b>" in snippet and "१५" in snippet and "कर्\u200dयाचा" in snippet
    res = service.search(conn, "15", mode="text")  # ASCII digits find the Devanagari ones
    assert "१५" in res.hits[0]["snippet"]


def test_latin_diacritics_still_fold(conn):
    a = indexed(conn, "Café receipt naïve")
    assert hit_ids(service.search(conn, "cafe", mode="text")) == [a]
    assert hit_ids(service.search(conn, "naive", mode="text")) == [a]


# ------------------------------------------------------------------ semantic engine
def test_non_latin_query_skips_the_semantic_pass(conn, fake_engine):
    a = indexed(conn, "भारत सरकार राहुल कुमार शर्मा")
    res = service.search(conn, "राहुल कुमार")
    assert hit_ids(res) == [a]
    assert res.hits[0]["source"] == "text"
    assert res.semantic_used is False
    assert fake_engine.texts == []
    # the semantic mode still asks the engine when chosen explicitly
    service.search(conn, "राहुल", mode="semantic")
    assert fake_engine.texts


def test_mixed_and_number_queries_still_use_the_engine(conn, fake_engine):
    indexed(conn, "राहुल passport 2024")
    service.search(conn, "राहुल passport")
    assert len(fake_engine.texts) >= 1
    before = len(fake_engine.texts)
    service.search(conn, "2024")
    assert len(fake_engine.texts) > before


def test_text_only_query_detection():
    assert service.text_only_query("राहुल कुमार")
    assert service.text_only_query("माझे नाव १९८८")
    assert not service.text_only_query("राहुल passport")
    assert not service.text_only_query("2024")
    assert not service.text_only_query("1234-5678")
    assert not service.text_only_query("café")
    assert not service.text_only_query("")
    assert service.text_only_query("राहुल × २")  # the multiplication sign is not a Latin letter


# ------------------------------------------------------------------ migration
def test_old_tokenizer_is_replaced_and_rebuilt_on_startup(conn):
    from app import ingest

    with tx(conn):
        conn.execute("DROP TABLE items_fts")
        conn.execute(OLD_DEFINITION)
    hindi = indexed(conn, "मुंबईत राहुल कुमार शर्मा", filename="kirayanama.pdf")
    trashed = indexed(conn, "माझे नाव प्रिया नायर आहे", deleted=True)
    english = indexed(conn, "electricity bill for october", filename="bill.pdf")
    assert text_terms(conn, hindi)[:2] == ["म", "बईत"]  # the old tokenizer splits at vowel signs

    db.init_db()
    assert FTS_TOKENIZE in fts_sql(conn)
    assert kv_get(conn, FTS_REBUILD_KEY) == "1"
    assert conn.execute("SELECT COUNT(*) FROM items_fts").fetchone()[0] == 0

    db.init_db()  # a second start before recovery keeps the flag and the new table
    assert kv_get(conn, FTS_REBUILD_KEY) == "1"

    ingest.recover_on_startup(conn)
    assert kv_get(conn, FTS_REBUILD_KEY) is None
    rows = {r[0] for r in conn.execute("SELECT rowid FROM items_fts")}
    assert rows == {hindi, trashed, english}
    assert text_terms(conn, hindi) == ["मुंबईत", "राहुल", "कुमार", "शर्मा"]
    assert hit_ids(service.search(conn, "मुंबई*", mode="text")) == [hindi]
    assert hit_ids(service.search(conn, "bill", mode="text")) == [english]
    assert hit_ids(service.search(conn, "kirayanama", mode="text")) == [hindi]

    db.init_db()  # nothing to do any more
    assert kv_get(conn, FTS_REBUILD_KEY) is None
    assert conn.execute("SELECT COUNT(*) FROM items_fts").fetchone()[0] == 3


def test_rebuild_is_a_no_op_without_the_flag(conn, monkeypatch):
    from app import ingest
    from app.documents import extract

    indexed(conn, "राहुल")
    calls: list[int] = []
    monkeypatch.setattr(extract, "refresh_fts", lambda c, i: calls.append(i))
    ingest.recover_on_startup(conn)
    assert calls == []


def test_rebuild_survives_a_failing_item(conn, monkeypatch):
    from app import ingest
    from app.documents import extract

    a = indexed(conn, "राहुल कुमार")
    b = indexed(conn, "प्रिया नायर")
    real = extract.refresh_fts

    def flaky(c, item_id):
        if item_id == a:
            raise ValueError("broken row")
        real(c, item_id)

    with tx(conn):
        conn.execute("DELETE FROM items_fts")
        db.kv_set(conn, FTS_REBUILD_KEY, 1)
    monkeypatch.setattr(extract, "refresh_fts", flaky)
    ingest.recover_on_startup(conn)
    assert kv_get(conn, FTS_REBUILD_KEY) is None
    assert {r[0] for r in conn.execute("SELECT rowid FROM items_fts")} == {b}
