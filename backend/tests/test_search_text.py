"""Full-text search (FTS5) of app.search.service: sanitising, ranking, snippets, filters, paging.
Runs without any model (the SigLIP engine is reported missing)."""
from __future__ import annotations

import pytest

from app.db import tx
from app.search import service
from tests.helpers import make_face, make_person
from tests.test_search_support import add_fts, hit_ids, item, no_engine  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_engine")


# ------------------------------------------------------------------ sanitising
def test_fts_terms_quote_each_piece_and_drop_operators():
    assert service.fts_terms("passport AND number") == ['"passport"', '"number"']
    assert service.fts_terms("pass* OR NOT x NEAR(a b)") == ['"pass"*', '"x"', '"a"', '"b"']
    assert service.fts_terms('say "hello') == ['"say"', '"hello"']
    assert service.fts_terms("1234-5678-9012") == ['"1234-5678-9012"']
    assert service.fts_terms("filename:secret") == ['"filename secret"']
    assert service.fts_terms("*** ( ) : ^ - AND ___") == []
    # lower-case words are ordinary words, not operators
    assert service.fts_terms("not and or") == ['"not"', '"and"', '"or"']


def test_fts_query_forms():
    assert service.fts_query("a b") == '"a" "b"'
    assert service.fts_query("a b", any_term=True) == '"a" OR "b"'
    assert service.fts_query("AND OR") is None
    assert len(service.fts_terms(" ".join(f"w{i}" for i in range(40)))) == service.MAX_TERMS


@pytest.mark.parametrize("q", ["AND", "foo OR", '"unbalanced', "NEAR(", "(a", "col:x", "^x", "-", "a*b*",
                               "'", "x NOT", "{filename}: y", "a\" OR \"b", "%_\\"])
def test_hostile_queries_never_raise(conn, q):
    row = item(conn, filename="report.pdf", kind="pdf")
    add_fts(conn, row["id"], filename="report.pdf", text="quarterly report numbers")
    res = service.search(conn, q, mode="text")
    assert res.total >= 0


# ------------------------------------------------------------------ matching and ranking
def test_text_and_filename_matches(conn):
    a = item(conn, kind="pdf", filename="passport_scan.pdf", category="documents")
    b = item(conn, kind="pdf", filename="bill.pdf", category="documents")
    c = item(conn, filename="beach.jpg")
    add_fts(conn, a["id"], filename="passport_scan.pdf", text="REPUBLIC OF INDIA passport number K1234567")
    add_fts(conn, b["id"], filename="bill.pdf", text="electricity bill for october")
    add_fts(conn, c["id"], filename="beach.jpg", labels="beach sea")

    res = service.search(conn, "K1234567", mode="text")
    assert hit_ids(res) == [a["id"]]
    assert res.hits[0]["source"] == "text"
    assert res.hits[0]["score"] == 1.0

    res = service.search(conn, "bill", mode="text")
    assert hit_ids(res) == [b["id"]]

    res = service.search(conn, "nothing-like-this", mode="text")
    assert res.total == 0 and res.hits == [] and res.next_offset is None


def test_prefix_search(conn):
    a = item(conn, kind="pdf", filename="a.pdf")
    add_fts(conn, a["id"], filename="a.pdf", text="passport renewal form")
    assert hit_ids(service.search(conn, "passp*", mode="text")) == [a["id"]]
    assert hit_ids(service.search(conn, "passp", mode="text")) == []


def test_phrase_from_punctuated_number(conn):
    a = item(conn, kind="pdf")
    b = item(conn, kind="pdf")
    add_fts(conn, a["id"], text="Aadhaar 2345 6789 0123 issued")
    add_fts(conn, b["id"], text="6789 then 2345 and 0123")
    assert hit_ids(service.search(conn, "2345-6789-0123", mode="text")) == [a["id"]]


def test_bm25_ranks_denser_match_first(conn):
    weak = item(conn, kind="pdf")
    strong = item(conn, kind="pdf")
    filler = " ".join(f"word{i}" for i in range(200))
    add_fts(conn, weak["id"], text=f"{filler} invoice {filler}")
    add_fts(conn, strong["id"], text="invoice invoice total invoice")
    res = service.search(conn, "invoice", mode="text")
    assert hit_ids(res) == [strong["id"], weak["id"]]
    assert res.hits[0]["score"] == 1.0 and 0 < res.hits[1]["score"] < 1.0


def test_relaxed_or_query_when_no_item_has_every_term(conn):
    a = item(conn, kind="pdf")
    add_fts(conn, a["id"], text="driving licence of the owner")
    res = service.search(conn, "licence unicorn", mode="text")
    assert hit_ids(res) == [a["id"]]
    assert res.hits[0]["score"] == pytest.approx(service.RELAXED_FACTOR)
    # every term present: the strict query wins and the OR query does not run
    assert service.search(conn, "licence owner", mode="text").hits[0]["score"] == 1.0


def test_items_without_fts_row_match_on_filename(conn):
    fresh = item(conn, kind="pdf", filename="Tax_Return_2025.pdf", text_status="pending")
    indexed = item(conn, kind="pdf", filename="tax_notes.pdf")
    add_fts(conn, indexed["id"], filename="tax_notes.pdf", text="notes about tax")
    res = service.search(conn, "tax", mode="text")
    assert hit_ids(res) == [indexed["id"], fresh["id"]]
    assert res.hits[1]["score"] == service.FILENAME_ONLY_SCORE
    # LIKE wildcards in the query are literal
    assert hit_ids(service.search(conn, "Tax%2025", mode="text")) == []


# ------------------------------------------------------------------ snippets
def test_snippet_highlights_and_escapes(conn):
    a = item(conn, kind="pdf")
    add_fts(conn, a["id"], text="<script>alert(1)</script> the quick brown fox jumps over the lazy dog")
    snip = service.search(conn, "fox", mode="text").hits[0]["snippet"]
    assert "<b>fox</b>" in snip
    assert "<script>" not in snip and "&lt;script&gt;" in snip


def test_snippet_is_trimmed_with_ellipsis(conn):
    a = item(conn, kind="pdf")
    words = " ".join(f"w{i}" for i in range(100))
    add_fts(conn, a["id"], text=f"{words} needle {words}")
    snip = service.search(conn, "needle", mode="text").hits[0]["snippet"]
    assert "<b>needle</b>" in snip and "..." in snip
    assert len(snip.split()) <= service.SNIPPET_TOKENS + 2


def test_snippet_falls_back_to_doc_column_then_none(conn):
    a = item(conn, kind="pdf", filename="scan.pdf")
    b = item(conn, kind="pdf", filename="holiday-plan.pdf")
    add_fts(conn, a["id"], filename="scan.pdf", text="", doc="PAN card ABCDE1234F")
    add_fts(conn, b["id"], filename="holiday-plan.pdf", text="unrelated words")
    assert "<b>ABCDE1234F</b>" in service.search(conn, "ABCDE1234F", mode="text").hits[0]["snippet"]
    assert service.search(conn, "holiday", mode="text").hits[0]["snippet"] is None


# ------------------------------------------------------------------ filters
def test_category_kind_and_date_filters(conn):
    a = item(conn, kind="pdf", category="documents", taken_at="2026-03-10T08:00:00.000Z")
    # dates are calendar days in settings.local_timezone (Asia/Kolkata): 10:30Z is 16:00 local on the 12th
    b = item(conn, kind="image", category="personal", taken_at="2026-03-12T10:30:00.000Z")
    c = item(conn, kind="video", category="personal", created_at="2026-04-01T10:00:00.000Z")
    for r in (a, b, c):
        add_fts(conn, r["id"], text="birthday party")

    def ids(**kw):
        return sorted(hit_ids(service.search(conn, "birthday", mode="text", **kw)))

    assert ids() == sorted([a["id"], b["id"], c["id"]])
    assert ids(category="personal") == sorted([b["id"], c["id"]])
    assert ids(kind="pdf") == [a["id"]]
    assert ids(category="personal", kind="video") == [c["id"]]
    # date range is inclusive on both ends; items without taken_at use created_at
    assert ids(date_from="2026-03-12", date_to="2026-03-12") == [b["id"]]
    assert ids(date_from="2026-03-11") == sorted([b["id"], c["id"]])
    assert ids(date_to="2026-03-10") == [a["id"]]
    assert ids(date_from="2026-04-01", date_to="2026-04-01") == [c["id"]]


def test_person_filter_needs_every_person(conn):
    alice, bob = make_person(conn, "Alice"), make_person(conn, "Bob")
    both = item(conn)
    only_alice = item(conn)
    tagged_bob = item(conn)
    make_face(conn, both["id"], alice, write_crop=False)
    make_face(conn, both["id"], bob, write_crop=False)
    make_face(conn, only_alice["id"], alice, write_crop=False)
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', '2026')",
                     (tagged_bob["id"], bob))
    for r in (both, only_alice, tagged_bob):
        add_fts(conn, r["id"], text="trip to goa")

    def ids(people):
        return sorted(hit_ids(service.search(conn, "goa", mode="text", person_ids=people)))

    assert ids([alice]) == sorted([both["id"], only_alice["id"]])
    assert ids([bob]) == sorted([both["id"], tagged_bob["id"]])
    assert ids([alice, bob]) == [both["id"]]


def test_vault_trash_and_duplicate_copies(conn):
    plain = item(conn)
    private = item(conn, nsfw=1, category="nsfw")
    trashed = item(conn, deleted_at="2026-01-01T00:00:00.000Z")
    blurry = item(conn, misclick=1)
    best = item(conn)
    copy = item(conn, dup_group_id=best["id"], dup_role="copy")
    with tx(conn):
        conn.execute("UPDATE items SET dup_group_id = id, dup_role = 'best' WHERE id = ?", (best["id"],))
    for r in (plain, private, trashed, blurry, best, copy):
        add_fts(conn, r["id"], text="sunset photo")

    normal = sorted(hit_ids(service.search(conn, "sunset", mode="text")))
    assert normal == sorted([plain["id"], blurry["id"], best["id"]])
    assert hit_ids(service.search(conn, "sunset", mode="text", vault=True)) == [private["id"]]


# ------------------------------------------------------------------ paging and status
def test_offset_paging(conn):
    rows = [item(conn, kind="pdf") for _ in range(5)]
    for r in rows:
        add_fts(conn, r["id"], text="receipt")
    first = service.search(conn, "receipt", mode="text", limit=2)
    second = service.search(conn, "receipt", mode="text", limit=2, offset=2)
    third = service.search(conn, "receipt", mode="text", limit=2, offset=4)
    assert first.total == second.total == third.total == 5
    assert (first.next_offset, second.next_offset, third.next_offset) == (2, 4, None)
    seen = hit_ids(first) + hit_ids(second) + hit_ids(third)
    assert sorted(seen) == sorted(r["id"] for r in rows)
    assert service.search(conn, "receipt", mode="text", offset=10).hits == []


def test_indexing_flag_and_indexed_without_engine(conn):
    a = item(conn)
    add_fts(conn, a["id"], text="hello")
    res = service.search(conn, "hello")
    assert res.indexing is False and res.indexed == 0 and res.semantic_used is False
    item(conn, analysis_status="pending")
    assert service.search(conn, "hello").indexing is True


def test_auto_mode_without_engine_is_text_only_and_semantic_mode_fails(conn):
    a = item(conn, kind="pdf")
    add_fts(conn, a["id"], text="vaccination certificate")
    res = service.search(conn, "vaccination", mode="auto")
    assert hit_ids(res) == [a["id"]] and res.hits[0]["source"] == "text"
    with pytest.raises(service.SearchUnavailable):
        service.search(conn, "vaccination", mode="semantic")


def test_search_disabled_skips_semantic_without_importing_ml(conn, monkeypatch):
    from app.config import settings

    def boom():
        raise AssertionError("ml module must not be touched when search is disabled")

    monkeypatch.setattr(settings, "search_enabled", False)
    monkeypatch.setattr(service, "_siglip_module", boom)
    a = item(conn, kind="pdf")
    add_fts(conn, a["id"], text="policy number")
    assert hit_ids(service.search(conn, "policy")) == [a["id"]]


def test_unknown_mode_is_rejected(conn):
    with pytest.raises(ValueError):
        service.search(conn, "x", mode="fuzzy")
