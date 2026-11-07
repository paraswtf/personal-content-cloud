"""Regression tests for the document parsing limits (review findings C7 to C12 and C22)."""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import pytest

from tests.helpers import pdf_bytes


def _docx(path: Path, paragraphs: list[str]) -> Path:
    import docx

    d = docx.Document()
    for text in paragraphs:
        d.add_paragraph(text)
    d.save(path)
    return path


# ------------------------------------------------------------------ C12: date filter edges
def test_local_day_start_utc_clamps_calendar_edges():
    from app.util import local_day_start_utc

    # A normal day in Asia/Kolkata (UTC+05:30) is unchanged by the clamp.
    assert local_day_start_utc("2024-03-10", "Asia/Kolkata") == "2024-03-09T18:30:00.000Z"
    assert local_day_start_utc("2024-03-10", "Asia/Kolkata", next_day=True) == "2024-03-10T18:30:00.000Z"
    # Both edges used to raise OverflowError (HTTP 500 from the search and library date filters).
    end = local_day_start_utc("9999-12-31", "Asia/Kolkata", next_day=True)
    start = local_day_start_utc("0001-01-01", "Asia/Kolkata")
    assert end.startswith("9999-") or end.startswith("9998-")
    assert start.startswith("0001-") or start.startswith("0002-")
    assert start < "1900" < "2100" < end
    assert local_day_start_utc("0001-01-01", "Pacific/Kiritimati")  # UTC+14
    assert local_day_start_utc("9999-12-31", "Pacific/Pago_Pago", next_day=True)  # UTC-11
    with pytest.raises(ValueError):
        local_day_start_utc("not a date", "Asia/Kolkata")


# ------------------------------------------------------------------ C7: xlsx declared dimension
def test_xlsx_huge_declared_dimension_is_fast(tmp_path):
    import openpyxl

    from app.media import documents

    wb = openpyxl.Workbook()
    wb.active["A1"] = "first"
    wb.active["XFD1048576"] = "last"  # openpyxl writes <dimension ref="A1:XFD1048576"/>
    path = tmp_path / "dim.xlsx"
    wb.save(path)
    with zipfile.ZipFile(path) as zf:
        assert b'ref="A1:XFD1048576"' in zf.read("xl/worksheets/sheet1.xml")
    t = time.monotonic()
    pages = documents.logical_pages(path)
    assert time.monotonic() - t < 5.0  # was unbounded (still running after 20 s)
    assert pages[0].startswith("Sheet: Sheet\nfirst")


def test_xlsx_row_and_cell_caps(tmp_path, monkeypatch):
    import openpyxl

    from app.media import documents

    monkeypatch.setattr(documents, "XLSX_MAX_ROWS", 50)
    monkeypatch.setattr(documents, "XLSX_MAX_COLS", 3)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["a", "b", "c", "d", "e"])
    ws["A200"] = "past the row cap"
    path = tmp_path / "caps.xlsx"
    wb.save(path)
    assert documents.logical_pages(path) == ["Sheet: Sheet\na\tb\tc\n..."]


def test_xlsx_empty_rows_use_the_workbook_scan_budget(tmp_path, monkeypatch):
    import openpyxl

    from app.media import documents

    monkeypatch.setattr(documents, "XLSX_MAX_CELLS", 1000)
    wb = openpyxl.Workbook()
    for i in range(3):
        ws = wb.active if i == 0 else wb.create_sheet(f"S{i}")
        ws["A1"] = f"top {i}"
        ws["A5000"] = f"bottom {i}"  # 4998 empty rows in between, no cells in them
    path = tmp_path / "gaps.xlsx"
    wb.save(path)
    pages = documents.logical_pages(path)
    assert len(pages) == 3  # still one page per sheet
    assert pages[0] == "Sheet: Sheet\ntop 0\n..."
    assert pages[1:] == ["Sheet: S1\n...", "Sheet: S2\n..."]


# ------------------------------------------------------------------ C10: many-entry zips
def _zip_with_entries(path: Path, n: int, top: str = "word") -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        for i in range(n):
            zf.writestr(f"{top}/e{i}.bin", b"")
    return path


def test_many_entry_zip_is_refused_before_the_directory_is_read(tmp_path, monkeypatch):
    from app.media import documents
    from app.media.probe import probe

    monkeypatch.setattr(documents, "ZIP_MAX_ENTRIES", 1000)
    path = _zip_with_entries(tmp_path / "many.docx", 3000)
    data = path.read_bytes()[:4096]

    def no_directory(*a, **k):
        raise AssertionError("zipfile.ZipFile was opened for a zip over the entry limit")

    monkeypatch.setattr(documents.zipfile, "ZipFile", no_directory)
    assert documents.sniff_document(data, path, "many.docx") is None
    monkeypatch.undo()

    monkeypatch.setattr(documents, "ZIP_MAX_ENTRIES", 1000)
    info = probe(path, "many.docx")
    assert info.kind == "other"
    with pytest.raises(ValueError):
        documents.logical_pages(path)
    # Under the limit the same layout still sniffs as DOCX.
    small = _zip_with_entries(tmp_path / "small.docx", 10)
    assert documents.sniff_document(small.read_bytes()[:4096], small, "small.docx") == documents.DOCX


def test_zip_directory_size_limit_bounds_a_lying_entry_count(tmp_path, monkeypatch):
    from app.media import documents

    path = _zip_with_entries(tmp_path / "lie.docx", 400)
    raw = bytearray(path.read_bytes())
    eocd = raw.rfind(b"PK\x05\x06")
    raw[eocd + 8:eocd + 12] = (3).to_bytes(2, "little") * 2  # claim 3 entries; zipfile reads by size
    path.write_bytes(bytes(raw))
    monkeypatch.setattr(documents, "ZIP_MAX_ENTRIES", 100)
    monkeypatch.setattr(documents, "ZIP_MAX_DIR_BYTES", 4096)
    with pytest.raises(documents.ZipRefused):
        documents._office_zip_names(path)


# ------------------------------------------------------------------ C8: office zip bombs
def _bomb_docx(path: Path, mb: int) -> Path:
    """A real docx whose word/document.xml expands to about `mb` MB of one repeated paragraph."""
    _docx(path, ["seed"])
    with zipfile.ZipFile(path) as zf:
        parts = {i.filename: zf.read(i.filename) for i in zf.infolist()}
    para = b"<w:p><w:r><w:t>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</w:t></w:r></w:p>"
    body = para * (mb * 1024 * 1024 // len(para))
    xml = parts["word/document.xml"]
    parts["word/document.xml"] = xml.replace(b"<w:body>", b"<w:body>" + body, 1)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in parts.items():
            zf.writestr(name, blob)
    return path


def test_docx_bomb_is_kept_as_other(tmp_path, monkeypatch):
    from app.media import documents
    from app.media.probe import probe

    path = _bomb_docx(tmp_path / "bomb.docx", 3)
    assert path.stat().st_size < 200_000
    monkeypatch.setattr(documents, "ZIP_MAX_TOTAL_BYTES", 1024 * 1024)
    assert documents.sniff_document(path.read_bytes()[:4096], path, "bomb.docx") is None
    info = probe(path, "bomb.docx")
    assert info.kind == "other"
    with pytest.raises(ValueError):
        documents.logical_pages(path)  # stored items and reprocess jobs are refused as well


def test_docx_part_size_and_ratio_limits(tmp_path, monkeypatch):
    from app.media import documents

    path = _bomb_docx(tmp_path / "ratio.docx", 3)
    with pytest.raises(documents.ZipRefused, match="expands"):
        documents._office_zip_names(path)  # default limits: about 3 MB from a few KB is over 200x
    monkeypatch.setattr(documents, "ZIP_MAX_RATIO", 10**6)
    monkeypatch.setattr(documents, "ZIP_MAX_PART_BYTES", 1024 * 1024)
    with pytest.raises(documents.ZipRefused, match="word/document.xml"):
        documents._office_zip_names(path)
    monkeypatch.setattr(documents, "ZIP_MAX_PART_BYTES", 64 * 1024 * 1024)
    assert "word/document.xml" in documents._office_zip_names(path)


def test_docx_blocks_stop_after_the_page_cap(tmp_path, monkeypatch):
    from app.media import documents

    monkeypatch.setattr(documents, "MAX_LOGICAL_PAGES", 2)
    path = _docx(tmp_path / "many.docx", [""] * 400)  # empty paragraphs: no characters, one line each
    blocks = documents._docx_blocks(path)
    assert len(blocks) == 2 * documents.LINES_PER_PAGE
    path = _docx(tmp_path / "long.docx", ["word " * 2000] * 40)
    blocks = documents._docx_blocks(path)
    assert len(blocks) < 40
    assert len(documents.logical_pages(path)) == 2
    # One paragraph far larger than the cap: wrapping stops early and no page holds the whole block.
    path = _docx(tmp_path / "giant.docx", ["word " * 50_000])
    pages = documents.logical_pages(path)
    assert len(pages) == 2
    assert all(len(documents._wrap(p)) <= documents.LINES_PER_PAGE for p in pages)


# ------------------------------------------------------------------ C9: PDF page cap
class _CountingLock:
    def __init__(self, inner):
        self.inner = inner
        self.entered = 0

    def __enter__(self):
        self.inner.acquire()
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.inner.release()
        return False

    def acquire(self, *a, **k):
        return self.inner.acquire(*a, **k)

    def release(self):
        self.inner.release()


def test_pdf_page_count_is_capped_and_text_locks_per_page(tmp_path, monkeypatch):
    from app.media import documents

    monkeypatch.setattr(documents, "PDF_MAX_PAGES", 4)
    path = tmp_path / "many.pdf"
    path.write_bytes(pdf_bytes([[f"page {i}"] for i in range(1, 11)]))
    count, w, h = documents.probe_document(path, "pdf")
    assert count == 4 and w > 0 and h > 0
    lock = _CountingLock(documents.PDFIUM_LOCK)
    monkeypatch.setattr(documents, "PDFIUM_LOCK", lock)
    layer = documents.text_layer(path, "pdf")
    assert layer == ["page 1", "page 2", "page 3", "page 4"]
    assert lock.entered >= 4  # one hold per page (plus open and close), not one for the document
    assert documents.page_image(path, "pdf", 5, max_side=200) is None
    assert documents.page_image(path, "pdf", 4, max_side=200) is not None


# ------------------------------------------------------------------ C11: long lines
def test_wrap_of_a_long_line_is_linear():
    from app.media import documents

    text = "x" * 4_000_000  # took about 3 s with textwrap's long-word path
    t = time.monotonic()
    lines = documents._wrap(text)
    assert time.monotonic() - t < 1.0
    assert all(len(line) <= documents.CHARS_PER_LINE for line in lines)
    assert "".join(lines) == text

    mixed = ("y" * 20_000 + " ") * 200
    t = time.monotonic()
    lines = documents._wrap(mixed)
    assert time.monotonic() - t < 1.0
    assert all(len(line) <= documents.CHARS_PER_LINE for line in lines)
    assert "".join(lines) == mixed.replace(" ", "")


def test_wrap_stops_at_max_lines():
    from app.media import documents

    text = "word " * 100_000
    full = documents._wrap(text)
    capped = documents._wrap(text, 120)
    assert capped == full[:120]
    assert documents._wrap("a\nb\nc\nd", 2) == ["a", "b"]


# ------------------------------------------------------------------ C22: plain text past the page cap
def test_plain_text_past_the_page_cap_reaches_the_text_layer(tmp_path, monkeypatch):
    from app.media import documents

    monkeypatch.setattr(documents, "MAX_LOGICAL_PAGES", 5)
    lines = [f"log line {i:07d}" for i in range(1000)]
    path = tmp_path / "big.log.txt"
    path.write_text("\n".join(lines))
    pages = documents.logical_pages(path)
    assert len(pages) == 5
    assert documents.probe_document(path, "document")[0] == 5
    layer = documents.text_layer(path, "document")
    assert len(layer) == 5
    assert layer[:4] == pages[:4]
    assert layer[-1].startswith(pages[-1])
    assert "log line 0000999" in layer[-1]
    assert "\n".join(layer).splitlines() == lines


def test_plain_text_single_giant_line_is_kept_whole(tmp_path, monkeypatch):
    from app.media import documents

    monkeypatch.setattr(documents, "MAX_LOGICAL_PAGES", 3)
    words = [f"w{i}" for i in range(200_000)]
    path = tmp_path / "giant.txt"
    path.write_text(" ".join(words))
    assert len(documents.logical_pages(path)) == 3
    layer = documents.text_layer(path, "document")
    assert len(layer) == 3
    assert " ".join(layer).split() == words

    path = tmp_path / "nospace.txt"
    path.write_text("q" * 300_000 + "END")
    for page in documents.logical_pages(path):
        assert len(documents._wrap(page)) <= documents.LINES_PER_PAGE  # renders show the whole page
    layer = documents.text_layer(path, "document")
    assert len(layer) == 3
    assert "".join(layer).replace("\n", "") == "q" * 300_000 + "END"
