"""Core media modules: probing every kind, document renders and text layers, RAW sniffing, dedup groups."""
from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest
from PIL import Image

from tests.helpers import flat_image, jpeg_bytes, make_item, make_video, noise_image, pdf_bytes, png_bytes


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _docx(path: Path, paragraphs: list[str], table: list[list[str]] | None = None) -> Path:
    import docx

    d = docx.Document()
    for text in paragraphs:
        d.add_paragraph(text)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, val in enumerate(row):
                t.cell(r, c).text = val
    d.save(path)
    return path


# ------------------------------------------------------------------ probe
def test_probe_jpeg_and_png(data_dir, tmp_path):
    from app.media.probe import probe

    j = probe(_write(tmp_path, "a.part", jpeg_bytes(noise_image(640, 480))), "photo.JPG")
    assert (j.kind, j.mime, j.ext, j.width, j.height) == ("image", "image/jpeg", ".jpg", 640, 480)
    assert j.visual and not j.text_capable
    p = probe(_write(tmp_path, "b.part", png_bytes(noise_image(100, 80))), None)
    assert (p.kind, p.ext, p.width, p.height) == ("image", ".png", 100, 80)


def test_probe_pdf(data_dir, tmp_path):
    from app.config import settings
    from app.media.probe import probe

    info = probe(_write(tmp_path, "c.part", pdf_bytes([["Hello"], ["World"], ["Three"]])), "scan.pdf")
    assert (info.kind, info.mime, info.ext, info.page_count) == ("pdf", "application/pdf", ".pdf", 3)
    assert max(info.width, info.height) == settings.page_render_size
    assert info.text_capable


def test_probe_office_and_text(data_dir, tmp_path):
    from app.media.probe import probe

    d = probe(_docx(tmp_path / "d.part", ["Hello docx"]), "letter.docx")
    assert (d.kind, d.ext, d.page_count) == ("document", ".docx", 1)
    t = probe(_write(tmp_path, "e.part", "first line\nsecond line\n".encode()), "notes.txt")
    assert (t.kind, t.mime, t.ext) == ("document", "text/plain", ".txt")
    md = probe(_write(tmp_path, "f.part", b"# Title\n\nbody"), "README.markdown")
    assert (md.kind, md.ext) == ("document", ".md")
    csv = probe(_write(tmp_path, "g.part", b"a,b\n1,2\n"), "data.csv")
    assert (csv.kind, csv.mime) == ("document", "text/csv")

    import openpyxl
    import pptx

    wb = openpyxl.Workbook()
    wb.active.append(["name", 3])
    wb.save(tmp_path / "h.part")
    x = probe(tmp_path / "h.part", "sheet.xlsx")
    assert (x.kind, x.ext, x.page_count) == ("document", ".xlsx", 1)
    pr = pptx.Presentation()
    for title in ("One", "Two"):
        pr.slides.add_slide(pr.slide_layouts[1]).shapes.title.text = title
    pr.save(tmp_path / "i.part")
    s = probe(tmp_path / "i.part", "deck.pptx")
    assert (s.kind, s.ext, s.page_count) == ("document", ".pptx", 2)


def test_probe_text_needs_a_text_extension(data_dir, tmp_path):
    from app.media.probe import probe

    path = _write(tmp_path, "e.part", b"just some words")
    assert probe(path, "notes.txt").kind == "document"
    other = probe(path, None)
    assert (other.kind, other.mime, other.ext) == ("other", "application/octet-stream", ".bin")


def test_probe_video(data_dir, tmp_path):
    from app.media.probe import probe

    path = make_video(tmp_path / "v.mp4", seconds=2)
    info = probe(path, "clip.mp4")
    assert (info.kind, info.ext, info.width, info.height) == ("video", ".mp4", 640, 360)
    assert info.duration_s == pytest.approx(2.0, abs=0.2)
    assert info.video_codec == "h264"


def test_probe_unknown_broken_and_empty(data_dir, tmp_path):
    from app.media.probe import probe

    unknown = probe(_write(tmp_path, "u.part", b"\x01\x02 binary blob " * 20), "Thing.XYZ")
    assert (unknown.kind, unknown.mime, unknown.ext) == ("other", "application/octet-stream", ".xyz")
    weird = probe(_write(tmp_path, "w.part", b"\x01\x02 binary blob " * 20), "evil.<script>")
    assert weird.ext == ".bin"
    broken = probe(_write(tmp_path, "b.part", b"\xff\xd8\xff\xe0 not really a jpeg" * 4), "IMG_1.jpg")
    assert (broken.kind, broken.mime, broken.ext) == ("other", "image/jpeg", ".jpg")
    assert broken.reason
    assert probe(_write(tmp_path, "z.part", b""), "empty.txt") is None


def test_probe_reads_exif(data_dir, tmp_path):
    from app.media.probe import probe

    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "Canon EOS R6"
    exif.get_ifd(0x8769)[0x9003] = "2024:05:01 10:30:00"
    img = noise_image(320, 240)
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    info = probe(_write(tmp_path, "x.part", buf.getvalue()), "x.jpg")
    assert info.camera == "Canon EOS R6"
    assert info.exif_datetime == "2024:05:01 10:30:00"
    assert info.taken_at == "2024-05-01T05:00:00.000Z"  # Asia/Kolkata is UTC+5:30


def _tiff(tags: list[int]) -> bytes:
    """A little-endian TIFF header whose first IFD holds the given tag numbers."""
    body = struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHII", t, 3, 1, 1) for t in tags)
    return b"II*\x00" + struct.pack("<I", 8) + body + b"\x00" * 68


def test_raw_sniffing():
    from app.media.raw import sniff_raw

    assert sniff_raw(b"FUJIFILMCCD-RAW 0201" + b"\x00" * 64)[2] == ".raf"
    assert sniff_raw(b"IIRO\x08\x00\x00\x00" + b"\x00" * 64)[2] == ".orf"
    assert sniff_raw(b"IIU\x00\x08\x00\x00\x00" + b"\x00" * 64)[2] == ".rw2"
    assert sniff_raw(b"\x00\x00\x00\x18ftypcrx " + b"\x00" * 64)[2] == ".cr3"
    assert sniff_raw(b"II*\x00\x10\x00\x00\x00CR\x02\x00" + b"\x00" * 64)[2] == ".cr2"
    assert sniff_raw(_tiff([0x0100, 0xC612]))[2] == ".dng"
    plain_tiff = _tiff([0x0100])
    assert sniff_raw(plain_tiff, "scan.tif") is None
    assert sniff_raw(plain_tiff, "DSC_0001.NEF")[1] == "image/x-nikon-nef"
    assert sniff_raw(plain_tiff, "DSC_0001.arw")[2] == ".arw"


def test_probe_damaged_raw_is_kept_as_other(data_dir, tmp_path):
    from app.media.probe import probe

    info = probe(_write(tmp_path, "r.part", b"FUJIFILMCCD-RAW 0201" + b"\x00" * 200), "DSCF0001.RAF")
    assert (info.kind, info.ext, info.mime) == ("other", ".raf", "image/x-fuji-raf")


# ------------------------------------------------------------------ documents
def test_pdf_renders_and_text_layer(data_dir, tmp_path):
    from app import storage
    from app.media import documents

    path = _write(tmp_path, "doc.pdf", pdf_bytes([["Invoice 42", "Total 100"], ["Second page here"]]))
    layer = documents.text_layer(path, "pdf")
    assert len(layer) == 2
    assert "Invoice 42" in layer[0] and "Total 100" in layer[0]
    assert "Second page" in layer[1]
    out = documents.render_pages(path, "pdf", "abcdefgh", 1, max_side=800, max_pages=10)
    assert [n for n, _, _ in out] == [1, 2]
    for n, w, h in out:
        f = storage.page_path("abcdefgh", 1, n)
        assert f.is_file()
        with Image.open(f) as im:
            assert im.size == (w, h) and max(w, h) == 800
    assert documents.rendered_pages("abcdefgh", 1) == [1, 2]
    assert documents.render_pages(path, "pdf", "abcdefgh", 2, max_side=400, max_pages=1)[0][0] == 1
    assert documents.render_page(path, "pdf", "abcdefgh", 1, 5, max_side=400) is None


def test_pdf_page_without_text_layer_is_none(data_dir, tmp_path):
    from app.media import documents

    path = _write(tmp_path, "blank.pdf", pdf_bytes([[], ["text"]]))
    assert documents.text_layer(path, "pdf") == [None, "text"]


def test_office_pages_match_renders(data_dir, tmp_path):
    from app import storage
    from app.media import documents

    long_para = "word " * 900  # about 4500 characters: more than one logical page
    path = _docx(tmp_path / "long.docx", ["Title line", long_para, "Closing line"], table=[["Name", "Ravi"]])
    pages = documents.logical_pages(path)
    assert len(pages) >= 2
    assert pages[0].startswith("Title line")
    assert "Name | Ravi" in pages[-1] and "Closing line" in pages[-1]
    layer = documents.text_layer(path, "document")
    assert layer == pages
    out = documents.render_pages(path, "document", "docuid22", 1, max_side=600, max_pages=50)
    assert len(out) == len(pages)
    assert storage.page_path("docuid22", 1, len(pages)).is_file()


def test_plain_text_pagination_and_wrapping(data_dir, tmp_path):
    from app.media import documents

    lines = [f"line {i}" for i in range(150)]
    path = _write(tmp_path, "log.txt", "\n".join(lines).encode())
    pages = documents.logical_pages(path)
    assert len(pages) == 3
    assert pages[0].splitlines()[0] == "line 0"
    assert "\n".join(pages).splitlines() == lines
    img = documents.text_page_image(pages[0], 500)
    assert img.size[1] == 500 and img.mode == "RGB"


def test_xlsx_one_page_per_sheet(data_dir, tmp_path):
    import openpyxl

    from app.media import documents

    wb = openpyxl.Workbook()
    wb.active.title = "Budget"
    wb.active.append(["Rent", 25000])
    wb.active.append([None, None])
    ws = wb.create_sheet("Notes")
    ws.append(["hello", 1.0])
    path = tmp_path / "book.xlsx"
    wb.save(path)
    pages = documents.logical_pages(path)
    assert pages == ["Sheet: Budget\nRent\t25000", "Sheet: Notes\nhello\t1"]


# ------------------------------------------------------------------ dedup groups
def test_regroup_picks_the_best_and_dissolves(conn):
    from app.db import tx
    from app.media import dedup

    small = make_item(conn, width=320, height=400)
    large = make_item(conn, width=720, height=900)
    third = make_item(conn, width=360, height=450)
    with tx(conn):
        dedup.link_into_group(conn, large["id"], small["id"], "test")
    rows = {r["id"]: r for r in conn.execute("SELECT id, dup_group_id, dup_role FROM items")}
    assert rows[large["id"]]["dup_role"] == "best"
    assert rows[small["id"]]["dup_group_id"] == large["id"] and rows[small["id"]]["dup_role"] == "copy"
    with tx(conn):
        dedup.link_into_group(conn, third["id"], small["id"])
    assert set(dedup.group_member_ids(conn, small["id"])) == {small["id"], large["id"], third["id"]}
    with tx(conn):
        assert dedup.set_best(conn, third["id"])
    assert conn.execute("SELECT dup_role FROM items WHERE id = ?", (third["id"],)).fetchone()[0] == "best"
    with tx(conn):
        dedup.regroup(conn, [small["id"]])
    assert conn.execute("SELECT dup_group_id FROM items WHERE id = ?", (small["id"],)).fetchone()[0] is None


def test_find_similar_links_a_resized_copy(conn, fixture_file):
    from app.media import dedup
    from app.media.hashing import image_fingerprint
    from app.media.probe import probe

    src = fixture_file("armstrong.jpg")
    img = Image.open(src).convert("RGB")
    fp = image_fingerprint(img)
    item = make_item(conn, data=src.read_bytes(), width=img.width, height=img.height, phash=fp.phash,
                     dhash=fp.dhash, sig=fp.sig)
    small = img.resize((396, 495))
    data = jpeg_bytes(small, quality=60)
    info = probe(_bytes_path(data), "copy.jpg")
    decision = dedup.find_similar(conn, info, image_fingerprint(small), len(data))
    assert decision.action == "link" and decision.target_id == item["id"]
    other = Image.open(fixture_file("aldrin.jpg")).convert("RGB")
    assert dedup.find_similar(conn, info, image_fingerprint(other), len(data)).action == "new"
    flat = flat_image(396, 495)
    assert dedup.find_similar(conn, info, image_fingerprint(flat), len(data)).action == "new"


def _bytes_path(data: bytes) -> Path:
    from app import storage

    p = storage.tmp_path(".jpg")
    p.write_bytes(data)
    return p
