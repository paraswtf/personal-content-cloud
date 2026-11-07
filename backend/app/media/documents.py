"""PDF and office documents: sniffing, probing, page renders and text layers.

CONTRACT.
- sniff_document(data, path, filename) -> (kind, mime, ext) | None: PDF ('%PDF-' in the first
  KiB), DOCX / XLSX / PPTX (zip with [Content_Types].xml and a word/, xl/ or ppt/ folder) and plain
  text (.txt/.text/.log, .md/.markdown, .csv by the file name's extension, UTF-8 decodable, no NUL
  bytes). Kinds: 'pdf' for PDF, 'document' for the rest. A zip that breaks the ZIP_MAX_* limits (too many
  entries, too large a central directory, too much uncompressed data, an extreme compression ratio) is
  not sniffed as an office file (None, so it is kept as kind 'other'); the reason is logged.
- probe_document(path, kind, ext) -> (page_count, width, height): width/height are the size of
  the page 1 render (long side settings.page_render_size). ValueError when the file cannot be read.
  PDFs count at most PDF_MAX_PAGES pages; office and plain-text files at most MAX_LOGICAL_PAGES.
- logical_pages(path, ext) -> list[str]: the text of an office or plain-text file split into
  logical pages (docx and plain text: about PAGE_CHARS characters or LINES_PER_PAGE wrapped lines,
  xlsx: one per sheet, pptx: one per slide). Renders and text layers both come from this one
  function, so page N's render always shows page N's text.
- text_layer(path, kind, ext=None) -> list[str | None]: one entry per page; for PDFs the embedded
  text layer (None when a page has none, so OCR can take over); office kinds return their logical
  pages' text. Plain text past MAX_LOGICAL_PAGES pages (up to PLAIN_MAX_BYTES of file) is added to the
  last entry, so it still reaches item_text and the search index; that page's render shows only its
  own logical page.
- render_pages(path, kind, uid, version, *, max_side, max_pages, ext=None) -> [(page_no, w, h)]:
  writes storage.page_path files (WebP); office pages are their text drawn on a white page.
- render_page(path, kind, uid, version, page_no, *, max_side, ext=None) -> (w, h) | None: one page
  (the lazy /files/page route).
- page_image(path, kind, page_no, *, max_side, ext=None) -> RGB PIL image | None.
pdfium is not thread-safe: every pypdfium2 call in the process must hold PDFIUM_LOCK (exported for
other modules that use pypdfium2 directly). `ext` defaults to the path's suffix; pass it for files
without one (staged uploads).
"""
from __future__ import annotations

import logging
import re
import struct
import textwrap
import threading
import zipfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app import storage
from app.config import settings
from app.media import images

log = logging.getLogger(__name__)

PDFIUM_LOCK = threading.RLock()

PDF = ("pdf", "application/pdf", ".pdf")
DOCX = ("document", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
XLSX = ("document", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")
PPTX = ("document", "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx")
TXT = ("document", "text/plain", ".txt")
MD = ("document", "text/markdown", ".md")
CSV = ("document", "text/csv", ".csv")
_PLAIN_BY_EXT = {".txt": TXT, ".text": TXT, ".log": TXT, ".md": MD, ".markdown": MD, ".csv": CSV}
OFFICE_EXTS = frozenset({".docx", ".xlsx", ".pptx"})
PLAIN_EXTS = frozenset({".txt", ".md", ".csv"})
DOCUMENT_EXTS = OFFICE_EXTS | PLAIN_EXTS
# File picker extensions (the upload page's accept list).
ACCEPT_EXTS = (".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".csv")

PAGE_CHARS = 3000
CHARS_PER_LINE = 90
LINES_PER_PAGE = 60
MAX_LOGICAL_PAGES = 1000
PLAIN_MAX_BYTES = 16 * 1024 * 1024  # longer plain-text files are indexed up to this size
PLAIN_BLOCK_CHARS = CHARS_PER_LINE * LINES_PER_PAGE  # longer plain-text lines are cut into blocks this long
PDF_MAX_PAGES = 2000  # pages counted, text-extracted and offered for a PDF
# Office files are zips of a few hundred XML parts. These limits keep zip bombs and huge directories
# away from python-docx, openpyxl and python-pptx, which parse whole parts into memory.
ZIP_MAX_ENTRIES = 10_000
ZIP_MAX_DIR_BYTES = 2 * 1024 * 1024  # central directory size (zipfile reads entries until it is used up)
ZIP_MAX_TOTAL_BYTES = 256 * 1024 * 1024  # sum of uncompressed sizes
ZIP_MAX_PART_BYTES = 64 * 1024 * 1024  # one uncompressed XML part (.xml, .rels)
ZIP_MAX_RATIO = 200  # uncompressed / compressed, checked on entries of ZIP_RATIO_MIN_BYTES or more
ZIP_RATIO_MIN_BYTES = 1024 * 1024
XLSX_MAX_ROWS = 100_000  # rows scanned per sheet, empty rows included
XLSX_MAX_COLS = 1000  # cells read per row
XLSX_MAX_CELLS = 2_000_000  # cells scanned per workbook
PAGE_ASPECT = 2 ** -0.5  # A4 portrait: width = height / sqrt(2)
_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


# ------------------------------------------------------------------ sniffing
def _ext_of(filename: str | None) -> str:
    if not filename:
        return ""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


class ZipRefused(ValueError):
    """A zip that breaks the ZIP_MAX_* limits."""


def _zip_end_record(fh, size: int) -> None:
    """Check the end-of-central-directory record before zipfile reads the directory. It is found the
    way zipfile finds it (the last 22 bytes, else the last signature in the final 64 KiB)."""
    eocd = 22
    if size < eocd:
        raise zipfile.BadZipFile("too short for a zip")
    fh.seek(size - eocd)
    rec = fh.read(eocd)
    start = size - eocd
    if not (rec[:4] == b"PK\x05\x06" and rec[-2:] == b"\x00\x00"):
        tail_start = max(size - eocd - 65535, 0)
        fh.seek(tail_start)
        tail = fh.read()
        pos = tail.rfind(b"PK\x05\x06")
        if pos < 0 or len(tail) - pos < eocd:
            raise zipfile.BadZipFile("no end of central directory record")
        rec = tail[pos:pos + eocd]
        start = tail_start + pos
    _sig, _disk, _disk_cd, n_disk, n_total, dir_size, dir_offset, _comment = struct.unpack("<4s4H2LH", rec)
    if start >= 20:
        fh.seek(start - 20)
        if fh.read(4) == b"PK\x06\x07":
            raise ZipRefused("zip64 directory")
    if 0xFFFF in (n_disk, n_total) or 0xFFFFFFFF in (dir_size, dir_offset):
        raise ZipRefused("zip64 directory")
    if n_total > ZIP_MAX_ENTRIES or n_disk > ZIP_MAX_ENTRIES:
        raise ZipRefused(f"{max(n_total, n_disk)} entries (limit {ZIP_MAX_ENTRIES})")
    if dir_size > ZIP_MAX_DIR_BYTES:
        raise ZipRefused(f"central directory of {dir_size} bytes (limit {ZIP_MAX_DIR_BYTES})")


def _office_zip_names(path: Path) -> list[str]:
    """Entry names of a zip that stays within the ZIP_MAX_* limits. ZipRefused (a ValueError) when it
    does not; zipfile.BadZipFile or OSError when it is not a readable zip. The central directory is read
    once, and only after the end record showed it is small."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        _zip_end_record(fh, fh.tell())
        fh.seek(0)
        with zipfile.ZipFile(fh) as zf:
            infos = zf.infolist()
    if len(infos) > ZIP_MAX_ENTRIES:
        raise ZipRefused(f"{len(infos)} entries (limit {ZIP_MAX_ENTRIES})")
    total = 0
    for info in infos:
        total += info.file_size
        if total > ZIP_MAX_TOTAL_BYTES:
            raise ZipRefused(f"more than {ZIP_MAX_TOTAL_BYTES} bytes uncompressed")
        if info.filename.lower().endswith((".xml", ".rels")) and info.file_size > ZIP_MAX_PART_BYTES:
            raise ZipRefused(f"{info.filename} is {info.file_size} bytes uncompressed (limit {ZIP_MAX_PART_BYTES})")
        if info.file_size >= ZIP_RATIO_MIN_BYTES and info.file_size > ZIP_MAX_RATIO * max(info.compress_size, 1):
            raise ZipRefused(f"{info.filename} expands {info.file_size // max(info.compress_size, 1)} times")
    return [info.filename for info in infos]


def _sniff_zip(path: Path) -> tuple[str, str, str] | None:
    try:
        names = _office_zip_names(path)
    except ZipRefused as e:
        log.warning("zip %s is not read as an office file: %s; keeping it as a plain file", path, e)
        return None
    except (zipfile.BadZipFile, OSError, ValueError):
        return None
    if "[Content_Types].xml" not in names:
        return None
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    if "word" in tops:
        return DOCX
    if "xl" in tops:
        return XLSX
    if "ppt" in tops:
        return PPTX
    return None


def _looks_like_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        # The sniff buffer may cut a multi-byte character at its end.
        return e.start >= len(data) - 3 and e.reason == "unexpected end of data"


def sniff_document(data: bytes, path: Path, filename: str | None = None) -> tuple[str, str, str] | None:
    """(kind, mime, ext) of a PDF, office or plain-text file, or None."""
    if b"%PDF-" in data[:1024]:
        return PDF
    if data[:4] == b"PK\x03\x04":
        return _sniff_zip(path)
    plain = _PLAIN_BY_EXT.get(_ext_of(filename))
    if plain is not None and data and _looks_like_text(data):
        return plain
    return None


# ------------------------------------------------------------------ PDFs
def _open_pdf(path: Path):
    import pypdfium2 as pdfium

    try:
        return pdfium.PdfDocument(str(path))
    except Exception as e:  # PdfiumError: damaged or password protected
        raise ValueError(f"cannot open PDF: {e}") from e


def _render_size(w: float, h: float, max_side: int) -> tuple[int, int, float]:
    scale = max_side / max(w, h, 1.0)
    return max(1, round(w * scale)), max(1, round(h * scale)), scale


def _pdf_probe(path: Path) -> tuple[int, int, int]:
    with PDFIUM_LOCK:
        pdf = _open_pdf(path)
        try:
            n = min(len(pdf), PDF_MAX_PAGES)
            if n <= 0:
                raise ValueError("PDF has no pages")
            page = pdf[0]
            try:
                w, h = page.get_size()
                if page.get_rotation() in (90, 270):
                    w, h = h, w
            finally:
                page.close()
        finally:
            pdf.close()
    width, height, _ = _render_size(w, h, settings.page_render_size)
    return n, width, height


def _pdf_page_image(path: Path, page_no: int, max_side: int) -> Image.Image | None:
    with PDFIUM_LOCK:
        pdf = _open_pdf(path)
        try:
            if page_no < 1 or page_no > min(len(pdf), PDF_MAX_PAGES):
                return None
            page = pdf[page_no - 1]
            try:
                w, h = page.get_size()
                scale = max_side / max(w, h, 1.0)
                bitmap = page.render(scale=scale)
                try:
                    img = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            pdf.close()
    return img


def _pdf_text(path: Path) -> list[str | None]:
    """Text of the first PDF_MAX_PAGES pages. PDFIUM_LOCK is taken per page, so lazy page renders
    for other items are not held up for the whole document."""
    out: list[str | None] = []
    with PDFIUM_LOCK:
        pdf = _open_pdf(path)
        n = min(len(pdf), PDF_MAX_PAGES)
    try:
        for i in range(n):
            with PDFIUM_LOCK:
                page = pdf[i]
                try:
                    tp = page.get_textpage()
                    try:
                        text = tp.get_text_range()
                    finally:
                        tp.close()
                finally:
                    page.close()
            text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            out.append(text or None)
    finally:
        with PDFIUM_LOCK:
            pdf.close()
    return out


# ------------------------------------------------------------------ office and plain text
# textwrap's own whitespace set: runs of anything else are one "word" to it.
_LONG_WORD = re.compile(r"[^\t\n\x0b\x0c\r ]{%d,}" % (CHARS_PER_LINE + 1))
_LONG_SPACE = re.compile(r"[\t\n\x0b\x0c\r ]{%d,}" % (CHARS_PER_LINE + 1))


def _split_long(m: re.Match) -> str:
    run = m.group(0)
    return " ".join(run[i:i + CHARS_PER_LINE] for i in range(0, len(run), CHARS_PER_LINE))


def _wrap(text: str, max_lines: int | None = None) -> list[str]:
    """Wrap to CHARS_PER_LINE in linear time. textwrap slices a long word once per output line
    (quadratic), so runs longer than a line are cut into line-sized pieces first and long blank runs
    become one space. With max_lines, stop once that many lines exist."""
    lines: list[str] = []
    for raw in text.split("\n"):
        if max_lines is not None and len(lines) >= max_lines:
            break
        raw = raw.rstrip()
        if not raw:
            lines.append("")
            continue
        if len(raw) > CHARS_PER_LINE:
            raw = raw.expandtabs()
            raw = _LONG_SPACE.sub(" ", raw)
            raw = _LONG_WORD.sub(_split_long, raw)
            if max_lines is not None:
                # A wrapped line uses at most one line of text and one blank run (each at most
                # CHARS_PER_LINE long now), so this prefix still yields every line that is kept.
                keep = (max_lines - len(lines) + 1) * 2 * CHARS_PER_LINE
                raw = raw[:keep]
        lines.extend(textwrap.wrap(raw, CHARS_PER_LINE, break_long_words=True, break_on_hyphens=False,
                                   replace_whitespace=False, drop_whitespace=True) or [""])
    if max_lines is not None:
        del lines[max_lines:]
    return lines


def _flow_cut(blocks: list[str], *, cap_wrap: bool = True) -> tuple[list[str], tuple[int, list[str]] | None]:
    """_flow, also returning where it stopped: None when every block was placed, else (index of the
    block that did not fit, its wrapped lines that were not placed). With cap_wrap False each block
    is wrapped whole, so those lines are complete (the caller keeps blocks short)."""
    pages: list[list[str]] = []
    cur: list[str] = []
    cur_lines = 0
    cur_chars = 0

    def flush() -> None:
        nonlocal cur, cur_lines, cur_chars
        if cur:
            pages.append(cur)
        cur, cur_lines, cur_chars = [], 0, 0

    for index, block in enumerate(blocks):
        block = block.rstrip()
        budget = (MAX_LOGICAL_PAGES - len(pages)) * LINES_PER_PAGE if cap_wrap else None
        lines = _wrap(block, budget)
        whole = budget is None or len(lines) < budget  # False when _wrap may have stopped early
        pos = 0
        while pos < len(lines):
            room = LINES_PER_PAGE - cur_lines
            if room <= 0 or (cur and cur_chars >= PAGE_CHARS):
                flush()
                if len(pages) >= MAX_LOGICAL_PAGES:
                    return ["\n".join(p) for p in pages], (index, lines[pos:])
                continue
            take = lines[pos:pos + room]
            pos += len(take)
            # A block that fits keeps its own text (rendering wraps it the same way again).
            piece = block if whole and len(take) == len(lines) else "\n".join(take)
            cur.append(piece)
            cur_lines += len(take)
            cur_chars += len(piece)
    flush()
    return ["\n".join(p) for p in pages], None


def _flow(blocks: list[str]) -> list[str]:
    """Split text blocks (paragraphs) into pages of at most LINES_PER_PAGE wrapped lines and about
    PAGE_CHARS characters. A block longer than a page is split at wrapped-line boundaries."""
    return _flow_cut(blocks)[0]


def _docx_blocks(path: Path) -> list[str]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = docx.Document(str(path))
    except Exception as e:
        raise ValueError(f"cannot open DOCX: {e}") from e
    blocks: list[str] = []
    # Every block takes at least one wrapped line and a page holds at most LINES_PER_PAGE lines of
    # CHARS_PER_LINE characters, so nothing past these limits can reach a logical page.
    max_blocks = MAX_LOGICAL_PAGES * LINES_PER_PAGE
    max_chars = MAX_LOGICAL_PAGES * LINES_PER_PAGE * CHARS_PER_LINE
    chars = 0
    body = doc.element.body
    for child in body.iterchildren():
        if len(blocks) >= max_blocks or chars >= max_chars:
            break
        before = len(blocks)
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            blocks.append(Paragraph(child, doc).text)
        elif tag == "tbl":
            table = Table(child, doc)
            for row in table.rows:
                cells: list[str] = []
                for cell in row.cells:
                    text = cell.text.strip()
                    if text and (not cells or cells[-1] != text):  # merged cells repeat their text
                        cells.append(text)
                if cells:
                    blocks.append(" | ".join(cells))
                if len(blocks) >= max_blocks:
                    break
        chars += sum(len(b) for b in blocks[before:])
    return blocks


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _xlsx_pages(path: Path) -> list[str]:
    import openpyxl

    pages: list[str] = []
    try:
        fh = open(path, "rb")  # a file object: openpyxl refuses names without an Excel extension
    except OSError as e:
        raise ValueError(f"cannot read XLSX: {e}") from e
    try:
        try:
            wb = openpyxl.load_workbook(fh, read_only=True, data_only=True)
        except Exception as e:
            raise ValueError(f"cannot open XLSX: {e}") from e
        try:
            pages = _sheet_pages(wb)
        finally:
            wb.close()
    finally:
        fh.close()
    return pages


def _sheet_pages(wb) -> list[str]:
    pages: list[str] = []
    cells_left = XLSX_MAX_CELLS
    for ws in wb.worksheets[:MAX_LOGICAL_PAGES]:
        lines = [f"Sheet: {ws.title}"]
        chars = 0
        if cells_left <= 0:  # the workbook's scan budget is used up: keep one page per sheet
            lines.append("...")
            pages.append("\n".join(lines))
            continue
        if hasattr(ws, "reset_dimensions"):
            # Read-only sheets pad every row to the declared <dimension> (A1:XFD1048576 is legal for
            # a sheet of two cells); without it rows have their own width.
            ws.reset_dimensions()
        for row_no, row in enumerate(ws.iter_rows(values_only=True), 1):
            if row_no > XLSX_MAX_ROWS or cells_left <= 0:
                lines.append("...")
                break
            cells_left -= max(len(row), 1)  # empty rows (gaps between <row> elements) count too
            cells = [_cell_text(v) for v in row[:XLSX_MAX_COLS]]
            while cells and not cells[-1]:
                cells.pop()
            if not any(cells):
                continue
            line = "\t".join(cells)
            lines.append(line)
            chars += len(line)
            if chars > PAGE_CHARS * 20:  # very large sheets: index the first part
                lines.append("...")
                break
        pages.append("\n".join(lines))
    return pages


def _pptx_pages(path: Path) -> list[str]:
    import pptx

    try:
        prs = pptx.Presentation(str(path))
    except Exception as e:
        raise ValueError(f"cannot open PPTX: {e}") from e
    pages: list[str] = []
    for slide in list(prs.slides)[:MAX_LOGICAL_PAGES]:
        parts: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
        pages.append("\n".join(parts))
    return pages


def _plain_text(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(PLAIN_MAX_BYTES)
    except OSError as e:
        raise ValueError(f"cannot read text file: {e}") from e
    return data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _plain_blocks(text: str) -> tuple[list[str], list[int]]:
    """Lines of the text, lines longer than PLAIN_BLOCK_CHARS cut into blocks (before a blank where
    one is near), and the offset in `text` where each block ends."""
    blocks: list[str] = []
    ends: list[int] = []
    start = 0
    for line in text.split("\n"):
        end = start + len(line)
        pos = start
        while end - pos > PLAIN_BLOCK_CHARS:
            limit = pos + PLAIN_BLOCK_CHARS
            cut = max(text.rfind(" ", pos + 1, limit + 1), text.rfind("\t", pos + 1, limit + 1))
            if cut <= pos:
                cut = limit
            blocks.append(text[pos:cut])
            ends.append(cut)
            pos = cut
        blocks.append(text[pos:end])
        ends.append(end)
        start = end + 1
    return blocks, ends


def _plain_pages(path: Path) -> tuple[list[str], str]:
    """Logical pages of a plain-text file, and the text that did not fit in MAX_LOGICAL_PAGES."""
    text = _plain_text(path)
    blocks, ends = _plain_blocks(text)
    pages, cut = _flow_cut(blocks, cap_wrap=False)
    if cut is None:
        return pages, ""
    index, unplaced = cut
    return pages, "\n".join(unplaced) + text[ends[index]:]


def _check_office_zip(path: Path) -> None:
    try:
        _office_zip_names(path)
    except (zipfile.BadZipFile, OSError) as e:  # ZipRefused is a ValueError and passes through
        raise ValueError(f"cannot open {path.name}: {e}") from e


@lru_cache(maxsize=16)
def _logical_pages_cached(path: str, ext: str, mtime_ns: int, size: int) -> tuple[tuple[str, ...], str]:
    """(logical pages, plain text past the last page)."""
    p = Path(path)
    rest = ""
    if ext in OFFICE_EXTS:
        _check_office_zip(p)  # files stored before the zip limits existed come through here too
    if ext == ".docx":
        pages = _flow(_docx_blocks(p))
    elif ext == ".xlsx":
        pages = _xlsx_pages(p)
    elif ext == ".pptx":
        pages = _pptx_pages(p)
    elif ext in PLAIN_EXTS:
        pages, rest = _plain_pages(p)
    else:
        raise ValueError(f"not an office or text document: {ext or 'no extension'}")
    return (tuple(pages) or ("",)), rest


def _pages_and_rest(path: Path, ext: str | None) -> tuple[tuple[str, ...], str]:
    p = Path(path)
    ext = (ext or p.suffix).lower()
    try:
        st = p.stat()
    except OSError as e:
        raise ValueError(f"cannot read {p.name}: {e}") from e
    return _logical_pages_cached(str(p), ext, st.st_mtime_ns, st.st_size)


def logical_pages(path: Path, ext: str | None = None) -> list[str]:
    """Logical pages of an office or plain-text file (at least one, possibly empty)."""
    return list(_pages_and_rest(path, ext)[0])


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for candidate in _FONT_PATHS:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType support for the default font
        return ImageFont.load_default()


def text_page_image(text: str, max_side: int) -> Image.Image:
    """A white portrait page with `text` drawn in black, wrapped the same way pagination wraps it."""
    height = int(max_side)
    width = max(1, round(height * PAGE_ASPECT))
    margin = max(8, round(width * 0.06))
    line_h = max(4, (height - 2 * margin) // LINES_PER_PAGE)
    font = _font(max(6, int(line_h * 0.78)))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    lines = _wrap(text)
    for i, line in enumerate(lines):
        if y + line_h > height - margin:
            draw.text((margin, y - line_h), "...", fill=(90, 90, 90), font=font)
            break
        if line:
            draw.text((margin, y), line.replace("\t", "    "), fill=(20, 20, 20), font=font)
        y += line_h
    return img


# ------------------------------------------------------------------ public API
def probe_document(path: Path, kind: str, ext: str | None = None) -> tuple[int, int, int]:
    """(page_count, width, height of the page 1 render). ValueError when unreadable."""
    if kind == "pdf":
        return _pdf_probe(Path(path))
    pages = logical_pages(path, ext)
    height = int(settings.page_render_size)
    return len(pages), max(1, round(height * PAGE_ASPECT)), height


def text_layer(path: Path, kind: str, ext: str | None = None) -> list[str | None]:
    """One entry per page: PDF text layer (None where a page has none) or office/plain page text."""
    if kind == "pdf":
        return _pdf_text(Path(path))
    pages, rest = _pages_and_rest(path, ext)
    texts = list(pages)
    if rest.strip():
        texts[-1] = texts[-1] + "\n" + rest
    return [t if t.strip() else None for t in texts]


def page_image(path: Path, kind: str, page_no: int, *, max_side: int, ext: str | None = None) -> Image.Image | None:
    """RGB render of one page, or None when the page does not exist."""
    if kind == "pdf":
        return _pdf_page_image(Path(path), page_no, max_side)
    pages = logical_pages(path, ext)
    if page_no < 1 or page_no > len(pages):
        return None
    return text_page_image(pages[page_no - 1], max_side)


def _save_page(img: Image.Image, uid: str, version: int, page_no: int, max_side: int) -> tuple[int, int]:
    dest = storage.page_path(uid, version, page_no)
    images.save_webp(img, dest, max_side, settings.preview_quality)
    w, h = img.size
    s = min(1.0, max_side / max(w, h, 1))
    return max(1, round(w * s)), max(1, round(h * s))


def render_page(path: Path, kind: str, uid: str, version: int, page_no: int, *, max_side: int,
                ext: str | None = None) -> tuple[int, int] | None:
    """Write one page render to storage.page_path; (width, height) or None if no such page."""
    img = page_image(path, kind, page_no, max_side=max_side, ext=ext)
    if img is None:
        return None
    return _save_page(img, uid, version, page_no, max_side)


def render_pages(path: Path, kind: str, uid: str, version: int, *, max_side: int, max_pages: int,
                 ext: str | None = None) -> list[tuple[int, int, int]]:
    """Render up to max_pages pages to storage.page_path; returns [(page_no, width, height)]."""
    out: list[tuple[int, int, int]] = []
    if max_pages <= 0:
        return out
    if kind == "pdf":
        with PDFIUM_LOCK:
            n = min(_pdf_probe(Path(path))[0], max_pages)
        for page_no in range(1, n + 1):
            img = _pdf_page_image(Path(path), page_no, max_side)
            if img is None:
                break
            w, h = _save_page(img, uid, version, page_no, max_side)
            out.append((page_no, w, h))
        return out
    pages = logical_pages(path, ext)
    for page_no, text in enumerate(pages[:max_pages], 1):
        w, h = _save_page(text_page_image(text, max_side), uid, version, page_no, max_side)
        out.append((page_no, w, h))
    return out


_PAGE_NAME = re.compile(r"^(?P<page>\d+)\.webp$")


def rendered_pages(uid: str, version: int) -> list[int]:
    """Page numbers that have a render on disk."""
    d = storage.pages_dir(uid, version)
    if not d.is_dir():
        return []
    return sorted(int(m["page"]) for p in d.iterdir() if (m := _PAGE_NAME.match(p.name)))
