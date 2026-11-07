"""PDF and office documents: sniffing, probing, page renders and text layers.

CONTRACT.
- sniff_document(data, path, filename) -> (kind, mime, ext) | None: PDF ('%PDF-' in the first
  KiB), DOCX / XLSX / PPTX (zip with [Content_Types].xml and a word/, xl/ or ppt/ folder) and plain
  text (.txt/.text/.log, .md/.markdown, .csv by the file name's extension, UTF-8 decodable, no NUL
  bytes). Kinds: 'pdf' for PDF, 'document' for the rest.
- probe_document(path, kind, ext) -> (page_count, width, height): width/height are the size of
  the page 1 render (long side settings.page_render_size). ValueError when the file cannot be read.
- logical_pages(path, ext) -> list[str]: the text of an office or plain-text file split into
  logical pages (docx and plain text: about PAGE_CHARS characters or LINES_PER_PAGE wrapped lines,
  xlsx: one per sheet, pptx: one per slide). Renders and text layers both come from this one
  function, so page N's render always shows page N's text.
- text_layer(path, kind, ext=None) -> list[str | None]: one entry per page; for PDFs the embedded
  text layer (None when a page has none, so OCR can take over); office kinds return their logical
  pages' text.
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


def _sniff_zip(path: Path) -> tuple[str, str, str] | None:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
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
            n = len(pdf)
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
            if page_no < 1 or page_no > len(pdf):
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
    out: list[str | None] = []
    with PDFIUM_LOCK:
        pdf = _open_pdf(path)
        try:
            for i in range(len(pdf)):
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
            pdf.close()
    return out


# ------------------------------------------------------------------ office and plain text
def _wrap(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        raw = raw.rstrip()
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, CHARS_PER_LINE, break_long_words=True, break_on_hyphens=False,
                                   replace_whitespace=False, drop_whitespace=True) or [""])
    return lines


def _flow(blocks: list[str]) -> list[str]:
    """Split text blocks (paragraphs) into pages of at most LINES_PER_PAGE wrapped lines and about
    PAGE_CHARS characters. A block longer than a page is split at wrapped-line boundaries."""
    pages: list[list[str]] = []
    cur: list[str] = []
    cur_lines = 0
    cur_chars = 0

    def flush() -> None:
        nonlocal cur, cur_lines, cur_chars
        if cur:
            pages.append(cur)
        cur, cur_lines, cur_chars = [], 0, 0

    for block in blocks:
        block = block.rstrip()
        lines = _wrap(block)
        first = True
        while lines:
            room = LINES_PER_PAGE - cur_lines
            if room <= 0 or (cur and cur_chars >= PAGE_CHARS):
                flush()
                if len(pages) >= MAX_LOGICAL_PAGES:
                    return ["\n".join(p) for p in pages]
                continue
            take, lines = lines[:room], lines[room:]
            # A block that fits keeps its own text (rendering wraps it the same way again).
            piece = block if first and not lines else "\n".join(take)
            first = False
            cur.append(piece)
            cur_lines += len(take)
            cur_chars += len(piece)
    flush()
    return ["\n".join(p) for p in pages[:MAX_LOGICAL_PAGES]]


def _docx_blocks(path: Path) -> list[str]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = docx.Document(str(path))
    except Exception as e:
        raise ValueError(f"cannot open DOCX: {e}") from e
    blocks: list[str] = []
    body = doc.element.body
    for child in body.iterchildren():
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
    for ws in wb.worksheets[:MAX_LOGICAL_PAGES]:
        lines = [f"Sheet: {ws.title}"]
        chars = 0
        for row in ws.iter_rows(values_only=True):
            cells = [_cell_text(v) for v in row]
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


def _plain_blocks(path: Path) -> list[str]:
    try:
        with open(path, "rb") as f:
            data = f.read(PLAIN_MAX_BYTES)
    except OSError as e:
        raise ValueError(f"cannot read text file: {e}") from e
    text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n")


@lru_cache(maxsize=16)
def _logical_pages_cached(path: str, ext: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    p = Path(path)
    if ext == ".docx":
        pages = _flow(_docx_blocks(p))
    elif ext == ".xlsx":
        pages = _xlsx_pages(p)
    elif ext == ".pptx":
        pages = _pptx_pages(p)
    elif ext in PLAIN_EXTS:
        pages = _flow(_plain_blocks(p))
    else:
        raise ValueError(f"not an office or text document: {ext or 'no extension'}")
    return tuple(pages) or ("",)


def logical_pages(path: Path, ext: str | None = None) -> list[str]:
    """Logical pages of an office or plain-text file (at least one, possibly empty)."""
    p = Path(path)
    ext = (ext or p.suffix).lower()
    try:
        st = p.stat()
    except OSError as e:
        raise ValueError(f"cannot read {p.name}: {e}") from e
    return list(_logical_pages_cached(str(p), ext, st.st_mtime_ns, st.st_size))


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
    return [t if t.strip() else None for t in logical_pages(path, ext)]


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
