"""Field models and text helpers shared by document detection and the parsers.

Contract:
- `Box`, `PageText`, `FieldValue`, `ParsedDocument`: the data passed between extract.py, detect.py and
  the parsers (`parse(text, pages, boxes) -> ParsedDocument`).
- `FieldSet`: ordered field builder used by every parser; `to_json()` gives the stored
  `{key: {value, label, confidence}}` mapping (user-edited fields also carry `"source": "user"`).
- Checksums: `verhoeff_valid` (Aadhaar), `mrz_check_digit` / `mrz_valid` (ICAO 9303), `gstin_valid`.
- Normalisers: `parse_date` / `find_dates` (dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, dd MMM yyyy, MMM dd yyyy,
  yyyy-mm-dd), `parse_month`, `parse_amount` / `find_amounts` / `format_inr` (Rs, INR, the rupee sign,
  lakh and crore figures and words), `after_label`, `relation_name`, `find_gender`, `address_after`.
All functions are pure.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ data passed around

@dataclass
class Box:
    """One OCR text box, coordinates normalised to the page render (0..1)."""
    page_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float = 1.0

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


@dataclass
class PageText:
    page_no: int
    text: str
    source: str = "none"                       # layer | ocr | render | none
    boxes: list[Box] = field(default_factory=list)
    barcodes: list[tuple[str, str]] = field(default_factory=list)  # (format, payload)
    width: int | None = None
    height: int | None = None


@dataclass
class FieldValue:
    value: str
    label: str
    confidence: float = 0.9
    source: str = "auto"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "label": self.label, "confidence": round(float(self.confidence), 2)}
        if self.source == "user":
            out["source"] = "user"
        return out


@dataclass
class ParsedDocument:
    doc_type: str
    confidence: float
    fields: dict[str, FieldValue]
    title: str | None = None
    doc_date: str | None = None
    issuer: str | None = None
    person_name: str | None = None
    barcode: str | None = None
    barcode_format: str | None = None

    def fields_json(self) -> dict[str, dict[str, Any]]:
        return {k: v.to_json() for k, v in self.fields.items()}


class FieldSet:
    """Ordered field builder. The first value set for a key wins unless override=True."""

    def __init__(self, spec: Iterable[tuple[str, str]]):
        self.spec = list(spec)
        self.labels = dict(self.spec)
        self.values: dict[str, FieldValue] = {}

    def set(self, key: str, value: Any, confidence: float = 0.9, *, label: str | None = None,
            override: bool = False) -> str | None:
        if value is None:
            return None
        text = clean_value(str(value))
        if not text:
            return None
        if key in self.values and not override:
            return self.values[key].value
        self.values[key] = FieldValue(text, label or self.labels.get(key) or key.replace("_", " ").capitalize(),
                                      max(0.0, min(1.0, confidence)))
        return text

    def get(self, key: str) -> str | None:
        v = self.values.get(key)
        return v.value if v else None

    def has(self, key: str) -> bool:
        return key in self.values

    def ordered(self) -> dict[str, FieldValue]:
        out = {k: self.values[k] for k, _ in self.spec if k in self.values}
        for k, v in self.values.items():
            if k not in out:
                out[k] = v
        return out

    def found_fraction(self, keys: Iterable[str]) -> float:
        keys = list(keys)
        if not keys:
            return 0.0
        return sum(1 for k in keys if k in self.values) / len(keys)


def confidence_from(fs: FieldSet, key_fields: Iterable[str], *, base: float = 0.25, bonus: float = 0.0) -> float:
    """base + share of key fields found (scaled to the rest), plus a bonus for valid checksums."""
    frac = fs.found_fraction(key_fields)
    return round(max(0.05, min(1.0, base + (1.0 - base) * frac + bonus)), 2)


# ------------------------------------------------------------------ text basics

_DASHES = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_SPACES = re.compile("[ \t\u00a0\u2000-\u200b]+")
_ODD_SPACES = re.compile("[\t\u00a0\u2000-\u200b]")
_GAP = re.compile(r" {3,}")
_PAIR = re.compile(r"(?<! ) {2}(?! )")


def normalize_text(text: str) -> str:
    """Unify dashes and spaces; keep line breaks (form feeds become line breaks). A run of 3+ spaces
    (a column gap, also what extract.py writes between distant OCR boxes) stays as exactly 3 spaces,
    so after_label can end a value there; shorter runs become one space."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = _DASHES.sub("-", text).replace("\u20b9", "\u20b9 ").replace("\u00ad", "")
    text = _ODD_SPACES.sub("   ", text)
    return "\n".join(_PAIR.sub(" ", _GAP.sub("   ", ln)).rstrip() for ln in text.split("\n"))


def split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in normalize_text(text).split("\n") if ln.strip()]


def clean_value(value: str) -> str:
    value = _SPACES.sub(" ", value.replace("\n", " ")).strip()
    return value.strip(" :;,|-=").strip()


def has_any(text: str, words: Iterable[str]) -> bool:
    low = text.lower()
    return any(w in low for w in words)


def nice_name(value: str | None) -> str | None:
    """Clean a person's or organisation's name; title-case it when OCR gave all capitals."""
    if not value:
        return None
    v = re.sub(r"[^A-Za-z .,'&()-]", " ", value)
    v = _SPACES.sub(" ", v).strip(" .,-'")
    if len(v) < 2 or len(v) > 80:
        return None
    if v.isupper() and len(v) > 3:
        v = " ".join(w.capitalize() if len(w) > 1 else w for w in v.split(" "))
    return v


def plausible_name(value: str | None) -> bool:
    if not value:
        return False
    if re.search(r"\d", value):
        return False
    words = [w for w in re.split(r"\s+", value.strip()) if w]
    if not 1 <= len(words) <= 6:
        return False
    letters = sum(c.isalpha() for c in value)
    return letters >= 3 and letters / max(1, len(value.replace(" ", ""))) > 0.8


# ------------------------------------------------------------------ labels

# Labels that end a value when they appear later on the same line.
STOP_LABELS = [
    r"d\.?\s*o\.?\s*b", r"date of birth", r"year of birth", r"yob", r"gender", r"sex", r"age", r"father'?s? name",
    r"mother'?s? name", r"husband'?s? name", r"name", r"address", r"phone", r"mobile", r"email", r"blood group",
    r"valid(?:ity)?(?: till| upto| up to)?", r"issue date", r"date of issue", r"doi", r"dated?", r"time", r"seat",
    r"gate", r"row", r"section", r"roll no", r"seat no", r"pnr", r"flight", r"invoice no", r"gstin", r"pan",
    r"ifsc", r"branch", r"a/c no", r"account no", r"policy no", r"employee id", r"emp(?:loyee)? code",
    r"designation", r"department", r"uan", r"booking id", r"order id", r"venue", r"amount", r"total", r"s/o",
    r"d/o", r"w/o", r"c/o", r"ref(?:erence)?\.? no", r"reg(?:istration)?\.? no", r"ref\.? by", r"referred by",
]
_STOP_RE = re.compile(r"\s(?:" + "|".join(STOP_LABELS) + r")\s*[:.\-]", re.I)
_LABEL_SEP = r"\s*(?:[:\-=.]|\bis\b)?\s*"


def _label_regex(label: str) -> str:
    """Label text -> regex: spaces match any spacing, dots and apostrophes are optional."""
    out = []
    for ch in label:
        if ch == " ":
            out.append(r"\s*")
        elif ch in ".'":
            out.append(re.escape(ch) + "?")
        elif ch == "/":
            out.append(r"\s*/\s*")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def after_label(text: str, labels: Iterable[str], *, next_line: bool = True, anchored: bool = False,
                max_len: int = 120, stop: bool = True, regex: bool = False, colon: bool = False) -> str | None:
    """Value written after a label: the rest of that line (cut at a gap of 2+ spaces, a ' | ' or another
    known label), or the next line when the label stands alone. anchored=True needs the label at the
    start of a line (or after a gap), so 'Name' does not match inside "Father's Name". colon=True needs
    a ':', '-' or '=' after the label (or the label alone at the end of the line)."""
    raw_lines = [ln.strip() for ln in normalize_text(text).split("\n")]
    raw_lines = [ln for ln in raw_lines if ln]
    for lab in labels:
        pat = lab if regex else _label_regex(lab)
        lead = r"(?:^|\s{2,}|\|\s*)" if anchored else r"(?<![A-Za-z])"
        sep = r"\s*(?:[:\-=]|$)\s*" if colon else _LABEL_SEP
        rx = re.compile(lead + r"(?:" + pat + r")(?![A-Za-z])" + sep, re.I)
        for i, line in enumerate(raw_lines):
            m = rx.search(line)
            if not m:
                continue
            rest = line[m.end():]
            rest = re.split(r"\s{3,}|\s\|\s|\t", rest)[0] if rest else ""
            if stop and rest:
                sm = _STOP_RE.search(" " + rest)
                if sm and sm.start() > 0:
                    rest = (" " + rest)[: sm.start()]
            rest = clean_value(rest)
            if rest:
                return rest[:max_len]
            if next_line and i + 1 < len(raw_lines):
                nxt = clean_value(raw_lines[i + 1])
                if nxt and not nxt.endswith(":"):
                    return nxt[:max_len]
    return None


def name_after(text: str, labels: Iterable[str], **kw) -> str | None:
    """A person's name after a label (rejects values with digits)."""
    val = after_label(text, labels, **kw)
    if val:
        val = re.split(r"\s+(?:s/o|d/o|w/o|c/o)\b", val, flags=re.I)[0]
    return nice_name(val) if plausible_name(val) else None


_RELATION_RE = re.compile(
    r"(?<![A-Za-z])(S|D|W|C)\s*/\s*O\s*[:.\-]?\s*([A-Za-z][A-Za-z .']{1,60}?)(?=\s*(?:,|\n|$|\s{2,}|\d))", re.I)
_REL_WORD_RE = re.compile(r"(?<![A-Za-z])(son|daughter|wife|husband|care) of\s*[:\-]?\s*([A-Za-z][A-Za-z .']{1,60}?)"
                          r"(?=\s*(?:,|\n|$|\s{2,}|\d))", re.I)
RELATIONS = {"s": "Son of", "d": "Daughter of", "w": "Wife of", "c": "Care of"}


def relation_name(text: str) -> tuple[str, str] | None:
    """('Son of', 'Ramesh Kumar') from 'S/O Ramesh Kumar,' or 'Son of Ramesh Kumar'."""
    t = normalize_text(text)
    m = _RELATION_RE.search(t)
    if m:
        name = nice_name(m.group(2))
        if name and plausible_name(name):
            return RELATIONS[m.group(1).lower()], name
    m = _REL_WORD_RE.search(t)
    if m:
        name = nice_name(m.group(2))
        if name and plausible_name(name):
            return f"{m.group(1).capitalize()} of", name
    return None


_GENDER_WORDS = {"male": "Male", "female": "Female", "transgender": "Transgender", "m": "Male", "f": "Female",
                 "t": "Transgender"}


def find_gender(text: str) -> str | None:
    val = after_label(text, ["gender", "sex"], next_line=True)
    if val:
        w = re.split(r"[\s/,]+", val.strip().lower())[0]
        if w in _GENDER_WORDS:
            return _GENDER_WORDS[w]
    m = re.search(r"(?<![A-Za-z])(female|male|transgender)(?![A-Za-z])", text, re.I)
    return _GENDER_WORDS[m.group(1).lower()] if m else None


PIN_RE = re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)")


def address_after(text: str, labels: Iterable[str] = ("address",), max_lines: int = 5) -> str | None:
    """Address block after a label: that line's rest plus following lines up to a PIN code."""
    raw = [ln.strip() for ln in normalize_text(text).split("\n") if ln.strip()]
    for lab in labels:
        rx = re.compile(r"(?<![A-Za-z])" + _label_regex(lab) + r"(?![A-Za-z])" + _LABEL_SEP, re.I)
        for i, line in enumerate(raw):
            m = rx.search(line)
            if not m:
                continue
            parts = [line[m.end():].strip()]
            j = i + 1
            while not PIN_RE.search(" ".join(parts)) and j < len(raw) and len(parts) < max_lines:
                nxt = raw[j]
                if re.match(r"^\s*(?:\d{4}\s\d{4}\s\d{4}|vid|www\.|help@|1947|issue date)", nxt, re.I):
                    break
                parts.append(nxt)
                j += 1
            addr = clean_value(", ".join(p for p in parts if p))
            addr = re.sub(r"\s*,\s*,+", ",", addr).strip(", ")
            if len(addr) >= 8:
                return addr[:300]
    return None


def find_pincode(text: str | None) -> str | None:
    if not text:
        return None
    found = PIN_RE.findall(text)
    return "".join(found[-1]) if found else None


# ------------------------------------------------------------------ dates

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
          "oct": 10, "nov": 11, "dec": 12}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)[a-z]*\.?"
_DATE_RES: list[tuple[str, re.Pattern[str]]] = [
    ("ymd", re.compile(r"(?<!\d)((?:19|20)\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)")),
    ("dmy", re.compile(r"(?<![\d/])(\d{1,2})\s?[/\-.]\s?(\d{1,2})\s?[/\-.]\s?(\d{4}|\d{2})(?![\d/])")),
    ("dMy", re.compile(r"(?<![A-Za-z0-9])(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*" + _MON + r"[\s\-/.,]*(\d{4}|\d{2})(?!\d)",
                       re.I)),
    ("Mdy", re.compile(r"(?<![A-Za-z])" + _MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?!\d)", re.I)),
]


def _year(y: str, past: bool) -> int:
    v = int(y)
    if len(y) == 4:
        return v
    this = date.today().year % 100
    if past:
        return 2000 + v if v <= this else 1900 + v
    return 2000 + v


def _mk(y: int, m: int, d: int) -> str | None:
    try:
        if not 1900 <= y <= 2100:
            return None
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str, *, past: bool = False) -> list[tuple[str, int, int]]:
    """All dates in text as (ISO date, start, end), in text order. Day-first for numeric dates."""
    t = normalize_text(text)
    found: list[tuple[str, int, int]] = []
    taken: list[tuple[int, int]] = []
    for kind, rx in _DATE_RES:
        for m in rx.finditer(t):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            g = m.groups()
            iso = None
            if kind == "ymd":
                iso = _mk(int(g[0]), int(g[1]), int(g[2]))
            elif kind == "dmy":
                iso = _mk(_year(g[2], past), int(g[1]), int(g[0]))
            elif kind == "dMy":
                iso = _mk(_year(g[2], past), MONTHS[g[1].lower()[:4] if g[1].lower().startswith("sept") else g[1].lower()[:3]], int(g[0]))
            elif kind == "Mdy":
                iso = _mk(int(g[2]), MONTHS[g[0].lower()[:3]], int(g[1]))
            if iso:
                found.append((iso, m.start(), m.end()))
                taken.append((m.start(), m.end()))
    found.sort(key=lambda x: x[1])
    return found


def parse_date(value: str | None, *, past: bool = False) -> str | None:
    if not value:
        return None
    d = find_dates(value, past=past)
    return d[0][0] if d else None


def date_after(text: str, labels: Iterable[str], *, past: bool = False, **kw) -> str | None:
    val = after_label(text, labels, stop=False, **kw)
    return parse_date(val, past=past) if val else None


def iso_valid(value: str | None) -> bool:
    if not value or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


_MONTH_YEAR = re.compile(r"(?<![A-Za-z])" + _MON + r"[\s\-'/,]*((?:19|20)\d{2}|'\d{2})(?!\d)", re.I)
_NUM_MONTH_YEAR = re.compile(r"(?<![\d/])(0?[1-9]|1[0-2])[/\-]((?:19|20)\d{2})(?!\d)")
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


def parse_month(value: str | None) -> str | None:
    """'March 2024', 'Mar-24', '03/2024' -> '2024-03'."""
    if not value:
        return None
    m = _MONTH_YEAR.search(value)
    if m:
        y = m.group(2).lstrip("'")
        year = int(y) if len(y) == 4 else 2000 + int(y)
        mon = MONTHS[m.group(1).lower()[:3]]
        return f"{year:04d}-{mon:02d}"
    m = _NUM_MONTH_YEAR.search(value)
    if m:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"
    return None


def month_label(ym: str | None) -> str | None:
    if not ym or not re.fullmatch(r"\d{4}-\d{2}", ym):
        return ym
    return f"{MONTH_NAMES[int(ym[5:]) - 1]} {ym[:4]}"


TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:.]([0-5]\d)\s*(am|pm|a\.m\.|p\.m\.|hrs|hours)?(?![\d])", re.I)


def parse_time(value: str | None) -> str | None:
    """'7:30 PM' -> '19:30'; '06.45 hrs' -> '06:45'."""
    if not value:
        return None
    m = TIME_RE.search(value)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower().replace(".", "")
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    if not 0 <= h <= 23:
        return None
    return f"{h:02d}:{mi:02d}"


# ------------------------------------------------------------------ amounts

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
           "crore": 10000000, "crores": 10000000, "million": 1000000, "billion": 1000000000}
_MULT = {"lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000, "l": 100000, "lpa": 100000,
         "crore": 10000000, "crores": 10000000, "cr": 10000000, "k": 1000}
CURRENCY = r"(?:rs\.?|inr|\u20b9|rupees)"
_AMOUNT_RE = re.compile(
    r"(" + CURRENCY + r")?\s*(-?\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|-?\d+(?:\.\d{1,2})?)"
    r"(?:\s*(lakhs?|lacs?|crores?|cr|lpa|l|k)(?![A-Za-z]))?(\s*(?:/-|cr\b|dr\b))?", re.I)


def words_to_number(text: str) -> int | None:
    """'Rupees One Lakh Twenty Thousand Five Hundred Only' -> 120500."""
    words = re.findall(r"[a-z]+", text.lower())
    total, current, seen = 0, 0, False
    for w in words:
        if w in ("and", "rupees", "rupee", "rs", "inr", "only", "paise"):
            if w == "paise":
                break
            continue
        if w in _NUM_WORDS:
            current += _NUM_WORDS[w]
            seen = True
        elif w == "hundred":
            current = max(current, 1) * 100
            seen = True
        elif w in _SCALES:
            total += max(current, 1) * _SCALES[w]
            current = 0
            seen = True
        elif seen:
            break
    return total + current if seen else None


def find_amounts(text: str) -> list[tuple[Decimal, bool, int, int]]:
    """Amounts in text: (value, had_currency_marker, start, end). Lakh/crore suffixes are applied."""
    out = []
    for m in _AMOUNT_RE.finditer(normalize_text(text)):
        cur, num, mult = m.group(1), m.group(2), (m.group(3) or "").lower()
        try:
            val = Decimal(num.replace(",", ""))
        except InvalidOperation:
            continue
        if mult in _MULT:
            val *= _MULT[mult]
        out.append((val, bool(cur) or bool(mult and mult != "k"), m.start(), m.end()))
    return out


def parse_amount(value: str | None) -> Decimal | None:
    """First amount in value, preferring one with a currency marker; falls back to amount words."""
    if not value:
        return None
    found = [a for a in find_amounts(value) if not _looks_like_date_or_id(value, a)]
    with_cur = [a for a in found if a[1]]
    pick = with_cur[0] if with_cur else (found[0] if found else None)
    if pick is not None:
        return pick[0]
    n = words_to_number(value)
    return Decimal(n) if n else None


def _looks_like_date_or_id(text: str, a: tuple[Decimal, bool, int, int]) -> bool:
    after = text[a[3]:a[3] + 2]
    before = text[max(0, a[2] - 1):a[2]]
    return bool(re.match(r"[/\-]\d", after)) or before == "/" and bool(re.match(r"\d", text[a[2]:a[2] + 1]))


def format_inr(value: Decimal | int | float | None) -> str | None:
    """Indian digit grouping with the rupee sign: 123456.5 -> '\u20b91,23,456.50'."""
    if value is None:
        return None
    v = Decimal(str(value)).quantize(Decimal("0.01"))
    neg = v < 0
    v = abs(v)
    whole, frac = divmod(v, 1)
    s = str(int(whole))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    out = "\u20b9" + s
    if frac:
        out += "." + f"{int((frac * 100).to_integral_value()):02d}"
    return ("-" if neg else "") + out


def amount_after(text: str, labels: Iterable[str], **kw) -> str | None:
    val = after_label(text, labels, stop=False, **kw)
    amt = parse_amount(val) if val else None
    return format_inr(amt) if amt is not None else None


# ------------------------------------------------------------------ checksums and identifiers

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


AADHAAR_RE = re.compile(r"(?<![\d])([2-9]\d{3})[ \-]?(\d{4})[ \-]?(\d{4})(?![\d])")
AADHAAR_GROUPED_RE = re.compile(r"(?<![\d ])([2-9]\d{3}) (\d{4}) (\d{4})(?![\d])")
AADHAAR_MASKED_RE = re.compile(r"(?<![\dX])([Xx*]{4})[ \-]?([Xx*]{4})[ \-]?(\d{4})(?![\d])")
VID_RE = re.compile(r"(?<!\d)(\d{4}) ?(\d{4}) ?(\d{4}) ?(\d{4})(?!\d)")


def find_aadhaar(text: str, *, grouped_only: bool = False) -> str | None:
    """First Verhoeff-valid 12-digit Aadhaar number, formatted 'dddd dddd dddd'. VIDs are skipped."""
    t = normalize_text(text)
    rx = AADHAAR_GROUPED_RE if grouped_only else AADHAAR_RE
    for m in rx.finditer(t):
        number = "".join(m.groups())
        # part of a 16-digit VID: digits continue after a space
        tail = t[m.end():m.end() + 5]
        if re.match(r" ?\d{4}", tail):
            continue
        if verhoeff_valid(number):
            return f"{number[:4]} {number[4:8]} {number[8:]}"
    return None


def mrz_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - 55
    return 0


def mrz_check_digit(data: str) -> int:
    weights = (7, 3, 1)
    return sum(mrz_value(c) * weights[i % 3] for i, c in enumerate(data)) % 10


def mrz_valid(data: str, check: str) -> bool:
    if check == "<":
        check = "0"
    return check.isdigit() and mrz_check_digit(data) == int(check)


_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"(?<![A-Z0-9])(\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z])(?![A-Z0-9])")


def gstin_check_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        v = _GST_CHARS.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return _GST_CHARS[(36 - total % 36) % 36]


def gstin_valid(gstin: str) -> bool:
    g = (gstin or "").upper()
    if not GSTIN_RE.fullmatch(g):
        return False
    return gstin_check_char(g[:14]) == g[14]


def find_gstins(text: str) -> list[str]:
    t = normalize_text(text).upper()
    out: list[str] = []
    for m in GSTIN_RE.finditer(t):
        g = m.group(1)
        if g not in out:
            out.append(g)
    return out


PAN_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{3}[ABCFGHLJPTE][A-Z]\d{4}[A-Z])(?![A-Z0-9])")
PAN_HOLDER = {"P": "Individual", "C": "Company", "H": "Hindu undivided family", "F": "Firm",
              "A": "Association of persons", "T": "Trust", "B": "Body of individuals", "L": "Local authority",
              "J": "Artificial juridical person", "G": "Government", "E": "Limited liability partnership"}


def find_pans(text: str) -> list[str]:
    """PAN numbers outside GSTINs (a GSTIN embeds its holder's PAN at characters 3 to 12)."""
    t = normalize_text(text).upper()
    t = GSTIN_RE.sub(lambda m: "#" * len(m.group(0)), t)
    out: list[str] = []
    for m in PAN_RE.finditer(t):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


EPIC_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{3}\d{7})(?![A-Z0-9])")
IFSC_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{4}0[A-Z0-9]{6})(?![A-Z0-9])")

STATE_CODES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh", "DD": "Daman and Diu", "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli", "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana",
    "JH": "Jharkhand", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh",
    "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur", "MP": "Madhya Pradesh",
    "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha", "PB": "Punjab", "PY": "Puducherry",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana", "TG": "Telangana",
    "UK": "Uttarakhand", "UA": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}
VEHICLE_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2})[ \-]?(\d{1,2})[ \-]?([A-Z]{1,3})[ \-]?(\d{4})(?![0-9])")
BH_VEHICLE_RE = re.compile(r"(?<![A-Z0-9])(\d{2})[ \-]?BH[ \-]?(\d{4})[ \-]?([A-Z]{1,2})(?![A-Z0-9])")


def find_vehicle_number(text: str) -> str | None:
    t = normalize_text(text).upper()
    for m in VEHICLE_RE.finditer(t):
        if m.group(1) in STATE_CODES:
            return f"{m.group(1)} {int(m.group(2)):02d} {m.group(3)} {m.group(4)}"
    m = BH_VEHICLE_RE.search(t)
    if m:
        return f"{m.group(1)} BH {m.group(2)} {m.group(3)}"
    return None


# Driving licence numbers: the national format SS RR YYYY NNNNNNN (Maharashtra MH12 20110012345,
# Delhi DL-0420110149646, Karnataka KA01 20200001234, Tamil Nadu TN-01-2008-0012345 ...), the
# 13-digit form without the year split, and older slash formats (MH/12/001234/2011).
DL_RES = [
    re.compile(r"(?<![A-Z0-9])([A-Z]{2})[ \-]?(\d{2})[ \-/]?((?:19|20)\d{2})[ \-/]?(\d{7})(?!\d)"),
    re.compile(r"(?<![A-Z0-9])([A-Z]{2})[ \-]?(\d{2})[ \-]?(\d{11})(?!\d)"),
    re.compile(r"(?<![A-Z0-9])([A-Z]{2})/(\d{2})/(\d{3,7})/((?:19|20)\d{2})(?!\d)"),
    re.compile(r"(?<![A-Z0-9])([A-Z]{2})(\d{2})\s?(\d{4,7})(?!\d)"),
]


def find_dl_number(text: str) -> tuple[str, str | None] | None:
    """(normalised licence number, state name) or None."""
    t = normalize_text(text).upper()
    for i, rx in enumerate(DL_RES):
        for m in rx.finditer(t):
            st = m.group(1)
            if st not in STATE_CODES:
                continue
            g = m.groups()
            if i == 0:
                num = f"{st}{g[1]} {g[2]}{g[3]}"
            elif i == 1:
                num = f"{st}{g[1]} {g[2]}"
            elif i == 2:
                num = f"{st}/{g[1]}/{g[2]}/{g[3]}"
            else:
                if i == 3 and not re.search(r"(?:dl|licen[cs]e)\s*(?:no|number)", t, re.I):
                    continue
                num = f"{st}{g[1]} {g[2]}"
            return num, STATE_CODES[st]
    return None


# ------------------------------------------------------------------ boxes

def largest_text(boxes: list[Box], page_no: int = 1, *, min_len: int = 3, skip: Iterable[str] = ()) -> str | None:
    """Text of the tallest OCR box on a page (usually the headline: an event name, a title)."""
    skip_l = [s.lower() for s in skip]
    best: Box | None = None
    for b in boxes:
        if b.page_no != page_no or len(b.text.strip()) < min_len:
            continue
        low = b.text.lower()
        if any(s in low for s in skip_l) or not re.search(r"[A-Za-z]{3}", b.text):
            continue
        if best is None or b.height > best.height * 1.05:
            best = b
    return clean_value(best.text) if best else None


def all_barcodes(pages: list[PageText]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in pages:
        for b in p.barcodes:
            if b not in out:
                out.append(b)
    return out


def first_line_with(text: str, words: Iterable[str], *, max_len: int = 100) -> str | None:
    """The first line containing one of the words (case-insensitive), cleaned."""
    words_l = [w.lower() for w in words]
    for ln in split_lines(text):
        low = ln.lower()
        if any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low) for w in words_l):
            v = clean_value(ln)
            if 3 <= len(v) <= max_len:
                return v
    return None


def known_name(text: str, names: Iterable[str]) -> str | None:
    """The first of a list of known organisation names found in the text (case-insensitive)."""
    low = normalize_text(text).lower()
    best: tuple[int, str] | None = None
    for n in names:
        i = low.find(n.lower())
        if i >= 0 and (best is None or i < best[0]):
            best = (i, n)
    return best[1] if best else None
