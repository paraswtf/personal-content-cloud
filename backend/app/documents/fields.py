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
- Hindi and Marathi (Devanagari): `normalize_text` folds Devanagari digits to ASCII, the danda to '.',
  drops zero-width joiners and decomposes nukta letters; month names, labels, gender words, relation
  words and names in Devanagari are understood by the helpers above. `word_tokens` is the mark-aware
  tokenizer (a token is a run of word characters, combining marks or U+0900 to U+0DFF letters),
  `languages_of(text)` names the languages of a text ('Hindi', 'Marathi', 'Hindi or Marathi', 'English'),
  `latin_part` / `local_part` split a mixed-script name, `local_name_after` reads a Devanagari name.
All functions are pure.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable

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

_DASHES = re.compile("[‐‑‒–—―−]")
_SPACES = re.compile("[ \t  -​]+")
_ODD_SPACES = re.compile("[\t  -​]")
_GAP = re.compile(r" {3,}")
_PAIR = re.compile(r"(?<! ) {2}(?! )")

# Devanagari: the block, its letters (no marks, digits or punctuation) and a fold table used by
# normalize_text: digits to ASCII, danda, double danda and the abbreviation sign to '.', zero-width
# (non-)joiners dropped, precomposed nukta letters (U+0958 to U+095F, NFC composition exclusions)
# split into letter + nukta so both spellings of a word compare equal.
DEV = "ऀ-ॿ"
_DEV_RE = re.compile("[" + DEV + "]")
_DEV_LETTER_RE = re.compile("[ऄ-हऽॐक़-ॡॲ-ॿ]")
_LATIN_LETTER_RE = re.compile("[A-Za-z]")
_DEV_FOLD: dict[int, Any] = {0x0966 + i: str(i) for i in range(10)}
_DEV_FOLD.update({0x0964: ".", 0x0965: ".", 0x0970: ".", 0x200C: None, 0x200D: None})
_DEV_FOLD.update({cp: unicodedata.normalize("NFD", chr(cp)) for cp in range(0x0958, 0x0960)})


def dev_fold(text: str) -> str:
    """Devanagari digits -> ASCII, danda -> '.', no zero-width joiners, nukta letters decomposed."""
    return text.translate(_DEV_FOLD) if text else text


def has_devanagari(text: str | None) -> bool:
    return bool(text) and bool(_DEV_LETTER_RE.search(text))


def is_letter(ch: str) -> bool:
    """A letter, or a combining mark (Devanagari vowel signs, virama, anusvara are marks, not letters)."""
    return ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc")


def word_guards(word: str) -> tuple[str, str]:
    """Lookbehind and lookahead that keep `word` (a label, a keyword) from matching inside a longer word
    of its own script: Latin words are guarded by Latin letters, Devanagari words by Devanagari letters
    and marks (so 'नाम' does not match in 'उपनाम' and 'कुल' not in 'कुलकर्णी')."""
    w = word.strip()
    lead = r"(?<![" + DEV + "])" if w[:1] and _DEV_RE.match(w[:1]) else r"(?<![A-Za-z])"
    trail = r"(?![" + DEV + "])" if w[-1:] and _DEV_RE.match(w[-1:]) else r"(?![A-Za-z])"
    return lead, trail


def word_regex(word: str) -> str:
    """Regex for a whole word or phrase (spaces match any spacing), guarded by word_guards."""
    lead, trail = word_guards(word)
    return lead + re.escape(dev_fold(word)).replace(r"\ ", r"\s+") + trail


def normalize_text(text: str) -> str:
    """Unify dashes and spaces; keep line breaks (form feeds become line breaks). A run of 3+ spaces
    (a column gap, also what extract.py writes between distant OCR boxes) stays as exactly 3 spaces,
    so after_label can end a value there; shorter runs become one space. Devanagari is folded
    (dev_fold) so labels, dates and amounts written with Devanagari digits match. Idempotent."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = dev_fold(text)
    text = _DASHES.sub("-", text).replace("₹", "₹ ").replace("­", "")
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


# a token: a run of word characters, combining marks (Latin and Indic) or Indic letters U+0900 to U+0DFF
# (dandas and the abbreviation sign excluded). Python's \w leaves out marks, so it splits 'मुंबई'.
TOKEN_RE = re.compile("[\\ẁ-ͯऀ-ॣ०-९ॱ-෿᪰-᫿᷀-᷿"
                      "⃐-⃿︠-︯]+")


def word_tokens(text: str | None) -> list[str]:
    """Words of a text for matching and search: marks stay inside words, zero-width joiners are dropped."""
    return TOKEN_RE.findall((text or "").replace("‌", "").replace("‍", ""))


_NAME_DEV = "ऀ-ॣॱ-ॿ"  # Devanagari without dandas, digits and the abbreviation sign


def latin_part(value: str | None) -> str:
    """The Latin-script part of a name ('राहुल शर्मा / Rahul Sharma' -> 'Rahul Sharma')."""
    v = re.sub(r"[^A-Za-z .,'&()-]", " ", value or "")
    return _SPACES.sub(" ", v).strip(" .,-'/")


def local_part(value: str | None) -> str:
    """The Devanagari part of a name ('राहुल शर्मा / Rahul Sharma' -> 'राहुल शर्मा')."""
    v = re.sub("[^" + _NAME_DEV + r" .,'&()-]", " ", dev_fold(value or ""))
    return _SPACES.sub(" ", v).strip(" .,-'/")


def nice_name(value: str | None) -> str | None:
    """Clean a person's or organisation's name; title-case it when OCR gave all capitals. A name in
    both scripts keeps its Latin spelling; a Devanagari-only name is kept as written (no casing)."""
    if not value:
        return None
    v = latin_part(value)
    if len(_LATIN_LETTER_RE.findall(v)) < 2 and has_devanagari(value):
        v = local_part(value)
        return v if 2 <= len(v) <= 80 else None
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
    letters = sum(is_letter(c) for c in value)
    return letters >= 3 and letters / max(1, len(value.replace(" ", ""))) > 0.8


def local_name(value: str | None) -> str | None:
    """The Devanagari name in a value, or None."""
    if not value or not has_devanagari(value):
        return None
    v = local_part(value)
    return v if plausible_name(v) and 2 <= len(v) <= 80 else None


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
# The same in Hindi and Marathi. They may be followed by '/' (a bilingual 'जन्म तिथि/DOB:' label).
STOP_LABELS_LOCAL = [
    "जन्म तिथि", "जन्मतिथि", "जन्म तारीख", "जन्मतारीख", "जन्म दिनांक", "जन्मदिनांक", "जन्म की तारीख", "जन्म वर्ष",
    "पिता का नाम", "माता का नाम", "पति का नाम", "वडिलांचे नाव", "आईचे नाव", "पतीचे नाव", "नाम", "नाव", "लिंग",
    "आयु", "उम्र", "वय", "पता", "पत्ता", "मोबाइल", "मोबाईल", "फोन", "दिनांक", "तारीख", "तिथि", "वैध", "राशि", "रकम",
    "रक्कम", "कुल", "एकूण", "पुत्र", "पुत्री", "पत्नी", "मुलगा", "मुलगी", "अनुक्रमांक", "आसन क्रमांक",
]
_STOP_RE = re.compile(r"\s(?:(?:" + "|".join(STOP_LABELS) + r")\s*[:.\-]|(?:"
                      + "|".join(re.escape(x).replace(r"\ ", r"\s*") for x in STOP_LABELS_LOCAL)
                      + r")\s*[:.\-/])", re.I)
_LABEL_SEP = r"\s*(?:[:\-=.]|\bis\b)?\s*"
# the other half of a bilingual label: 'नाम / Name :' after 'नाम', '/ नाम :' after 'Name'. It ends at a
# ':' '-' '='; at the end of the line it must end in a label word, so a value written after the slash
# ('Name / राहुल शर्मा') is not taken for a label.
_LATIN_TAIL_END = (r"(?:name\(s\)|names?|birth|dob|date|address|gender|sex|no\.?|number|year|place|issue|expiry|"
                   r"nationality|surname|age|amount|total|validity|group|id|code)")
_LOCAL_TAIL_END = ("(?:नाम|नाव|नांव|तिथि|तिथी|तारीख|दिनांक|पता|पत्ता|लिंग|संख्या|क्रमांक|सं\\.?|वर्ष|स्थान|राष्ट्रीयता|"
                   "उपनाम|आयु|उम्र|वय|राशि|रक्कम|रकम|वैधता|समूह|गट)")
_LATIN_TAIL = (r"\s*/\s*(?:[A-Za-z][A-Za-z .'()]{0,40}?\s*[:\-=]\s*|(?:[A-Za-z][A-Za-z .'()]{0,40}?\s)?"
               + _LATIN_TAIL_END + r"\s*$)")
_LOCAL_TAIL = (r"\s*/\s*(?:[" + DEV + r"][" + DEV + r" .'()]{0,40}?\s*[:\-=]\s*|(?:[" + DEV + r"][" + DEV
               + r" .'()]{0,40}?\s)?" + _LOCAL_TAIL_END + r"\s*$)")


def _label_regex(label: str) -> str:
    """Label text -> regex: spaces match any spacing, dots and apostrophes are optional."""
    out = []
    for ch in dev_fold(label):
        if ch == " ":
            out.append(r"\s*")
        elif ch in ".'":
            out.append(re.escape(ch) + "?")
        elif ch == "/":
            out.append(r"\s*/\s*")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _is_local(label: str) -> bool:
    return bool(_DEV_RE.search(label))


# Hindi and Marathi label synonyms shared by the parsers (each parser appends them after its English
# labels, so an English label still wins on a bilingual document).
NAME_LABELS_LOCAL = ["नाम", "नाव", "नांव"]
FATHER_LABELS_LOCAL = ["पिता का नाम", "पिता/पति का नाम", "वडिलांचे नाव", "वडिलांचे नांव", "पिता", "वडील"]
MOTHER_LABELS_LOCAL = ["माता का नाम", "आईचे नाव", "आईचे नांव", "माता", "आई"]
HUSBAND_LABELS_LOCAL = ["पति का नाम", "पतीचे नाव", "पति", "पती"]
DOB_LABELS_LOCAL = ["जन्म तिथि", "जन्म तिथी", "जन्मतिथि", "जन्मतिथी", "जन्म तारीख", "जन्मतारीख", "जन्म की तारीख",
                    "जन्म दिनांक", "जन्मदिनांक"]
ISSUE_LABELS_LOCAL = ["जारी करने की तिथि", "जारी करने की तारीख", "जारी करने का दिनांक", "जारी दिनांक", "जारी तिथि",
                      "जारी केल्याचा दिनांक", "दिनांक जारी"]
VALID_LABELS_LOCAL = ["तक वैध", "पर्यंत वैध", "वैधता", "वैध"]
# generic date words: a hit right after 'जन्म' (birth) is a date of birth, not the document's date
DATE_LABELS_LOCAL = ["दिनांक", "तारीख", "तिथि", "तिथी"]
AMOUNT_LABELS_LOCAL = ["कुल राशि", "कुल रकम", "एकूण रक्कम", "राशि", "रकम", "रक्कम"]
_GENERIC_DATE = tuple(dev_fold(x) for x in DATE_LABELS_LOCAL)
_NOT_BIRTH = r"(?<!जन्म)(?<!जन्म\s)(?<!जन्म\sकी\s)(?<!जन्म\sका\s)"


def after_label(text: str, labels: Iterable[str], *, next_line: bool = True, anchored: bool = False,
                max_len: int = 120, stop: bool = True, regex: bool = False, colon: bool = False,
                accept: Callable[[str], bool] | None = None) -> str | None:
    """Value written after a label: the rest of that line (cut at a gap of 2+ spaces, a ' | ' or another
    known label), or the next line when the label stands alone. anchored=True needs the label at the
    start of a line (or after a gap, or right after a Devanagari label and a '/', as in 'नाम / Name'),
    so 'Name' does not match inside "Father's Name". colon=True needs a ':', '-' or '=' after the label
    (or the label alone at the end of the line). accept(value) -> False skips that hit and goes on to
    the next line or label (so 'Seat' in 'Seat No. 14C' gives way to 'Seat No', and 'Boarding' in a
    'BOARDING PASS' heading to a later 'Boarding 06:15'). Labels may be Hindi or Marathi; the other
    half of a bilingual label ('जन्म तिथि/DOB:') is skipped before the value."""
    raw_lines = [ln.strip() for ln in normalize_text(text).split("\n")]
    raw_lines = [ln for ln in raw_lines if ln]
    for lab in labels:
        pat = lab if regex else _label_regex(lab)
        local = _is_local(lab)
        guard_lead, guard_trail = word_guards(lab if not regex or local else "a")
        if anchored:
            lead = r"(?:^|\s{2,}|\|\s*|(?<=[" + DEV + r"])\s*/\s*)"
        else:
            lead = guard_lead
        if local and dev_fold(lab).startswith(_GENERIC_DATE):
            lead = _NOT_BIRTH + lead
        sep = r"\s*(?:[:\-=]|$)\s*" if colon else _LABEL_SEP
        tail = _LATIN_TAIL if local else _LOCAL_TAIL
        rx = re.compile(lead + r"(?:" + pat + r")" + guard_trail + r"(?:" + tail + "|" + sep + ")", re.I)
        for i, line in enumerate(raw_lines):
            m = rx.search(line)
            if not m:
                continue
            rest = line[m.end():]
            # 'Name / राहुल शर्मा': a value in the other script written after the slash
            rest = re.sub(r"^\s*/\s*(?=[A-Za-z])" if local else "^\\s*/\\s*(?=[" + DEV + "])", "", rest)
            rest = re.split(r"\s{3,}|\s\|\s|\t", rest)[0] if rest else ""
            if stop and rest:
                sm = _STOP_RE.search(" " + rest)
                if sm and sm.start() > 0:
                    rest = (" " + rest)[: sm.start()]
            rest = clean_value(rest)
            if rest:
                if accept is None or accept(rest[:max_len]):
                    return rest[:max_len]
                continue
            if next_line and i + 1 < len(raw_lines):
                nxt = clean_value(raw_lines[i + 1])
                if nxt and not nxt.endswith(":") and (accept is None or accept(nxt[:max_len])):
                    return nxt[:max_len]
    return None


_REL_SPLIT = re.compile(r"\s+(?:(?:s/o|d/o|w/o|c/o)\b|(?:पुत्र|पुत्री|पत्नी|पति|मुलगा|मुलगी)(?![" + DEV + "]))", re.I)


def _latin_name_ok(v: str) -> bool:
    return plausible_name(latin_part(_REL_SPLIT.split(v)[0])) and len(_LATIN_LETTER_RE.findall(v)) >= 3


def name_after(text: str, labels: Iterable[str], **kw) -> str | None:
    """A person's name after a label (rejects values with digits). When the text carries Devanagari and
    the first hit gives no Latin spelling, a Latin spelling after any of the labels is preferred."""
    labels = list(labels)
    val = after_label(text, labels, **kw)
    if val:
        val = _REL_SPLIT.split(val)[0]
    name = nice_name(val) if plausible_name(val) else None
    if (name is None or not _LATIN_LETTER_RE.search(name)) and has_devanagari(text) and "accept" not in kw:
        lat = after_label(text, labels, accept=_latin_name_ok, **kw)
        if lat:
            return nice_name(latin_part(_REL_SPLIT.split(lat)[0])) or name
    return name


def local_name_after(text: str, labels: Iterable[str], **kw) -> str | None:
    """The Devanagari spelling of a name after one of the labels (Hindi or English labels), or None."""
    if not has_devanagari(text):
        return None
    val = after_label(text, labels, accept=lambda v: bool(local_name(_REL_SPLIT.split(v)[0])), **kw)
    return local_name(_REL_SPLIT.split(val)[0]) if val else None


_RELATION_RE = re.compile(
    r"(?<![A-Za-z])(S|D|W|C)\s*/\s*O\s*[:.\-]?\s*([A-Za-z][A-Za-z .']{1,60}?)(?=\s*(?:,|\n|$|\s{2,}|\d))", re.I)
_REL_WORD_RE = re.compile(r"(?<![A-Za-z])(son|daughter|wife|husband|care) of\s*[:\-]?\s*([A-Za-z][A-Za-z .']{1,60}?)"
                          r"(?=\s*(?:,|\n|$|\s{2,}|\d))", re.I)
RELATIONS = {"s": "Son of", "d": "Daughter of", "w": "Wife of", "c": "Care of"}
# Hindi and Marathi relation words before the relative's name: 'पुत्र श्री रमेश कुमार', 'पत्नी: सुनीता'
LOCAL_RELATIONS = {"पुत्र": "Son of", "पुत्री": "Daughter of", "पत्नी": "Wife of", "पति": "Husband of",
                   "मुलगा": "Son of", "मुलगी": "Daughter of"}
_REL_LOCAL_RE = re.compile(
    r"(?<![" + DEV + r"])(" + "|".join(LOCAL_RELATIONS) + r")(?![" + DEV + r"])"
    r"(?!\s*(?:का|की|के|चे|ची|च्या)(?![" + DEV + r"]))\s*[:\-]?\s*"
    r"(?:(?:श्रीमती|श्री|स्व\.?|कै\.?)\s+)?([" + _NAME_DEV + r"][" + _NAME_DEV + r" .']{1,60}?)"
    r"(?=\s*(?:,|\n|$|\s{2,}|\d))")


def relation_name(text: str) -> tuple[str, str] | None:
    """('Son of', 'Ramesh Kumar') from 'S/O Ramesh Kumar,' or 'Son of Ramesh Kumar' (or from Hindi and
    Marathi 'पुत्र रमेश कुमार', 'मुलगी: ...'; the name is then kept in Devanagari)."""
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
    m = _REL_LOCAL_RE.search(t)
    if m:
        name = local_name(m.group(2))
        if name:
            return LOCAL_RELATIONS[m.group(1)], name
    return None


_GENDER_WORDS = {"male": "Male", "female": "Female", "transgender": "Transgender", "m": "Male", "f": "Female",
                 "t": "Transgender"}
LOCAL_GENDER = {"पुरुष": "Male", "पुरूष": "Male", "महिला": "Female", "स्त्री": "Female", "अन्य": "Transgender",
                 "तृतीयपंथी": "Transgender", "ट्रांसजेंडर": "Transgender"}
GENDER_LABELS = ["gender", "sex", "लिंग"]
_LOCAL_GENDER_RE = re.compile(r"(?<![" + DEV + r"])(पुरुष|पुरूष|महिला|स्त्री)(?![" + DEV + "])")


def find_gender(text: str) -> str | None:
    val = after_label(text, GENDER_LABELS, next_line=True)
    if val:
        w = re.split(r"[\s/,]+", val.strip().lower())[0]
        if w in _GENDER_WORDS:
            return _GENDER_WORDS[w]
        if w in LOCAL_GENDER:
            return LOCAL_GENDER[w]
    m = re.search(r"(?<![A-Za-z])(female|male|transgender)(?![A-Za-z])", text, re.I)
    if m:
        return _GENDER_WORDS[m.group(1).lower()]
    m = _LOCAL_GENDER_RE.search(dev_fold(text))
    return LOCAL_GENDER[m.group(1)] if m else None


PIN_RE = re.compile(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)")


ADDRESS_LABELS_LOCAL = ["पता", "पत्ता"]


def address_after(text: str, labels: Iterable[str] = ("address",), max_lines: int = 5) -> str | None:
    """Address block after a label: that line's rest plus following lines up to a PIN code. Labels may be
    Hindi or Marathi ('पता', 'पत्ता'); the other half of a bilingual 'पता / Address:' label is skipped."""
    raw = [ln.strip() for ln in normalize_text(text).split("\n") if ln.strip()]
    for lab in labels:
        lead, trail = word_guards(lab)
        tail = _LATIN_TAIL if _is_local(lab) else _LOCAL_TAIL
        rx = re.compile(lead + _label_regex(lab) + trail + "(?:" + tail + "|" + _LABEL_SEP + ")", re.I)
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
# Hindi and Marathi month names (spellings with and without nukta, anusvara or half nasal)
LOCAL_MONTHS: dict[str, int] = {}
for _no, _words in enumerate((
        ("जनवरी", "जानेवारी"), ("फ़रवरी", "फरवरी", "फेब्रुवारी"), ("मार्च",), ("अप्रैल", "अप्रेल", "एप्रिल"),
        ("मई", "मे"), ("जून",), ("जुलाई", "जुलै"), ("अगस्त", "ऑगस्ट", "अगस्ट"),
        ("सितंबर", "सितम्बर", "सप्टेंबर", "सप्टेम्बर"), ("अक्टूबर", "अक्तूबर", "ऑक्टोबर", "अक्टोबर"),
        ("नवंबर", "नवम्बर", "नोव्हेंबर", "नोव्हेम्बर"), ("दिसंबर", "दिसम्बर", "डिसेंबर", "डिसेम्बर")), start=1):
    for _w in _words:
        LOCAL_MONTHS[dev_fold(_w)] = _no
_LOCAL_MON = "|".join(sorted((re.escape(w) for w in LOCAL_MONTHS), key=len, reverse=True))
# whole month words only ('Marks', 'Marathi', 'Junior' are not months; 'मे' is not in 'में'); compiled
# with re.A so that U+017F (long s) does not match 's' under re.I
_MON = (r"((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?=\.?(?![a-z]))|(?<![" + DEV + r"])(?:" + _LOCAL_MON + r")(?!["
        + DEV + r"]))\.?")
_DATE_RES: list[tuple[str, re.Pattern[str]]] = [
    ("ymd", re.compile(r"(?<!\d)((?:19|20)\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)")),
    ("dmy", re.compile(r"(?<![\d/])(\d{1,2})\s?[/\-.]\s?(\d{1,2})\s?[/\-.]\s?(\d{4}|\d{2})(?![\d/])")),
    ("dMy", re.compile(r"(?<![A-Za-z0-9])(\d{1,2})(?:st|nd|rd|th)?[\s\-/.,]*" + _MON + r"[\s\-/.,]*(\d{4}|\d{2})(?!\d)",
                       re.I | re.A)),
    ("Mdy", re.compile(r"(?<![A-Za-z])" + _MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?!\d)", re.I | re.A)),
]


def _month_no(word: str) -> int | None:
    if word in LOCAL_MONTHS:
        return LOCAL_MONTHS[word]
    return MONTHS.get(word.lower()[:3])


def _year(y: str, past: bool) -> int:
    """Two-digit year: past=True never lands after this year; otherwise a window up to 30 years ahead
    (expiry dates), so an old '98' is 1998, not 2098."""
    v = int(y)
    if len(y) == 4:
        return v
    this = date.today().year % 100
    if past:
        return 2000 + v if v <= this else 1900 + v
    return 2000 + v if v <= this + 30 else 1900 + v


def _mk(y: int, m: int, d: int) -> str | None:
    try:
        if not 1900 <= y <= 2100:
            return None
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str, *, past: bool = False) -> list[tuple[str, int, int]]:
    """All dates in text as (ISO date, start, end), in text order. Day-first for numeric dates.
    Offsets are into normalize_text(text)."""
    return _dates_in(normalize_text(text), past=past)


def _dates_in(t: str, *, past: bool = False) -> list[tuple[str, int, int]]:
    """find_dates on text that is already normalised."""
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
                mon = _month_no(g[1])
                iso = _mk(_year(g[2], past), mon, int(g[0])) if mon else None
            elif kind == "Mdy":
                mon = _month_no(g[0])
                iso = _mk(int(g[2]), mon, int(g[1])) if mon else None
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


# 'March 2024', 'Mar-24', "Mar'24", 'Mar/24'; a bare two-digit number needs - / or ' so that the day in
# 'Mar 15, 2024' is not read as a year
_MONTH_YEAR = re.compile(r"(?<![A-Za-z])" + _MON + r"(?:[\s\-'/,]*((?:19|20)\d{2})(?!\d)|"
                         r"[\s,]*[\-/'][\s\-']*(\d{2})(?!\d|\s*[\-/]\s*\d))", re.I | re.A)
_NUM_MONTH_YEAR = re.compile(r"(?<![\d/])(0?[1-9]|1[0-2])[/\-]((?:19|20)\d{2})(?!\d)")
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


def parse_month(value: str | None) -> str | None:
    """'March 2024', 'Mar-24', '03/2024', 'मार्च २०२४' -> '2024-03'."""
    if not value:
        return None
    value = dev_fold(value)
    m = _MONTH_YEAR.search(value)
    if m:
        y = m.group(2) or m.group(3)
        year = int(y) if len(y) == 4 else 2000 + int(y)
        mon = _month_no(m.group(1))
        if mon:
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
CURRENCY = r"(?:rs\.?|inr|\u20b9|rupees|रुपये|रुपए|रुपया|रु\.?|रू\.?)"
# a '-' right after a digit is a separator ('01-04-2024'), not a minus sign. A trailing 'Cr'/'Dr' after
# a balance ('1,000.00 Cr', '250.00 Dr') is a credit/debit marker, not crore (see _amounts_in).
_AMOUNT_RE = re.compile(
    r"(" + CURRENCY + r")?\s*((?:(?<!\d)-)?\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|(?:(?<!\d)-)?\d+(?:\.\d{1,2})?)"
    r"(?:\s*(lakhs?|lacs?|crores?|cr|lpa|l|k)(?![A-Za-z]))?(\s*(?:/-|cr\b|dr\b(?!\.[ ]*[A-Z])))?", re.I)
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")


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
    """Amounts in text: (value, had_currency_marker, start, end). Lakh/crore suffixes are applied; a 'Dr'
    (debit) marker makes the value negative. Offsets are into normalize_text(text)."""
    return _amounts_in(normalize_text(text))


def _amounts_in(t: str) -> list[tuple[Decimal, bool, int, int]]:
    """find_amounts on text that is already normalised."""
    out = []
    for m in _AMOUNT_RE.finditer(t):
        cur, num, mult = m.group(1), m.group(2), (m.group(3) or "").lower()
        mark = (m.group(4) or "").strip().lower()
        if mult == "cr" and ("," in num or (re.search(r"\.\d{2}$", num) and not cur)):
            # '12,345.00 Cr' on a statement is a credit balance; '2.5 Cr' / 'Rs 1.25 Cr' stay crore
            mult, mark = "", "cr"
        try:
            val = Decimal(num.replace(",", ""))
        except InvalidOperation:
            continue
        if mult in _MULT:
            val *= _MULT[mult]
        if mark == "dr":
            val = -abs(val)
        out.append((val, bool(cur) or bool(mult and mult != "k") or mark in ("cr", "dr"), m.start(), m.end()))
    return out


def parse_amount(value: str | None) -> Decimal | None:
    """First amount in value, preferring one with a currency marker, then one written like money (grouping
    commas or paise). Dates, month-years and percentages are not amounts. Falls back to amount words."""
    if not value:
        return None
    t = normalize_text(value)
    skip = [(s, e) for _iso, s, e in _dates_in(t)]
    skip += [m.span() for m in _MONTH_YEAR.finditer(t)] + [m.span() for m in _PERCENT_RE.finditer(t)]
    found = [a for a in _amounts_in(t) if not _looks_like_date_or_id(t, a)
             and not any(s < a[3] and a[2] < e for s, e in skip)]
    with_cur = [a for a in found if a[1]]
    money_like = [a for a in found if re.search(r",\d|\.\d{2}(?!\d)", t[a[2]:a[3]])]
    pick = (with_cur or money_like or found or [None])[0]
    if pick is not None:
        return pick[0]
    n = words_to_number(value)
    return Decimal(n) if n else None


def _looks_like_date_or_id(text: str, a: tuple[Decimal, bool, int, int]) -> bool:
    after = text[a[3]:a[3] + 2]
    before = text[max(0, a[2] - 1):a[2]]
    return bool(re.match(r"[/\-.]\d", after)) or before == "/" and bool(re.match(r"\d", text[a[2]:a[2] + 1]))


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
        # ... or digits come before it (a VID whose first group starts with 0 or 1)
        if re.search(r"\d{4}[ \-]?$", t[max(0, m.start() - 5):m.start()]):
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
    """Text of the tallest OCR box on a page (usually the headline: an event name, a title). Boxes with
    Latin words come first; a Devanagari box is used only when no Latin box qualifies."""
    skip_l = [s.lower() for s in skip]
    for script in (re.compile(r"[A-Za-z]{3}"), re.compile("[" + DEV + "]{3}")):
        best: Box | None = None
        for b in boxes:
            if b.page_no != page_no or len(b.text.strip()) < min_len:
                continue
            low = b.text.lower()
            if any(s in low for s in skip_l) or not script.search(b.text):
                continue
            if best is None or b.height > best.height * 1.05:
                best = b
        if best:
            return clean_value(dev_fold(best.text))
    return None


def all_barcodes(pages: list[PageText]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in pages:
        for b in p.barcodes:
            if b not in out:
                out.append(b)
    return out


def first_line_with(text: str, words: Iterable[str], *, max_len: int = 100) -> str | None:
    """The first line containing one of the words (case-insensitive), cleaned."""
    rxs = [re.compile(word_regex(w), re.I) for w in words]
    for ln in split_lines(text):
        if any(rx.search(ln) for rx in rxs):
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


# ------------------------------------------------------------------ languages

# Stop words and common document words that tell Hindi from Marathi (both are written in Devanagari).
HINDI_WORDS = frozenset(dev_fold(w) for w in (
    "है", "हैं", "और", "का", "की", "के", "में", "को", "से",
    "नाम", "पता", "पिता", "राशि", "रकम", "रसीद", "कुल", "किराया", "किरायानामा", "वेतन", "पर्ची", "पुत्र", "पुत्री",
    "पति", "नहीं", "यह", "एवं", "द्वारा", "किया", "गया", "हुआ", "प्राप्तांक", "अंक", "सूची", "तिथि", "मेरा", "मेरी",
))
MARATHI_WORDS = frozenset(dev_fold(w) for w in (
    "आहे", "आहेत", "आणि", "चे", "ची", "च्या", "मध्ये", "ला", "ने", "व",
    "नाव", "पत्ता", "वडिलांचे", "आईचे", "पतीचे", "एकूण", "रक्कम", "पावती", "गुणपत्रक", "गुणपत्रिका", "जन्मतारीख",
    "भाडे", "पगार", "मुलगा", "मुलगी", "नाही", "हे", "झाले", "केले", "माझे", "माझी", "आमचे", "गुण", "निकाल",
))  # words both languages use on documents (दिनांक, क्रमांक, शासन, करार) are left out
_MARATHI_SUFFIX = tuple(dev_fold(x) for x in ("चे", "ची", "च्या"))
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def languages_of(text: str | None) -> list[str]:
    """Languages of a text, the Devanagari one first: ['Hindi', 'English'], ['Marathi'],
    ['Hindi or Marathi'] (Devanagari, but the stop words do not decide), ['English'] or [].
    Devanagari counts from two words of two or more letters; English from two Latin words of three or
    more letters (so 'Rs.' or 'UPI' alone does not make a Hindi receipt English)."""
    toks = word_tokens(dev_fold(text or ""))
    dev = [w for w in toks if has_devanagari(w) and len(w) >= 2]
    latin = sum(1 for w in toks if _LATIN_WORD.fullmatch(w))
    out: list[str] = []
    if len(dev) >= 2:
        hi = sum(1 for w in dev if w in HINDI_WORDS)
        mr = sum(1 for w in dev if w in MARATHI_WORDS or (len(w) > 3 and w.endswith(_MARATHI_SUFFIX)))
        out.append("Hindi" if hi > mr else "Marathi" if mr > hi else "Hindi or Marathi")
    if latin >= 2:
        out.append("English")
    return out
