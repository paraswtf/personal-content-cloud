"""Apple Wallet pass builder and signer (docs/ARCHITECTURE.md section 11).

Contract:
- configured() -> (ok, missing): needs settings.wallet_pass_type_id, settings.wallet_team_id and cert.pem,
  key.pem, wwdr.pem in storage.wallet_dir(). `missing` names what is absent (env names and file names).
- build_pass(conn, item_row, doc_row) -> bytes of a signed .pkpass (a zip, entries stored uncompressed):
  pass.json (formatVersion 1, passTypeIdentifier, teamIdentifier, serialNumber = item uid,
  organizationName, description, eventTicket or boardingPass with primary, secondary, auxiliary and back
  fields from the document fields, barcode + barcodes, colours from the thumbnail's dominant colour,
  relevantDate when the document has a date and a time), icon.png, icon@2x.png, logo.png, logo@2x.png
  and (event tickets) thumbnail.png, thumbnail@2x.png drawn with Pillow from the item thumbnail,
  manifest.json (sha1 of every other file) and signature (detached DER PKCS#7 over manifest.json,
  SHA-256, signer certificate + key, WWDR certificate included).
- Raises WalletNotConfigured (missing setup), NotAPass (document group is not 'pass'), WalletError
  (the certificates or key cannot be loaded or used).
The key may be encrypted: its password is read from the WALLET_KEY_PASSWORD environment variable.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image

from app import storage
from app.config import settings
from app.util import json_loads

log = logging.getLogger(__name__)

REQUIRED_FILES = ("cert.pem", "key.pem", "wwdr.pem")
MIME = "application/vnd.apple.pkpass"
_FORMATS = {"QRCode": "PKBarcodeFormatQR", "PDF417": "PKBarcodeFormatPDF417", "CompactPDF417": "PKBarcodeFormatPDF417",
            "Aztec": "PKBarcodeFormatAztec", "Code128": "PKBarcodeFormatCode128"}
_LEGACY = ("PKBarcodeFormatQR", "PKBarcodeFormatPDF417", "PKBarcodeFormatAztec")
_IMAGES_COMMON = {"icon.png": (29, 29), "icon@2x.png": (58, 58), "logo.png": (160, 50), "logo@2x.png": (320, 100)}
_IMAGES_EVENT = {"thumbnail.png": (90, 90), "thumbnail@2x.png": (180, 180)}


class WalletError(Exception):
    pass


class WalletNotConfigured(WalletError):
    def __init__(self, missing: list[str]):
        super().__init__("wallet_not_configured: " + ", ".join(missing))
        self.missing = missing


class NotAPass(WalletError):
    pass


def configured() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not settings.wallet_pass_type_id:
        missing.append("WALLET_PASS_TYPE_ID")
    if not settings.wallet_team_id:
        missing.append("WALLET_TEAM_ID")
    d = storage.wallet_dir()
    for name in REQUIRED_FILES:
        if not (d / name).is_file():
            missing.append(name)
    return not missing, missing


# ------------------------------------------------------------------ pass.json

def _field(key: str, label: str, value: Any) -> dict[str, str]:
    return {"key": key, "label": label.upper() if len(label) <= 14 else label, "value": str(value)}


def _values(doc_row: sqlite3.Row) -> dict[str, tuple[str, str]]:
    """key -> (label, value) from the documents.fields JSON, non-empty only."""
    out: dict[str, tuple[str, str]] = {}
    for key, f in (json_loads(doc_row["fields"], {}) or {}).items():
        if isinstance(f, dict):
            val, label = str(f.get("value", "")).strip(), str(f.get("label", key))
        else:
            val, label = str(f).strip(), key
        if val:
            out[str(key)] = (label, val)
    return out


def _relevant_date(day: str | None, time: str | None) -> str | None:
    if not day or not time or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or not re.fullmatch(r"\d{2}:\d{2}", time):
        return None
    try:
        zone = ZoneInfo(settings.local_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone = ZoneInfo("UTC")
    try:
        dt = datetime.fromisoformat(f"{day}T{time}").replace(tzinfo=zone)
    except ValueError:
        return None
    return dt.isoformat(timespec="minutes")


def _barcode(doc_row: sqlite3.Row, vals: dict[str, tuple[str, str]]) -> dict[str, str] | None:
    message = doc_row["barcode"]
    fmt = _FORMATS.get(doc_row["barcode_format"] or "", "PKBarcodeFormatQR")
    if not message:
        for key in ("booking_id", "pnr"):
            if key in vals:
                message, fmt = vals[key][1], "PKBarcodeFormatQR"
                break
    if not message:
        return None
    out = {"format": fmt, "message": str(message), "messageEncoding": "iso-8859-1"}
    alt = (vals.get("booking_id") or vals.get("pnr") or ("", ""))[1]
    if alt:
        out["altText"] = alt
    return out


def _layout(doc_type: str, vals: dict[str, tuple[str, str]]) -> tuple[str, dict[str, Any], str | None]:
    """(style key, structure, time for relevantDate)."""
    used: set[str] = set()

    def pick(keys: list[str], labels: dict[str, str] | None = None) -> list[dict[str, str]]:
        out = []
        for k in keys:
            if k in vals and k not in used:
                used.add(k)
                label = (labels or {}).get(k, vals[k][0])
                out.append(_field(k, label, vals[k][1]))
        return out

    if doc_type == "boarding_pass":
        struct: dict[str, Any] = {
            "transitType": "PKTransitTypeAir",
            "primaryFields": pick(["from", "to"]),
            "secondaryFields": pick(["passenger", "flight"]),
            "auxiliaryFields": pick(["date", "boarding_time", "gate", "seat", "zone"]),
            "headerFields": pick(["cabin"]),
        }
        time = vals.get("boarding_time", vals.get("departure_time", ("", "")))[1] or None
        style = "boardingPass"
    else:
        struct = {
            "primaryFields": pick(["event"]),
            "secondaryFields": pick(["venue", "date"]),
            "auxiliaryFields": pick(["time", "section", "row", "seat", "gate"]),
            "headerFields": pick(["quantity"]),
        }
        time = vals.get("time", ("", ""))[1] or None
        style = "eventTicket"
    struct["backFields"] = [_field(k, lab, v) for k, (lab, v) in vals.items() if k not in used]
    struct = {k: v for k, v in struct.items() if v or k in ("primaryFields", "transitType")}
    return style, struct, time


def pass_json(item_row: sqlite3.Row, doc_row: sqlite3.Row, colours: dict[str, str]) -> dict[str, Any]:
    vals = _values(doc_row)
    style, struct, time = _layout(doc_row["doc_type"], vals)
    from app.documents import parsers

    label = parsers.get(doc_row["doc_type"]).label
    data: dict[str, Any] = {
        "formatVersion": 1,
        "passTypeIdentifier": settings.wallet_pass_type_id,
        "teamIdentifier": settings.wallet_team_id,
        "serialNumber": item_row["uid"],
        "organizationName": settings.wallet_org_name or settings.site_title,
        "description": (doc_row["title"] or label)[:200],
        "logoText": (vals.get("event") or vals.get("airline") or (label, label))[1][:40],
        **colours,
        style: struct,
    }
    bc = _barcode(doc_row, vals)
    if bc:
        data["barcodes"] = [bc]
        if bc["format"] in _LEGACY:
            data["barcode"] = dict(bc)
    day = doc_row["doc_date"] or vals.get("date", ("", ""))[1] or None
    rel = _relevant_date(day, time)
    if rel:
        data["relevantDate"] = rel
    return data


# ------------------------------------------------------------------ images and colours

def _thumb(item_row: sqlite3.Row) -> Image.Image | None:
    p = storage.thumb_path(item_row["uid"], int(item_row["version"]))
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            return im.convert("RGB")
    except Exception:
        log.warning("cannot read thumbnail %s", storage.rel(p), exc_info=True)
        return None


def dominant_colour(img: Image.Image | None) -> tuple[int, int, int]:
    """The most common colour of the image after quantising to a small palette, darkened when very
    light so white text stays readable."""
    if img is None:
        return (40, 44, 52)
    small = img.convert("RGB").resize((64, 64))
    q = small.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = sorted(q.getcolors() or [], reverse=True)
    if not counts:
        return (40, 44, 52)
    idx = counts[0][1]
    r, g, b = pal[idx * 3: idx * 3 + 3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 200:
        r, g, b = (int(c * 0.55) for c in (r, g, b))
    return int(r), int(g), int(b)


def colours_for(img: Image.Image | None) -> dict[str, str]:
    r, g, b = dominant_colour(img)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    fg = (255, 255, 255) if lum < 150 else (17, 17, 17)
    label = (220, 220, 220) if lum < 150 else (60, 60, 60)
    return {"backgroundColor": f"rgb({r}, {g}, {b})", "foregroundColor": "rgb({}, {}, {})".format(*fg),
            "labelColor": "rgb({}, {}, {})".format(*label)}


def _png(img: Image.Image | None, size: tuple[int, int], bg: tuple[int, int, int], *, fit: bool) -> bytes:
    w, h = size
    if img is None:
        canvas = Image.new("RGB", size, bg)
    elif fit:
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        im = img.copy()
        im.thumbnail(size, Image.LANCZOS)
        canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    else:
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        canvas = img.crop((left, top, left + side, top + side)).resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    canvas.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ------------------------------------------------------------------ signing

def _load_cert(path: Path):
    from cryptography import x509

    data = path.read_bytes()
    try:
        return x509.load_pem_x509_certificate(data)
    except ValueError:
        return x509.load_der_x509_certificate(data)


def sign_manifest(manifest: bytes, wallet_dir: Path | None = None) -> bytes:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.serialization import pkcs7

    d = wallet_dir or storage.wallet_dir()
    try:
        cert = _load_cert(d / "cert.pem")
        wwdr = _load_cert(d / "wwdr.pem")
        pw = os.environ.get("WALLET_KEY_PASSWORD")
        key = serialization.load_pem_private_key((d / "key.pem").read_bytes(), password=pw.encode() if pw else None)
        return (
            pkcs7.PKCS7SignatureBuilder()
            .set_data(manifest)
            .add_signer(cert, key, hashes.SHA256())
            .add_certificate(wwdr)
            .sign(serialization.Encoding.DER, [pkcs7.PKCS7Options.DetachedSignature, pkcs7.PKCS7Options.Binary])
        )
    except (ValueError, TypeError, OSError) as e:
        raise WalletError(f"cannot sign the pass: {e}") from e


def build_pass(conn: sqlite3.Connection, item_row: sqlite3.Row, doc_row: sqlite3.Row) -> bytes:
    ok, missing = configured()
    if not ok:
        raise WalletNotConfigured(missing)
    if doc_row["doc_group"] != "pass":
        raise NotAPass(doc_row["doc_type"])
    thumb = _thumb(item_row)
    colours = colours_for(thumb)
    bg = tuple(int(x) for x in re.findall(r"\d+", colours["backgroundColor"]))
    files: dict[str, bytes] = {
        "pass.json": json.dumps(pass_json(item_row, doc_row, colours), ensure_ascii=False, indent=1).encode("utf-8"),
    }
    images = dict(_IMAGES_COMMON)
    if doc_row["doc_type"] != "boarding_pass":
        images.update(_IMAGES_EVENT)
    for name, size in images.items():
        files[name] = _png(thumb, size, bg, fit=name.startswith("logo"))  # type: ignore[arg-type]
    manifest = json.dumps({n: hashlib.sha1(b).hexdigest() for n, b in files.items()}, indent=1).encode()
    signature = sign_manifest(manifest)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("manifest.json", manifest)
        z.writestr("signature", signature)
    return buf.getvalue()
