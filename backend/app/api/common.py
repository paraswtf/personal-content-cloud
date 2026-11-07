"""Query helpers and row -> schema converters shared by the API routers.

Visibility: every item query goes through `visible_sql()`: live (not in the trash), not NSFW unless
the caller may see the vault, and mis-clicks and duplicate copies filtered as asked.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from app.config import settings
from app.schemas import (
    BodyOut,
    DocumentField,
    DocumentOut,
    FaceOut,
    GuessOut,
    ItemCard,
    ItemUrls,
    PersonOut,
    PersonRef,
    UploadOut,
)
from app.urls import (
    body_url,
    doc_photo_url,
    download_url,
    face_url,
    original_url,
    page_url,
    playable_video_url,
    preview_small_url,
    preview_url,
    thumb_url,
)
from app.util import json_loads

UPLOAD_MESSAGES = {
    "uploading": "Uploading...",
    "queued": "Waiting to be processed...",
    "processing": "Processing...",
    "stored": "Added to the library.",
    "duplicate": "Already in the library (identical file), skipped.",
    "rejected": "This file could not be read.",
    "error": "Processing failed.",
    "cancelled": "Cancelled.",
}

# Columns every item query must select (alias `i`) for item_card(). dup_count comes from a scalar subquery.
ITEM_CARD_COLUMNS = (
    "i.id, i.uid, i.kind, i.version, i.original_filename, i.width, i.height, i.duration_s, i.taken_at, i.created_at, "
    "i.category, i.nsfw, i.misclick, i.misclick_reasons, i.favorite, i.dup_group_id, i.dup_role, i.page_count, "
    "i.preview_status, i.video_codec, i.ext, i.bitrate, "
    "(SELECT d.doc_type FROM documents d WHERE d.item_id = i.id) AS doc_type, "
    "(SELECT g.uid FROM items g WHERE g.id = i.dup_group_id) AS dup_group_uid, "
    "(SELECT COUNT(*) FROM items c WHERE c.dup_group_id = i.dup_group_id AND c.id != i.dup_group_id "
    " AND c.deleted_at IS NULL) AS dup_count"
)

DOC_TYPE_LABELS = {
    "aadhaar": "Aadhaar card", "pan": "PAN card", "passport": "Passport", "driving_licence": "Driving licence",
    "voter_id": "Voter ID", "vehicle_rc": "Vehicle registration", "bank_statement": "Bank statement",
    "invoice": "Invoice", "receipt": "Receipt", "salary_slip": "Salary slip", "marksheet": "Marksheet",
    "degree": "Degree", "certificate": "Certificate", "offer_letter": "Offer letter",
    "insurance_policy": "Insurance policy", "prescription": "Prescription", "lab_report": "Lab report",
    "vaccination": "Vaccination certificate", "event_pass": "Event pass", "boarding_pass": "Boarding pass",
    "other": "Document",
}
DOC_GROUP_OF = {
    "aadhaar": "id", "pan": "id", "passport": "id", "driving_licence": "id", "voter_id": "id", "vehicle_rc": "id",
    "bank_statement": "money", "invoice": "money", "receipt": "money", "salary_slip": "money",
    "marksheet": "education_work", "degree": "education_work", "certificate": "education_work",
    "offer_letter": "education_work",
    "insurance_policy": "health", "prescription": "health", "lab_report": "health", "vaccination": "health",
    "event_pass": "pass", "boarding_pass": "pass", "other": "other",
}


def visible_sql(*, vault: bool = False, misclicks: str = "exclude", dups: str = "collapse",
                alias: str = "i") -> str:
    """WHERE fragment (no leading AND) selecting the items a listing may show.
    vault: True shows only NSFW items (the vault), False hides them.
    misclicks: exclude | include | only. dups: collapse (best of each group only) | all."""
    parts = [f"{alias}.deleted_at IS NULL", f"{alias}.nsfw = {1 if vault else 0}"]
    if misclicks == "exclude":
        parts.append(f"{alias}.misclick = 0")
    elif misclicks == "only":
        parts.append(f"{alias}.misclick = 1")
    if dups == "collapse":
        parts.append(f"({alias}.dup_group_id IS NULL OR {alias}.dup_role = 'best')")
    return " AND ".join(parts)


def stream_friendly_row(row: sqlite3.Row) -> bool:
    try:
        from app.media.video import stream_friendly

        return bool(stream_friendly(row["video_codec"], row["ext"], row["width"], row["height"], row["bitrate"]))
    except Exception:
        return False


def item_urls(row: sqlite3.Row, *, full: bool = False) -> ItemUrls:
    uid, v = row["uid"], int(row["version"])
    urls = ItemUrls(thumb=thumb_url(uid, v), preview=preview_url(uid, v), preview_small=preview_small_url(uid, v))
    if full:
        urls.original = original_url(uid, v)
        urls.download = download_url(uid, v)
        if row["kind"] == "video":
            urls.video = playable_video_url(uid, v, row["preview_status"], stream_friendly_row(row))
        if row["kind"] in ("pdf", "document") and row["page_count"]:
            urls.pages = [page_url(uid, v, n) for n in range(1, int(row["page_count"]) + 1)]
    return urls


def item_card(row: sqlite3.Row) -> ItemCard:
    """row needs ITEM_CARD_COLUMNS."""
    keys = row.keys()
    return ItemCard(
        uid=row["uid"],
        kind=row["kind"],
        version=row["version"],
        filename=row["original_filename"],
        width=row["width"],
        height=row["height"],
        duration_s=row["duration_s"],
        taken_at=row["taken_at"],
        created_at=row["created_at"],
        category=row["category"],
        nsfw=bool(row["nsfw"]),
        misclick=bool(row["misclick"]),
        misclick_reasons=json_loads(row["misclick_reasons"]) if row["misclick"] else None,
        favorite=bool(row["favorite"]),
        dup_group=row["dup_group_uid"] if "dup_group_uid" in keys else None,
        dup_count=int(row["dup_count"] or 0) if "dup_count" in keys else 0,
        page_count=row["page_count"],
        doc_type=row["doc_type"] if "doc_type" in keys else None,
        urls=item_urls(row),
    )


def display_name(person_id: int, name: str | None) -> str:
    return name if name else f"Unnamed #{person_id}"


def person_ref(conn: sqlite3.Connection, person_id: int | None, name: str | None = None,
               cover_face_id: int | None = None) -> PersonRef | None:
    """PersonRef with its cover; looks the person up when only the id is known."""
    if person_id is None:
        return None
    if name is None and cover_face_id is None:
        row = conn.execute("SELECT id, name, cover_face_id FROM persons WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            return None
        name, cover_face_id = row["name"], row["cover_face_id"]
    cover = cover_face_id or auto_cover(conn, person_id)
    return PersonRef(id=person_id, name=name, cover_url=face_url(cover) if cover else None)


def auto_cover(conn: sqlite3.Connection, person_id: int) -> int | None:
    row = conn.execute(
        "SELECT f.id FROM faces f JOIN items i ON i.id = f.item_id AND i.deleted_at IS NULL AND i.nsfw = 0 "
        "WHERE f.person_id = ? ORDER BY f.quality DESC, f.id LIMIT 1",
        (person_id,),
    ).fetchone()
    return int(row["id"]) if row else None


def person_out(conn: sqlite3.Connection, row: sqlite3.Row) -> PersonOut:
    """row from persons (id, name, cover_face_id, created_at)."""
    pid = int(row["id"])
    face_count = conn.execute(
        "SELECT COUNT(*) FROM faces f JOIN items i ON i.id = f.item_id AND i.deleted_at IS NULL WHERE f.person_id = ?",
        (pid,),
    ).fetchone()[0]
    item_count = conn.execute(
        "SELECT COUNT(DISTINCT item_id) FROM ("
        "SELECT f.item_id FROM faces f WHERE f.person_id = ? UNION ALL "
        "SELECT b.item_id FROM bodies b WHERE b.person_id = ? UNION ALL "
        "SELECT t.item_id FROM item_tags t WHERE t.person_id = ?) x "
        "JOIN items i ON i.id = x.item_id AND i.deleted_at IS NULL",
        (pid, pid, pid),
    ).fetchone()[0]
    guess_count = conn.execute(
        "SELECT (SELECT COUNT(*) FROM faces WHERE guess_person_id = ? AND person_id IS NULL) + "
        "(SELECT COUNT(*) FROM bodies WHERE guess_person_id = ? AND person_id IS NULL)",
        (pid, pid),
    ).fetchone()[0]
    cover = row["cover_face_id"] or auto_cover(conn, pid)
    return PersonOut(
        id=pid, name=row["name"], cover_url=face_url(cover) if cover else None, face_count=face_count,
        item_count=item_count, guess_count=guess_count, created_at=row["created_at"],
    )


def guess_out(conn: sqlite3.Connection, row: sqlite3.Row) -> GuessOut | None:
    """row needs guess_person_id, guess_score, guess_evidence, person_id."""
    if row["person_id"] is not None or row["guess_person_id"] is None:
        return None
    ref = person_ref(conn, int(row["guess_person_id"]))
    if ref is None:
        return None
    return GuessOut(person=ref, score=float(row["guess_score"] or 0), evidence=json_loads(row["guess_evidence"], {}) or {})


def face_out(conn: sqlite3.Connection, row: sqlite3.Row) -> FaceOut:
    """row from faces."""
    return FaceOut(
        id=row["id"], x=row["x"], y=row["y"], w=row["w"], h=row["h"], det_score=row["det_score"],
        size_px=row["size_px"], quality=row["quality"], frame_time_s=row["frame_time_s"],
        spans=json_loads(row["spans"]), person=person_ref(conn, row["person_id"]), assigned_by=row["assigned_by"],
        guess=guess_out(conn, row), url=face_url(int(row["id"])),
    )


def body_out(conn: sqlite3.Connection, row: sqlite3.Row) -> BodyOut:
    """row from bodies."""
    return BodyOut(
        id=row["id"], x=row["x"], y=row["y"], w=row["w"], h=row["h"], det_score=row["det_score"],
        face_id=row["face_id"], person=person_ref(conn, row["person_id"]), assigned_by=row["assigned_by"],
        guess=guess_out(conn, row), frame_time_s=row["frame_time_s"], url=body_url(int(row["id"])),
    )


def document_out(row: sqlite3.Row, item_row: sqlite3.Row, *, wallet_configured: bool = False) -> DocumentOut:
    """row from documents; item_row needs uid, version."""
    raw = json_loads(row["fields"], {}) or {}
    fields: dict[str, DocumentField] = {}
    order: list[str] = []
    for key, val in raw.items():
        if isinstance(val, dict):
            fields[key] = DocumentField(value=str(val.get("value", "")), label=str(val.get("label", key)),
                                        confidence=float(val.get("confidence", 1.0)))
        else:
            fields[key] = DocumentField(value=str(val), label=key, confidence=1.0)
        order.append(key)
    doc_type = row["doc_type"]
    return DocumentOut(
        doc_type=doc_type, doc_group=row["doc_group"], label=DOC_TYPE_LABELS.get(doc_type, doc_type.replace("_", " ").title()),
        confidence=float(row["confidence"]), fields=fields, field_order=order, title=row["title"], doc_date=row["doc_date"],
        issuer=row["issuer"], person_name=row["person_name"], barcode=row["barcode"], barcode_format=row["barcode_format"],
        photo_url=doc_photo_url(item_row["uid"], int(item_row["version"])) if row["has_photo"] else None,
        source=row["source"], wallet_available=bool(wallet_configured and row["doc_group"] == "pass" and row["barcode"]),
        updated_at=row["updated_at"],
    )


def people_filter_sql(person_ids: list[int], alias: str = "i") -> tuple[str, list[int]]:
    """SQL condition: the item contains ALL the given people (faces, bodies or tags)."""
    ids = sorted({int(p) for p in person_ids})
    ph = ",".join("?" * len(ids))
    sql = (
        f"{alias}.id IN (SELECT item_id FROM ("
        f"SELECT item_id, person_id FROM faces WHERE person_id IN ({ph}) UNION "
        f"SELECT item_id, person_id FROM bodies WHERE person_id IN ({ph}) UNION "
        f"SELECT item_id, person_id FROM item_tags WHERE person_id IN ({ph})"
        ") GROUP BY item_id HAVING COUNT(DISTINCT person_id) = ?)"
    )
    return sql, [*ids, *ids, *ids, len(ids)]


def valid_cursor_scalar(value: object) -> bool:
    """A keyset cursor element that SQLite can bind: str, float, None, or a 64-bit int."""
    if value is None or isinstance(value, (str, float)):
        return True
    return isinstance(value, int) and not isinstance(value, bool) and -(2 ** 63) <= value < 2 ** 63


UPLOAD_SELECT = (
    "SELECT u.*, i.uid AS item_uid, i.version AS item_version, i.deleted_at AS item_deleted FROM uploads u "
    "LEFT JOIN items i ON i.id = u.item_id"
)


def upload_out(row: sqlite3.Row) -> UploadOut:
    """row from UPLOAD_SELECT."""
    item_uid = None if row["item_deleted"] else row["item_uid"]
    return UploadOut(
        id=row["uid"],
        filename=row["original_filename"],
        size=row["size_bytes"],
        received_bytes=row["received_bytes"],
        status=row["status"],
        message=row["message"] or UPLOAD_MESSAGES.get(row["status"]),
        chunk_size=settings.chunk_size,
        item_uid=item_uid,
        item_thumb_url=thumb_url(item_uid, row["item_version"]) if item_uid else None,
        created_at=row["created_at"],
    )


def json_field(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return json_loads(row[key], default)
    except (IndexError, KeyError):
        return default
