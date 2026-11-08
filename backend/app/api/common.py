"""Query helpers and row -> schema converters shared by the API routers.

Visibility: every item query goes through `visible_sql()`: live (not in the trash), not NSFW unless
the caller may see the vault, and mis-clicks and duplicate copies filtered as asked.
"""
from __future__ import annotations

import sqlite3

import numpy as np
from collections.abc import Iterable
from typing import Any

from app.config import settings
from app.schemas import (
    AnimalOut,
    BodyOut,
    DocumentField,
    DocumentOut,
    FaceOut,
    GuessOut,
    ItemCard,
    ItemUrls,
    LiveVideoOut,
    PersonOut,
    PersonRef,
    UploadOut,
)
from app.urls import (
    animal_url,
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

# Columns every item query must select (alias `i`) for item_card(). dup_count comes from a scalar subquery:
# the other live members of the item's duplicate group on the same side of the vault (none without a group).
# dup_group_uid (the best's uid) is NULL when the best is on the other side, so no vault uid reaches the library.
ITEM_CARD_COLUMNS = (
    "i.id, i.uid, i.kind, i.version, i.original_filename, i.width, i.height, i.duration_s, i.taken_at, i.created_at, "
    "i.category, i.nsfw, i.misclick, i.misclick_reasons, i.favorite, i.dup_group_id, i.dup_role, i.page_count, "
    "i.received_via, i.forward, i.forward_reasons, "
    "i.preview_status, i.video_codec, i.ext, i.bitrate, i.live, i.live_video_item_id, "
    "(SELECT c.uid FROM items c WHERE c.id = i.live_video_item_id) AS live_uid, "
    "(SELECT d.doc_type FROM documents d WHERE d.item_id = i.id) AS doc_type, "
    "CASE WHEN i.dup_group_id IS NULL THEN NULL "
    "ELSE (SELECT g.uid FROM items g WHERE g.id = i.dup_group_id AND g.nsfw = i.nsfw) END AS dup_group_uid, "
    "CASE WHEN i.dup_group_id IS NULL THEN 0 "
    "ELSE (SELECT COUNT(*) FROM items c WHERE c.dup_group_id = i.dup_group_id AND c.id != i.id "
    " AND c.deleted_at IS NULL AND c.nsfw = i.nsfw) END AS dup_count"
)

DOC_TYPE_LABELS = {
    "aadhaar": "Aadhaar card", "pan": "PAN card", "passport": "Passport", "driving_licence": "Driving licence",
    "voter_id": "Voter ID", "vehicle_rc": "Vehicle registration", "id_card": "ID card", "bank_statement": "Bank statement",
    "invoice": "Invoice", "receipt": "Receipt", "salary_slip": "Salary slip", "traffic_challan": "Traffic challan",
    "parking_receipt": "Parking receipt", "marksheet": "Marksheet",
    "degree": "Degree", "certificate": "Certificate", "offer_letter": "Offer letter",
    "insurance_policy": "Insurance policy", "prescription": "Prescription", "lab_report": "Lab report",
    "vaccination": "Vaccination certificate", "event_pass": "Event pass", "boarding_pass": "Boarding pass",
    "other": "Document",
}
DOC_GROUP_OF = {
    "aadhaar": "id", "pan": "id", "passport": "id", "driving_licence": "id", "voter_id": "id", "vehicle_rc": "id",
    "id_card": "id",
    "bank_statement": "money", "invoice": "money", "receipt": "money", "salary_slip": "money",
    "traffic_challan": "money", "parking_receipt": "money",
    "marksheet": "education_work", "degree": "education_work", "certificate": "education_work",
    "offer_letter": "education_work",
    "insurance_policy": "health", "prescription": "health", "lab_report": "health", "vaccination": "health",
    "event_pass": "pass", "boarding_pass": "pass", "other": "other",
}


def visible_sql(*, vault: bool = False, misclicks: str = "exclude", dups: str = "collapse",
                alias: str = "i", forwards: str = "exclude") -> str:
    """WHERE fragment (no leading AND) selecting the items a listing may show.
    vault: True shows only NSFW items (the vault), False hides them.
    misclicks: exclude | include | only. forwards: exclude | include | only (section 29). dups: collapse | all.
    collapse works within the visible side of the vault: an item of a duplicate group shows when it is
    the group's best, or when the best is not a live item on the same side (a normal copy under a vault
    best shows in the library, a vault copy under a normal best shows in the vault)."""
    parts = [f"{alias}.deleted_at IS NULL", f"{alias}.nsfw = {1 if vault else 0}", f"{alias}.companion_of IS NULL"]
    if misclicks == "exclude":
        parts.append(f"{alias}.misclick = 0")
    elif misclicks == "only":
        parts.append(f"{alias}.misclick = 1")
    if forwards == "exclude":
        parts.append(f"{alias}.forward = 0")
    elif forwards == "only":
        parts.append(f"{alias}.forward = 1")
    if dups == "collapse":
        parts.append(
            f"({alias}.dup_group_id IS NULL OR {alias}.dup_role = 'best' OR NOT EXISTS ("
            f"SELECT 1 FROM items dup_best WHERE dup_best.id = {alias}.dup_group_id AND dup_best.dup_role = 'best' "
            f"AND dup_best.deleted_at IS NULL AND dup_best.nsfw = {alias}.nsfw))"
        )
    return " AND ".join(parts)


def page_sql(where: list[str], sort_expr: str, after: list[Any] | None) -> tuple[str, list[Any]]:
    """(SQL, extra params) of one keyset page: the inner query picks the page's ids in index order
    (items_order / items_added serve the default and added sorts with `nsfw = 0|1` and `deleted_at
    IS NULL` in `where`), and the card columns are computed only for those rows. The params of
    `where` come first, then the returned ones, then the limit."""
    page_where, extra = list(where), []
    if after is not None:
        # The plain bound lets SQLite seek the index; the row value keeps the order exact on ties.
        page_where.append(f"{sort_expr} <= ? AND ({sort_expr}, i.id) < (?, ?)")
        extra = [after[0], after[0], after[1]]
    inner = (f"SELECT i.id, {sort_expr} AS sort_key FROM items i "
             f"WHERE {' AND '.join(page_where) if page_where else '1'} "
             f"ORDER BY {sort_expr} DESC, i.id DESC LIMIT ?")
    sql = (f"SELECT {ITEM_CARD_COLUMNS}, p.sort_key FROM ({inner}) p JOIN items i ON i.id = p.id "
           "ORDER BY p.sort_key DESC, p.id DESC")
    return sql, extra


def stream_friendly_row(row: sqlite3.Row) -> bool:
    try:
        from app.media.video import stream_friendly

        return bool(stream_friendly(row["video_codec"], row["ext"], row["width"], row["height"], row["bitrate"]))
    except Exception:
        return False


def url_version(version: int, nsfw: object) -> str:
    """The ?v= cache key of an item's file URLs: the version, with an 'n' for vault items. Flagging an
    item NSFW changes its URLs, so a copy cached under the old URL (immutable) is never shown again."""
    return f"{int(version)}n" if nsfw else str(int(version))


def _row_nsfw(row: sqlite3.Row) -> bool:
    return bool(row["nsfw"]) if "nsfw" in row.keys() else False


def item_urls(row: sqlite3.Row, *, full: bool = False) -> ItemUrls:
    uid, v = row["uid"], url_version(row["version"], _row_nsfw(row))
    urls = ItemUrls(thumb=thumb_url(uid, v), preview=preview_url(uid, v), preview_small=preview_small_url(uid, v))
    keys = row.keys()
    if "live" in keys and row["live"]:
        # Section 20: the clip of a Live Photo (the companion's video route adapts to its preview state).
        live_uid = row["live_uid"] if "live_uid" in keys else None
        urls.live = f"/files/video/{live_uid}?v={v}" if live_uid else f"/files/live/{uid}?v={v}"
    if full:
        urls.original = original_url(uid, v)
        urls.download = download_url(uid, v)
        if row["kind"] == "video":
            urls.video = playable_video_url(uid, v, row["preview_status"], stream_friendly_row(row))
        if row["kind"] in ("pdf", "document") and row["page_count"]:
            from app.media.documents import PDF_MAX_PAGES  # items probed before the cap keep a larger count

            urls.pages = [page_url(uid, v, n) for n in range(1, min(int(row["page_count"]), PDF_MAX_PAGES) + 1)]
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
        received_via=row["received_via"] if "received_via" in keys else None,
        forward=bool(row["forward"]) if "forward" in keys else False,
        forward_reasons=json_loads(row["forward_reasons"]) if "forward" in keys and row["forward"] else None,
        favorite=bool(row["favorite"]),
        dup_group=row["dup_group_uid"] if "dup_group_uid" in keys else None,
        dup_count=int(row["dup_count"] or 0) if "dup_count" in keys else 0,
        page_count=row["page_count"],
        doc_type=row["doc_type"] if "doc_type" in keys else None,
        live=bool(row["live"]) if "live" in keys else False,
        urls=item_urls(row),
    )


def display_name(person_id: int, name: str | None) -> str:
    return name if name else f"Unnamed #{person_id}"


# A face may serve as a cover only while its item is live and outside the vault (as people._covers).
COVER_ITEM_OK = "ci.deleted_at IS NULL AND ci.nsfw = 0"


def cover_visible(conn: sqlite3.Connection, face_id: int | None) -> bool:
    """True when the face exists and its item is live and not in the vault."""
    if face_id is None:
        return False
    return conn.execute(
        f"SELECT 1 FROM faces f JOIN items ci ON ci.id = f.item_id AND {COVER_ITEM_OK} WHERE f.id = ?", (face_id,)
    ).fetchone() is not None


def person_ref(conn: sqlite3.Connection, person_id: int | None, name: str | None = None,
               cover_face_id: int | None = None) -> PersonRef | None:
    """PersonRef with its cover; looks the person up when only the id is known. A chosen cover from
    a trashed or vault item is skipped for the automatic one. For many people use person_refs()."""
    if person_id is None:
        return None
    if name is None and cover_face_id is None:
        row = conn.execute("SELECT id, name, cover_face_id FROM persons WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            return None
        name, cover_face_id = row["name"], row["cover_face_id"]
    cover = cover_face_id if cover_visible(conn, cover_face_id) else auto_cover(conn, person_id)
    return PersonRef(id=person_id, name=name, cover_url=face_url(cover) if cover else None)


def person_refs(conn: sqlite3.Connection, person_ids: Iterable[int | None]) -> dict[int, PersonRef]:
    """{person_id: PersonRef} for many people with two queries per 500 ids (names and valid chosen
    covers, then one ROW_NUMBER query for the automatic covers). Unknown ids are left out."""
    ids = sorted({int(p) for p in person_ids if p is not None})
    out: dict[int, PersonRef] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT p.id, p.name, p.kind, ci.id IS NOT NULL AS cover_ok, p.cover_face_id FROM persons p "
            f"LEFT JOIN faces f ON f.id = p.cover_face_id LEFT JOIN items ci ON ci.id = f.item_id AND {COVER_ITEM_OK} "
            f"WHERE p.id IN ({ph})",
            chunk,
        ).fetchall()
        covers = {int(r["id"]): int(r["cover_face_id"]) for r in rows if r["cover_ok"]}
        need = [int(r["id"]) for r in rows if int(r["id"]) not in covers and r["kind"] != "pet"]
        if need:
            for r in conn.execute(
                "SELECT pid, fid FROM ("
                "  SELECT f.person_id AS pid, f.id AS fid, "
                "         ROW_NUMBER() OVER (PARTITION BY f.person_id ORDER BY f.quality DESC, f.id) AS rn "
                f"  FROM faces f JOIN items ci ON ci.id = f.item_id AND {COVER_ITEM_OK} "
                f"  WHERE f.person_id IN ({','.join('?' * len(need))})"
                ") WHERE rn = 1",
                need,
            ):
                covers[int(r["pid"])] = int(r["fid"])
        pets = [int(r["id"]) for r in rows if r["kind"] == "pet"]
        pet_covers = pet_cover_ids(conn, pets) if pets else {}
        for r in rows:
            pid = int(r["id"])
            if r["kind"] == "pet":
                aid = pet_covers.get(pid)
                out[pid] = PersonRef(id=pid, name=r["name"], cover_url=animal_url(aid) if aid else None, kind="pet")
                continue
            cover = covers.get(pid)
            out[pid] = PersonRef(id=pid, name=r["name"], cover_url=face_url(cover) if cover else None)
    return out


def pet_cover_ids(conn: sqlite3.Connection, pet_ids: list[int]) -> dict[int, int]:
    """{pet id: its best animal crop in a visible item} (section 22)."""
    out: dict[int, int] = {}
    ids = sorted({int(p) for p in pet_ids})
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for r in conn.execute(
            "SELECT pid, aid FROM ("
            "  SELECT a.person_id AS pid, a.id AS aid, "
            "         ROW_NUMBER() OVER (PARTITION BY a.person_id ORDER BY a.assigned_by = 'user' DESC, a.det_score DESC, a.id) AS rn "
            f"  FROM animals a JOIN items ci ON ci.id = a.item_id AND {COVER_ITEM_OK} "
            f"  WHERE a.person_id IN ({','.join('?' * len(chunk))})"
            ") WHERE rn = 1",
            chunk,
        ):
            out[int(r["pid"])] = int(r["aid"])
    return out


def box_person_ids(rows: Iterable[sqlite3.Row]) -> set[int]:
    """Person ids (assigned and guessed) of face or body rows, for person_refs()."""
    out: set[int] = set()
    for r in rows:
        for key in ("person_id", "guess_person_id"):
            if r[key] is not None:
                out.add(int(r[key]))
    return out


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
    cover = row["cover_face_id"] if cover_visible(conn, row["cover_face_id"]) else auto_cover(conn, pid)
    return PersonOut(
        id=pid, name=row["name"], cover_url=face_url(cover) if cover else None, cover_face_id=row["cover_face_id"],
        face_count=face_count,
        item_count=item_count, guess_count=guess_count, created_at=row["created_at"],
    )


def _ref(conn: sqlite3.Connection, person_id: int | None, refs: dict[int, PersonRef] | None) -> PersonRef | None:
    if person_id is None:
        return None
    if refs is not None:
        return refs.get(int(person_id))
    return person_ref(conn, int(person_id))


def guess_out(conn: sqlite3.Connection, row: sqlite3.Row, refs: dict[int, PersonRef] | None = None) -> GuessOut | None:
    """row needs guess_person_id, guess_score, guess_evidence, person_id. refs: from person_refs()
    (covering box_person_ids of the rows), so a page of boxes looks people up once."""
    if row["person_id"] is not None or row["guess_person_id"] is None:
        return None
    ref = _ref(conn, row["guess_person_id"], refs)
    if ref is None:
        return None
    return GuessOut(person=ref, score=float(row["guess_score"] or 0), evidence=json_loads(row["guess_evidence"], {}) or {})


VIDEO_GROUP_SIM = 0.5  # section 10: two tracks of a video this alike are one person for whole-video editing


def _track_vec(blob: bytes | None) -> "np.ndarray | None":
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def video_groups(faces: list[sqlite3.Row], bodies: list[sqlite3.Row], animals: list[sqlite3.Row]) -> dict[tuple[str, int], int]:
    """{(kind, id): group} for the tracks of a video (section 10): tracks assigned to a person share that
    person's group across kinds; among the rest (unassigned and guessed), a track joins the group of the
    most similar earlier track of its kind at cosine >= VIDEO_GROUP_SIM (faces by face embedding, bodies by
    torso, animals by crop), else starts one. Groups are numbered in order of first appearance."""
    groups: dict[tuple[str, int], int] = {}
    person_group: dict[int, int] = {}
    counter = 0

    def fresh() -> int:
        nonlocal counter
        counter += 1
        return counter

    for kind, rows, col in (("face", faces, "embedding"), ("body", bodies, "torso_embedding"), ("animal", animals, "embedding")):
        reps: list[tuple[int, "np.ndarray"]] = []
        for r in sorted(rows, key=lambda r: (float(r["frame_time_s"] or 0.0), int(r["id"]))):
            pid = r["person_id"]
            if pid is not None:
                g = person_group.get(int(pid))
                if g is None:
                    g = person_group[int(pid)] = fresh()
            else:
                emb = _track_vec(r[col]) if col in r.keys() else None
                g = None
                if emb is not None:
                    best_sim = VIDEO_GROUP_SIM
                    for gid, e in reps:
                        if e.shape == emb.shape:
                            sim = float(e @ emb)
                            if sim >= best_sim:
                                g, best_sim = gid, sim
                    if g is None:
                        g = fresh()
                    reps.append((g, emb))
                else:
                    g = fresh()
            groups[(kind, int(r["id"]))] = g
    return groups


def face_out(conn: sqlite3.Connection, row: sqlite3.Row, refs: dict[int, PersonRef] | None = None) -> FaceOut:
    """row from faces. refs: as for guess_out."""
    return FaceOut(
        id=row["id"], x=row["x"], y=row["y"], w=row["w"], h=row["h"], det_score=row["det_score"],
        size_px=row["size_px"], quality=row["quality"], frame_time_s=row["frame_time_s"],
        spans=json_loads(row["spans"]), person=_ref(conn, row["person_id"], refs), assigned_by=row["assigned_by"],
        guess=guess_out(conn, row, refs), url=face_url(int(row["id"])),
    )


def body_out(conn: sqlite3.Connection, row: sqlite3.Row, refs: dict[int, PersonRef] | None = None) -> BodyOut:
    """row from bodies. refs: as for guess_out."""
    return BodyOut(
        id=row["id"], x=row["x"], y=row["y"], w=row["w"], h=row["h"], det_score=row["det_score"],
        face_id=row["face_id"], person=_ref(conn, row["person_id"], refs), assigned_by=row["assigned_by"],
        guess=guess_out(conn, row, refs), frame_time_s=row["frame_time_s"], url=body_url(int(row["id"])),
    )


def animal_out(conn: sqlite3.Connection, row: sqlite3.Row, refs: dict[int, PersonRef] | None = None) -> AnimalOut:
    """row from animals (section 22)."""
    return AnimalOut(
        id=row["id"], x=row["x"], y=row["y"], w=row["w"], h=row["h"], det_score=row["det_score"],
        species=row["species"], frame_time_s=row["frame_time_s"], person=_ref(conn, row["person_id"], refs),
        assigned_by=row["assigned_by"], species_source=_species_source(row), source=_box_source(row),
        url=animal_url(int(row["id"])),
    )


def _box_source(row: sqlite3.Row) -> str:
    try:
        return "user" if row["source"] == "user" else "detector"
    except (IndexError, KeyError):
        return "detector"


def _species_source(row: sqlite3.Row) -> str:
    try:
        return "user" if row["species_source"] == "user" else "auto"
    except (IndexError, KeyError):
        return "auto"


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
        photo_url=doc_photo_url(item_row["uid"], url_version(item_row["version"], _row_nsfw(item_row)))  # type: ignore[arg-type]
        if row["has_photo"] else None,
        source=row["source"], wallet_available=bool(wallet_configured and row["doc_group"] == "pass" and row["barcode"]),
        updated_at=row["updated_at"],
    )


def people_filter_sql(person_ids: list[int], alias: str = "i") -> tuple[str, list[int]]:
    """SQL condition: the item contains ALL the given people (faces, bodies, animal boxes or tags)."""
    ids = sorted({int(p) for p in person_ids})
    ph = ",".join("?" * len(ids))
    sql = (
        f"{alias}.id IN (SELECT item_id FROM ("
        f"SELECT item_id, person_id FROM faces WHERE person_id IN ({ph}) UNION "
        f"SELECT item_id, person_id FROM bodies WHERE person_id IN ({ph}) UNION "
        f"SELECT item_id, person_id FROM animals WHERE person_id IN ({ph}) UNION "
        f"SELECT item_id, person_id FROM item_tags WHERE person_id IN ({ph})"
        ") GROUP BY item_id HAVING COUNT(DISTINCT person_id) = ?)"
    )
    return sql, [*ids, *ids, *ids, *ids, len(ids)]


def valid_cursor_scalar(value: object) -> bool:
    """A keyset cursor element that SQLite can bind: str, float, None, or a 64-bit int."""
    if value is None or isinstance(value, (str, float)):
        return True
    return isinstance(value, int) and not isinstance(value, bool) and -(2 ** 63) <= value < 2 ** 63


UPLOAD_SELECT = (
    "SELECT u.*, i.uid AS item_uid, i.version AS item_version, i.deleted_at AS item_deleted, "
    "i.nsfw AS item_nsfw FROM uploads u LEFT JOIN items i ON i.id = u.item_id"
)


def upload_out(row: sqlite3.Row, vault_open: bool = False) -> UploadOut:
    """row from UPLOAD_SELECT. vault_open: the session may see vault items. Otherwise an upload whose
    item is in the vault gets no item uid or thumb and the plain status message (nothing that says
    the item is private)."""
    item_uid = None if row["item_deleted"] else row["item_uid"]
    nsfw = bool(row["item_nsfw"]) and item_uid is not None
    message = row["message"] or UPLOAD_MESSAGES.get(row["status"])
    if nsfw and not vault_open:
        item_uid, message = None, UPLOAD_MESSAGES.get(row["status"])
    return UploadOut(
        id=row["uid"],
        filename=row["original_filename"],
        size=row["size_bytes"],
        received_bytes=row["received_bytes"],
        status=row["status"],
        message=message,
        chunk_size=settings.chunk_size,
        item_uid=item_uid,
        item_thumb_url=thumb_url(item_uid, url_version(row["item_version"], nsfw)) if item_uid else None,  # type: ignore[arg-type]
        item_nsfw=bool(nsfw and item_uid),
        created_at=row["created_at"],
    )


def json_field(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return json_loads(row[key], default)
    except (IndexError, KeyError):
        return default


def tags_filter_sql(names: list[str], alias: str = "i") -> tuple[str, list[Any]]:
    """SQL condition: the item carries ALL the named tags (user or suggested rows; section 19)."""
    from app.util import name_key

    keys = sorted({name_key(n) for n in names if name_key(n)})
    if not keys:
        return "1 = 1", []
    ph = ",".join("?" * len(keys))
    sql = (
        f"{alias}.id IN (SELECT ut.item_id FROM item_user_tags ut JOIN tags t ON t.id = ut.tag_id "
        f"WHERE t.name_key IN ({ph}) GROUP BY ut.item_id HAVING COUNT(DISTINCT t.id) = ?)"
    )
    return sql, [*keys, len(keys)]


def live_video_out(conn: sqlite3.Connection, row: sqlite3.Row) -> LiveVideoOut | None:
    """The clip of a Live Photo still (section 20): the companion's playable video, or the extracted
    motion clip served by /files/live. None for items without a clip."""
    keys = row.keys()
    if "live" not in keys or not row["live"]:
        return None
    v = url_version(row["version"], _row_nsfw(row))
    comp_id = row["live_video_item_id"] if "live_video_item_id" in keys else None
    if comp_id:
        comp = conn.execute(
            "SELECT uid, version, nsfw, preview_status, video_codec, ext, width, height, bitrate, duration_s "
            "FROM items WHERE id = ?", (comp_id,)
        ).fetchone()
        if comp is not None:
            cv = url_version(comp["version"], bool(comp["nsfw"]))
            url = playable_video_url(comp["uid"], cv, comp["preview_status"], stream_friendly_row(comp))
            return LiveVideoOut(uid=comp["uid"], url=url, duration_s=comp["duration_s"])
    return LiveVideoOut(uid=None, url=f"/files/live/{row['uid']}?v={v}", duration_s=None)
