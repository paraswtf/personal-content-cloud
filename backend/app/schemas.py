"""Pydantic models of the HTTP API (docs/ARCHITECTURE.md section 13). frontend/src/lib/types.ts
mirrors these by hand: change both together."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal["documents", "screenshots", "personal", "places", "food", "pets", "objects", "misc", "nsfw"]
ItemKind = Literal["image", "video", "pdf", "document", "other"]
AssignedBy = Literal["auto", "user", "guess"]


# ---------------------------------------------------------------- auth

class AuthState(BaseModel):
    setup_required: bool
    authenticated: bool
    username: str | None = None
    passkeys: int = 0
    vault_until: str | None = None
    session_id: int | None = None
    site_title: str


class SetupIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=10, max_length=200)


class LoginIn(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=200)


class PasswordChangeIn(BaseModel):
    current_password: str = Field(max_length=200)
    new_password: str = Field(min_length=10, max_length=200)


class SessionOut(BaseModel):
    id: int
    created_at: str
    last_seen_at: str
    user_agent: str | None = None
    ip: str | None = None
    current: bool = False
    device_name: str | None = None


class DeviceIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class DeviceCreated(BaseModel):
    id: int
    name: str
    token: str  # shown once


class SyncCheckIn(BaseModel):
    sha256: list[str] = Field(default=[], max_length=2000)
    asset_ids: list[str] = Field(default=[], max_length=2000)


class SyncCheckOut(BaseModel):
    known: dict[str, str] = {}   # sha256 -> item uid (live or in the trash)
    assets: dict[str, str] = {}  # asset id (the upload pair key) -> item uid


class PasskeyOut(BaseModel):
    id: int
    name: str
    created_at: str
    last_used_at: str | None = None
    backup_eligible: bool = False


class PasskeyRegisterVerifyIn(BaseModel):
    challenge_id: str = Field(max_length=64)
    name: str = Field(min_length=1, max_length=64)
    credential: dict[str, Any]


class PasskeyRenameIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ChallengeOut(BaseModel):
    challenge_id: str
    options: dict[str, Any]


class PasskeyLoginVerifyIn(BaseModel):
    challenge_id: str = Field(max_length=64)
    credential: dict[str, Any]


class VaultUnlockIn(BaseModel):
    password: str | None = Field(default=None, max_length=200)
    challenge_id: str | None = Field(default=None, max_length=64)
    credential: dict[str, Any] | None = None


class Ok(BaseModel):
    ok: bool = True


# ---------------------------------------------------------------- items

class PersonRef(BaseModel):
    id: int
    name: str | None = None
    cover_url: str | None = None


class GuessOut(BaseModel):
    person: PersonRef
    score: float
    evidence: dict[str, Any] = {}


class FaceOut(BaseModel):
    id: int
    x: float
    y: float
    w: float
    h: float
    det_score: float
    size_px: int
    quality: float
    frame_time_s: float | None = None
    spans: list[list[float]] | None = None
    person: PersonRef | None = None
    assigned_by: AssignedBy = "auto"
    guess: GuessOut | None = None
    url: str


class BodyOut(BaseModel):
    id: int
    x: float
    y: float
    w: float
    h: float
    det_score: float
    face_id: int | None = None
    person: PersonRef | None = None
    assigned_by: AssignedBy = "auto"
    guess: GuessOut | None = None
    frame_time_s: float | None = None
    url: str


class ItemUrls(BaseModel):
    thumb: str
    preview: str
    preview_small: str
    original: str | None = None
    download: str | None = None
    video: str | None = None
    pages: list[str] = []
    live: str | None = None  # the Live Photo clip (section 20), on details


class LiveVideoOut(BaseModel):
    uid: str | None = None  # the companion item for an iPhone pair, None for an extracted motion clip
    url: str
    duration_s: float | None = None


class ItemCard(BaseModel):
    uid: str
    kind: ItemKind
    version: int
    filename: str
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    taken_at: str | None = None
    created_at: str
    category: Category
    nsfw: bool = False
    misclick: bool = False
    misclick_reasons: dict[str, Any] | None = None
    favorite: bool = False
    dup_group: str | None = None   # uid of the group's best item
    dup_count: int = 0             # copies in the group (0 when not in a group)
    page_count: int | None = None
    doc_type: str | None = None
    live: bool = False
    urls: ItemUrls


class DocumentField(BaseModel):
    value: str
    label: str
    confidence: float = 1.0


class DocumentOut(BaseModel):
    doc_type: str
    doc_group: str
    label: str
    confidence: float
    fields: dict[str, DocumentField]
    field_order: list[str] = []
    title: str | None = None
    doc_date: str | None = None
    issuer: str | None = None
    person_name: str | None = None
    barcode: str | None = None
    barcode_format: str | None = None
    photo_url: str | None = None
    source: Literal["auto", "user"] = "auto"
    wallet_available: bool = False
    updated_at: str


class GpsOut(BaseModel):
    lat: float
    lon: float
    alt: float | None = None


class LabelOut(BaseModel):
    label: str
    score: float


class ItemDetail(ItemCard):
    mime: str
    size_bytes: int
    bitrate: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    has_audio: bool | None = None
    camera: str | None = None
    gps: GpsOut | None = None
    exif_datetime: str | None = None
    category_source: Literal["auto", "user"] = "auto"
    category_scores: dict[str, float] | None = None
    nsfw_score: float | None = None
    quality_scores: dict[str, Any] | None = None
    labels: list[LabelOut] = []
    faces: list[FaceOut] = []
    bodies: list[BodyOut] = []
    tags: list[PersonRef] = []
    people: list[PersonRef] = []
    user_tags: list[ItemTagOut] = []
    live_video: LiveVideoOut | None = None
    document: DocumentOut | None = None
    text_status: Literal["none", "pending", "done", "error"] = "none"
    analysis_status: Literal["pending", "done", "error"] = "pending"
    preview_status: Literal["none", "pending", "done", "error"] = "none"
    deleted_at: str | None = None
    updated_at: str


class ItemPage(BaseModel):
    items: list[ItemCard]
    next_cursor: str | None = None
    total: int


class ItemUpdate(BaseModel):
    category: Category | None = None
    nsfw: bool | None = None
    misclick: bool | None = None
    favorite: bool | None = None
    filename: str | None = Field(default=None, min_length=1, max_length=255)


class BulkIn(BaseModel):
    uids: list[str] = Field(min_length=1, max_length=1000)
    action: Literal["delete", "restore", "keep", "category", "nsfw", "favorite", "unfavorite", "purge", "tag", "untag"]
    value: str | None = None


class BulkOut(BaseModel):
    updated: int


class ReprocessIn(BaseModel):
    stages: list[Literal["analyze", "text", "preview"]] = ["analyze", "text"]


class PageOut(BaseModel):
    page_no: int
    width: int | None = None
    height: int | None = None
    text: str | None = None
    text_source: str = "none"
    url: str


class ItemTextOut(BaseModel):
    text: str
    source: str
    char_count: int


class DupGroup(BaseModel):
    best: ItemCard
    copies: list[ItemCard]


class DupGroupPage(BaseModel):
    groups: list[DupGroup]
    next_cursor: str | None = None
    total: int


class GuessItem(BaseModel):
    kind: Literal["face", "body"]
    id: int
    item: ItemCard
    box: FaceOut | BodyOut
    guess: GuessOut


class GuessPage(BaseModel):
    items: list[GuessItem]
    next_cursor: str | None = None
    total: int


class GuessIdsIn(BaseModel):
    faces: list[int] = []
    bodies: list[int] = []


class DiskStats(BaseModel):
    free_bytes: int
    total_bytes: int
    originals_bytes: int


class Stats(BaseModel):
    items: int
    by_kind: dict[str, int]
    by_category: dict[str, int]
    people: int
    named_people: int
    documents: int
    misclicks: int
    guesses: int
    duplicates: int
    trash: int
    nsfw: int
    queue: dict[str, Any]
    disk: DiskStats
    analysis: dict[str, int]
    text: dict[str, int]
    recent: list[ItemCard] = []


# ---------------------------------------------------------------- people

class PersonOut(BaseModel):
    id: int
    name: str | None = None
    cover_url: str | None = None
    cover_face_id: int | None = None
    face_count: int = 0
    item_count: int = 0
    guess_count: int = 0
    created_at: str


class PersonPage(BaseModel):
    items: list[PersonOut]
    next_cursor: str | None = None
    total: int


class PersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    face_ids: list[int] = []


class PersonUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    cover_face_id: int | None = None


class MergeIn(BaseModel):
    target_id: int
    source_ids: list[int] = Field(min_length=1)


class FaceListItem(BaseModel):
    id: int
    item_uid: str
    item_kind: ItemKind
    thumb_url: str
    url: str
    det_score: float
    size_px: int
    quality: float
    frame_time_s: float | None = None
    person_id: int | None = None
    assigned_by: AssignedBy
    guess: GuessOut | None = None
    created_at: str


class FacePage(BaseModel):
    items: list[FaceListItem]
    next_cursor: str | None = None
    total: int


class AssignFacesIn(BaseModel):
    face_ids: list[int] = Field(min_length=1, max_length=1000)
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


class AssignBodiesIn(BaseModel):
    body_ids: list[int] = Field(min_length=1, max_length=1000)
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


class AssignOut(BaseModel):
    updated: int
    person: PersonOut | None = None


class Suggestion(BaseModel):
    a: PersonOut
    b: PersonOut
    similarity: float
    a_samples: list[str] = []
    b_samples: list[str] = []


class SuggestionList(BaseModel):
    items: list[Suggestion]


class DismissIn(BaseModel):
    a: int
    b: int


class TagIn(BaseModel):
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


# ---------------------------------------------------------------- documents

class DocumentListItem(BaseModel):
    item: ItemCard
    document: DocumentOut


class DocumentPage(BaseModel):
    items: list[DocumentListItem]
    next_cursor: str | None = None
    total: int


class DocumentTypeOut(BaseModel):
    type: str
    group: str
    label: str
    count: int


class DocumentUpdate(BaseModel):
    doc_type: str | None = Field(default=None, max_length=40)
    fields: dict[str, str] | None = None     # key -> value; empty string removes the field
    title: str | None = Field(default=None, max_length=200)
    doc_date: str | None = Field(default=None, max_length=10)
    person_name: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------- search

class SearchHit(ItemCard):
    score: float
    match_time_s: float | None = None
    snippet: str | None = None
    source: Literal["semantic", "text", "both"] = "semantic"


class SearchPage(BaseModel):
    query: str
    mode: Literal["auto", "semantic", "text"]
    items: list[SearchHit]
    total: int
    next_offset: int | None = None
    indexed: int
    indexing: bool


# ---------------------------------------------------------------- uploads

UploadStatus = Literal["uploading", "queued", "processing", "stored", "duplicate", "rejected", "error", "cancelled"]


class UploadCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    mime: str | None = Field(default=None, max_length=128)
    client_id: str | None = Field(default=None, max_length=64)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source: Literal["web", "share", "camera", "api"] = "web"
    pair_key: str | None = Field(default=None, max_length=128)  # both halves of a Live Photo carry the same key


class UploadOut(BaseModel):
    id: str
    filename: str
    size: int
    received_bytes: int
    status: UploadStatus
    message: str | None = None
    chunk_size: int
    item_uid: str | None = None
    item_thumb_url: str | None = None
    # True only when the session may see vault items and the stored item is in the vault (the
    # link then needs ?vault=1); a locked session gets no uid or thumb for such an item.
    item_nsfw: bool = False
    created_at: str


class UploadList(BaseModel):
    uploads: list[UploadOut]


class ChunkOut(BaseModel):
    received_bytes: int


# ---------------------------------------------------------------- system

class ModelStatus(BaseModel):
    key: str
    label: str
    present: bool
    loaded: bool
    provider: str | None = None
    size_bytes: int = 0
    downloading: bool = False
    error: str | None = None


class SystemOut(BaseModel):
    version: str
    site_title: str
    gpu: dict[str, Any]
    models: list[ModelStatus]
    queues: dict[str, Any]
    disk: DiskStats
    wallet: dict[str, Any]
    timezone: str
    ffmpeg: bool
    workers: bool
    download: dict[str, Any] | None = None


class TagOut(BaseModel):
    id: int
    name: str
    items: int = 0
    auto_items: int = 0


class TagIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class UserTagIn(BaseModel):
    tag_id: int | None = None
    name: str | None = Field(default=None, max_length=80)


class ItemTagOut(BaseModel):
    id: int
    name: str
    source: Literal["user", "auto"] = "user"
    score: float | None = None


class LearningStatus(BaseModel):
    enabled: bool
    trained_at: str | None = None
    pending: int = 0
    learners: dict[str, Any]
    events: dict[str, int] = {}
    errors: dict[str, str] = {}


class LearningEnabledIn(BaseModel):
    enabled: bool


class LearningApplyOut(BaseModel):
    items: int
    documents: int
    tags: int = 0


class ConfigOut(BaseModel):
    site_title: str
    max_upload_bytes: int
    simple_upload_max_bytes: int
    chunk_size: int
    accept: str
    categories: list[str]
    vault_unlock_min: int
    trash_days: int
