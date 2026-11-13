"""Pydantic models of the HTTP API (docs/ARCHITECTURE.md section 13). frontend/src/lib/types.ts
mirrors these by hand: change both together."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    # app.build_id.API_BUILD of the running server; the site compares it with its own (section 14)
    api_build: str = ""


class RelyingPartyOut(BaseModel):
    """Where the passkeys belong (section 4): the RP ID (the server's own host) and the server's own address. The site
    compares its location with them: on a custom domain a browser without WebAuthn related origins cannot use them."""

    rp_id: str
    home_url: str


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


class SyncPending(BaseModel):
    """One of the caller's unfinished uploads of an asset (section 28): continue it at received_bytes."""
    upload_id: str
    resource: Literal["still", "video", "file"] = "file"
    received_bytes: int
    size: int


class SyncCheckOut(BaseModel):
    known: dict[str, str] = {}   # sha256 -> item uid (live or in the trash)
    assets: dict[str, str] = {}  # asset id (the upload pair key) -> item uid
    # section 28: the caller's own 'uploading' rows per asked asset id, and the asked asset ids with
    # an item in the trash (they still count in known and assets)
    pending: dict[str, list[SyncPending]] = {}
    trashed: list[str] = []


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


class UndoOut(BaseModel):
    """POST /api/actions/{id}/undo (section 33): undo_id undoes the undo (a redo); partial when some changes
    were left as they are now because something else changed them since; items = the uids touched."""
    undo_id: str
    changes: int
    partial: bool
    items: list[str]


class FaceDeleteOut(Ok):
    """DELETE /faces/{id} ("Not a face", section 10): followed = unassigned look-alikes removed with it."""
    followed: int = 0


# ---------------------------------------------------------------- items

PersonKind = Literal["person", "pet"]
Species = Literal["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "rabbit", "hamster", "guinea_pig", "turtle", "fish", "goat", "pig", "donkey", "camel", "alpaca",
                  "llama", "deer", "monkey", "chicken", "duck"]


TagRelation = Literal["in", "related"]


class PersonRef(BaseModel):
    id: int
    name: str | None = None
    cover_url: str | None = None
    kind: PersonKind = "person"
    # In an item's people and tags lists (section 10): 'related' when the item is about them without them
    # in it. A reference from a box is always 'in'.
    relation: TagRelation = "in"
    # In an item's tags: true for a tag made by a guess the owner confirmed (the system's find, section 10).
    by_guess: bool | None = None


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
    group: int | None = None    # videos: the tracks of one person or look-alike share a number (section 10)
    source: Literal["detector", "user"] = "detector"  # section 10: 'user' when the owner drew the face box
    by_guess: bool = False  # section 10: the person came from a guess the owner confirmed


TraitKind = Literal["tattoo", "nails", "mehndi"]


class TraitOut(BaseModel):
    """A body trait (section 34): a tattoo, mehndi or painted nails seen on a person box. The region is
    normalised to the photo; hidden when the owner said it is not a trait or not this person's (or a
    look-alike of one they hid); hex is the nail colour (nails only)."""
    id: int
    kind: TraitKind
    part: str
    x: float
    y: float
    w: float
    h: float
    label: str
    crop_url: str
    hidden: bool = False
    hex: str | None = None
    item_uid: str | None = None  # the person page only: the row's photo, to open it
    taken_at: str | None = None


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
    group: int | None = None    # videos: the tracks of one person or look-alike share a number (section 10)
    by_guess: bool = False  # section 10: the person came from a guess the owner confirmed
    source: Literal["detector", "user"] = "detector"  # section 10: 'user' when the owner drew the person box
    traits: list[TraitOut] | None = None  # section 34: photos' item detail only (None elsewhere)


class AnimalOut(BaseModel):
    """An animal box (section 22); person is the pet identity it belongs to."""
    id: int
    x: float
    y: float
    w: float
    h: float
    det_score: float
    species: str
    frame_time_s: float | None = None
    person: PersonRef | None = None
    assigned_by: Literal["auto", "user"] = "auto"
    species_source: Literal["auto", "user"] = "auto"  # section 22: 'user' when the owner set the species
    source: Literal["detector", "user"] = "detector"  # section 22: 'user' when the owner drew the box
    url: str
    group: int | None = None    # videos: the tracks of one person or look-alike share a number (section 10)


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
    received_via: str | None = None            # section 29: the messaging app the file came through
    forward: bool = False                      # section 29: set apart as a forward
    forward_reasons: dict[str, Any] | None = None
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


class AlbumRef(BaseModel):
    id: int
    name: str


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
    forward_source: Literal["auto", "user"] = "auto"  # section 29
    category_scores: dict[str, float] | None = None
    nsfw_score: float | None = None
    quality_scores: dict[str, Any] | None = None
    labels: list[LabelOut] = []
    faces: list[FaceOut] = []
    bodies: list[BodyOut] = []
    animals: list[AnimalOut] = []
    albums: list[AlbumRef] = []  # section 24
    place: str | None = None  # section 25: "Panaji, Goa, India" when the gazetteer knows the spot
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
    forward: bool | None = None  # section 29
    favorite: bool | None = None
    filename: str | None = Field(default=None, min_length=1, max_length=255)


class BulkIn(BaseModel):
    uids: list[str] = Field(min_length=1, max_length=1000)
    action: Literal["delete", "restore", "keep", "category", "nsfw", "favorite", "unfavorite", "purge", "tag", "untag",
                    "album", "unalbum", "forward", "unforward"]
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


class GuessCorrectIn(GuessIdsIn):
    """"Someone else" on a guess (section 10): the boxes and who they really are, an existing person or a new name."""

    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


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
    forwards: int = 0  # section 29
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
    kind: PersonKind = "person"
    species: str | None = None    # pets only
    animal_count: int = 0         # pets: crops in visible items
    confirmed_faces: int = 0      # faces (or crops) the owner assigned: the anchors of the reference
    followed: int = 0             # merge responses (section 10): unnamed look-alikes merged along with the owner's merge
    related_count: int = 0        # live items tagged 'related' to them (section 10); item_count keeps meaning 'in'
    trait_count: int | None = None  # section 34: tattoos, mehndi and nails seen on them (GET /people/{id} only)


class TraitGroupOut(BaseModel):
    """One tattoo of a person (section 34): the photos it was seen in, first and last, and the best crop."""
    kind: TraitKind
    part: str
    label: str
    photos: int
    first_seen: str | None = None
    last_seen: str | None = None
    best: TraitOut
    item_uids: list[str] = []


class PersonTraitsOut(BaseModel):
    """GET /people/{id}/traits: tattoos grouped by design; mehndi and nails of the last 60 days (12 at most)."""
    tattoos: list[TraitGroupOut] = []
    mehndi: list[TraitOut] = []
    nails: list[TraitOut] = []


class TraitHideIn(BaseModel):
    """PUT /traits/{id}: hide a trait ("Not a tattoo", "Not this person's") or show it again on this photo."""
    hidden: bool
    verdict: Literal["not_trait", "not_theirs"] = "not_trait"


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
    source: Literal["detector", "user"] = "detector"  # section 10: 'user' when the owner drew the face box
    by_guess: bool = False  # section 10: the person came from a guess the owner confirmed
    guess: GuessOut | None = None
    created_at: str


class FacePage(BaseModel):
    items: list[FaceListItem]
    next_cursor: str | None = None
    total: int


class AnimalListItem(BaseModel):
    id: int
    item_uid: str
    item_kind: ItemKind
    thumb_url: str
    url: str
    det_score: float
    species: str
    frame_time_s: float | None = None
    person_id: int | None = None
    assigned_by: Literal["auto", "user"] = "auto"
    source: Literal["detector", "user"] = "detector"  # section 22: 'user' when the owner drew the box
    created_at: str


class AnimalPage(BaseModel):
    items: list[AnimalListItem]
    next_cursor: str | None = None
    total: int


class BoxIn(BaseModel):
    """A box's new place (section 10): x, y, w, h normalised to the picture (display orientation)."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)


class AssignFacesIn(BaseModel):
    face_ids: list[int] = Field(min_length=1, max_length=1000)
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


class AssignBodiesIn(BaseModel):
    body_ids: list[int] = Field(min_length=1, max_length=1000)
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


class AssignAnimalsIn(BaseModel):
    animal_ids: list[int] = Field(min_length=1, max_length=1000)
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)
    # section 22: a pet of another species first becomes the boxes' species (every crop of it, as the
    # owner's choice), then the boxes move to it
    fix_species: bool = False


SPECIES_NAMES = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                 "rabbit", "hamster", "guinea_pig", "turtle", "fish", "goat", "pig", "donkey", "camel", "alpaca",
                 "llama", "deer", "monkey", "chicken", "duck")
MIN_BOX_SHARE = 0.005  # analysis.animals.MIN_BOX_SHARE


class PersonBoxIn(BaseModel):
    """A person box the owner draws on a photo or a video frame (section 10): normalised, inside the frame, at
    least half a percent of it; `t`, the frame's time in seconds, for a video (400 frame_time_needed without
    it); the person: an existing one (person_id) or a new name (400 person_needed without either)."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)
    t: float | None = Field(default=None, ge=0.0, le=86400.0)
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def _inside_and_large_enough(self) -> "PersonBoxIn":
        if self.x + self.w > 1.0005 or self.y + self.h > 1.0005:
            raise ValueError("The box must lie inside the photo")
        if self.w * self.h < MIN_BOX_SHARE:
            raise ValueError("The box must cover at least half a percent of the photo")
        return self


class FaceBoxIn(BaseModel):
    """A face box the owner draws around a face the detector missed (section 10): normalised, inside the frame (the
    route checks at least 12 pixels a side); `t` for a video; the person: person_id or new_person_name."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)
    t: float | None = Field(default=None, ge=0.0, le=86400.0)
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


class PersonBoxOut(BaseModel):
    """The drawn person box and the face found inside it (null for a helmet, a back, someone far away)."""
    body: BodyOut
    face: FaceOut | None = None


class AnimalBoxIn(BaseModel):
    """A box the owner draws on a photo or a video frame (section 22): normalised, inside the frame, at least
    half a percent of it; the species when the owner picked one (else the zero-shot answers, else 400
    species_needed); `t`, the frame's time in seconds, for a video (400 frame_time_needed without it)."""
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)
    species: str | None = Field(default=None, max_length=20)
    t: float | None = Field(default=None, ge=0.0, le=86400.0)

    @model_validator(mode="after")
    def _inside_and_large_enough(self) -> "AnimalBoxIn":
        if self.x + self.w > 1.0005 or self.y + self.h > 1.0005:
            raise ValueError("The box must lie inside the photo")
        if self.w * self.h < MIN_BOX_SHARE:
            raise ValueError("The box is too small")
        if self.species is not None and self.species not in SPECIES_NAMES:
            raise ValueError("Unknown species")
        return self


class RedetectIn(BaseModel):
    """The second look's detector threshold (section 22), lower than the pipeline's 0.45."""
    threshold: float = Field(default=0.2, ge=0.1, le=0.9)


class SpeciesIn(BaseModel):
    """POST animals/species (section 22): the owner's species for these boxes."""
    animal_ids: list[int] = Field(min_length=1, max_length=1000)
    species: Species


class PersonSpeciesIn(BaseModel):
    """POST people/{id}/species (section 22): every crop of the pet takes this species."""
    species: Species


class AssignOut(BaseModel):
    updated: int
    person: PersonOut | None = None
    followed: int = 0   # faces that looked like the ones taken away from a person and went with them


class Suggestion(BaseModel):
    a: PersonOut
    b: PersonOut
    similarity: float
    a_samples: list[str] = []
    b_samples: list[str] = []
    kind: Literal["same", "childhood"] = "same"  # 'childhood': their childhood photos look alike (section 10)


class RelatedSuggestionOut(BaseModel):
    """An item that may be related to someone (section 10): it looks like an item the owner marked as related to
    them ('alike'), or somewhat like it and taken at the same moment ('moment') or place ('place'); `because_uid` is
    that item."""

    item: ItemCard
    reason: Literal["alike", "moment", "place"]
    because_uid: str


class RelatedSuggestionList(BaseModel):
    items: list[RelatedSuggestionOut]


class RelatedDismissIn(BaseModel):
    uids: list[str] = Field(min_length=1, max_length=500)


class ItemRelatedSuggestion(BaseModel):
    person: PersonRef
    reason: Literal["alike", "moment", "place"]
    because_uid: str


class ItemRelatedSuggestions(BaseModel):
    people: list[ItemRelatedSuggestion]


class ChildhoodCandidate(BaseModel):
    person: PersonOut
    similarity: float


class ChildhoodOut(BaseModel):
    """Who an unnamed group of childhood photos may be as a child (section 10), best first; empty when the group
    is not mostly childhood photos."""

    candidates: list[ChildhoodCandidate]


class RecognitionCut(BaseModel):
    """One cut of the check (section 10): of the faces the model was not trained on, how many of the person's it
    found at probability p or more, and how many faces of other people it mistook for them."""

    p: float
    found: int
    mistaken: int
    unclear_found: int = 0
    unclear_mistaken: int = 0


class RecognitionCheck(BaseModel):
    folds: Literal["day", "photo"] | None = None  # what it was tested on: days, or photos, it was not trained on
    positives: int
    negatives: int
    unclear_positives: int = 0
    unclear_negatives: int = 0
    cuts: list[RecognitionCut] = []
    claim: RecognitionCut | None = None  # at the cut where the automatic matching keeps a face for the person


class RecognitionProgress(BaseModel):
    done: int
    total: int
    step: str  # read | fit | check | save (training), scan | write (looking again)


class RecognitionScan(BaseModel):
    """The last look: faces looked at, the model's guesses standing after it (sure: at the sure cut or more),
    faces taken out of unnamed groups for a guess, and faces on other named people that look like the person."""

    at: str
    looked: int
    guesses: int
    sure: int
    detached: int
    elsewhere: int


class RecognitionOut(BaseModel):
    """Recognition trained on one person (section 10, app.learning.person_model)."""

    available: bool
    reason: str | None = None  # why it cannot be trained or used (learning off, too few clear faces)
    learning: bool = True  # learning is on in Settings; off, a trained model is not used
    trained: bool = False
    trained_at: str | None = None
    counts: dict[str, float] = {}
    check: RecognitionCheck | None = None
    guess_p: float | None = None
    sure_p: float | None = None
    conflicts: int = 0  # faces on the person that look like faces the owner took off them, still standing
    elsewhere: int = 0  # faces on other named people that look like the person, still standing
    guesses: int = 0  # the model's open guesses now
    withdrawn: bool = False  # its guesses were taken back: it makes none until the next look
    scan: RecognitionScan | None = None
    running: Literal["train", "scan"] | None = None
    progress: RecognitionProgress | None = None
    error: str | None = None  # why the last run failed, or that it stopped when the app was restarted
    answers_since: int = 0  # the owner's answers about this person since the training
    busy: bool = False  # a run for another person is going


class RecognitionStartIn(BaseModel):
    scan: bool = True


class RecognitionStartOut(BaseModel):
    started: bool


class RecognitionLeaveIn(BaseModel):
    face_ids: list[int] = Field(min_length=1, max_length=1000)


class RecognitionConflict(BaseModel):
    face: FaceListItem
    like: FaceListItem | None = None  # the face it looks like; None when that one cannot be shown
    like_person: PersonRef | None = None  # who that face is now
    sim: float


class RecognitionConflictPage(BaseModel):
    items: list[RecognitionConflict]
    next_cursor: str | None = None
    total: int


class RecognitionElsewhere(BaseModel):
    face: FaceListItem
    person: PersonRef
    p: float


class RecognitionElsewhereList(BaseModel):
    items: list[RecognitionElsewhere]


class SuggestionList(BaseModel):
    items: list[Suggestion]


class DismissIn(BaseModel):
    a: int
    b: int


class TagIn(BaseModel):
    """A person or pet tagged on an item: an existing `person_id` (a person or a pet), or `new_person_name`
    (with `kind` 'pet' and its `species` for a new pet, section 22). `relation` 'related' says the item is
    about them without them in it (section 10); posting the other relation for someone already tagged
    switches it."""
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)
    kind: PersonKind = "person"
    species: str | None = Field(default=None, max_length=20)
    relation: TagRelation = "in"

    @model_validator(mode="after")
    def _known_species(self) -> "TagIn":
        if self.species is not None and self.species not in SPECIES_NAMES:
            raise ValueError("Unknown species")
        return self


# People on a selection (section 10): who the selected items show, and changing it on all of them at once.
class ItemsPeopleIn(BaseModel):
    """POST items/people: the selected items (unknown and trashed uids are skipped)."""
    uids: list[str] = Field(min_length=1, max_length=1000)


class ItemPersonOut(BaseModel):
    """A person or pet on the selected items: `items` counts the items they are on in any way (a face, a
    body, a pet box or a tag of either relation); `tags` and `related` count their tags by relation;
    `by_you` counts the items where the owner put them (a box they tagged or drew, or a tag)."""
    id: int
    name: str | None = None
    kind: PersonKind = "person"
    species: str | None = None
    cover_url: str | None = None
    items: int = 0
    faces: int = 0
    bodies: int = 0
    animals: int = 0
    tags: int = 0
    related: int = 0
    by_you: int = 0


class ItemsPeopleOut(BaseModel):
    items: int                      # the selected items found (live, not in the trash)
    unnamed_faces: int = 0          # faces on them that nobody is named for (no person)
    people: list[ItemPersonOut] = []


class PeopleMoveIn(BaseModel):
    """POST items/people/move: on these items, everything of `from_person_id` goes to `person_id` or to
    `new_person_name` (exactly one of them)."""
    uids: list[str] = Field(min_length=1, max_length=1000)
    from_person_id: int
    person_id: int | None = None
    new_person_name: str | None = Field(default=None, max_length=120)


class PeopleRemoveIn(BaseModel):
    """POST items/people/remove: on these items, the person's boxes are unassigned and their tags deleted."""
    uids: list[str] = Field(min_length=1, max_length=1000)
    person_id: int


class PeopleTagIn(TagIn):
    """POST items/people/tag: the per-item tag (TagIn) on every selected item."""
    uids: list[str] = Field(min_length=1, max_length=1000)


class PeopleBulkOut(BaseModel):
    updated: int                    # selected items where something changed
    person: PersonOut | None = None  # the target (move, tag); None for a removal
    followed: int = 0               # look-alike faces and pet crops that followed the boxes moved away
    pulled: int = 0                 # unnamed look-alike groups merged into the target person (a move, section 10)


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
    resource: Literal["still", "video", "file"] = "file"  # which half of a Live Photo (section 28)


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


class TierOut(BaseModel):
    """One storage tier (section 31)."""
    key: str                        # originals | data | backup
    path: str
    holds: str
    free_bytes: int
    total_bytes: int
    shared_with: str | None = None  # an earlier tier on the same filesystem: one disk, not two


class StorageOut(BaseModel):
    """Section 31: the tiers, and the move of an earlier layout's originals (app.tiers.status)."""
    tiered: bool
    tiers: list[TierOut]
    migration: dict[str, Any]


class BackupOut(BaseModel):
    """Section 31: app.backup.status."""
    enabled: bool
    dir: str | None = None
    hour: int
    keep: int
    running: bool
    progress: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    snapshots: list[str] = []
    next_run_at: str | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None


class BackupStartOut(BaseModel):
    started: bool
    running: bool


class ThroughputType(BaseModel):
    """Section 32: one job type's finished runs per bucket."""
    type: str
    label: str
    total: int
    failed: int
    avg_ms: int | None = None
    counts: list[int]
    failed_counts: list[int]


class ThroughputOut(BaseModel):
    range: str          # hour | day | week
    bucket_s: int
    start: str
    end: str
    buckets: list[str]  # UTC ISO start of each bucket, aligned to the local clock
    types: list[ThroughputType]
    per_minute: dict[str, float]  # finished per minute over the last ten minutes, by type


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
    video: dict[str, Any] | None = None  # section 27: encoder and web copy counts
    forwards: dict[str, Any] | None = None  # section 29: {enabled, unchecked, forwards, received, running}
    pets: dict[str, Any] | None = None  # section 22: {enabled, unchecked, running}: the species backfill
    storage: StorageOut | None = None  # section 31: tiers and the move of legacy originals
    backup: BackupOut | None = None  # section 31: nightly snapshots


class TagOut(BaseModel):
    id: int
    name: str
    items: int = 0
    auto_items: int = 0


class TagNameIn(BaseModel):
    """A free-form tag's name (section 19); TagIn above is the person tag of section 10."""
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
    vault_sensitivity: Literal["low", "normal", "high"] = "normal"
    # Set on the reply to a sensitivity change only: what the re-sort of the automatically decided
    # items did (checked, moved_in, moved_out, stale = analysed before the signals existed).
    vault_moved: dict[str, int] | None = None


class LearningEnabledIn(BaseModel):
    enabled: bool | None = None
    vault_sensitivity: Literal["low", "normal", "high"] | None = None


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
    map: dict[str, str] = {}  # section 25: {tiles, style_url}
