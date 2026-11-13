"""Runtime settings, read from environment variables (docs/ARCHITECTURE.md section 15).

`settings` is a module-level singleton. Import it as `from app.config import settings` and read
attributes at call time (tests call `reload_settings(...)`, which updates the same object in place).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_TITLE = "Personal Content Cloud"
APP_VERSION = "0.1.0"

# Relevance (app.ml.siglip.relevance, 0-1) a semantic search hit needs. Calibrated on real photos.
DEFAULT_SEARCH_MIN_RELEVANCE = 0.001

CATEGORIES = ("documents", "screenshots", "personal", "places", "food", "pets", "objects", "misc", "nsfw")
ITEM_KINDS = ("image", "video", "pdf", "document", "other")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v is not None else default


def _float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v is not None else default


def _bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _sizes(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    v = _env(name)
    if v is None:
        return default
    return tuple(int(x) for x in v.replace(" ", "").split(",") if x)


@dataclass
class Settings:
    site_title: str
    data_dir: Path
    models_dir: Path
    frontend_dist: Path

    # auth
    rp_id: str | None          # WebAuthn relying party id (host); derived from the request when None
    rp_name: str
    origin: str | None         # expected WebAuthn origin(s), comma-separated (scheme://host[:port]); the first is the
                               # server's own address (app.auth.passkeys.home_origin); derived from the request when None
    session_days: int
    vault_unlock_min: int
    trash_days: int
    local_timezone: str        # naive capture times and "same day" grouping

    # uploads
    max_upload_bytes: int
    simple_upload_max_bytes: int
    chunk_size: int
    min_free_disk_bytes: int
    staging_ttl_hours: int

    # ML runtime
    gpu_enabled: str           # 'auto' | 'true' | 'false'
    gpu_arena_shrink: bool  # free CUDA arena memory after every run
    learning_enabled: bool  # learn from the owner's corrections (section 18)
    vault_sensitivity: str  # low | normal | high (section 9)
    ort_threads: int

    # derivatives
    thumb_size: int
    preview_size: int
    preview_small_size: int
    preview_quality: int
    page_render_size: int      # long side of document page renders
    page_render_max: int       # pages rendered per document

    # faces (as the hub)
    face_det_sizes: tuple[int, ...]
    face_video_det_size: int
    face_det_threshold: float
    face_min_size_px: int
    face_good_min_size_px: int
    face_good_det_score: float
    cluster_match_threshold: float
    cluster_match_threshold_low: float
    cluster_topk: int
    suggest_threshold: float
    face_age_matching: bool  # section 10: children's faces matched within the child band
    cluster_child_threshold: float  # the cut for a face in the child band (against childhood photos only)
    child_lead: float  # a face in the child band joins only a person it matches this much better than the next
    child_suggest_threshold: float  # "Same child?" suggestions: childhood centroids at least this alike

    # bodies
    bodies_enabled: bool
    body_det_threshold: float

    # pets (section 22)
    pets_enabled: bool
    animal_det_threshold: float
    pet_match_threshold: float
    pet_new_threshold: float

    # categories, nsfw, mis-clicks
    nsfw_threshold: float
    doc_likeness_min: float
    blur_var_min: float
    burst_window_s: float

    # person guessing
    guess_window_h: float
    guess_scene_sim: float
    guess_clothes_sim: float
    guess_gps_m: float

    # body traits (section 34): tattoos, mehndi and nails as evidence for people boxes
    traits_enabled: bool       # measure traits (the runner in app.analysis.traits)
    trait_matching: bool       # use them in guesses; off = guesses exactly as before
    trait_only_guesses: bool   # a guess from a tattoo alone, under the admission rule of section 34
    trait_only_mehndi: bool    # the same from mehndi alone (off: mehndi only agrees, breaks ties and supports)
    traits_mehndi: bool        # measure mehndi at all (off by default: SigLIP read plain hands as mehndi)
    traits_threads: int        # CPU threads per trait model session (pose, hands, dino)
    trait_min_body_px: int     # smallest box long side, in pixels of the 2048 px preview
    trait_min_limb_px: int     # shortest limb segment, preview pixels
    nail_min_hand_px: int      # smallest hand span for nails
    tattoo_gate_p: float       # SigLIP tattoo cut (measured in step 0; from the unclear cut up to here is "unclear")
    tattoo_min_inliers: int    # SIFT inliers for a verified tattoo pair
    mehndi_min_inliers: int    # SIFT inliers for a verified mehndi pair
    trait_min_p: float         # floor of the learned trait certainty (owner-outcome tightening)
    trait_margin_llr: float    # lead over the runner-up (log likelihood ratio) for a trait-only guess
    trait_prior_people: int    # prior odds 1:N for the stored trait-only score
    nails_max_days: int        # nails are compared within this many days
    mehndi_max_days: int       # mehndi is compared within this many days
    trait_propagate_max: int   # items a person's tagging queues for a trait re-guess

    # position (section 35): people followed from one photo to the next at the same place
    place_enabled: bool        # line photos up (the runner in app.analysis.place) and use positions in guesses
    place_only_guesses: bool   # a guess from the position alone, under the admission rule of section 35
    place_threads: int         # OpenCV threads while a photo is lined up: 1 keeps the work on the low-priority thread

    # video
    video_sample_interval_s: float
    video_face_interval_s: float   # faces are followed this densely (section 10)
    video_face_max_frames: int
    video_max_samples: int
    video_track_threshold: float
    video_previews: bool
    video_preview_max_height: int
    video_preview_crf: int
    video_preview_maxrate_kbps: int
    video_preview_audio_kbps: int
    video_transcode_threads: int
    video_stream_max_kbps: int
    video_transcode_max_s: float
    video_faces_max_s: float
    video_encoder: str         # auto | h264_nvenc | h264_vaapi | h264_qsv | libx264 (section 27)
    vaapi_device: str

    # forwards (section 29)
    forwards_enabled: bool
    forward_max_side: int
    forward_jpeg_quality_max: float
    forward_meme_min: float

    # map and places (section 25)
    geo_enabled: bool
    map_tiles: str             # offline | online
    map_tile_style: str
    challan_portal_url: str  # section 30: another state's e-challan portal in place of the Maharashtra link

    # search
    search_enabled: bool
    search_threads: int
    search_min_relevance: float
    search_contrast: float
    search_color_check: bool
    search_video_frames: int
    search_person_min_relevance: float

    # text and documents
    ocr_enabled: bool
    ocr_max_pages: int

    # storage tiers and nightly backups (section 31)
    originals_dir: Path | None   # ORIGINALS_DIR: originals, staging and failed uploads (a fast disk); None = data_dir
    backup_dir: Path | None      # BACKUP_DIR: nightly snapshots (a large, slow disk); None = backups off
    backup_hour: int             # BACKUP_HOUR: local hour (0-23) of the nightly run
    backup_keep: int             # BACKUP_KEEP: how many nightly snapshots are kept

    # Apple Wallet
    wallet_pass_type_id: str | None
    wallet_team_id: str | None
    wallet_org_name: str | None
    wallet_key_password: str | None  # passphrase of key.pem, when it has one
    wallet_dir: Path | None

    # 3D heads (section 36): the generator is a separate program with its own Python environment
    gen3d_dir: Path              # GEN3D_DIR: the generator's venv/, weights/ and src/ (default MODELS_DIR/gen3d)
    gen3d_python: Path | None    # GEN3D_PYTHON: the generator's Python (default GEN3D_DIR/venv/bin/python)
    gen3d_generator: str         # GEN3D_GENERATOR: auto | a backend's name
    gen3d_sandbox: str           # GEN3D_SANDBOX: auto | on | off (the generator runs inside bwrap)
    gen3d_max_rss_mb: int        # the generator is stopped above this much resident memory
    gen3d_look_timeout_s: float  # ... and when one head takes longer than this
    gen3d_fake: bool             # GEN3D_FAKE: the stand-in generator that needs no model (tests, a first look)

    # tests turn off the background worker threads
    workers_enabled: bool

    @property
    def db_path(self) -> Path:
        return self.data_dir / "cloud.db"

    @property
    def originals_root(self) -> Path:
        """Where originals/, staging/ and failed/ live: ORIGINALS_DIR, or DATA_DIR when it is not set."""
        return self.originals_dir or self.data_dir

    @property
    def wallet_path(self) -> Path:
        return self.wallet_dir or (self.data_dir / "wallet")


def _load(**overrides) -> Settings:
    cpu = os.cpu_count() or 4
    data_dir = _env("DATA_DIR") or str(REPO_ROOT / "data")
    wallet_dir = _env("WALLET_DIR")
    originals_dir = _env("ORIGINALS_DIR")
    backup_dir = _env("BACKUP_DIR")
    values = dict(
        site_title=_env("SITE_TITLE", SITE_TITLE),
        data_dir=Path(data_dir).resolve(),
        models_dir=Path(_env("MODELS_DIR", str(REPO_ROOT / "models"))).resolve(),
        frontend_dist=Path(_env("FRONTEND_DIST", str(REPO_ROOT / "frontend" / "dist"))).resolve(),
        rp_id=_env("RP_ID"),
        rp_name=_env("RP_NAME", _env("SITE_TITLE", SITE_TITLE)),
        origin=_env("ORIGIN"),
        session_days=_int("SESSION_DAYS", 30),
        vault_unlock_min=_int("VAULT_UNLOCK_MIN", 15),
        trash_days=_int("TRASH_DAYS", 30),
        local_timezone=_env("LOCAL_TIMEZONE", "Asia/Kolkata"),
        max_upload_bytes=_int("MAX_UPLOAD_MB", 16 * 1024) * 1024 * 1024,
        simple_upload_max_bytes=_int("SIMPLE_UPLOAD_MAX_MB", 256) * 1024 * 1024,
        chunk_size=_int("CHUNK_SIZE_MB", 8) * 1024 * 1024,
        min_free_disk_bytes=_int("MIN_FREE_DISK_MB", 2048) * 1024 * 1024,
        staging_ttl_hours=_int("STAGING_TTL_HOURS", 48),
        gpu_enabled=(_env("GPU_ENABLED", "auto") or "auto").strip().lower(),
        ort_threads=_int("ORT_THREADS", max(1, min(6, cpu - 2))),
        thumb_size=_int("THUMB_SIZE", 480),
        preview_size=_int("PREVIEW_SIZE", 2048),
        preview_small_size=_int("PREVIEW_SMALL_SIZE", 1280),
        preview_quality=_int("PREVIEW_QUALITY", 80),
        page_render_size=_int("PAGE_RENDER_SIZE", 1600),
        page_render_max=_int("PAGE_RENDER_MAX", 200),
        face_det_sizes=_sizes("FACE_DET_SIZES", (640, 1280, 1920)),
        face_video_det_size=_int("FACE_VIDEO_DET_SIZE", 640),
        face_det_threshold=_float("FACE_DET_THRESHOLD", 0.5),
        face_min_size_px=_int("FACE_MIN_SIZE_PX", 24),
        face_good_min_size_px=_int("FACE_GOOD_MIN_SIZE_PX", 48),
        face_good_det_score=_float("FACE_GOOD_DET_SCORE", 0.7),
        cluster_match_threshold=_float("CLUSTER_MATCH_THRESHOLD", 0.45),
        cluster_match_threshold_low=_float("CLUSTER_MATCH_THRESHOLD_LOW", 0.52),
        cluster_topk=_int("CLUSTER_TOPK", 3),
        suggest_threshold=_float("SUGGEST_THRESHOLD", 0.35),
        face_age_matching=_bool("FACE_AGE_MATCHING", True),
        cluster_child_threshold=_float("CLUSTER_CHILD_THRESHOLD", 0.48),
        child_lead=_float("CHILD_LEAD", 0.08),
        child_suggest_threshold=_float("CHILD_SUGGEST_THRESHOLD", 0.40),
        bodies_enabled=_bool("BODIES_ENABLED", True),
        body_det_threshold=_float("BODY_DET_THRESHOLD", 0.5),
        pets_enabled=_bool("PETS_ENABLED", True),
        animal_det_threshold=_float("ANIMAL_DET_THRESHOLD", 0.45),
        pet_match_threshold=_float("PET_MATCH_THRESHOLD", 0.82),
        pet_new_threshold=_float("PET_NEW_THRESHOLD", 0.86),
        nsfw_threshold=_float("NSFW_THRESHOLD", 0.7),
        doc_likeness_min=_float("DOC_LIKENESS_MIN", 0.55),
        blur_var_min=_float("BLUR_VAR_MIN", 40.0),
        burst_window_s=_float("BURST_WINDOW_S", 3.0),
        guess_window_h=_float("GUESS_WINDOW_H", 1.0),  # clothes are evidence only close in time
        guess_scene_sim=_float("GUESS_SCENE_SIM", 0.80),
        guess_clothes_sim=_float("GUESS_CLOTHES_SIM", 0.85),
        guess_gps_m=_float("GUESS_GPS_M", 250.0),
        traits_enabled=_bool("TRAITS_ENABLED", True),
        trait_matching=_bool("TRAIT_MATCHING", True),
        trait_only_guesses=_bool("TRAIT_ONLY_GUESSES", True),
        trait_only_mehndi=_bool("TRAIT_ONLY_MEHNDI", False),
        traits_mehndi=_bool("TRAITS_MEHNDI", False),
        traits_threads=max(1, _int("TRAITS_THREADS", 2)),
        trait_min_body_px=_int("TRAIT_MIN_BODY_PX", 200),
        trait_min_limb_px=_int("TRAIT_MIN_LIMB_PX", 64),
        nail_min_hand_px=_int("NAIL_MIN_HAND_PX", 96),
        tattoo_gate_p=_float("TATTOO_GATE_P", 0.80),  # step 0: 706 limb crops pass it with the cloth and jewel cuts
        tattoo_min_inliers=_int("TATTOO_MIN_INLIERS", 12),  # native crops: 12+ inliers, same 36 of 900, different 2 of 1,500
        mehndi_min_inliers=_int("MEHNDI_MIN_INLIERS", 30),
        trait_min_p=_float("TRAIT_MIN_P", 0.90),
        trait_margin_llr=_float("TRAIT_MARGIN_LLR", 2.0),
        trait_prior_people=max(1, _int("TRAIT_PRIOR_PEOPLE", 20)),
        nails_max_days=_int("NAILS_MAX_DAYS", 14),
        mehndi_max_days=_int("MEHNDI_MAX_DAYS", 21),
        trait_propagate_max=_int("TRAIT_PROPAGATE_MAX", 400),
        place_enabled=_bool("PLACE_ENABLED", True),
        place_only_guesses=_bool("PLACE_ONLY_GUESSES", True),
        place_threads=max(1, _int("PLACE_THREADS", 1)),
        video_sample_interval_s=_float("VIDEO_SAMPLE_INTERVAL_S", 1.0),
        video_face_interval_s=_float("VIDEO_FACE_INTERVAL_S", 0.2),
        video_face_max_frames=_int("VIDEO_FACE_MAX_FRAMES", 1500),
        video_max_samples=_int("VIDEO_MAX_SAMPLES", 150),
        video_track_threshold=_float("VIDEO_TRACK_THRESHOLD", 0.5),
        video_previews=_bool("VIDEO_PREVIEWS", True),
        video_preview_max_height=_int("VIDEO_PREVIEW_MAX_HEIGHT", 1080),
        video_preview_crf=_int("VIDEO_PREVIEW_CRF", 23),
        video_preview_maxrate_kbps=_int("VIDEO_PREVIEW_MAXRATE_KBPS", 6000),
        video_preview_audio_kbps=_int("VIDEO_PREVIEW_AUDIO_KBPS", 96),
        video_transcode_threads=_int("VIDEO_TRANSCODE_THREADS", 4),
        video_stream_max_kbps=_int("VIDEO_STREAM_MAX_KBPS", 8000),
        video_transcode_max_s=_float("VIDEO_TRANSCODE_MAX_S", 3 * 3600.0),
        video_faces_max_s=_float("VIDEO_FACES_MAX_S", 900.0),
        video_encoder=(_env("VIDEO_ENCODER", "auto") or "auto").strip().lower(),
        vaapi_device=_env("VAAPI_DEVICE", "/dev/dri/renderD128"),
        forwards_enabled=_bool("FORWARDS_ENABLED", True),
        forward_max_side=_int("FORWARD_MAX_SIDE", 1600),
        forward_jpeg_quality_max=_float("FORWARD_JPEG_QUALITY_MAX", 82.0),
        forward_meme_min=_float("FORWARD_MEME_MIN", 0.5),
        geo_enabled=_bool("GEO_ENABLED", True),
        map_tiles=(_env("MAP_TILES", "offline") or "offline").strip().lower(),
        map_tile_style=_env("MAP_TILE_STYLE", "https://tiles.openfreemap.org/styles/liberty"),
        challan_portal_url=_env("CHALLAN_PORTAL_URL", "") or "",
        search_enabled=_bool("SEARCH_ENABLED", True),
        search_threads=max(1, _int("SEARCH_THREADS", 2)),
        search_min_relevance=_float("SEARCH_MIN_RELEVANCE", DEFAULT_SEARCH_MIN_RELEVANCE),
        search_contrast=max(0.0, _float("SEARCH_CONTRAST", 20.0)),
        search_color_check=_bool("SEARCH_COLOR_CHECK", True),
        search_video_frames=max(1, _int("SEARCH_VIDEO_FRAMES", 6)),
        search_person_min_relevance=_float("SEARCH_PERSON_MIN_RELEVANCE", 0.4),
        ocr_enabled=_bool("OCR_ENABLED", True),
        ocr_max_pages=_int("OCR_MAX_PAGES", 50),
        wallet_pass_type_id=_env("WALLET_PASS_TYPE_ID"),
        wallet_team_id=_env("WALLET_TEAM_ID"),
        wallet_org_name=_env("WALLET_ORG_NAME"),
        wallet_key_password=_env("WALLET_KEY_PASSWORD"),
        wallet_dir=Path(wallet_dir).resolve() if wallet_dir else None,
        originals_dir=Path(originals_dir).resolve() if originals_dir else None,
        backup_dir=Path(backup_dir).resolve() if backup_dir else None,
        backup_hour=min(23, max(0, _int("BACKUP_HOUR", 3))),
        backup_keep=max(1, _int("BACKUP_KEEP", 14)),
        workers_enabled=_bool("WORKERS_ENABLED", True),
        gpu_arena_shrink=_bool("GPU_ARENA_SHRINK", True),
        learning_enabled=_bool("LEARNING_ENABLED", True),
        vault_sensitivity=(_env("VAULT_SENSITIVITY", "normal") or "normal").strip().lower(),
        gen3d_dir=Path(_env("GEN3D_DIR") or (Path(_env("MODELS_DIR", str(REPO_ROOT / "models"))) / "gen3d")).resolve(),
        # not resolved: a venv's python is a symlink, and following it would lose the venv
        gen3d_python=Path(_env("GEN3D_PYTHON")).absolute() if _env("GEN3D_PYTHON") else None,
        gen3d_generator=(_env("GEN3D_GENERATOR", "auto") or "auto").strip().lower(),
        gen3d_sandbox=(_env("GEN3D_SANDBOX", "auto") or "auto").strip().lower(),
        gen3d_max_rss_mb=max(256, _int("GEN3D_MAX_RSS_MB", 5000)),
        gen3d_look_timeout_s=max(5.0, _float("GEN3D_LOOK_TIMEOUT_S", 600.0)),
        gen3d_fake=_bool("GEN3D_FAKE", False),
    )
    values.update(overrides)
    if values.get("originals_dir") is not None:
        values["originals_dir"] = Path(values["originals_dir"]).resolve()
        if values["originals_dir"] == Path(values["data_dir"]).resolve():
            values["originals_dir"] = None  # the same folder: one tier, as before
    if values.get("backup_dir") is not None:
        values["backup_dir"] = Path(values["backup_dir"]).resolve()
    return Settings(**values)


settings = _load()


def reload_settings(**overrides) -> Settings:
    """Re-read the environment (plus overrides) into the existing settings object."""
    fresh = _load(**overrides)
    for f in fields(Settings):
        setattr(settings, f.name, getattr(fresh, f.name))
    return settings
