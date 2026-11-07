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
    origin: str | None         # expected WebAuthn origin (scheme://host[:port]); derived when None
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

    # bodies
    bodies_enabled: bool
    body_det_threshold: float

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

    # video
    video_sample_interval_s: float
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

    # Apple Wallet
    wallet_pass_type_id: str | None
    wallet_team_id: str | None
    wallet_org_name: str | None
    wallet_key_password: str | None  # passphrase of key.pem, when it has one
    wallet_dir: Path | None

    # tests turn off the background worker threads
    workers_enabled: bool

    @property
    def db_path(self) -> Path:
        return self.data_dir / "cloud.db"

    @property
    def wallet_path(self) -> Path:
        return self.wallet_dir or (self.data_dir / "wallet")


def _load(**overrides) -> Settings:
    cpu = os.cpu_count() or 4
    data_dir = _env("DATA_DIR") or str(REPO_ROOT / "data")
    wallet_dir = _env("WALLET_DIR")
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
        bodies_enabled=_bool("BODIES_ENABLED", True),
        body_det_threshold=_float("BODY_DET_THRESHOLD", 0.5),
        nsfw_threshold=_float("NSFW_THRESHOLD", 0.7),
        doc_likeness_min=_float("DOC_LIKENESS_MIN", 0.55),
        blur_var_min=_float("BLUR_VAR_MIN", 40.0),
        burst_window_s=_float("BURST_WINDOW_S", 3.0),
        guess_window_h=_float("GUESS_WINDOW_H", 12.0),
        guess_scene_sim=_float("GUESS_SCENE_SIM", 0.80),
        guess_clothes_sim=_float("GUESS_CLOTHES_SIM", 0.85),
        guess_gps_m=_float("GUESS_GPS_M", 250.0),
        video_sample_interval_s=_float("VIDEO_SAMPLE_INTERVAL_S", 1.0),
        video_max_samples=_int("VIDEO_MAX_SAMPLES", 150),
        video_track_threshold=_float("VIDEO_TRACK_THRESHOLD", 0.5),
        video_previews=_bool("VIDEO_PREVIEWS", True),
        video_preview_max_height=_int("VIDEO_PREVIEW_MAX_HEIGHT", 720),
        video_preview_crf=_int("VIDEO_PREVIEW_CRF", 27),
        video_preview_maxrate_kbps=_int("VIDEO_PREVIEW_MAXRATE_KBPS", 2500),
        video_preview_audio_kbps=_int("VIDEO_PREVIEW_AUDIO_KBPS", 96),
        video_transcode_threads=_int("VIDEO_TRANSCODE_THREADS", 4),
        video_stream_max_kbps=_int("VIDEO_STREAM_MAX_KBPS", 3000),
        video_transcode_max_s=_float("VIDEO_TRANSCODE_MAX_S", 3 * 3600.0),
        video_faces_max_s=_float("VIDEO_FACES_MAX_S", 900.0),
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
        workers_enabled=_bool("WORKERS_ENABLED", True),
        gpu_arena_shrink=_bool("GPU_ARENA_SHRINK", True),
        learning_enabled=_bool("LEARNING_ENABLED", True),
    )
    values.update(overrides)
    return Settings(**values)


settings = _load()


def reload_settings(**overrides) -> Settings:
    """Re-read the environment (plus overrides) into the existing settings object."""
    fresh = _load(**overrides)
    for f in fields(Settings):
        setattr(settings, f.name, getattr(fresh, f.name))
    return settings
