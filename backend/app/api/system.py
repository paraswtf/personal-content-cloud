"""System status and maintenance (docs/ARCHITECTURE.md section 13).

CONTRACT. Router without a prefix, mounted at /api/system; every route requires a logged-in session.
GET  /api/system                  SystemOut: version, GPU (app.ml.runtime.gpu_info), models
                                  (app.ml.models.statuses), queues, disk, Apple Wallet configuration
                                  (app.wallet.pkpass.configured), time zone, ffmpeg presence, workers,
                                  model download state. Each optional module is imported lazily; when it
                                  is missing the field reports empty or unknown values.
POST /api/system/models/download  starts the model downloader in a background thread -> Ok
POST /api/system/housekeeping     runs the worker's housekeeping now (not the daily orphan file
                                  sweep) -> Ok
POST /api/system/retry-failed     retries what failed for good (ingest.retry_failed: uploads in 'error'
                                  whose bytes are kept, errored jobs with their item status set back to
                                  'pending') -> {"requeued": int}. Holds worker.housekeeping_lock so it
                                  never runs beside housekeeping.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import jobs, storage
from app.api.deps import get_db, require_user
from app.config import APP_VERSION, settings
from app.schemas import DiskStats, LearningApplyOut, LearningEnabledIn, LearningStatus, ModelStatus, Ok, SystemOut

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_user)])  # require_user also checks the CSRF header


class RetryOut(BaseModel):
    requeued: int

_download_thread: threading.Thread | None = None
_download_guard = threading.Lock()


def _gpu() -> dict[str, Any]:
    try:
        from app.ml import runtime
    except ImportError:
        return {"available": False, "name": None, "provider": None}
    try:
        info = dict(runtime.gpu_info())
    except Exception:
        log.exception("could not read the GPU status")
        return {"available": False, "name": None, "provider": None}
    info.setdefault("available", False)
    info.setdefault("name", None)
    info.setdefault("provider", None)
    return info


def _models() -> list[ModelStatus]:
    try:
        from app.ml import models
    except ImportError:
        return []
    try:
        raw = models.statuses()
    except Exception:
        log.exception("could not read the model status")
        return []
    fields = set(ModelStatus.model_fields)
    out: list[ModelStatus] = []
    for entry in raw or []:
        data = {k: v for k, v in dict(entry).items() if k in fields}
        data.setdefault("label", data.get("key", ""))
        data.setdefault("present", False)
        data.setdefault("loaded", False)
        try:
            out.append(ModelStatus(**data))
        except Exception:
            log.warning("skipping malformed model status %r", entry)
    return out


def _download_state() -> dict[str, Any] | None:
    try:
        from app.ml import models
    except ImportError:
        return None
    try:
        return dict(models.download_state())
    except Exception:
        log.exception("could not read the model download state")
        return None


def _wallet() -> dict[str, Any]:
    try:
        from app.wallet.pkpass import configured
    except ImportError:
        return {"configured": False, "missing": ["wallet module"]}
    try:
        ok, missing = configured()
        return {"configured": bool(ok), "missing": list(missing)}
    except Exception:
        log.exception("could not read the wallet configuration")
        return {"configured": False, "missing": []}


@router.get("", response_model=SystemOut)
def system(conn: sqlite3.Connection = Depends(get_db)) -> SystemOut:
    free, total = storage.disk_usage()
    originals = int(conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM items").fetchone()[0])
    return SystemOut(
        version=APP_VERSION, site_title=settings.site_title, gpu=_gpu(), models=_models(),
        queues=jobs.counts(conn), disk=DiskStats(free_bytes=free, total_bytes=total, originals_bytes=originals),
        wallet=_wallet(), timezone=settings.local_timezone,
        ffmpeg=shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
        workers=settings.workers_enabled, download=_download_state(),
    )


def _run_download() -> None:
    try:
        from app.ml import models

        models.download()
    except Exception:
        log.exception("model download failed")


@router.post("/models/download", response_model=Ok)
def download_models() -> Ok:
    global _download_thread
    with _download_guard:
        if _download_thread is None or not _download_thread.is_alive():
            _download_thread = threading.Thread(target=_run_download, name="pcc-model-download", daemon=True)
            _download_thread.start()
    return Ok()


@router.post("/housekeeping", response_model=Ok)
def run_housekeeping(conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    from app import worker

    worker.housekeeping(conn)
    return Ok()


@router.post("/retry-failed", response_model=RetryOut)
def retry_failed(conn: sqlite3.Connection = Depends(get_db)) -> RetryOut:
    from app import ingest, worker

    with worker.housekeeping_lock:
        n = ingest.retry_failed(conn)
    log.info("retry-failed: queued %d uploads and jobs again", n)
    return RetryOut(requeued=n)


# ------------------------------------------------------------------ learning (section 18)
@router.get("/learning", response_model=LearningStatus)
def learning_status(conn: sqlite3.Connection = Depends(get_db)) -> LearningStatus:
    from app.learning import train as learning_train

    return LearningStatus(**learning_train.status(conn))


@router.patch("/learning", response_model=LearningStatus)
def learning_enabled(body: LearningEnabledIn, conn: sqlite3.Connection = Depends(get_db)) -> LearningStatus:
    from app.learning import store as learning_store
    from app.learning import train as learning_train

    learning_store.set_enabled(conn, body.enabled)
    return LearningStatus(**learning_train.status(conn))


@router.post("/learning/train", response_model=LearningStatus)
def learning_train_now(conn: sqlite3.Connection = Depends(get_db)) -> LearningStatus:
    from app.learning import train as learning_train

    return LearningStatus(**learning_train.train_all(conn))


@router.post("/learning/apply", response_model=LearningApplyOut)
def learning_apply(conn: sqlite3.Connection = Depends(get_db)) -> LearningApplyOut:
    """Queue a fresh analysis for live photos and videos whose category, vault or mis-click decision
    was automatic, and a fresh text stage for documents typed automatically, so the learned models
    reach existing items. Tags learned for existing items are applied directly."""
    from app import ingest

    item_ids = [int(r["id"]) for r in conn.execute(
        "SELECT id FROM items WHERE deleted_at IS NULL AND kind IN ('image', 'video') AND analysis_status = 'done' "
        "AND (category_source = 'auto' OR nsfw_source = 'auto' OR misclick_source = 'auto') ORDER BY id"
    )]
    doc_ids = [int(r["item_id"]) for r in conn.execute(
        "SELECT d.item_id FROM documents d JOIN items i ON i.id = d.item_id WHERE d.source = 'auto' "
        "AND i.deleted_at IS NULL AND i.text_status = 'done' ORDER BY d.item_id"
    )]
    for item_id in item_ids:
        ingest.enqueue_reprocess(conn, item_id, ["analyze"])
    for item_id in doc_ids:
        ingest.enqueue_reprocess(conn, item_id, ["text"])
    tags = 0
    try:
        from app.learning import tags as learning_tags

        tags = int(learning_tags.apply_all(conn))
    except ImportError:
        pass
    return LearningApplyOut(items=len(item_ids), documents=len(doc_ids), tags=tags)


@router.delete("/learning", response_model=Ok)
def learning_reset() -> Ok:
    from app.learning import train as learning_train

    learning_train.reset()
    return Ok()
