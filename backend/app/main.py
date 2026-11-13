"""FastAPI application: API routers, file serving, the built SPA and the worker lifecycle.

CONTRACT (docs/ARCHITECTURE.md sections 1, 4 and 13).
- create_app() -> FastAPI. lifespan: exclusive flock on DATA_DIR/.instance.lock (refuses to start
  without it); storage.ensure_dirs(); db.init_db(); images.register_heif(); a warning when ffmpeg
  or ffprobe is missing; precompress_assets(); tiers.move_staged() (staged uploads an earlier layout
  left under DATA_DIR go to ORIGINALS_DIR, section 31); ingest.recover_on_startup(); when
  settings.workers_enabled: worker.start(), tiers.start() (the background move of legacy originals) and
  backup.start() (the nightly snapshot thread); all three stopped and the lock released on shutdown.
- Routers (each a plain APIRouter without a prefix; main mounts it): app.api.auth at /api/auth,
  app.api.library at /api, app.api.people at /api, app.api.documents at /api/documents,
  app.api.search at /api/search, app.api.uploads at /api/uploads, app.api.files at /files,
  app.api.system at /api/system. A router whose own paths already carry its prefix is mounted as it
  is. A router module that is missing or fails to import is logged and skipped.
- Middleware (pure ASGI, so uploads and file responses stream untouched): security headers on every
  response (nosniff, Referrer-Policy same-origin, X-Frame-Options DENY unless the response set its
  own) and Cache-Control "private, no-store" on /api responses that set none. JSON under /api is
  gzipped (>= 1 KB); /files is never compressed, so Range/206 keeps working.
- Unknown /api/* paths return a JSON 404. Every other GET serves frontend_dist: /assets/* (immutable,
  precompressed .gz siblings), real files, and index.html (no-cache) for client routes.
- POST /share-target (the PWA share target) is handled by the service worker in the browser; a
  browser without the worker gets a 303 to /upload?shared=1.
"""
from __future__ import annotations

import fcntl
import gzip
import importlib
import logging
import mimetypes
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import db, storage
from app.api.deps import require_user
from app.auth import sessions
from app.auth.sessions import SessionInfo
from app.config import settings

log = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "same-origin",
    "x-frame-options": "DENY",
}
API_CACHE = "private, no-store"
ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
IMMUTABLE = "public, max-age=31536000, immutable"
API_GZIP_MIN_BYTES = 1024
API_GZIP_LEVEL = 6  # dynamic JSON: level 6 is most of level 9's gain for much less CPU
# /assets/* types worth precompressing; images, fonts and media are compressed already.
COMPRESSIBLE_SUFFIXES = frozenset({".js", ".css", ".svg", ".json", ".map", ".html", ".txt", ".webmanifest"})
_MEDIA_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml", ".json": "application/json", ".map": "application/json",
                ".html": "text/html; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                ".webmanifest": "application/manifest+json"}
BUILD_HINT = (
    "Personal Content Cloud: the web app is not built yet.\n"
    "Run `npm install && npm run build` in frontend/ and reload this page.\n"
)
# (module, mount prefix). Each module exposes `router`.
ROUTERS: tuple[tuple[str, str], ...] = (
    ("app.api.auth", "/api/auth"),
    ("app.api.library", "/api"),
    ("app.api.people", "/api"),
    ("app.api.tags", ""),
    ("app.api.documents", "/api/documents"),
    ("app.api.search", "/api/search"),
    ("app.api.uploads", "/api/uploads"),
    ("app.api.files", "/files"),
    ("app.api.system", "/api/system"),
    ("app.api.timeline", "/api/timeline"),
    ("app.api.albums", "/api/albums"),
    ("app.api.map", "/api/map"),
    ("app.api.shares", "/api/shares"),
    ("app.api.share", ""),  # public share routes carry their own /api/s and /files/s paths (section 26)
    ("app.api.actions", ""),  # undo (section 33)
    ("app.api.model3d", ""),  # 3D heads carry their own /api/people/{id}/model3d and /files/model3d paths (section 36)
)


# ------------------------------------------------------------------ lifespan
INSTANCE_LOCK = ".instance.lock"


def _acquire_instance_lock() -> int:
    """Exclusive, non-blocking flock on DATA_DIR/.instance.lock; returns the open fd.

    Two processes on one DATA_DIR would requeue each other's running jobs, wipe tmp/ under each
    other and ingest the same upload twice (the loser's cleanup deleting the winner's files).
    uvicorn runs the lifespan before it binds the port, so this must come before recovery.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(settings.data_dir / INSTANCE_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError(
            f"Another instance is already using the data folder {settings.data_dir}. Stop it first."
        ) from None
    return fd


def _release_instance_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _startup_recovery() -> None:
    from app import ingest

    conn = db.connect()
    try:
        ingest.recover_on_startup(conn)
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app import backup, tiers, worker
    from app.media import images

    lock_fd = _acquire_instance_lock()
    started = False
    try:
        storage.ensure_dirs()
        db.init_db()
        try:
            images.register_heif()
        except Exception:
            log.exception("Could not register the HEIF opener; HEIC photos will be kept without previews")
        missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
        if missing:
            log.error("%s not found on PATH: videos cannot be processed until it is installed",
                      " and ".join(missing))
        try:
            snapshot_site()
            precompress_assets()
        except Exception:  # never block startup: the plain assets are served then
            log.exception("Could not precompress the frontend assets")
        try:
            tiers.move_staged()
        except Exception:  # the uploads concerned start over; the server still runs
            log.exception("Could not move the staged uploads to ORIGINALS_DIR; they stay under DATA_DIR")
        # Not swallowed: without it, queued uploads and pending jobs would never run.
        _startup_recovery()
        if settings.workers_enabled:
            worker.start()
            started = True
            tiers.start()
            backup.start()
        yield
    finally:
        try:
            if started:
                tiers.stop()
                backup.stop()
                worker.stop()
        finally:
            _release_instance_lock(lock_fd)


# ---------------------------------------------------------------- middleware
def _is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


class SecurityHeadersMiddleware:
    """Pure ASGI middleware adding the security headers (and no-store on /api) to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        api = _is_api_path(scope.get("path", ""))

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for k, v in SECURITY_HEADERS.items():
                    if k not in headers:
                        headers[k] = v
                if api and "cache-control" not in headers:
                    headers["cache-control"] = API_CACHE
            await send(message)

        await self.app(scope, receive, wrapped)


class SessionCookieMiddleware:
    """Re-issues the session cookie when `authenticate` slid the session's expiry (it leaves the raw
    token in `request.state.refresh_cookie`), so the browser's cookie lifetime slides with the row.
    Responses that already set or clear the cookie (login, logout) are left alone."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                token = (scope.get("state") or {}).get("refresh_cookie")
                if token:
                    headers = MutableHeaders(scope=message)
                    if not any(v.startswith(sessions.COOKIE + "=") for v in headers.getlist("set-cookie")):
                        probe = Response()
                        sessions.set_cookie(probe, Request(scope), token)
                        # only the cookie: a blank Response also carries "content-length: 0", and a second
                        # Content-Length makes browsers, URLSession and h11 drop the whole response
                        for k, v in probe.raw_headers:
                            if k == b"set-cookie":
                                headers.append(k.decode("latin-1"), v.decode("latin-1"))
            await send(message)

        await self.app(scope, receive, wrapped)


SIMPLE_UPLOAD_PATH = "/api/uploads/simple"
SIMPLE_UPLOAD_MARGIN = 1024 * 1024  # multipart framing and the form fields


class BodyLimitMiddleware:
    """Rejects an oversized simple upload before Starlette spools its multipart body to disk: the
    request must declare Content-Length (411 otherwise) and it must fit the configured limit plus a
    small margin (413 otherwise). Chunked uploads read their bodies with their own caps."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "") == SIMPLE_UPLOAD_PATH and scope.get("method") == "POST":
            raw = Headers(scope=scope).get("content-length")
            try:
                length = int(raw) if raw is not None else None
            except ValueError:
                length = -1
            if length is None or length < 0:
                await JSONResponse({"detail": "length_required"}, status_code=411)(scope, receive, send)
                return
            if length > settings.simple_upload_max_bytes + SIMPLE_UPLOAD_MARGIN:
                limit_mb = settings.simple_upload_max_bytes // (1024 * 1024)
                await JSONResponse(
                    {"detail": f"The file is larger than the {limit_mb} MB limit of this upload route"}, status_code=413
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)


class ApiGZipMiddleware:
    """Starlette's GZipMiddleware, applied to /api paths only.

    /files responses are left alone: originals, previews and videos are already compressed, and
    gzip would break streaming and byte ranges (Safari needs 206 for video).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=API_GZIP_MIN_BYTES, compresslevel=API_GZIP_LEVEL)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and _is_api_path(scope.get("path", "")):
            await self.gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)


# ------------------------------------------------------------------- the SPA
def _gz_sibling(path: Path) -> Path:
    return path.with_name(path.name + ".gz")


def _gz_fresh(src: Path, gz: Path) -> bool:
    try:
        return gz.stat().st_mtime >= src.stat().st_mtime
    except OSError:
        return False


# The site served: DATA_DIR/site, a copy of frontend_dist taken at startup (snapshot_site), or
# frontend_dist itself when the copy could not be made. A later `npm run build` then changes
# frontend/dist without swapping the site under a running server whose API it may not match.
_site_dir: Path | None = None


def site_dir() -> Path:
    return _site_dir or settings.frontend_dist


def snapshot_site() -> Path | None:
    """Copy settings.frontend_dist to DATA_DIR/site and serve from there; None (serve the build
    directly) when there is no build or the copy fails. Called once per startup, after the lock."""
    global _site_dir
    _site_dir = None
    src = settings.frontend_dist
    if not (src / "index.html").is_file():
        return None
    dest = settings.data_dir / "site"
    tmp = dest.with_name(f".site.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copytree(src, tmp, symlinks=False, ignore=shutil.ignore_patterns("*.gz", ".*.tmp"))
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except OSError:
        log.exception("Could not copy the frontend build to %s; serving %s directly", dest, src)
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    _site_dir = dest
    return dest


def precompress_assets() -> int:
    """Write <file>.gz (level 9) next to every compressible file under the site's assets that has
    none or an outdated one. A failed write is logged and skipped: the plain file is served then.
    Files that do not get smaller are left without a .gz. Returns the number written."""
    assets = site_dir() / "assets"
    if not assets.is_dir():
        return 0
    written = 0
    for src in sorted(assets.rglob("*")):
        if src.suffix.lower() not in COMPRESSIBLE_SUFFIXES or not src.is_file() or src.is_symlink():
            continue
        gz = _gz_sibling(src)
        if _gz_fresh(src, gz):
            continue
        tmp = gz.with_name(f".{gz.name}.{uuid.uuid4().hex}.tmp")
        try:
            data = src.read_bytes()
            packed = gzip.compress(data, compresslevel=9, mtime=0)
            if len(packed) >= len(data):
                continue
            tmp.write_bytes(packed)
            os.replace(tmp, gz)
            written += 1
        except OSError as e:
            log.warning("could not precompress %s (%s); serving it uncompressed", src.name, e)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    if written:
        log.info("precompressed %d frontend assets", written)
    return written


def _accepts_gzip(request: Request) -> bool:
    """True if Accept-Encoding lists gzip (or *) without q=0."""
    for part in request.headers.get("accept-encoding", "").split(","):
        token, _, params = part.strip().partition(";")
        if token.strip().lower() not in ("gzip", "*"):
            continue
        q = 1.0
        for param in params.split(";"):
            k, _, val = param.strip().partition("=")
            if k.strip().lower() == "q":
                try:
                    q = float(val)
                except ValueError:
                    q = 0.0
        if q > 0:
            return True
    return False


def _asset_response(static: Path, request: Request) -> FileResponse:
    """An /assets/* file: the precompressed .gz sibling when the client accepts gzip, else plain."""
    headers = {"cache-control": IMMUTABLE}
    suffix = static.suffix.lower()
    if suffix not in COMPRESSIBLE_SUFFIXES:
        return FileResponse(static, headers=headers)
    headers["vary"] = "Accept-Encoding"
    media_type = _MEDIA_TYPES.get(suffix) or mimetypes.guess_type(static.name)[0] or "application/octet-stream"
    gz = _gz_sibling(static)
    if _accepts_gzip(request) and gz.is_file() and _gz_fresh(static, gz):
        return FileResponse(gz, media_type=media_type, headers={**headers, "content-encoding": "gzip"})
    return FileResponse(static, media_type=media_type, headers=headers)



def _dist_file(rel_path: str) -> Path | None:
    """Existing file inside the served site for a URL path, or None (no escaping)."""
    dist = site_dir().resolve()
    if not rel_path or rel_path.endswith("/"):
        return None
    try:
        p = (dist / rel_path).resolve()
    except (OSError, ValueError):
        return None
    if dist not in p.parents or not p.is_file():
        return None
    return p


def _register_spa(app: FastAPI) -> None:
    @app.post("/share-target", include_in_schema=False)
    def share_target_fallback() -> RedirectResponse:
        # The service worker answers this in the browser; without it, land on the upload page.
        return RedirectResponse("/upload?shared=1", status_code=303)

    @app.api_route("/api/{rest:path}", methods=ALL_METHODS, include_in_schema=False)
    def api_not_found(rest: str) -> None:
        raise HTTPException(status_code=404, detail="Not found")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def spa(full_path: str, request: Request) -> Response:
        if full_path == "api" or full_path.startswith("files/") or full_path == "files":
            raise HTTPException(status_code=404, detail="Not found")
        dist = site_dir()
        index = dist / "index.html"
        if not index.is_file():
            if full_path in ("", "index.html"):
                return PlainTextResponse(BUILD_HINT, headers={"cache-control": "no-cache"})
            raise HTTPException(status_code=404, detail="Not found")
        static = _dist_file(full_path)
        if static is not None:
            if full_path.startswith("assets/"):
                return _asset_response(static, request)
            # index.html, the service worker and the manifest must never be cached stale.
            cache = "no-cache" if full_path in ("index.html", "sw.js", "manifest.webmanifest") else "public, max-age=3600"
            media_type = _MEDIA_TYPES.get(static.suffix.lower()) or mimetypes.guess_type(static.name)[0]
            return FileResponse(static, media_type=media_type, headers={"cache-control": cache})
        if full_path.startswith("assets/"):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(index, media_type="text/html", headers={"cache-control": "no-cache"})


# ----------------------------------------------------------------- routers
def _effective_paths(router: APIRouter) -> list[str]:
    return [getattr(r, "path", "") for r in router.routes]


def mount_prefix(router: APIRouter, prefix: str) -> str:
    """The prefix to include a router with: '' when its paths already start with /api or /files,
    '/api' when they already start with the rest of the prefix (for example '/search/...' for
    '/api/search'), else the given prefix."""
    paths = [p for p in _effective_paths(router) if p]
    if paths and all(p.startswith(("/api/", "/files/")) or p in ("/api", "/files") for p in paths):
        return ""
    if prefix.startswith("/api/"):
        tail = prefix[len("/api"):]
        if paths and all(p == tail or p.startswith(tail + "/") for p in paths):
            return "/api"
    return prefix


def _include_routers(app: FastAPI) -> list[str]:
    mounted: list[str] = []
    for module_name, prefix in ROUTERS:
        try:
            module = importlib.import_module(module_name)
            router = getattr(module, "router")
        except ImportError as e:
            log.warning("router %s not available (%s); its routes are skipped", module_name, e)
            continue
        except Exception:
            log.exception("router %s failed to import; its routes are skipped", module_name)
            continue
        app.include_router(router, prefix=mount_prefix(router, prefix))
        mounted.append(module_name)
    return mounted


# ----------------------------------------------------------------------- app
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.site_title,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,  # served by the session-protected route below
    )
    from app.undo import ActionMiddleware  # section 33: each owner write is one undoable action

    app.add_middleware(ActionMiddleware)
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(ApiGZipMiddleware)
    app.add_middleware(SessionCookieMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)  # added last = outermost
    app.state.routers = _include_routers(app)

    @app.get("/api/openapi.json", include_in_schema=False)
    def openapi_schema(_session: SessionInfo = Depends(require_user)) -> JSONResponse:
        # The route map is for the signed-in owner, not for anyone who can reach the instance.
        return JSONResponse(app.openapi())

    _register_spa(app)  # catch-alls last
    return app
