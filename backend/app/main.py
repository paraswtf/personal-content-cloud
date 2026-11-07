"""FastAPI application: API routers, file serving, the built SPA, and the worker lifecycle.

CONTRACT (implemented by the API agent).
- lifespan: exclusive flock on DATA_DIR/.instance.lock (refuse to start without it);
  storage.ensure_dirs(); db.init_db(); images.register_heif(); ingest.recover_on_startup();
  geotag_video.startup() (forgets renders, deletes DATA_DIR/tmp/geotag-video);
  worker.start() (skipped when settings.workers_enabled is False); worker.stop() on shutdown.
  The geotag video worker (app.media.geotag_video) starts on its first render, whatever
  settings.workers_enabled says, and geotag_video.shutdown() kills a running render on shutdown.
  The semantic search indexer (app.search.indexer) starts with the workers when search is
  enabled and its model is present, and stops first on shutdown.
- Routers: public (/api), uploads (/api/uploads), admin (/api/admin), admin exports (/api/admin/exports),
  files (/files).
- Middleware: if a request has ?key=<settings.access_key>, set the access cookie (app.auth) and,
  for HTML page requests, redirect to the same URL without the key. Security headers on every
  response: X-Content-Type-Options nosniff, Referrer-Policy same-origin, X-Frame-Options DENY.
- SPA: serve settings.frontend_dist: /assets/* and /brand/* as static files (immutable cache for
  /assets), every other GET that is not /api/* or /files/* returns index.html (no-cache). If the
  dist folder is missing, GET / returns a short plain-text hint to run `npm run build`.
- Unknown /api/* paths return JSON 404.
- Compression: JSON under /api is gzipped (>= 1 KB, when the client accepts gzip). /files is never
  touched, so file streaming and Range/206 stay intact. Compressible /assets/* files are served from
  precompressed .gz siblings (made at startup) with Content-Encoding: gzip and Vary: Accept-Encoding.
"""
from __future__ import annotations

import fcntl
import gzip
import hmac
import logging
import mimetypes
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from starlette.datastructures import MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.changelog import actors, trash
from app.changelog import context as changelog_context
from app import auth, db, ingest, storage, worker
from app.api import admin, exports, files, history, public, uploads
from app.config import settings
from app.media import geotag_video, images
from app.search import indexer as search_indexer

log = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "same-origin",
    "x-frame-options": "DENY",
}
ALL_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
IMMUTABLE = "public, max-age=31536000, immutable"
API_GZIP_MIN_BYTES = 1024
API_GZIP_LEVEL = 6  # dynamic JSON: level 6 is most of level 9's gain for much less CPU
# /assets/* types worth precompressing; images, fonts and media are compressed already.
COMPRESSIBLE_SUFFIXES = frozenset({".js", ".css", ".svg", ".json", ".map", ".html", ".txt"})
_MEDIA_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml", ".json": "application/json", ".map": "application/json",
                ".html": "text/html; charset=utf-8", ".txt": "text/plain; charset=utf-8"}
BUILD_HINT = (
    "SICSR Alumni Meet - Content HUB: the web app is not built yet.\n"
    "Run `npm install && npm run build` in frontend/ and reload this page.\n"
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    lock_fd = _acquire_instance_lock()
    started = False
    search_started = False
    try:
        storage.ensure_dirs()
        db.init_db()
        try:
            images.register_heif()
        except Exception:
            log.exception("Could not register the HEIF opener; HEIC photos will be rejected")
        missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
        if missing:
            log.error(
                "%s not found on PATH: videos cannot be processed and will wait in DATA_DIR/failed "
                "until it is installed and an admin retries them",
                " and ".join(missing),
            )
        try:
            precompress_assets()
        except Exception:  # never block startup: the plain assets are served then
            log.exception("Could not precompress the frontend assets")
        conn = db.connect()
        try:
            # File moves of transactions a crash interrupted go back first (app/changelog/trash.py).
            trash.recover(conn)
            # Not swallowed: without it, queued uploads and pending face/preview jobs would never run.
            changelog_context.declare(conn, actors.system_spec(
                "maintenance.startup", "Startup checks and upgrades", maintenance=True))
            try:
                ingest.recover_on_startup(conn)
            finally:
                changelog_context.finish(conn)
        finally:
            conn.close()
        geotag_video.startup()
        if settings.workers_enabled:
            worker.start()
            started = True
            try:
                search_started = search_indexer.start()
            except Exception:  # search is optional: never block startup
                log.exception("Could not start the search indexer")
        yield
    finally:
        try:
            geotag_video.shutdown()
        except Exception:
            log.exception("Could not stop the geotag video renders")
        try:
            if search_started:
                search_indexer.stop()
        except Exception:
            log.exception("Could not stop the search indexer")
        try:
            if started:
                worker.stop()
        finally:
            _release_instance_lock(lock_fd)


# ---------------------------------------------------------------- middleware
def _is_page_request(scope: Scope, headers: MutableHeaders) -> bool:
    path = scope.get("path", "")
    if scope.get("method") != "GET":
        return False
    if path.startswith("/api/") or path == "/api" or path.startswith("/files/"):
        return False
    return "text/html" in headers.get("accept", "")


class AccessKeyAndHeadersMiddleware:
    """Pure ASGI middleware: handles ?key=<access key> and adds security headers.

    Pure ASGI (not BaseHTTPMiddleware) so upload bodies and file responses stream untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        cookie_headers: list[tuple[bytes, bytes]] = []
        query = scope.get("query_string", b"").decode("latin-1")
        expected = settings.access_key
        if expected and "key=" in query:
            pairs = parse_qsl(query, keep_blank_values=True)
            given = next((v for k, v in pairs if k == "key"), None)
            if given is not None and hmac.compare_digest(given.encode(), expected.encode()):
                request = Request(scope)
                cookie_resp = Response()
                auth.set_access_cookie(cookie_resp, request)
                cookie_headers = [(k, v) for k, v in cookie_resp.raw_headers if k == b"set-cookie"]
                if _is_page_request(scope, MutableHeaders(scope=scope)):
                    rest = urlencode([(k, v) for k, v in pairs if k != "key"])
                    # scope["path"] is percent-decoded: /%2F%2Fevil.example would become the
                    # scheme-relative //evil.example. Collapse leading slashes and backslashes.
                    path = "/" + scope.get("path", "/").lstrip("/\\")
                    target = scope.get("root_path", "") + path + (f"?{rest}" if rest else "")
                    redirect = RedirectResponse(target, status_code=303)
                    redirect.headers["cache-control"] = "no-store"
                    redirect.raw_headers.extend(cookie_headers)
                    await redirect(scope, receive, self._wrap_send(send, []))
                    return
                scope = self._with_access_cookie(scope)

        await self.app(scope, receive, self._wrap_send(send, cookie_headers))

    @staticmethod
    def _with_access_cookie(scope: Scope) -> Scope:
        """The request that carries a valid key is itself treated as unlocked."""
        crumb = f"{auth.ACCESS_COOKIE}={auth.access_token()}".encode("latin-1")
        headers = [(k, v) for k, v in scope.get("headers", []) if k != b"cookie"]
        existing = b"; ".join(v for k, v in scope.get("headers", []) if k == b"cookie")
        headers.append((b"cookie", existing + b"; " + crumb if existing else crumb))
        return {**scope, "headers": headers}

    @staticmethod
    def _wrap_send(send: Send, extra: list[tuple[bytes, bytes]]) -> Send:
        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for k, v in SECURITY_HEADERS.items():
                    if k not in headers:
                        headers[k] = v
                for k, v in extra:
                    headers.append(k.decode("latin-1"), v.decode("latin-1"))
            await send(message)

        return wrapped


def _is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


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


def precompress_assets() -> int:
    """Write <file>.gz (level 9) next to every compressible file under frontend_dist/assets that has
    none or an outdated one. A failed write is logged and skipped: the plain file is served then.
    Files that do not get smaller are left without a .gz. Returns the number written."""
    assets = settings.frontend_dist / "assets"
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
    """Existing file inside settings.frontend_dist for a URL path, or None (no escaping)."""
    dist = settings.frontend_dist.resolve()
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
    @app.api_route("/api/{rest:path}", methods=ALL_METHODS, include_in_schema=False)
    def api_not_found(rest: str) -> None:
        raise HTTPException(status_code=404, detail="Not found")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def spa(full_path: str, request: Request) -> Response:
        if full_path == "api" or full_path.startswith("files/") or full_path == "files":
            raise HTTPException(status_code=404, detail="Not found")
        dist = settings.frontend_dist
        index = dist / "index.html"
        if not index.is_file():
            if full_path in ("", "index.html"):
                return PlainTextResponse(BUILD_HINT, headers={"cache-control": "no-cache"})
            raise HTTPException(status_code=404, detail="Not found")
        static = _dist_file(full_path)
        if static is not None:
            if full_path.startswith("assets/"):
                return _asset_response(static, request)
            if full_path == "index.html":
                cache = "no-cache"
            else:
                cache = "public, max-age=3600"
            return FileResponse(static, headers={"cache-control": cache})
        if full_path.startswith(("assets/", "brand/")):
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(index, media_type="text/html", headers={"cache-control": "no-cache"})


# ----------------------------------------------------------------------- app
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.site_title,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(ApiGZipMiddleware)
    app.add_middleware(AccessKeyAndHeadersMiddleware)  # added last = outermost
    app.include_router(public.router, prefix="/api")
    app.include_router(uploads.router, prefix="/api/uploads")
    app.include_router(admin.router, prefix="/api/admin")
    app.include_router(exports.router, prefix="/api/admin")
    app.include_router(history.router, prefix="/api/admin")
    app.include_router(files.router, prefix="/files")
    _register_spa(app)  # catch-alls last
    return app


app = create_app()
