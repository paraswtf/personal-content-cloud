"""3D heads of a person, one per look (docs/ARCHITECTURE.md section 36): the job, its state and the stored model.

Nothing here runs until the owner presses Generate (start()). The job plans the looks (app.analysis.looks), writes a
head-and-shoulders picture per look, and starts the generator: a separate program with its own Python environment
(the gen3d package at the repo root, settings GEN3D_*), as a child process that lives only while the job runs, so
PyTorch stays out of the server and the GPU and the memory are free the rest of the time.

Files, under DATA_DIR/people3d/<person id>/:
  current              the name of the folder in use (a time stamp)
  <stamp>/model.json   the manifest: {rev, person_id, generated_at, seconds, generator{name, version}, options,
                       birth_year, scanned, usable, looks: [{key, label, age, age_band, hair, hair_label, facial_hair,
                       facial_hair_label, photos, face_id, item_id, item_uid, taken_at, faults, mesh, picture, bytes,
                       vertices, faces}]}
  <stamp>/<key>.glb    a look's mesh;  <stamp>/<key>.jpg  the picture it was made from (512 px)
  work/                the job's scratch folder (pictures, job.json, out/), removed when the job ends
  error.json           why the last run failed, until the next start

The child (see gen3d/__main__.py for the contract):
- its own process group, with oom_score_adj 1000 so a memory limit takes it and not the server;
- inside a sandbox when the real generator runs (settings.gen3d_sandbox, bwrap): the whole filesystem read-only
  but the work folder, no network, its own process ids. The generator runs third-party model code on the owner's
  face photos; the sandbox makes sure that code can neither send anything out nor change a file;
- watched once a second: stopped when the resident memory of its whole process tree passes GEN3D_MAX_RSS_MB, when a
  head takes longer than GEN3D_LOOK_TIMEOUT_S, on cancel, and when the server exits.

One job at a time for the whole app (one GPU). A look whose source photo was deleted, put in the trash or moved to
the vault since is neither listed nor served. Folders of people who no longer exist are removed by sweep()
(housekeeping); a deleted person's model stops being served at once, as its routes ask for the person first.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import storage
from app.analysis import looks as looks_mod
from app.config import REPO_ROOT, settings
from app.util import now_iso

log = logging.getLogger(__name__)

RUNNER_KEY = "model3d"
FOLDER = "people3d"
STAMP_RE = re.compile(r"^\d{8}T\d{6}$")
NAME_RE = re.compile(r"^[a-z0-9_-]{1,40}\.(glb|jpg)$")
PICTURE_SIDE = 512
PICTURE_QUALITY = 88
CHECK_TTL_S = 300.0        # how long "the generator is installed" is believed
CHECK_RETRY_S = 30.0       # ... and "it is not"
CHECK_TIMEOUT_S = 120.0
STOP_WAIT_S = 5.0
EXIT_WAIT_S = 30.0         # after its last line the child has this long to exit by itself
DETAIL_MIN, DETAIL_MAX, DETAIL_DEFAULT = 1, 3, 2
SANDBOX_BIN = "bwrap"
OPTION_KEYS = ("max_looks", "detail", "framing")

STEP_LOOKS = "Finding looks"
STEP_PHOTOS = "Preparing photos"
STEP_BUILD = "Building heads"
STEP_SAVE = "Saving"


class Busy(Exception):
    """Another generation is running."""


class GeneratorMissing(Exception):
    """The generator (or the sandbox it must run in) is not installed."""


class NotEnoughPhotos(Exception):
    """The person has too few good face photos."""


class _Cancelled(Exception):
    pass


class _Failed(Exception):
    """The job failed for a reason the owner can read."""


_lock = threading.Lock()
_state: dict[str, Any] | None = None       # the running job, or None
_proc: subprocess.Popen[bytes] | None = None
_check: tuple[float, tuple[Any, ...], dict[str, Any]] | None = None
_atexit_done = False


# ------------------------------------------------------------------ paths
def root() -> Path:
    return storage.root() / FOLDER


def person_dir(person_id: int) -> Path:
    return root() / str(int(person_id))


def current_rev(person_id: int) -> str | None:
    try:
        rev = (person_dir(person_id) / "current").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return rev if STAMP_RE.fullmatch(rev) and (person_dir(person_id) / rev / "model.json").is_file() else None


def manifest(person_id: int) -> dict[str, Any] | None:
    rev = current_rev(person_id)
    if rev is None:
        return None
    try:
        doc = json.loads((person_dir(person_id) / rev / "model.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("looks"), list) else None


def file_path(person_id: int, rev: str, name: str) -> Path | None:
    """The path of a model file when rev is the model in use and name is well-formed; None otherwise."""
    if not NAME_RE.fullmatch(name) or rev != current_rev(person_id):
        return None
    return person_dir(person_id) / rev / name


def file_url(person_id: int, rev: str, name: str) -> str:
    return f"/files/model3d/{int(person_id)}/{rev}/{name}"


# ------------------------------------------------------------------ the generator and its sandbox
def _python() -> Path:
    if settings.gen3d_fake:
        return Path(sys.executable)
    return settings.gen3d_python or (settings.gen3d_dir / "venv" / "bin" / "python")


def _weights_dir() -> Path:
    return settings.gen3d_dir / "weights"


def sandbox_bin() -> str | None:
    for candidate in (f"/usr/bin/{SANDBOX_BIN}", shutil.which(SANDBOX_BIN)):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return None


def sandbox_wanted(fake: bool) -> bool:
    """Whether the child runs inside the sandbox: "on" always, "off" never, "auto" for the real generator when bwrap
    is there (the stand-in generator is our own code and runs without it). GeneratorMissing when it is wanted with
    "on" and bwrap is not installed."""
    mode = settings.gen3d_sandbox
    if mode == "off":
        return False
    if mode == "on":
        if sandbox_bin() is None:
            raise GeneratorMissing("The sandbox for the generator (bwrap) is not installed.")
        return True
    return not fake and sandbox_bin() is not None


def sandbox_prefix(work: Path) -> list[str]:
    """The bwrap command line in front of the generator's: everything read-only except `work`, a private /tmp, no
    network, its own process ids and IPC, gone when this process goes."""
    return [
        sandbox_bin() or SANDBOX_BIN,
        "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        "--bind", str(work), str(work),
        "--unshare-net", "--unshare-ipc", "--unshare-pid", "--die-with-parent", "--new-session",
        "--setenv", "HOME", "/tmp", "--setenv", "XDG_CACHE_HOME", "/tmp/cache",
        "--setenv", "HF_HUB_OFFLINE", "1", "--setenv", "TRANSFORMERS_OFFLINE", "1",
        # libraries that keep a cache beside their code or in the home folder get a place they can write
        "--setenv", "MPLCONFIGDIR", "/tmp/mpl", "--setenv", "NUMBA_CACHE_DIR", "/tmp/numba",
    ]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    weights = _weights_dir()
    env.update(PYTHONUNBUFFERED="1", HF_HOME=str(weights / "hf"), U2NET_HOME=str(weights / "u2net"))
    env.pop("PYTHONHOME", None)
    return env


def _command(args: list[str], work: Path, fake: bool) -> list[str]:
    """The child's full command line. The shell in front raises the child's oom_score_adj before anything else
    runs (inherited by everything it starts), then becomes the command itself."""
    cmd = [str(_python()), "-m", "gen3d", *args]
    if sandbox_wanted(fake):
        cmd = [*sandbox_prefix(work), *cmd]
    return ["/bin/sh", "-c", '{ echo 1000 > /proc/self/oom_score_adj; } 2>/dev/null; exec "$@"', "gen3d", *cmd]


def generator(refresh: bool = False) -> dict[str, Any]:
    """{"installed", "name", "device", "reason"}: what a Generate would run with. The real generator is asked with
    `python -m gen3d --check` (remembered for CHECK_TTL_S; CHECK_RETRY_S while it is missing)."""
    global _check
    fake = bool(settings.gen3d_fake) or settings.gen3d_generator == "fake"
    if fake:
        try:
            sandbox_wanted(True)
        except GeneratorMissing as e:
            return {"installed": False, "name": None, "device": None, "reason": str(e)}
        return {"installed": True, "name": "fake", "device": "cpu", "reason": None}
    python = _python()
    key = (str(python), str(_weights_dir()), settings.gen3d_generator, settings.gen3d_sandbox)
    now = time.monotonic()
    with _lock:
        cached = _check
    if not refresh and cached is not None and cached[1] == key:
        ttl = CHECK_TTL_S if cached[2]["installed"] else CHECK_RETRY_S
        if now - cached[0] < ttl:
            return dict(cached[2])
    out = _ask_generator(python)
    with _lock:
        _check = (now, key, out)
    return dict(out)


MISSING = "The 3D generator is not installed. Run scripts/setup_3d.sh once on the server."


def _ask_generator(python: Path) -> dict[str, Any]:
    missing = {"installed": False, "name": None, "device": None, "reason": MISSING}
    if not python.is_file():
        return missing
    try:
        root().mkdir(parents=True, exist_ok=True)
        cmd = _command(["--check", "--weights-dir", str(_weights_dir())], root(), fake=False)
    except GeneratorMissing as e:
        return {**missing, "reason": str(e)}
    try:
        done = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_child_env(), capture_output=True, timeout=CHECK_TIMEOUT_S,
                              check=False)
        lines = [ln for ln in done.stdout.decode("utf-8", "replace").splitlines() if ln.strip().startswith("{")]
        info = json.loads(lines[-1]) if lines else None
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        log.warning("3D heads: the generator did not answer the check: %s", e)
        return missing
    if not isinstance(info, dict):
        log.warning("3D heads: the generator's check said nothing (exit %s): %s", done.returncode,
                    done.stderr.decode("utf-8", "replace")[-300:])
        return missing
    wanted = settings.gen3d_generator
    found = info.get("backends") or {}
    name = info.get("auto") if wanted == "auto" else (wanted if found.get(wanted) else None)
    if not name:
        return missing
    return {"installed": True, "name": str(name), "device": info.get("device"), "reason": None}


# ------------------------------------------------------------------ status
def _live_items(conn: sqlite3.Connection, item_ids: list[int]) -> set[int]:
    """The items among these that are in the library: not deleted, not in the trash, not in the vault."""
    ids = sorted({int(i) for i in item_ids})
    if not ids:
        return set()
    marks = ",".join("?" * len(ids))
    return {int(r[0]) for r in conn.execute(
        f"SELECT id FROM items WHERE id IN ({marks}) AND deleted_at IS NULL AND nsfw = 0", ids)}


def listed_looks(conn: sqlite3.Connection, person_id: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """(the manifest, its looks that may be shown): a look goes when its source photo left the library."""
    doc = manifest(person_id)
    if doc is None:
        return None, []
    looks = [lk for lk in doc["looks"] if isinstance(lk, dict)]
    live = _live_items(conn, [lk.get("item_id") or 0 for lk in looks])
    return doc, [lk for lk in looks if int(lk.get("item_id") or 0) in live]


def running() -> dict[str, Any] | None:
    with _lock:
        return dict(_state) if _state is not None else None


def _error(person_id: int) -> str | None:
    try:
        doc = json.loads((person_dir(person_id) / "error.json").read_text(encoding="utf-8"))
        return str(doc.get("error") or "") or None
    except (OSError, ValueError):
        return None


def status(conn: sqlite3.Connection, person_id: int) -> dict[str, Any]:
    """What the 3D tab shows. Reading it never starts anything."""
    gen = generator()
    photos = looks_mod.usable_photos(conn, person_id)
    job = running()
    mine = job is not None and int(job["person_id"]) == int(person_id)
    doc, shown = listed_looks(conn, person_id)
    error = None if mine else _error(person_id)
    reason = gen["reason"]
    if reason is None and photos < looks_mod.MIN_USABLE:
        reason = (f"There are too few clear, large face photos of this person: {photos} found, "
                  f"{looks_mod.MIN_USABLE} needed.")
    rev = doc.get("rev") if doc else None
    out_looks = []
    for lk in shown:
        out_looks.append({
            **{k: lk.get(k) for k in ("key", "label", "age", "age_band", "hair", "hair_label", "facial_hair",
                                      "facial_hair_label", "photos", "taken_at", "item_uid", "bytes", "vertices")},
            "picture_url": file_url(person_id, rev, lk["picture"]),
            "mesh_url": file_url(person_id, rev, lk["mesh"]),
        })
    return {
        "available": reason is None,
        "reason": reason,
        "generator": gen["name"],
        "usable_photos": photos,
        "state": "running" if mine else "done" if out_looks else "error" if error else "none",
        "busy": job is not None and not mine,
        "progress": {"step": job["step"], "done": job["done"], "total": job["total"]} if mine else None,
        "error": error,
        "generated_at": doc.get("generated_at") if doc and out_looks else None,
        "options": (doc.get("options") or None) if doc and out_looks else None,
        "looks": out_looks,
    }


# ------------------------------------------------------------------ start, cancel, remove
def clean_options(options: dict[str, Any] | None) -> dict[str, Any]:
    o = options or {}
    out: dict[str, Any] = {
        "max_looks": max(looks_mod.LOOKS_MIN, min(looks_mod.LOOKS_MAX, int(o.get("max_looks", looks_mod.LOOKS_DEFAULT)))),
        "detail": max(DETAIL_MIN, min(DETAIL_MAX, int(o.get("detail", DETAIL_DEFAULT)))),
        "framing": round(max(looks_mod.FRAMING_MIN, min(looks_mod.FRAMING_MAX,
                                                         float(o.get("framing", looks_mod.FRAMING_DEFAULT)))), 2),
    }
    if o.get("generator") == "fake":
        out["generator"] = "fake"
    return out


def start(conn: sqlite3.Connection, person_id: int, options: dict[str, Any] | None = None) -> None:
    """Start a generation for the person. Busy, GeneratorMissing or NotEnoughPhotos when it cannot. With the worker
    threads off (tests) the job runs here, before this returns."""
    global _state
    from app import worker

    opts = clean_options(options)
    fake = opts.get("generator") == "fake" or bool(settings.gen3d_fake) or settings.gen3d_generator == "fake"
    if fake:
        sandbox_wanted(True)  # GeneratorMissing when the sandbox is demanded and missing
    else:
        gen = generator(refresh=True)
        if not gen["installed"]:
            raise GeneratorMissing(gen["reason"] or MISSING)
    if looks_mod.usable_photos(conn, person_id) < looks_mod.MIN_USABLE:
        raise NotEnoughPhotos()
    with _lock:
        if _state is not None:
            raise Busy()
        _state = {"person_id": int(person_id), "step": STEP_LOOKS, "done": 0, "total": 0, "started_at": now_iso(),
                  "cancel": False, "remove": False}
    if not settings.workers_enabled:
        _run(int(person_id), opts, fake)
        return
    if worker.run_in_background(RUNNER_KEY, _run, int(person_id), opts, fake) is None:
        with _lock:
            _state = None
        raise Busy()


def cancel(person_id: int | None = None) -> bool:
    """Stop the running job (the one of this person when given). True when there was one."""
    with _lock:
        if _state is None or (person_id is not None and int(_state["person_id"]) != int(person_id)):
            return False
        _state["cancel"] = True
        proc = _proc
    if proc is not None:
        _stop_child(proc)
    return True


def remove(conn: sqlite3.Connection, person_id: int) -> None:  # noqa: ARG001 - conn kept for the callers' shape
    """Delete the person's model, stopping their running job first."""
    with _lock:
        mine = _state is not None and int(_state["person_id"]) == int(person_id)
        if mine:
            _state["remove"] = True  # type: ignore[index]
    if mine:
        cancel(person_id)  # the job removes the folder as it ends
    else:
        storage.remove_tree(person_dir(person_id))


def sweep(conn: sqlite3.Connection) -> int:
    """Remove the model folders of people who no longer exist (housekeeping). Returns how many went."""
    base = root()
    if not base.is_dir():
        return 0
    job = running()
    gone = 0
    for entry in base.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if job is not None and int(job["person_id"]) == pid:
            continue
        row = conn.execute("SELECT kind FROM persons WHERE id = ?", (pid,)).fetchone()
        if row is None or row["kind"] == "pet":
            storage.remove_tree(entry)
            gone += 1
    return gone


def shutdown() -> None:
    """Stop the child when the server exits."""
    with _lock:
        proc = _proc
        if _state is not None:
            _state["cancel"] = True
    if proc is not None:
        _stop_child(proc)


# ------------------------------------------------------------------ the child process
def _tree(pid: int) -> list[int]:
    """pid and every process that descends from it (Linux /proc; just [pid] elsewhere)."""
    children: dict[int, list[int]] = {}
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as f:
                    stat = f.read().decode("ascii", "replace")
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(ppid, []).append(int(name))
    except OSError:
        return [pid]
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, ()))
    return out


def tree_rss_mb(pid: int) -> float:
    """The resident memory of pid and all its descendants, in MB (the sandbox puts the generator two processes
    below the one that was started)."""
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    total = 0
    for p in _tree(pid):
        try:
            with open(f"/proc/{p}/statm", "rb") as f:
                total += int(f.read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
    return total / (1024.0 * 1024.0)


def _running(pid: int) -> bool:
    """Whether the process exists and is not a zombie waiting to be collected."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return f.read().decode("ascii", "replace").rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _stop_child(proc: subprocess.Popen[bytes]) -> None:
    """Stop the child and everything under it: SIGTERM to its process group, then SIGKILL to the group and to any
    descendant that sits in another group (the sandbox starts a new session inside), and wait until they are
    gone, so the memory and the GPU are free when this returns."""
    if proc.poll() is not None:
        return
    below = _tree(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        pass
    for kill in (lambda: os.killpg(proc.pid, signal.SIGKILL), *[(lambda p=p: os.kill(p, signal.SIGKILL)) for p in below]):
        try:
            kill()
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        log.warning("3D heads: the generator (pid %s) did not stop", proc.pid)
    deadline = time.monotonic() + STOP_WAIT_S
    others = [p for p in below if p != proc.pid]
    while any(_running(p) for p in others) and time.monotonic() < deadline:
        time.sleep(0.05)


def _set(**values: Any) -> None:
    with _lock:
        if _state is not None:
            _state.update(values)


def _cancelled() -> bool:
    with _lock:
        return _state is None or bool(_state["cancel"])


def _tail(path: Path, limit: int = 400) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][-limit:] if lines else ""


def _run_child(work: Path, job_path: Path, total: int, fake: bool) -> None:
    """Start the generator on the job file and follow it to its end. _Failed with the reason when it ends badly,
    _Cancelled when it was stopped on request."""
    global _proc, _atexit_done
    cmd = _command(["--job", str(job_path)], work, fake)
    err_path = work / "stderr.log"
    with open(err_path, "wb") as err:
        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=_child_env(), stdout=subprocess.PIPE, stderr=err,
                                    stdin=subprocess.DEVNULL, process_group=0)
        except OSError as e:
            raise _Failed(f"The generator could not be started: {e}") from e
    with _lock:
        _proc = proc
        if not _atexit_done:
            atexit.register(shutdown)
            _atexit_done = True
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw.decode("utf-8", "replace").strip())
        lines.put(None)

    threading.Thread(target=pump, name="pcc-model3d-out", daemon=True).start()
    limit_mb = float(settings.gen3d_max_rss_mb)
    timeout = float(settings.gen3d_look_timeout_s)
    last_look = time.monotonic()
    reason: str | None = None
    ended = False
    try:
        while not ended:
            try:
                line = lines.get(timeout=1.0)
            except queue.Empty:
                line = ""
            if line is None:
                ended = True
            elif line.startswith("progress:"):
                try:
                    done = int(line.split(":", 1)[1].split("/")[0])
                    _set(step=STEP_BUILD, done=done, total=total)
                except ValueError:
                    pass
            elif line.startswith("look:"):
                last_look = time.monotonic()
            if ended:
                break
            if _cancelled():
                raise _Cancelled()
            used = tree_rss_mb(proc.pid)
            if used > limit_mb:
                reason = (f"The generator needed more memory than it may use ({int(used)} MB of {int(limit_mb)} MB) "
                          "and was stopped.")
                break
            if time.monotonic() - last_look > timeout:
                reason = f"A head took longer than {int(timeout // 60) or 1} minutes and the generator was stopped."
                break
        if ended:  # it closed its output: let it exit by itself, so its exit code is its own
            try:
                proc.wait(EXIT_WAIT_S)
            except subprocess.TimeoutExpired:
                reason = "The generator did not exit and was stopped."
    finally:
        _stop_child(proc)
        with _lock:
            _proc = None
    if reason is not None:
        raise _Failed(reason)
    if _cancelled():
        raise _Cancelled()
    code = proc.returncode
    if code not in (0, 3):
        said = _tail(err_path)
        raise _Failed(f"The generator stopped with an error{': ' + said if said else f' (exit {code})'}.")


# ------------------------------------------------------------------ the job
def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    storage.write_bytes_atomic(path, json.dumps(doc, indent=1, sort_keys=True).encode("utf-8"))


def _prepare(conn: sqlite3.Connection, plan: looks_mod.Plan, work: Path, framing: float) -> list[dict[str, Any]]:
    """Write each look's picture (the look's best photo, else a runner-up whose pixels read) and return the looks
    that have one, as manifest rows."""
    rows: list[dict[str, Any]] = []
    for n, look in enumerate(plan.looks, start=1):
        if _cancelled():
            raise _Cancelled()
        by_id = {r.face_id: r for r in plan.readings}
        chosen = None
        for face_id in [look.face_id, *look.runners_up]:
            if looks_mod.write_picture(conn, face_id, work / f"{look.key}.png", framing):
                chosen = by_id.get(face_id)
                break
        _set(step=STEP_PHOTOS, done=n, total=len(plan.looks))
        if chosen is None:
            continue
        rows.append({
            "key": look.key, "label": look.label, "age": look.age, "age_band": look.age_band, "hair": look.hair,
            "hair_label": look.hair_label, "facial_hair": look.facial_hair, "facial_hair_label": look.facial_hair_label,
            "photos": look.photos, "face_id": chosen.face_id, "item_id": chosen.item_id, "item_uid": chosen.item_uid,
            "taken_at": chosen.taken_at, "faults": list(chosen.faults),
        })
    return rows


def _collect(person_id: int, work: Path, rows: list[dict[str, Any]], plan: looks_mod.Plan, options: dict[str, Any],
             seconds: float) -> None:
    """Move the meshes and pictures of the looks that worked into a new folder and make it the model in use."""
    from PIL import Image

    try:
        result = json.loads((work / "out" / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise _Failed("The generator left no result.") from e
    built = {r.get("key"): r for r in result.get("looks") or [] if isinstance(r, dict) and r.get("ok")}
    rev = _stamp()
    base = person_dir(person_id)
    dest = base / rev
    if dest.exists():
        storage.remove_tree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    looks: list[dict[str, Any]] = []
    for row in rows:
        made = built.get(row["key"])
        mesh = work / "out" / f"{row['key']}.glb"
        if made is None or not mesh.is_file():
            continue
        target = dest / f"{row['key']}.glb"
        storage.move_into_place(mesh, target)
        with Image.open(work / f"{row['key']}.png") as im:
            im.convert("RGB").resize((PICTURE_SIDE, PICTURE_SIDE), Image.Resampling.LANCZOS).save(
                dest / f"{row['key']}.jpg", "JPEG", quality=PICTURE_QUALITY)
        looks.append({**row, "mesh": target.name, "picture": f"{row['key']}.jpg", "bytes": target.stat().st_size,
                      "vertices": int(made.get("vertices") or 0), "faces": int(made.get("faces") or 0)})
    if not looks:
        storage.remove_tree(dest)
        failed = [r.get("error") for r in result.get("looks") or [] if isinstance(r, dict) and r.get("error")]
        raise _Failed("No head could be built" + (f": {failed[-1]}" if failed else "") + ".")
    _write_json(dest / "model.json", {
        "rev": rev, "person_id": int(person_id), "generated_at": now_iso(), "seconds": round(seconds, 1),
        "generator": result.get("generator") or {}, "options": {k: options[k] for k in OPTION_KEYS},
        "birth_year": plan.birth_year, "scanned": plan.scanned, "usable": plan.usable, "looks": looks,
    })
    storage.write_bytes_atomic(base / "current", rev.encode("ascii"))
    for entry in base.iterdir():  # the older models
        if entry.is_dir() and STAMP_RE.fullmatch(entry.name) and entry.name != rev:
            storage.remove_tree(entry)


def _run(person_id: int, options: dict[str, Any], fake: bool) -> None:
    """The whole job, on its own connection. Ends by clearing the running state whatever happened."""
    global _state
    from app import worker
    from app.db import connect

    if settings.workers_enabled:
        worker._lower_priority()
    base = person_dir(person_id)
    work = base / "work"
    started = time.monotonic()
    conn = connect()
    try:
        storage.remove_tree(work)
        (work / "out").mkdir(parents=True, exist_ok=True)
        storage.safe_unlink(base / "error.json")
        try:
            plan = looks_mod.plan(conn, person_id, options["max_looks"],
                                  progress=lambda done, total: _set(step=STEP_LOOKS, done=done, total=total))
        except looks_mod.NoPhotos as e:
            raise _Failed("There are too few clear, large face photos of this person.") from e
        if _cancelled():
            raise _Cancelled()
        if not plan.looks:
            raise _Failed("No look could be told from the photos: none of them shows the face clearly enough.")
        rows = _prepare(conn, plan, work, options["framing"])
        if not rows:
            raise _Failed("The photos of the looks could not be read.")
        job = {
            "out_dir": str(work / "out"), "weights_dir": str(_weights_dir()),
            "generator": "fake" if fake else settings.gen3d_generator, "device": "auto",
            "options": {"detail": options["detail"], "seed": 0},
            "looks": [{"key": r["key"], "image": str(work / f"{r['key']}.png")} for r in rows],
        }
        _write_json(work / "job.json", job)
        _set(step=STEP_BUILD, done=0, total=len(rows))
        _run_child(work, work / "job.json", len(rows), fake)
        _set(step=STEP_SAVE, done=len(rows), total=len(rows))
        _collect(person_id, work, rows, plan, options, time.monotonic() - started)
        log.info("3D heads: person %s done in %.0f s", person_id, time.monotonic() - started)
    except _Cancelled:
        log.info("3D heads: the job of person %s was stopped", person_id)
    except _Failed as e:
        log.warning("3D heads: person %s: %s", person_id, e)
        _note_error(base, str(e))
    except Exception as e:  # noqa: BLE001 - the owner sees that it failed; the log says why
        log.exception("3D heads: the job of person %s failed", person_id)
        _note_error(base, f"The generation failed ({type(e).__name__}).")
    finally:
        conn.close()
        storage.remove_tree(work)
        with _lock:
            drop = _state is not None and bool(_state.get("remove"))
            _state = None
        if drop:
            storage.remove_tree(base)


def _note_error(base: Path, text: str) -> None:
    try:
        _write_json(base / "error.json", {"error": text, "at": now_iso()})
    except OSError:
        log.warning("3D heads: could not write %s", base / "error.json")
