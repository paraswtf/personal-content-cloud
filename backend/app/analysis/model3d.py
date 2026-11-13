"""3D heads of a person, one per look (docs/ARCHITECTURE.md section 36): the job, its state and the stored model.

Nothing here runs until the owner presses Generate (start()). The job plans the looks (app.analysis.looks), writes a
head-and-shoulders picture per look at the photo's own resolution (1024 to 3072 px a side), and starts the
generator: a separate program with its own Python environment (the gen3d package at the repo root, settings
GEN3D_*), as a child process that lives only while the job runs, so PyTorch stays out of the server and the GPU and
the memory are free the rest of the time.

Options (OPTION_KEYS, clean_options): max_looks, detail, framing, and for the generator's finishing of a head
`smoothing` (0 off, 1 light, 2 normal, 3 strong), `texture` (0 colours only; 1 and 2 the photo laid over the
front as a texture of at most 1024 and 2048 px; 3 the photo at its own size, which is the picture's: at most
3072 px) and `relief` (1: the face's shape corrected from a depth model when its files are installed; 0: not).
The server only passes the last three on.

Files, under DATA_DIR/people3d/<person id>/:
  current              the name of the folder in use (a time stamp)
  <stamp>/model.json   the manifest: {rev, person_id, generated_at, seconds, generator{name, version}, options,
                       birth_year, scanned, usable, peak_rss_mb, generator_memory_mb{peak, with_model, released},
                       looks: [{key, label, age, age_band, hair, hair_label,
                       facial_hair, facial_hair_label, photos, face_id, item_id, item_uid, taken_at, faults, mesh,
                       picture, bytes, vertices, faces, textured, texture, overlap, overlap_head, relief,
                       note}]}. `textured`: the head carries the photo as a texture; `texture` its [width,
                       height]; `overlap` how well the head's outline covered the person's in the picture, and
                       `overlap_head` the same over the head alone (kept here to judge a run by: the generator's
                       own result file goes with the work folder); `relief`: the face's shape was corrected
                       from the depth model; `note` the generator's remark on the look (why a photo was not
                       used, or why the face was not shaped). A manifest written before the finishing existed
                       has none of the six, nor the three newer options: it reads as not textured and without
                       relief, and the status leaves the three options out. `peak_rss_mb` is what the server saw
                       the child's process tree hold at most; `generator_memory_mb` is the generator's own
                       count: its peak, what it held with the model loaded, and what it held once the model was
                       let go and before the heads were finished (kept to judge a run by, like the overlaps).
  <stamp>/<key>.glb    a look's mesh;  <stamp>/<key>.jpg  the picture it was made from (512 px)
  work/                the job's scratch folder (pictures, job.json, out/), removed when the job ends
  error.json           why the last run failed, until the next start

The child (see gen3d/__main__.py for the contract):
- its own process group, with oom_score_adj 1000 so a memory limit takes it and not the server;
- an environment of its own: a short list (_child_env), never the server's;
- inside a sandbox when the real generator runs (settings.gen3d_sandbox, bwrap; sandbox_prefix says exactly what
  it is made of). The generator runs third-party model code on the owner's face photos. Inside the sandbox that
  code sees the system's programs and libraries, its own Python environment, weights and code, and the job's work
  folder, which is the only place it can write. It does not see the home folder, the library, the database, the
  rest of the repository, the machine's sockets (Docker, the session bus) or the network;
- watched once a second: stopped when the resident memory of its whole process tree passes GEN3D_MAX_RSS_MB, when a
  head takes longer than GEN3D_LOOK_TIMEOUT_S, on Stop or Delete, when its person is deleted or merged away, and
  when the server exits. The heads that were finished before a limit stopped it are kept.

One job at a time for the whole app (one GPU). A look is neither listed nor served once its source photo is deleted,
in the trash or in the vault, or once its source face is no longer this person's. Folders of people who no longer
exist, and what an interrupted run left behind, are removed by sweep() (housekeeping); a deleted person's model
stops being served at once, as its routes ask for the person first.

A head holds the photo's own pixels as its texture, so its files follow the photo:
- the files are served with "private, no-cache" and an ETag (app.api.model3d), so a browser asks before every use
  and a look that stopped being listed is not shown again from its cache;
- sweep() removes the mesh and the picture of a look whose photo is gone for good (its item row no longer exists:
  the trash was emptied, or the photo was deleted outright) and takes the look out of model.json; when no look is
  left the model goes;
- a look whose photo is in the trash or the vault, or whose face is no longer this person's, keeps its files on
  disk, unlisted and unserved, because the photo or the face can come back; they go with the next generation, with
  Delete, with the person, or when the photo is gone for good.
"""
from __future__ import annotations

import atexit
import json
import logging
import math
import os
import queue
import re
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
CHECK_TIMEOUT_S = 30.0
GONE_CHECK_S = 1.0         # how often a running job looks whether its person still exists
LOOK_FACE_IOU = 0.5        # how much one of the person's faces must overlap a look's source face to stand in for it
STOP_WAIT_S = 5.0
EXIT_WAIT_S = 30.0         # after its last line the child has this long to exit by itself
DETAIL_MIN, DETAIL_MAX, DETAIL_DEFAULT = 1, 3, 2
SMOOTHING_MIN, SMOOTHING_MAX, SMOOTHING_DEFAULT = 0, 3, 2   # the finishing's smoothing: off, light, normal, strong
TEXTURE_MIN, TEXTURE_MAX, TEXTURE_DEFAULT = 0, 3, 2         # colours only; the photo to 1024, to 2048, at its own size
RELIEF_MIN, RELIEF_MAX, RELIEF_DEFAULT = 0, 1, 1            # the face's shape from a depth model: off, on
NOTE_MAX = 200             # characters of a look's note that are kept
SANDBOX_BIN = "bwrap"
OPTION_KEYS = ("max_looks", "detail", "framing", "smoothing", "texture", "relief")
FIRST_OPTIONS = ("max_looks", "detail", "framing")   # every model has these; the rest came with the finishing
JOB_ONLY = ("inside",)     # keys of a look's row that go into the job file and not into model.json

STEP_LOOKS = "Finding looks"
STEP_PHOTOS = "Preparing photos"
STEP_BUILD = "Building heads"       # the child's first pass: the model makes every look's raw head
STEP_FINISH = "Finishing heads"     # its second: each head is smoothed, shaped, given its photo and written
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
_peak_mb = 0.0                             # the most memory the last child's process tree was seen to hold
_check: tuple[float, tuple[Any, ...], dict[str, Any]] | None = None
_checking = False
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
def _python(fake: bool = False) -> Path:
    """The Python that runs the generator. The stand-in needs only numpy and Pillow, so it runs with the server's
    own Python however it was asked for (GEN3D_FAKE, GEN3D_GENERATOR=fake, or the job's own option); the real
    generator runs with GEN3D_PYTHON, or GEN3D_DIR/venv/bin/python."""
    if fake or settings.gen3d_fake or settings.gen3d_generator == "fake":
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


SANDBOX_HOME = "/tmp"      # the sandbox's own, empty /tmp
ENV_PASSED = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "GEN3D_THREADS")
SYSTEM_DIRS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
LOADER_FILES = ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives")


def _child_env(home: str | Path = SANDBOX_HOME) -> dict[str, str]:
    """The child's whole environment: a short list, never the server's own, which holds the server's settings and
    can hold secrets and the address of a Docker socket. `home` is where HOME and the libraries' caches point: the
    sandbox's private /tmp, or a folder inside the work folder when there is no sandbox (the machine's /tmp may be
    shared with other users). The Hugging Face libraries are told to stay offline either way, so a run never asks
    the hub for a newer file than the installer pinned. Four variables come from the server when it has them: the
    GPU choice (two names), the library path and the generator's thread count."""
    home = str(home)
    weights = _weights_dir()
    env = {
        "PATH": "/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1",
        # the sandbox has no /etc/passwd: a library that asks who is running it (getpass.getuser) reads these
        "USER": "gen3d", "LOGNAME": "gen3d",
        "XDG_CACHE_HOME": f"{home}/cache", "MPLCONFIGDIR": f"{home}/mpl", "NUMBA_CACHE_DIR": f"{home}/numba",
        "HF_HOME": str(weights / "hf"), "U2NET_HOME": str(weights / "u2net"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    env.update({name: os.environ[name] for name in ENV_PASSED if name in os.environ})
    return env


def _env_for(work: Path, fake: bool) -> dict[str, str]:
    """The environment a child that works in `work` is started with: its home is the sandbox's /tmp when it runs
    inside the sandbox, else a folder in `work`, which goes when `work` goes."""
    try:
        boxed = sandbox_wanted(fake)
    except GeneratorMissing:
        boxed = False
    if boxed:
        return _child_env(SANDBOX_HOME)
    home = Path(work) / "home"
    home.mkdir(parents=True, exist_ok=True)
    return _child_env(home)


def _gpu_nodes() -> list[str]:
    """The NVIDIA device nodes of this machine, /dev/nvidia-caps included (none without the driver)."""
    try:
        return [f"/dev/{name}" for name in sorted(os.listdir("/dev")) if name.startswith("nvidia")]
    except OSError:
        return []


def _python_root(root: str) -> bool:
    """A folder that is a Python environment or installation, and not, say, a home folder that holds a bin/python."""
    try:
        return os.path.isfile(os.path.join(root, "pyvenv.cfg")) or any(Path(root, "lib").glob("python3*"))
    except OSError:
        return False


def sandbox_reads(python: Path) -> list[str]:
    """What the sandbox shows read-only besides the system's programs and libraries: the Python environment that
    runs the child (where it is named and where it really is), the installation its interpreter comes from, the
    generator's folder (weights and model code, as named and as it really is) and this repository's gen3d package.
    An environment's python is a link, often to a link (uv names an installation by its minor version and links
    that name to the exact one): the installation is shown under every name on that way, or the link leads
    nowhere inside. Only what exists; never / and nothing that /usr already covers."""
    python = Path(python)
    roots = [str(python.parent.parent), os.path.realpath(python.parent.parent)]
    hop = str(python)
    for _ in range(16):
        if not os.path.islink(hop):
            break
        hop = os.path.normpath(os.path.join(os.path.dirname(hop), os.readlink(hop)))
        roots.append(str(Path(hop).parent.parent))
    roots.append(str(Path(os.path.realpath(python)).parent.parent))
    wanted = [r for r in roots if _python_root(r)]
    wanted += [str(settings.gen3d_dir), os.path.realpath(settings.gen3d_dir), str(REPO_ROOT / "gen3d")]
    out: list[str] = []
    for path in wanted:
        path = os.path.abspath(path)
        if path in ("/", "/usr") or path.startswith("/usr/") or path in out or not os.path.exists(path):
            continue
        out.append(path)
    return out


def sandbox_prefix(work: Path, python: Path | None = None) -> list[str]:
    """The bwrap command line in front of the generator's. The generator sees a system made of exactly these parts
    and nothing else of the machine:
    - /usr and the loader's files from /etc, read-only: the system's programs and libraries;
    - its own /proc, a new /dev with the NVIDIA device nodes added, /sys read-only, an empty /tmp and /run;
    - read-only: the Python environment that runs it and the installation behind it, GEN3D_DIR (weights and model
      code) and this repository's gen3d package (sandbox_reads);
    - read and write: `work` (and its own /tmp and /dev/shm, which are memory and go with it).
    So it has no home folder, no DATA_DIR, no other part of the repository and none of the machine's sockets (they
    live in /run and in the home folder); no network; its own process ids, IPC, host name and user namespace; and
    an environment of its own (_child_env) in place of the server's."""
    python = Path(python) if python is not None else _python()
    args = [
        sandbox_bin() or SANDBOX_BIN,
        "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-net", "--unshare-uts", "--unshare-cgroup-try",
        "--die-with-parent", "--new-session", "--clearenv",
    ]
    for name, value in _child_env(SANDBOX_HOME).items():
        args += ["--setenv", name, value]
    args += ["--ro-bind", "/usr", "/usr"]
    for path in SYSTEM_DIRS:
        if os.path.islink(path):
            args += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            args += ["--ro-bind", path, path]
    for path in LOADER_FILES:
        if os.path.exists(path):
            args += ["--ro-bind", path, path]
    args += ["--proc", "/proc", "--dev", "/dev"]
    for node in _gpu_nodes():
        args += ["--dev-bind", node, node]
    args += ["--ro-bind", "/sys", "/sys", "--tmpfs", "/tmp", "--tmpfs", "/run"]
    for path in sandbox_reads(python):  # after the empty /tmp and /run, so a folder that lives there still shows
        args += ["--ro-bind", path, path]
    # the folders that only lead to the parts above sit on the sandbox's own root: read-only as well, so the work
    # folder, /tmp and /dev/shm (both in memory, both its own) are the only places it can write at all
    args += ["--bind", str(work), str(work), "--remount-ro", "/", "--chdir", str(REPO_ROOT)]
    return args


def _command(args: list[str], work: Path, fake: bool) -> list[str]:
    """The child's full command line. The shell in front raises the child's oom_score_adj before anything else
    runs (inherited by everything it starts), then becomes the command itself."""
    python = _python(fake)
    cmd = [str(python), "-m", "gen3d", *args]
    if sandbox_wanted(fake):
        cmd = [*sandbox_prefix(work, python), *cmd]
    return ["/bin/sh", "-c", '{ echo 1000 > /proc/self/oom_score_adj; } 2>/dev/null; exec "$@"', "gen3d", *cmd]


def generator(refresh: bool = False) -> dict[str, Any]:
    """{"installed", "name", "device", "reason", "relief"}: what a Generate would run with. `relief`: the two depth
    model files the face relief needs are installed (None when the generator does not say: the stand-in, an
    older generator, a failed check). The real generator is asked with `python -m gen3d --check`, which only looks
    at files, and the answer is remembered for CHECK_TTL_S (CHECK_RETRY_S while it is missing). A status read waits
    for that at most once: after the first answer an old one is given at once and a new one is fetched in the
    background. start() asks again itself (`refresh`)."""
    fake = bool(settings.gen3d_fake) or settings.gen3d_generator == "fake"
    if fake:
        try:
            sandbox_wanted(True)
        except GeneratorMissing as e:
            return {"installed": False, "name": None, "device": None, "reason": str(e), "relief": None}
        return {"installed": True, "name": "fake", "device": "cpu", "reason": None, "relief": None}
    python = _python()
    key = (str(python), str(_weights_dir()), settings.gen3d_generator, settings.gen3d_sandbox)
    with _lock:
        cached = _check
    if refresh or cached is None or cached[1] != key:
        return dict(_ask_and_remember(python, key))
    ttl = CHECK_TTL_S if cached[2]["installed"] else CHECK_RETRY_S
    if time.monotonic() - cached[0] >= ttl:
        if not settings.workers_enabled:  # no background threads (tests): ask here
            return dict(_ask_and_remember(python, key))
        _ask_in_background(python, key)
    return dict(cached[2])


def _ask_and_remember(python: Path, key: tuple[Any, ...]) -> dict[str, Any]:
    global _check
    out = _ask_generator(python)
    with _lock:
        _check = (time.monotonic(), key, out)
    return out


def _ask_in_background(python: Path, key: tuple[Any, ...]) -> None:
    global _checking
    with _lock:
        if _checking:
            return
        _checking = True

    def ask() -> None:
        global _checking
        try:
            _ask_and_remember(python, key)
        finally:
            with _lock:
                _checking = False

    threading.Thread(target=ask, name="pcc-model3d-check", daemon=True).start()


MISSING = "The 3D generator is not installed. Run scripts/setup_3d.sh once on the server."


def _ask_generator(python: Path) -> dict[str, Any]:
    """Ask the generator's own Python what is installed. The check runs as a job does: inside the sandbox when that
    is wanted, with an empty folder of its own as the only place it may write (never the models' folder)."""
    missing = {"installed": False, "name": None, "device": None, "reason": MISSING, "relief": None}
    if not python.is_file():
        return missing
    scratch = storage.tmp_path(".gen3d-check")
    info: Any = None
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        cmd = _command(["--check", "--weights-dir", str(_weights_dir())], scratch, False)
        done = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_env_for(scratch, False), capture_output=True,
                              timeout=CHECK_TIMEOUT_S, check=False)
        lines = [ln for ln in done.stdout.decode("utf-8", "replace").splitlines() if ln.strip().startswith("{")]
        info = json.loads(lines[-1]) if lines else None
        if not isinstance(info, dict):
            log.warning("3D heads: the generator's check said nothing (exit %s): %s", done.returncode,
                        done.stderr.decode("utf-8", "replace")[-300:])
    except GeneratorMissing as e:
        return {**missing, "reason": str(e)}
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        log.warning("3D heads: the generator did not answer the check: %s", e)
        return missing
    finally:
        storage.remove_tree(scratch)
    if not isinstance(info, dict):
        return missing
    wanted = settings.gen3d_generator
    found = info.get("backends") or {}
    name = info.get("auto") if wanted == "auto" else (wanted if found.get(wanted) else None)
    if not name:
        return missing
    relief = info.get("relief")
    return {"installed": True, "name": str(name), "device": info.get("device"), "reason": None,
            "relief": relief if isinstance(relief, bool) else None}


# ------------------------------------------------------------------ status
def _overlap(a: Any, b: Any) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(aw * ah + bw * bh - w * h, 1e-12)


def _own_looks(conn: sqlite3.Connection, person_id: int, looks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The looks that may be shown: the source face is still this person's, on a photo that is in the library (not
    deleted, not in the trash, not in the vault). A face that was moved to someone else, unassigned or removed
    takes its look with it. A photo that was processed again has new face rows; its look stays when one of this
    person's faces on that photo sits where the look's face was."""
    item_ids = sorted({int(lk.get("item_id") or 0) for lk in looks})
    if not item_ids:
        return []
    marks = ",".join("?" * len(item_ids))
    faces: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in conn.execute(
            f"SELECT f.id, f.item_id, f.x, f.y, f.w, f.h FROM faces f JOIN items i ON i.id = f.item_id "
            f"WHERE f.item_id IN ({marks}) AND f.person_id = ? AND i.deleted_at IS NULL AND i.nsfw = 0",
            (*item_ids, int(person_id))):
        faces[int(row["item_id"])].append(row)
    shown = []
    for lk in looks:
        theirs = faces.get(int(lk.get("item_id") or 0), [])
        box = lk.get("box")
        if any(int(f["id"]) == int(lk.get("face_id") or 0) for f in theirs):
            shown.append(lk)
        elif isinstance(box, list) and len(box) == 4 and any(
                _overlap(box, (f["x"], f["y"], f["w"], f["h"])) >= LOOK_FACE_IOU for f in theirs):
            shown.append(lk)
    return shown


def listed_looks(conn: sqlite3.Connection, person_id: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """(the manifest, its looks that may be shown, see _own_looks)."""
    doc = manifest(person_id)
    if doc is None:
        return None, []
    return doc, _own_looks(conn, person_id, [lk for lk in doc["looks"] if isinstance(lk, dict)])


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
        row = {
            **{k: lk.get(k) for k in ("key", "label", "age", "age_band", "hair", "hair_label", "facial_hair",
                                      "facial_hair_label", "photos", "taken_at", "item_uid", "bytes", "vertices")},
            "picture_url": file_url(person_id, rev, lk["picture"]),
            "mesh_url": file_url(person_id, rev, lk["mesh"]),
            "textured": lk.get("textured") is True,  # a head made before the finishing existed has no such key
            "relief": lk.get("relief") is True,
        }
        note = _note(lk.get("note"))
        if note:
            row["note"] = note
        out_looks.append(row)
    return {
        "available": reason is None,
        "reason": reason,
        "generator": gen["name"],
        # the depth model files of the face relief are there: false tells the owner why no head gets one
        "relief_installed": gen.get("relief"),
        "usable_photos": photos,
        "state": "running" if mine else "done" if out_looks else "error" if error else "none",
        "busy": job is not None and not mine,
        "stopping": bool(mine and job["cancel"]),  # Stop or Delete was pressed and the job is on its way out
        "progress": {"step": job["step"], "done": job["done"], "total": job["total"]} if mine else None,
        "error": error,
        "generated_at": doc.get("generated_at") if doc and out_looks else None,
        "options": shown_options(doc.get("options")) if doc and out_looks else None,
        "looks": out_looks,
    }


def shown_options(stored: Any) -> dict[str, Any] | None:
    """The options a model was made with, as the status gives them: every option the manifest has, inside its
    range. A model made before the finishing existed has no `smoothing`, `texture` or `relief`, and they are left
    out here, not filled in: that is how the tab can tell that these heads predate the finishing (it fills its
    sliders with the defaults for what is missing). None when the manifest holds no usable options."""
    if not isinstance(stored, dict) or not all(k in stored for k in FIRST_OPTIONS):
        return None
    cleaned = clean_options(stored)
    return {k: cleaned[k] for k in OPTION_KEYS if k in stored}


# ------------------------------------------------------------------ start, cancel, remove
def _number(value: Any, lo: float, hi: float, default: float, whole: bool) -> float:
    """An option inside its range: a value outside is moved to the nearest end, and a value that is no number at
    all (None, a word, a truth value, a list, not-a-number, infinity) gives the default. A whole option drops a
    fraction (2.9 is 2), and a number written as text counts as that number."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(int(value)) if whole else float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(lo, min(hi, number))


def clean_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """The options of a job, each inside its range (OPTION_KEYS; see _number for what a bad value becomes), and
    `generator: "fake"` when the stand-in was asked for. Other keys are dropped."""
    o = options if isinstance(options, dict) else {}

    def whole(key: str, lo: int, hi: int, default: int) -> int:
        return int(_number(o.get(key, default), lo, hi, default, True))

    out: dict[str, Any] = {
        "max_looks": whole("max_looks", looks_mod.LOOKS_MIN, looks_mod.LOOKS_MAX, looks_mod.LOOKS_DEFAULT),
        "detail": whole("detail", DETAIL_MIN, DETAIL_MAX, DETAIL_DEFAULT),
        "framing": round(_number(o.get("framing", looks_mod.FRAMING_DEFAULT), looks_mod.FRAMING_MIN,
                                 looks_mod.FRAMING_MAX, looks_mod.FRAMING_DEFAULT, False), 2),
        "smoothing": whole("smoothing", SMOOTHING_MIN, SMOOTHING_MAX, SMOOTHING_DEFAULT),
        "texture": whole("texture", TEXTURE_MIN, TEXTURE_MAX, TEXTURE_DEFAULT),
        "relief": whole("relief", RELIEF_MIN, RELIEF_MAX, RELIEF_DEFAULT),
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
        # not listed and not served from this moment; the job removes the folder as it ends
        storage.safe_unlink(person_dir(person_id) / "current")
        cancel(person_id)
    else:
        storage.remove_tree(person_dir(person_id))


def _drop_gone_looks(conn: sqlite3.Connection, person_id: int) -> int:
    """Take the looks whose photo is gone for good out of the person's model: the item row no longer exists (the
    trash was emptied, or the photo was deleted outright; an item id is never given out twice, and the uid is
    compared as well). The manifest is rewritten without them first, then their mesh and picture are removed, and
    so is any mesh or picture in the folder that the manifest does not name (what a run of this that was cut
    short left). With no look left the model itself goes, and the person's folder with it unless that still holds
    an error.json. A photo that is only in the trash or the vault still has its row, so its look stays as it is:
    not listed, not served, its files kept. A manifest that does not read decides nothing. Returns how many looks
    went. Call with _lock held and no job running."""
    rev = current_rev(person_id)
    doc = manifest(person_id)
    if rev is None or doc is None:
        return 0
    folder = person_dir(person_id) / rev
    looks = [lk for lk in doc["looks"] if isinstance(lk, dict)]
    ids = sorted({int(lk.get("item_id") or 0) for lk in looks})
    there: dict[int, str] = {}
    for start in range(0, len(ids), 400):
        part = ids[start:start + 400]
        for row in conn.execute(f"SELECT id, uid FROM items WHERE id IN ({','.join('?' * len(part))})", part):
            there[int(row["id"])] = str(row["uid"])
    kept = [lk for lk in looks
            if there.get(int(lk.get("item_id") or 0)) is not None
            and (lk.get("item_uid") is None or there[int(lk.get("item_id") or 0)] == str(lk.get("item_uid")))]
    gone = len(looks) - len(kept)
    if gone and not kept:
        storage.safe_unlink(person_dir(person_id) / "current")  # not listed from this moment
        storage.remove_tree(folder)
        try:
            person_dir(person_id).rmdir()  # the person's folder too, unless it still holds an error.json
        except OSError:
            pass
        return gone
    if gone:
        _write_json(folder / "model.json", {**doc, "looks": kept})
    named = {str(lk.get(k)) for lk in kept for k in ("mesh", "picture")}
    try:
        entries = list(folder.iterdir())
    except OSError:
        return gone
    for entry in entries:
        if entry.is_file() and NAME_RE.fullmatch(entry.name) and entry.name not in named:
            storage.safe_unlink(entry)
    return gone


def sweep(conn: sqlite3.Connection) -> int:
    """Housekeeping. Removes the model folders of people who no longer exist and, while no job runs, what an
    interrupted run left behind for everyone else: a work folder (a server that stopped in the middle of a job
    never got to remove it) and model folders that `current` does not name; and, of the models in use, the looks
    whose photo is gone for good with their files (_drop_gone_looks). Returns how many people's folders went."""
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
    left = 0
    looks_gone = 0
    with _lock:  # held, so no job can start and make a work folder while the leftovers go
        if _state is None:
            for entry in base.iterdir():
                if not entry.is_dir() or not entry.name.isdigit():
                    continue
                keep = current_rev(int(entry.name))
                for sub in entry.iterdir():
                    if sub.is_dir() and (sub.name == "work" or (STAMP_RE.fullmatch(sub.name) and sub.name != keep)):
                        storage.remove_tree(sub)
                        left += 1
                try:
                    looks_gone += _drop_gone_looks(conn, int(entry.name))
                except (OSError, sqlite3.Error, ValueError, TypeError):
                    log.exception("3D heads: could not check the looks of person %s against their photos", entry.name)
    if left:
        log.info("3D heads: removed %d folders that interrupted runs left behind", left)
    if looks_gone:
        log.info("3D heads: removed %d looks whose photo is gone for good", looks_gone)
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


def _run_child(work: Path, job_path: Path, total: int, fake: bool,
               stopped: Callable[[], bool] | None = None) -> tuple[set[str], str | None]:
    """Start the generator on the job file and follow it to its end. Returns (the keys of the looks it said it
    finished, why it ended badly or None): a child that was stopped at a limit, or that broke, may have finished
    some heads first. _Failed when it cannot be started, _Cancelled when `stopped` (Stop, Delete, the person gone)
    says so."""
    global _proc, _atexit_done, _peak_mb
    stopped = stopped or _cancelled
    with _lock:
        _peak_mb = 0.0
    cmd = _command(["--job", str(job_path)], work, fake)
    err_path = work / "stderr.log"
    with open(err_path, "wb") as err:
        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=_env_for(work, fake), stdout=subprocess.PIPE, stderr=err,
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
    finished: set[str] = set()
    made = 0
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
                except ValueError:
                    done = 0
                if done > 0:    # heads finished so far; the "0/M" at the start says nothing new
                    _set(step=STEP_FINISH, done=min(done, total), total=total)
            elif line.startswith("made:"):
                # the first pass: the model is done with one more look. Nothing is finished by that, but the job
                # is alive and that far, so the bar moves and the look's time starts again
                made += 1
                last_look = time.monotonic()
                _set(step=STEP_BUILD, done=min(made, total), total=total)
            elif line.startswith("look:"):
                last_look = time.monotonic()
                words = line.split()
                if len(words) >= 3 and words[2] == "ok":
                    finished.add(words[1])
            if ended:
                break
            if stopped():
                raise _Cancelled()
            used = tree_rss_mb(proc.pid)
            if used > _peak_mb:
                with _lock:
                    _peak_mb = used
            if used > limit_mb:
                reason = (f"The generator needed more memory than it may use ({int(used)} MB of {int(limit_mb)} MB) "
                          "and was stopped.")
                break
            if time.monotonic() - last_look > timeout:
                limit = f"{int(timeout // 60)} minutes" if timeout >= 120 else f"{int(timeout)} seconds"
                reason = f"A head took longer than {limit} and the generator was stopped."
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
    if reason is None and _cancelled():
        raise _Cancelled()
    if reason is None and proc.returncode not in (0, 3):
        said = _tail(err_path)
        reason = f"The generator stopped with an error{': ' + said if said else f' (exit {proc.returncode})'}."
    log.info("3D heads: the generator's memory was at most %d MB (sampled once a second; the limit is %d MB)",
             int(_peak_mb), int(limit_mb))
    return finished, reason


def peak_rss_mb() -> int:
    """The most memory the last child's whole process tree was seen to hold, in MB (sampled once a second, so a
    short spike can be missed). It goes into model.json, where it says how near the job came to GEN3D_MAX_RSS_MB."""
    with _lock:
        return int(round(_peak_mb))


# ------------------------------------------------------------------ the job
def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    storage.write_bytes_atomic(path, json.dumps(doc, indent=1, sort_keys=True).encode("utf-8"))


def _prepare(conn: sqlite3.Connection, plan: looks_mod.Plan, work: Path, framing: float) -> list[dict[str, Any]]:
    """Write each look's picture (the look's best photo, else a runner-up whose pixels read), at the photo's own
    resolution (looks.picture_side: 1024 to 3072 px a side; at 3072 that is 28 MB of pixels, a file of about 15 MB
    for a photo and 28 MB at worst, written in about a second and a half), and return the looks that have one, as
    manifest rows. The files live in the work folder, which goes when the job ends. Each row also carries
    `inside`, for the job file only (JOB_ONLY): the part of the written picture that shows the photo itself
    (looks.inside), or None when that is not known."""
    rows: list[dict[str, Any]] = []
    for n, look in enumerate(plan.looks, start=1):
        if _cancelled():
            raise _Cancelled()
        by_id = {r.face_id: r for r in plan.readings}
        chosen = None
        inside = None
        for face_id in [look.face_id, *look.runners_up]:
            written = looks_mod.write_picture(conn, face_id, work / f"{look.key}.png", framing)
            if written:
                chosen = by_id.get(face_id)
                if isinstance(written, tuple) and len(written) == 4:
                    inside = [int(v) for v in written]
                break
        _set(step=STEP_PHOTOS, done=n, total=len(plan.looks))
        if chosen is None:
            continue
        rows.append({
            "key": look.key, "label": look.label, "age": look.age, "age_band": look.age_band, "hair": look.hair,
            "hair_label": look.hair_label, "facial_hair": look.facial_hair, "facial_hair_label": look.facial_hair_label,
            "photos": look.photos, "face_id": chosen.face_id, "item_id": chosen.item_id, "item_uid": chosen.item_uid,
            "taken_at": chosen.taken_at, "faults": list(chosen.faults), "box": [round(float(v), 5) for v in chosen.box],
            "inside": inside,
        })
    return rows


def _job_look(row: dict[str, Any], work: Path) -> dict[str, Any]:
    """A look as the job file gives it to the generator: its key, its picture and, when it is known, `inside`:
    [left, top, right, bottom] in pixels of that picture, the part that is the photo itself. Where the square of
    the picture left the photo, the rest is the photo's edge repeated (up to looks.CUT_OFF_SHARE of the square);
    the generator lays the photo on the head only inside this rectangle."""
    look: dict[str, Any] = {"key": row["key"], "image": str(work / f"{row['key']}.png")}
    if row.get("inside"):
        look["inside"] = list(row["inside"])
    return look


def _glb_doc(path: Path) -> dict[str, Any] | None:
    """A .glb's own description (its JSON part), or None when it cannot be read."""
    try:
        with open(path, "rb") as f:
            magic, _version, _total, n_json, kind = struct.unpack("<IIIII", f.read(20))
            if magic != 0x46546C67 or kind != 0x4E4F534A or n_json > 4_000_000:
                return None
            doc = json.loads(f.read(n_json).decode("utf-8"))
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError, struct.error):
        return None


def _glb_counts(path: Path) -> tuple[int, int]:
    """(vertices, triangles) of a .glb's first part (the whole surface; a photo layer lies over a share of it), as
    the file says them in its own description; (0, 0) when that cannot be read."""
    doc = _glb_doc(path)
    if doc is None:
        return 0, 0
    try:
        prim = doc["meshes"][0]["primitives"][0]
        return (int(doc["accessors"][prim["attributes"]["POSITION"]]["count"]),
                int(doc["accessors"][prim["indices"]]["count"]) // 3)
    except (ValueError, KeyError, IndexError, TypeError):
        return 0, 0


def _glb_textured(path: Path) -> bool:
    """Whether a .glb carries a picture as a texture: it holds an image, and a part's material uses one."""
    doc = _glb_doc(path)
    if doc is None:
        return False
    try:
        if not doc.get("images") or not doc.get("textures"):
            return False
        materials = doc.get("materials") or []
        for mesh in doc.get("meshes") or []:
            for prim in mesh.get("primitives") or []:
                index = prim.get("material")
                if isinstance(index, int) and 0 <= index < len(materials):
                    if "baseColorTexture" in (materials[index].get("pbrMetallicRoughness") or {}):
                        return True
        return False
    except (AttributeError, TypeError):
        return False


def _note(value: Any) -> str | None:
    """A look's note as one short line of plain text, or None. The text comes from the generator, which runs model
    code written by others: nothing but a short printable line is kept."""
    if not isinstance(value, str):
        return None
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in value).split())
    return text[:NOTE_MAX] or None


def _generator_memory(value: Any) -> dict[str, int | None] | None:
    """The generator's own memory numbers (its result file's `memory_mb`), checked: {"peak", "with_model",
    "released"} in MB, each a whole number or None; None when the generator said nothing of the kind."""
    if not isinstance(value, dict):
        return None
    out: dict[str, int | None] = {}
    for key in ("peak", "with_model", "released"):
        mb = value.get(key)
        ok = isinstance(mb, (int, float)) and not isinstance(mb, bool) and math.isfinite(mb) and 0 <= mb < 10_000_000
        out[key] = int(mb) if ok else None
    return out if any(v is not None for v in out.values()) else None


def _share(value: Any) -> float | None:
    """A share between 0 and 1 as the generator gave it, to four places; None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        return None
    return round(float(value), 4)


def _finishing(made: dict[str, Any], mesh: Path) -> dict[str, Any]:
    """What the generator said about a look's finishing (the keys of its result row), checked: `textured` (read
    from the file itself when the row does not say: a head that was finished before the generator was stopped has
    no row, and the generator before the finishing wrote no such key), `texture` as [width, height] and only with
    a texture, `overlap` and `overlap_head` as shares between 0 and 1, `relief` (the face's shape was corrected
    from the depth model; only a plain true counts), `note` as one short line."""
    said = made.get("textured")
    textured = said if isinstance(said, bool) else _glb_textured(mesh)
    size = made.get("texture")
    texture = None
    if (textured and isinstance(size, (list, tuple)) and len(size) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 16384 for v in size)):
        texture = [int(size[0]), int(size[1])]
    return {"textured": bool(textured), "texture": texture, "overlap": _share(made.get("overlap")),
            "overlap_head": _share(made.get("overlap_head")), "relief": made.get("relief") is True,
            "note": _note(made.get("note"))}


def _collect(person_id: int, work: Path, rows: list[dict[str, Any]], plan: looks_mod.Plan, options: dict[str, Any],
             seconds: float, finished: set[str] | None = None, peak_mb: int | None = None) -> int:
    """Move the meshes and pictures of the looks that worked into a new folder and make it the model in use.
    Returns how many looks it holds. `finished` is given when the child did not end by itself: then the looks it
    reported as finished count (a mesh is written whole before it is reported), with or without a result file.
    `peak_mb` is the most memory the child was seen to hold."""
    from PIL import Image

    try:
        result = json.loads((work / "out" / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        if finished is None:
            raise _Failed("The generator left no result.") from e
        result = {}
    built = {r.get("key"): r for r in result.get("looks") or [] if isinstance(r, dict) and r.get("ok")}
    if finished is not None:
        built = {key: built.get(key) or {} for key in finished}
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
        vertices, faces = int(made.get("vertices") or 0), int(made.get("faces") or 0)
        if not vertices:
            vertices, faces = _glb_counts(mesh)
        finishing = _finishing(made, mesh)
        target = dest / f"{row['key']}.glb"
        storage.move_into_place(mesh, target)
        with Image.open(work / f"{row['key']}.png") as im:
            im.convert("RGB").resize((PICTURE_SIDE, PICTURE_SIDE), Image.Resampling.LANCZOS).save(
                dest / f"{row['key']}.jpg", "JPEG", quality=PICTURE_QUALITY)
        kept = {k: v for k, v in row.items() if k not in JOB_ONLY}
        looks.append({**kept, "mesh": target.name, "picture": f"{row['key']}.jpg", "bytes": target.stat().st_size,
                      "vertices": vertices, "faces": faces, **finishing})
    if not looks:
        storage.remove_tree(dest)
        failed = [r.get("error") for r in result.get("looks") or [] if isinstance(r, dict) and r.get("error")]
        raise _Failed("No head could be built" + (f": {failed[-1]}" if failed else "") + ".")
    _write_json(dest / "model.json", {
        "rev": rev, "person_id": int(person_id), "generated_at": now_iso(), "seconds": round(seconds, 1),
        "generator": result.get("generator") or {}, "options": {k: options[k] for k in OPTION_KEYS},
        "birth_year": plan.birth_year, "scanned": plan.scanned, "usable": plan.usable,
        "peak_rss_mb": int(peak_mb) if peak_mb else None,
        "generator_memory_mb": _generator_memory(result.get("memory_mb")), "looks": looks,
    })
    storage.write_bytes_atomic(base / "current", rev.encode("ascii"))
    for entry in base.iterdir():  # the older models
        if entry.is_dir() and STAMP_RE.fullmatch(entry.name) and entry.name != rev:
            storage.remove_tree(entry)
    return len(looks)


def _gone(conn: sqlite3.Connection, person_id: int) -> bool:
    """The person no longer exists as a person: deleted, merged into someone else, or turned into a pet."""
    try:
        row = conn.execute("SELECT kind FROM persons WHERE id = ?", (int(person_id),)).fetchone()
    except sqlite3.Error:
        return False  # a busy database is no reason to stop a job
    return row is None or row["kind"] == "pet"


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
    looked = [0.0]

    def stopped(now: bool = False) -> bool:
        """Stop or Delete was pressed, or the person is gone (deleted, or merged into someone else, which no route
        could stop afterwards): either way the job ends here. The person is looked up at most once every
        GONE_CHECK_S, and at once with `now`."""
        if _cancelled():
            return True
        t = time.monotonic()
        if now or t - looked[0] >= GONE_CHECK_S:
            looked[0] = t
            if _gone(conn, person_id):
                log.info("3D heads: person %s is gone, their job stops", person_id)
                _set(cancel=True)
                return True
        return False

    def reading(done: int, total: int) -> None:
        """Called before the models load and after every photo: where Stop takes effect while looks are found."""
        _set(step=STEP_LOOKS, done=done, total=total)
        if stopped():
            raise _Cancelled()

    try:
        storage.remove_tree(work)
        (work / "out").mkdir(parents=True, exist_ok=True)
        storage.safe_unlink(base / "error.json")
        try:
            plan = looks_mod.plan(conn, person_id, options["max_looks"], progress=reading)
        except looks_mod.NoPhotos as e:
            raise _Failed("There are too few clear, large face photos of this person.") from e
        if stopped(now=True):
            raise _Cancelled()
        if not plan.looks:
            raise _Failed("No look could be told from the photos: none of them shows the face clearly enough.")
        rows = _prepare(conn, plan, work, options["framing"])
        if not rows:
            raise _Failed("The photos of the looks could not be read.")
        job = {
            "out_dir": str(work / "out"), "weights_dir": str(_weights_dir()),
            "generator": "fake" if fake else settings.gen3d_generator, "device": "auto",
            # what the watch below allows the child: a look whose depth models would not fit under it is finished
            # without the relief by the generator itself, which is better than the whole job being stopped here
            "max_rss_mb": int(settings.gen3d_max_rss_mb),
            # `framing` goes along because the face relief finds the face by it: the picture is that many face
            # sizes wide (looks.square_for)
            "options": {"detail": options["detail"], "framing": options["framing"], "smoothing": options["smoothing"],
                        "texture": options["texture"], "relief": options["relief"], "seed": 0},
            "looks": [_job_look(r, work) for r in rows],
        }
        _write_json(work / "job.json", job)
        _set(step=STEP_BUILD, done=0, total=len(rows))
        finished, reason = _run_child(work, work / "job.json", len(rows), fake, stopped)
        if stopped(now=True):
            raise _Cancelled()
        if reason is not None and not finished:
            raise _Failed(reason)
        _set(step=STEP_SAVE, done=len(rows), total=len(rows))
        try:
            kept = _collect(person_id, work, rows, plan, options, time.monotonic() - started,
                            finished if reason is not None else None, peak_rss_mb())
        except _Failed:
            if reason is not None:
                raise _Failed(reason) from None
            raise
        if reason is not None:  # the child was stopped part of the way: what it had finished is the new model
            _note_error(base, f"{reason} {kept} of {len(rows)} heads {'was' if kept == 1 else 'were'} finished "
                              f"before that and {'is' if kept == 1 else 'are'} kept.")
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
