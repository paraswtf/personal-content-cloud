"""`python -m gen3d --job JOB.json` builds one 3D head per look; `python -m gen3d --check --weights-dir DIR` says
what is installed by looking at files (add `--device` to have PyTorch imported and its version and the device
named; that takes seconds, and the app asks the plain check for status reads). The contract with the app
(docs/ARCHITECTURE.md section 36):

Job file:
    {"out_dir": "/abs/work", "weights_dir": "/abs/models/gen3d/weights", "generator": "auto", "device": "auto",
     "options": {"detail": 2, "seed": 0},
     "looks": [{"key": "l1", "image": "/abs/work/l1.png"}]}
  `image`: a square RGB picture of the head and shoulders, the face centred. `detail`: 1 to 3. `generator`: "auto"
  (the first installed backend of gen3d.backends.ORDER), a backend's name, or "fake".

Output: OUT/<key>.glb for every look that worked; OUT/result.json
    {"generator": {"name", "version"}, "looks": [{"key", "ok", "mesh", "vertices", "faces", "seconds", "error"}]}
  and on stdout, one line per event, flushed at once: "progress: N/M" and "look: <key> ok|error <seconds>".
  A look that fails does not stop the others.

Exit: 0 when at least one look was built, 3 when none was, 2 for a bad job file, 4 when the backend cannot load
(missing weights, not enough memory), 5 when it was told to stop (SIGTERM) or its parent went away. The reason is the
last line on stderr.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from gen3d import backends, glb
from gen3d.mesh import finish

EXIT_OK, EXIT_BAD_JOB, EXIT_NO_LOOK, EXIT_NO_BACKEND, EXIT_STOPPED = 0, 2, 3, 4, 5
KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


class BadJob(ValueError):
    pass


class Stopped(Exception):
    pass


def say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def fail(reason: str, code: int) -> int:
    sys.stderr.write(reason.strip().replace("\n", " ") + "\n")
    sys.stderr.flush()
    return code


def read_job(path: str) -> dict[str, Any]:
    try:
        job = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BadJob(f"the job file cannot be read: {e}") from e
    if not isinstance(job, dict):
        raise BadJob("the job file is not an object")
    out = job.get("out_dir")
    if not isinstance(out, str) or not Path(out).is_absolute():
        raise BadJob("out_dir must be an absolute path")
    looks = job.get("looks")
    if not isinstance(looks, list) or not looks:
        raise BadJob("the job has no looks")
    seen: set[str] = set()
    for look in looks:
        key = look.get("key") if isinstance(look, dict) else None
        if not isinstance(key, str) or not key or len(key) > 40 or not set(key) <= KEY_CHARS or key in seen:
            raise BadJob("every look needs its own key of lower-case letters, digits, - and _")
        seen.add(key)
        if not isinstance(look.get("image"), str) or not Path(look["image"]).is_absolute():
            raise BadJob(f"look {key}: image must be an absolute path")
    options = job.get("options") or {}
    if not isinstance(options, dict):
        raise BadJob("options must be an object")
    try:
        detail = int(options.get("detail", 2))
    except (TypeError, ValueError) as e:
        raise BadJob("detail must be 1, 2 or 3") from e
    if detail not in (1, 2, 3):
        raise BadJob("detail must be 1, 2 or 3")
    options = {**options, "detail": detail}
    generator = job.get("generator") or "auto"
    if generator != "auto" and generator not in backends.REGISTRY:
        raise BadJob(f"unknown generator: {generator}")
    return {"out_dir": Path(out), "weights_dir": Path(job.get("weights_dir") or out), "generator": generator,
            "device": str(job.get("device") or "auto"), "options": options, "looks": looks}


def _torch() -> tuple[str | None, str]:
    """(the installed PyTorch version or None, "cuda" or "cpu")."""
    try:
        import torch  # noqa: PLC0415 - only the generator's own Python has it
    except Exception:  # noqa: BLE001 - not installed, or broken: the fake backend still works
        return None, "cpu"
    try:
        return str(torch.__version__), "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return str(getattr(torch, "__version__", "")) or None, "cpu"


def check(weights_dir: Path, with_device: bool = False) -> int:
    """One JSON line: which backends are installed. PyTorch is imported only with `with_device`; without it
    "device" and "torch" are null."""
    found = {name: (True if name == "fake" else backends.is_available(name, weights_dir)) for name in backends.REGISTRY}
    version, device = _torch() if with_device else (None, None)
    auto = next((name for name in backends.ORDER if found.get(name)), None)
    from gen3d import relief  # noqa: PLC0415 - numpy at import; "relief": the two depth model files are there

    say(json.dumps({"backends": found, "auto": auto, "device": device, "relief": relief.available(weights_dir),
                    "torch": version}, sort_keys=True))
    return EXIT_OK


def _prepare(name: str):
    """gen3d.prep.prepare when that module is there (the real one cuts the background out), else None. The fake
    backend takes the picture as it is."""
    if name == "fake":
        return None
    try:
        from gen3d import prep  # noqa: PLC0415 - the real backends' companion, written with them
    except ModuleNotFoundError as e:
        if e.name == "gen3d.prep":
            return None
        raise
    return prep.prepare


def _watch_parent(stop: threading.Event) -> None:
    """Stop when the process that started this one goes away, so a killed server never leaves a model on the GPU."""
    parent = os.getppid()
    try:  # Linux: the kernel sends SIGTERM when the starting thread dies
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)  # 1 = PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - not Linux: the poll below covers it
        pass

    def poll() -> None:
        while not stop.wait(2.0):
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=poll, name="gen3d-parent", daemon=True).start()


def run(job: dict[str, Any]) -> int:
    from PIL import Image  # noqa: PLC0415

    out_dir: Path = job["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    looks = job["looks"]
    name = job["generator"]
    if name == "auto":
        name = backends.pick(job["weights_dir"])
        if name is None:
            return fail("no generator is installed (run scripts/setup_3d.sh)", EXIT_NO_BACKEND)
    try:
        backend = backends.get(name)
        if not backend.available(job["weights_dir"]):
            return fail(f"the {name} generator is not installed (run scripts/setup_3d.sh)", EXIT_NO_BACKEND)
        prepare = _prepare(name)
        backend.load(job["weights_dir"], job["device"], dict(job["options"]))
    except Stopped:
        return fail("stopped", EXIT_STOPPED)
    except Exception as e:  # noqa: BLE001 - missing library, missing weights, no memory
        return fail(f"the {name} generator could not start: {e}", EXIT_NO_BACKEND)

    results: list[dict[str, Any]] = []
    stopped = False
    say(f"progress: 0/{len(looks)}")
    try:
        for n, look in enumerate(looks, start=1):
            key = look["key"]
            started = time.monotonic()
            row: dict[str, Any] = {"key": key, "ok": False, "mesh": None, "vertices": 0, "faces": 0, "seconds": 0.0,
                                   "error": None}
            options = {**job["options"], "out_dir": str(out_dir), "key": key}
            try:
                with Image.open(look["image"]) as im:
                    picture = im.convert("RGB")
                if prepare is not None:
                    picture = prepare(picture, options)
                mesh = finish(backend.generate(picture, options))
                glb.write(out_dir / f"{key}.glb", mesh)
                row.update(ok=True, mesh=f"{key}.glb", vertices=int(mesh.vertices.shape[0]),
                           faces=int(mesh.faces.shape[0]))
            except Stopped:
                raise
            except Exception as e:  # noqa: BLE001 - this look failed; the others still run
                row["error"] = f"{type(e).__name__}: {e}"[:500]
            row["seconds"] = round(time.monotonic() - started, 2)
            results.append(row)
            say(f"look: {key} {'ok' if row['ok'] else 'error'} {row['seconds']}")
            say(f"progress: {n}/{len(looks)}")
    except Stopped:
        stopped = True
    finally:
        try:
            backend.close()
        except Exception:  # noqa: BLE001 - closing never hides the result
            pass
        doc = {"generator": {"name": backend.name, "version": str(backend.version)}, "looks": results}
        tmp = out_dir / "result.json.part"
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, out_dir / "result.json")
    if stopped:
        return fail("stopped", EXIT_STOPPED)
    if not any(r["ok"] for r in results):
        last = next((r["error"] for r in reversed(results) if r["error"]), "no look could be built")
        return fail(f"no look could be built: {last}", EXIT_NO_LOOK)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gen3d", description="Build 3D heads from prepared pictures.")
    parser.add_argument("--job", help="the job file (JSON)")
    parser.add_argument("--check", action="store_true", help="print what is installed as one JSON line")
    parser.add_argument("--weights-dir", default=".", help="where the backends' weights are (for --check)")
    parser.add_argument("--device", action="store_true",
                        help="with --check: also import PyTorch and name its version and the device")
    args = parser.parse_args(argv)
    if args.check:
        return check(Path(args.weights_dir), args.device)
    if not args.job:
        return fail("give --job JOB.json or --check", EXIT_BAD_JOB)
    try:
        job = read_job(args.job)
    except BadJob as e:
        return fail(str(e), EXIT_BAD_JOB)

    def on_term(_signum: int, _frame: Any) -> None:
        raise Stopped()

    signal.signal(signal.SIGTERM, on_term)
    stop = threading.Event()
    _watch_parent(stop)
    try:
        return run(job)
    except Stopped:
        return fail("stopped", EXIT_STOPPED)
    finally:
        stop.set()


if __name__ == "__main__":
    sys.exit(main())
