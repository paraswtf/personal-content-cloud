"""`python -m gen3d --job JOB.json` builds one 3D head per look; `python -m gen3d --check --weights-dir DIR` says
what is installed by looking at files (add `--device` to have PyTorch imported and its version and the device
named; that takes seconds, and the app asks the plain check for status reads). The contract with the app
(docs/ARCHITECTURE.md section 36):

Job file:
    {"out_dir": "/abs/work", "weights_dir": "/abs/models/gen3d/weights", "generator": "auto", "device": "auto",
     "options": {"detail": 2, "smoothing": 2, "texture": 2, "seed": 0},
     "looks": [{"key": "l1", "image": "/abs/work/l1.png", "inside": [0, 0, 2400, 2210]}]}
  `image`: a square RGB picture of the head and shoulders, the face centred, at the photo's own resolution (1024
  to 3072 pixels wide): the model is given a smaller copy, and the picture itself becomes the head's texture.
  `inside` (may be left out): [left, top, right, bottom] in pixels of that picture, the part that is the photo
  itself; right and bottom are the first pixels past it. Where the look's square left the photo the app filled the
  picture by repeating the photo's edge, and no photo is laid on the head from there. Left out: all of the picture.
  `detail`: 1 to 3. `smoothing`: 0 to 3 (off, light, normal, strong; 2 when left out). `texture`: 0 for a colour
  per vertex only, 1 and 2 for the photo laid over the front with its longest side at most 1024 and 2048 pixels,
  3 for the photo at its own size (2 when left out; the texture is never enlarged). `generator`: "auto" (the first
  installed backend of gen3d.backends.ORDER), a backend's name, or "fake".

A job runs in two passes, because the app's memory limit is shared with this process and the finishing must not
sit on top of the model. First every look goes through the cut-out and the model, and only its raw mesh and its
picture record are kept. Then the model is closed and its memory handed back. Then every look is finished and
written, one after the other, and reported as it is done.

Output: OUT/<key>.glb for every look that worked; OUT/result.json
    {"generator": {"name", "version"}, "memory_mb": {"peak", "with_model", "released"},
     "looks": [{"key", "ok", "mesh", "vertices", "faces", "seconds", "error", "textured", "texture", "overlap",
     "overlap_head", "note"}]}
  `memory_mb`: this process's resident memory by the kernel's own count: the most it held at any moment of the
  job, what it held at the end of the first pass with the model loaded, and what it held once the model was let
  go (each null where it could not be read or the job did not get there).
  `vertices` and `faces` count the whole surface. `textured`: the head has the photo layer; `texture`: its picture's
  [width, height], else null; `overlap`: how well the mesh's outline covered the person's in the picture (0 to 1,
  null when nothing was fitted) and `overlap_head` the same over the head alone: the photo is used when both reach
  0.92; `note`: why a head that was asked for a photo layer has none.
  `seconds`: the look's share of both passes.
  On stdout, one line per event, flushed at once: "progress: N/M" and "look: <key> ok|error <seconds>".
  "progress: 0/M" comes at the start and nothing during the first pass; in the second, every look gets its
  "look:" line and the next "progress:" line when it is done, in the job's order, and a look that worked has its
  file in place before its line is printed. A look that failed in the first pass is reported there too, in its
  turn. A look that fails does not stop the others, and no look fails because of its photo: it is written without.

Exit: 0 when at least one look was built, 3 when none was, 2 for a bad job file, 4 when the backend cannot load
(missing weights, not enough memory), 5 when it was told to stop (SIGTERM) or its parent went away. The reason is the
last line on stderr.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from gen3d import backends, glb
from gen3d.finishing import finish_head

EXIT_OK, EXIT_BAD_JOB, EXIT_NO_LOOK, EXIT_NO_BACKEND, EXIT_STOPPED = 0, 2, 3, 4, 5
KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


class BadJob(ValueError):
    pass


class Stopped(BaseException):
    """Raised by the SIGTERM handler wherever the job is. Not an Exception: nothing that keeps a look going after
    a failure (the photo stage's fallback, a library's own handler) may swallow the order to stop."""


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
        inside = look.get("inside")
        if inside is not None and not (isinstance(inside, list) and len(inside) == 4 and all(
                isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in inside)):
            raise BadJob(f"look {key}: inside must be [left, top, right, bottom] in pixels of the picture")
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
    for name in ("smoothing", "texture"):
        try:
            level = int(options.get(name, 2))
        except (TypeError, ValueError) as e:
            raise BadJob(f"{name} must be 0, 1, 2 or 3") from e
        if level not in (0, 1, 2, 3) or isinstance(options.get(name, 2), bool):
            raise BadJob(f"{name} must be 0, 1, 2 or 3")
        options[name] = level
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
    say(json.dumps({"backends": found, "auto": auto, "device": device, "torch": version}, sort_keys=True))
    return EXIT_OK


def _prepare(name: str):
    """gen3d.prep.prepare for a real backend (it cuts the background out), None for the fake one, which takes the
    picture as it is."""
    if name == "fake":
        return None
    from gen3d import prep  # noqa: PLC0415 - numpy and Pillow at import; its model only when it is used

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


def _resident_mb() -> int | None:
    """This process's resident memory now, in MB (Linux; None where it cannot be read)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 2**20)
    except (OSError, ValueError, IndexError):
        return None


def _peak_mb() -> int | None:
    """The most resident memory this process held so far, in MB, by the kernel's count."""
    try:
        import resource  # noqa: PLC0415 - not on every system

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak // 2**20) if sys.platform == "darwin" else int(peak // 1024)
    except Exception:  # noqa: BLE001 - a number for the record, never a reason to fail
        return None


def _release(backend: Any) -> None:
    """Close the backend and hand its memory back, before the heads are finished: the model, the cut-out model,
    what the collector still holds, the blocks PyTorch keeps on the graphics card for reuse, and the freed memory
    the C library keeps for itself."""
    try:
        backend.close()
    except Exception:  # noqa: BLE001 - closing never hides the result
        pass
    prep = sys.modules.get("gen3d.prep")
    if prep is not None:
        prep.release()
    gc.collect()
    torch = sys.modules.get("torch")   # only a real backend imported it; the stand-in never does
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - not glibc: nothing to ask
        pass


def run(job: dict[str, Any]) -> int:
    from PIL import Image  # noqa: PLC0415

    from gen3d.prep import source_of, whole  # noqa: PLC0415

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
    memory: dict[str, int | None] = {"peak": None, "with_model": None, "released": None}
    made: list[dict[str, Any]] = []   # per look, from the first pass: its mesh and picture record, or its error
    stopped = released = False
    say(f"progress: 0/{len(looks)}")
    try:
        # the first pass: the cut-out and the model. Nothing is reported from here
        for look in looks:
            started = time.monotonic()
            options = {**job["options"], "out_dir": str(out_dir), "key": look["key"], "inside": look.get("inside")}
            kept: dict[str, Any] = {"options": options, "mesh": None, "prepared": None, "error": None}
            try:
                with Image.open(look["image"]) as im:
                    picture = im.convert("RGB")
                if prepare is not None:
                    prepared = prepare(picture, options)
                    given = prepared.rgba
                else:
                    prepared, given = whole(picture, options), picture
                del picture
                kept["mesh"] = backend.generate(given, options)
                # the picture itself is read from its file again when the look is finished: up to 28 MB a look
                # that need not wait in memory beside the model
                prepared.source = None
                kept["prepared"] = prepared
                del prepared, given
            except Stopped:
                raise
            except Exception as e:  # noqa: BLE001 - this look failed; the others still run
                kept.update(mesh=None, prepared=None, error=f"{type(e).__name__}: {e}"[:500])
            kept["seconds"] = time.monotonic() - started
            made.append(kept)
        memory["with_model"] = _resident_mb()
        released = True
        _release(backend)
        memory["released"] = _resident_mb()
        # the second pass: every look finished, written and reported, in the job's order
        for n, (look, kept) in enumerate(zip(looks, made), start=1):
            key = look["key"]
            started = time.monotonic()
            row: dict[str, Any] = {"key": key, "ok": False, "mesh": None, "vertices": 0, "faces": 0, "seconds": 0.0,
                                   "error": kept["error"], "textured": False, "texture": None, "overlap": None,
                                   "overlap_head": None, "note": None}
            if kept["error"] is None:
                try:
                    mesh, prepared = kept.pop("mesh"), kept.pop("prepared")
                    with Image.open(look["image"]) as im:
                        prepared.source = source_of(im)
                    head = finish_head(mesh, prepared, kept["options"])
                    del mesh, prepared
                    glb.write(out_dir / f"{key}.glb", head)
                    row.update(ok=True, mesh=f"{key}.glb", vertices=int(head.base.vertices.shape[0]),
                               faces=int(head.base.faces.shape[0]), textured=head.photo is not None,
                               texture=None if head.photo is None or head.texture_size is None else list(head.texture_size),
                               overlap=None if head.overlap is None else round(float(head.overlap), 4),
                               overlap_head=None if head.overlap_head is None else round(float(head.overlap_head), 4),
                               note=head.note)
                    del head
                except Stopped:
                    raise
                except Exception as e:  # noqa: BLE001 - this look failed; the others still run
                    row["error"] = f"{type(e).__name__}: {e}"[:500]
            row["seconds"] = round(kept["seconds"] + time.monotonic() - started, 2)
            results.append(row)
            say(f"look: {key} {'ok' if row['ok'] else 'error'} {row['seconds']}")
            say(f"progress: {n}/{len(looks)}")
    except Stopped:
        stopped = True
    finally:
        if not released:
            try:
                backend.close()
            except Exception:  # noqa: BLE001 - closing never hides the result
                pass
        memory["peak"] = _peak_mb()
        doc = {"generator": {"name": backend.name, "version": str(backend.version)}, "memory_mb": memory,
               "looks": results}
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
