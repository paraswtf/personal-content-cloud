"""onnxruntime session factory shared by every model in app.ml (docs/ARCHITECTURE.md section 8).

Contract:
  providers() -> list[str]          provider names new sessions get, in order of preference
  session(path, *, threads=None)    an InferenceSession on CUDA when allowed and working, else CPU
  cuda_usable() -> bool             True when CUDA was verified to work in this process
  cuda_provider_options() -> dict   options used for the CUDA execution provider (also used by app.ml.ocr)
  gpu_info() -> dict                {available, name, provider} for /api/system
  status() -> dict                  gpu_info() plus the setting, the verification result and the last error
  prepare() -> None                 one-time process setup; every function above calls it first
  wrap_session(sess) -> session     a CUDA session whose run() returns the arena's unused memory to the
                                    device after every call (settings.gpu_arena_shrink); app.ml.ocr
                                    applies it to rapidocr's sessions

GPU selection follows settings.gpu_enabled: 'false' always uses the CPU; 'auto' and 'true' use CUDA when
onnxruntime-gpu lists CUDAExecutionProvider and a test session really runs on it. onnxruntime-gpu 1.2x+
finds its CUDA and cuDNN libraries (the nvidia-* pip wheels) only after ort.preload_dlls(), so prepare()
calls it once before any session is built. Without the preload, sessions silently come up on the CPU
with the provider bridge printing load errors, so the check reads sess.get_providers() rather than
relying on an exception. TensorRT is listed by the wheel too but is never used.

Session creation is serialised by one lock; running a session is the engine's business (each engine
keeps one lock per session). When CUDA fails for a session, that session is rebuilt on the CPU and
the failure is logged once.

GPU memory: onnxruntime keeps every device allocation in a per-session arena that grows with each new
input shape and never gives memory back, so a server that has seen many image sizes ends up holding
most of the card (7.7 GB of 8 GB was measured after ordinary use). CUDA sessions are therefore returned
wrapped: their run() passes the run option `memory.enable_memory_arena_shrinkage` for the GPU, which
frees the arena's unused chunks after each call. The cost is a device allocation per run, which the
background pipeline does not notice; GPU_ARENA_SHRINK=false turns it off.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.config import settings

log = logging.getLogger(__name__)

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"

# cuDNN benchmarking (EXHAUSTIVE) re-runs for every new input shape; face detection and OCR see a new
# shape for almost every image, so the heuristic choice is much faster overall. The arena grows only
# by what is requested, which keeps several models on one 8 GB card.
_CUDA_OPTIONS: dict[str, Any] = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "HEURISTIC",
    "do_copy_in_default_stream": True,
}

_prepare_lock = threading.Lock()
_create_lock = threading.Lock()
_prepared = False
_cuda_ok: bool | None = None       # None: not decided yet (no test model; decided by the first session)
_last_error: str | None = None
_cuda_fail_logged = False
_gpu_name: str | None = None
_gpu_name_checked = False


def _gpu_setting() -> str:
    value = (getattr(settings, "gpu_enabled", "auto") or "auto").strip().lower()
    return value if value in ("auto", "true", "false") else "auto"


def _cuda_listed() -> bool:
    try:
        return CUDA in ort.get_available_providers()
    except Exception:  # pragma: no cover - broken install
        return False


def _test_model() -> Path | None:
    """A tiny ONNX model to verify CUDA with: rapidocr's text direction classifier (about 0.6 MB)."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("rapidocr")
        if spec is None or not spec.origin:
            return None
        p = Path(spec.origin).parent / "models" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
        return p if p.is_file() else None
    except Exception:
        return None


def _base_options(threads: int | None) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    n = threads if threads is not None else getattr(settings, "ort_threads", 2)
    opts.intra_op_num_threads = max(1, int(n))
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return opts


def _verify_cuda(model: Path) -> tuple[bool, str | None]:
    """Build a session on CUDA and run it once; True when CUDA really executes it."""
    try:
        sess = ort.InferenceSession(str(model), _base_options(1),
                                    providers=[(CUDA, dict(_CUDA_OPTIONS)), CPU])
        if not sess.get_providers() or sess.get_providers()[0] != CUDA:
            return False, "CUDA execution provider did not initialise (libraries missing or GPU unsupported)"
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if len(shape) == 4:
            shape = [1, shape[1] or 3, 48, 192]
        sess.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
        del sess
        return True, None
    except Exception as e:  # noqa: BLE001 - any failure means CPU
        return False, f"CUDA test session failed: {e}"


def prepare() -> None:
    """One-time setup: quiet ORT logging, load the pip CUDA libraries, verify CUDA. Idempotent."""
    global _prepared, _cuda_ok, _last_error
    if _prepared:
        return
    with _prepare_lock:
        if _prepared:
            return
        try:
            ort.set_default_logger_severity(3)
        except Exception:  # pragma: no cover
            pass
        if _gpu_setting() == "false":
            _cuda_ok = False
        elif not _cuda_listed():
            _cuda_ok = False
            _last_error = "onnxruntime has no CUDA execution provider"
            if _gpu_setting() == "true":
                log.warning("GPU_ENABLED=true but %s", _last_error)
        else:
            try:
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls(cuda=True, cudnn=True, msvc=False, directory="")
            except Exception as e:  # noqa: BLE001
                log.info("onnxruntime preload_dlls failed (%s); relying on the system library path", e)
            model = _test_model()
            if model is None:
                _cuda_ok = None  # decided by the first real session
            else:
                ok, err = _verify_cuda(model)
                _cuda_ok = ok
                _last_error = err
                if ok:
                    log.info("onnxruntime: CUDA verified (%s)", gpu_name() or "GPU")
                else:
                    log.warning("onnxruntime: %s; using the CPU", err)
        _prepared = True


def reset() -> None:
    """Forget the verification result (tests change settings.gpu_enabled). Existing sessions stay as they are."""
    global _prepared, _cuda_ok, _last_error, _cuda_fail_logged
    with _prepare_lock:
        _prepared = False
        _cuda_ok = None
        _last_error = None
        _cuda_fail_logged = False


def cuda_usable() -> bool:
    """True when new sessions should be (and so far could be) placed on CUDA."""
    prepare()
    if _gpu_setting() == "false" or not _cuda_listed():
        return False
    return _cuda_ok is not False


def cuda_provider_options() -> dict[str, Any]:
    return dict(_CUDA_OPTIONS)


def providers() -> list[str]:
    """Provider names, best first, that session() asks for."""
    return [CUDA, CPU] if cuda_usable() else [CPU]


def session(path: Path | str, *, threads: int | None = None) -> ort.InferenceSession:
    """Build an InferenceSession for an ONNX file, on CUDA when usable, else on the CPU."""
    global _cuda_ok, _last_error, _cuda_fail_logged
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"model file missing: {p} (run scripts/download_models.py)")
    prepare()
    opts = _base_options(threads)
    with _create_lock:
        if cuda_usable():
            try:
                sess = ort.InferenceSession(str(p), opts, providers=[(CUDA, dict(_CUDA_OPTIONS)), CPU])
                if sess.get_providers() and sess.get_providers()[0] == CUDA:
                    if _cuda_ok is None:
                        _cuda_ok = True
                    return wrap_session(sess)
                err = f"CUDA execution provider did not initialise for {p.name}"
            except Exception as e:  # noqa: BLE001
                err = f"CUDA session for {p.name} failed: {e}"
            _last_error = err
            if _cuda_ok is None:
                _cuda_ok = False
            if not _cuda_fail_logged:
                _cuda_fail_logged = True
                log.warning("onnxruntime: %s; falling back to the CPU", err)
        return ort.InferenceSession(str(p), opts, providers=[CPU])


class _ShrinkingSession:
    """Proxy around an InferenceSession on CUDA: run() frees the arena's unused memory afterwards."""

    def __init__(self, sess: ort.InferenceSession) -> None:
        self._sess = sess
        self._ro = ort.RunOptions()
        self._ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{_CUDA_OPTIONS['device_id']}")

    def run(self, output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
        return self._sess.run(output_names, input_feed, run_options if run_options is not None else self._ro)

    @property
    def shrinks_arena(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sess, name)


def wrap_session(sess: Any) -> Any:
    """The session itself, or the shrinking proxy when it runs on CUDA and the setting is on."""
    if getattr(sess, "shrinks_arena", False) or not settings.gpu_arena_shrink:
        return sess
    if session_provider(sess) == CUDA:
        return _ShrinkingSession(sess)
    return sess


def session_provider(sess: Any) -> str | None:
    """The provider that runs most of a session ('CUDAExecutionProvider' or 'CPUExecutionProvider')."""
    try:
        provs = sess.get_providers()
        return provs[0] if provs else None
    except Exception:
        return None


def gpu_name() -> str | None:
    """GPU name from a quick nvidia-smi query, cached; None without nvidia-smi or a GPU."""
    global _gpu_name, _gpu_name_checked
    if _gpu_name_checked:
        return _gpu_name
    _gpu_name_checked = True
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5, check=False)
        line = (out.stdout or "").strip().splitlines()
        _gpu_name = line[0].strip() if out.returncode == 0 and line and line[0].strip() else None
    except Exception:  # noqa: BLE001
        _gpu_name = None
    return _gpu_name


def gpu_info() -> dict[str, Any]:
    """{available: CUDA usable, name: GPU name or None, provider: provider new sessions use}."""
    usable = cuda_usable()
    return {
        "available": bool(usable),
        "name": gpu_name(),
        "provider": CUDA if usable else CPU,
    }


def status() -> dict[str, Any]:
    info = gpu_info()
    info.update({
        "setting": _gpu_setting(),
        "cuda_listed": _cuda_listed(),
        "verified": _cuda_ok,
        "error": _last_error,
        "onnxruntime": ort.__version__,
    })
    return info
