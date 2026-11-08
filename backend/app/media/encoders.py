"""Video encoders for the web copy (docs/ARCHITECTURE.md section 27).

CONTRACT.
- candidates() -> [name, ...]: the encoders to try, in order, from settings.video_encoder: 'auto'
  means h264_nvenc, h264_vaapi, h264_qsv, libx264; a known name means that one, then libx264 (an
  unknown name is logged once and treated as auto).
- detect() -> EncoderInfo: the first candidate whose three-frame lavfi test encode works. Cached for
  the process behind a lock; computed lazily (the first transcode, on the video worker thread) and
  never at startup. redetect() clears the cache and the failure marks. note_failure(name, reason,
  device_error) records a failed real run: the encoder is skipped until redetect() when the
  failure is about the device or driver (is_device_error) or when two runs in a row failed;
  a single per-file failure (an odd pixel format, a size the chip refuses) only costs that copy,
  which transcode_preview makes with libx264. note_success(name) clears the run of failures;
  mark_failed(name, reason) skips the encoder unconditionally.
- build_video_args(name, max_height, crf, maxrate_kbps) -> (pre_input, video_args): the ffmpeg
  arguments before "-i" (the VAAPI device) and the filter plus codec options for that encoder; the
  quality knob is the CRF equivalent (-crf, -cq, -qp, -global_quality) capped by maxrate with a
  buffer of twice that. The short side of the picture is scaled to at most max_height, never up.
- status() -> dict: {encoder, hw, device, reason, checked_at, candidates} for GET /api/system
  (encoder is None until detected).
"""
from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
from app.util import now_iso

log = logging.getLogger(__name__)

SOFTWARE = "libx264"
HARDWARE = ("h264_nvenc", "h264_vaapi", "h264_qsv")
KNOWN = (*HARDWARE, SOFTWARE)
TEST_TIMEOUT_S = 20
TEST_FRAMES = 3

_lock = threading.Lock()
_cached: EncoderInfo | None = None
_failed: dict[str, str] = {}
_strikes: dict[str, int] = {}  # consecutive real-run failures per encoder (note_failure)
_warned_unknown: set[str] = set()
STRIKES_TO_MARK = 2
# stderr fragments that mean the device or driver cannot encode at all (not one bad file)
DEVICE_ERROR_MARKERS = (
    "cannot load", "no va display", "mfx session", "cannot init", "failed to initiali",
    "no such device", "device creation failed", "no capable devices", "no nvenc capable",
    "driver does not support", "cannot open the device", "could not open encoder", "timed out",
)


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    hw: bool
    device: str | None
    reason: str | None   # why the candidates before this one were not used (short), or None
    checked_at: str


def candidates() -> list[str]:
    """Encoder names in the order to try, from settings.video_encoder."""
    wanted = (settings.video_encoder or "auto").strip().lower()
    if wanted == "auto":
        return list(KNOWN)
    if wanted in KNOWN:
        return [wanted] if wanted == SOFTWARE else [wanted, SOFTWARE]
    if wanted not in _warned_unknown:
        _warned_unknown.add(wanted)
        log.warning("VIDEO_ENCODER=%r is not one of %s; trying them in order", wanted, ", ".join(KNOWN))
    return list(KNOWN)


def _scale_expr(max_height: int) -> str:
    """Short side at most max_height, even dimensions, never upscaled. ffmpeg autorotates before
    the filter, so iw/ih are display dimensions: a portrait clip keeps its orientation."""
    mh = int(max_height)
    return (f"w='if(gte(iw,ih),-2,trunc(min(iw,{mh})/2)*2)'"
            f":h='if(gte(iw,ih),trunc(min(ih,{mh})/2)*2,-2)'")


def build_video_args(name: str, max_height: int, crf: int, maxrate_kbps: int) -> tuple[list[str], list[str]]:
    """(arguments before -i, filter and codec arguments) for the encoder `name`."""
    scale = f"scale={_scale_expr(max_height)}"
    q = str(int(crf))
    rate = [f"-maxrate", f"{int(maxrate_kbps)}k", "-bufsize", f"{2 * int(maxrate_kbps)}k"]
    if name == "h264_nvenc":
        return [], ["-vf", f"{scale},format=yuv420p", "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                    "-rc", "vbr", "-cq", q, "-b:v", "0", *rate, "-forced-idr", "1", "-profile:v", "high",
                    "-pix_fmt", "yuv420p"]
    if name == "h264_vaapi":
        return (["-vaapi_device", settings.vaapi_device],
                ["-vf", f"format=nv12,hwupload,scale_vaapi={_scale_expr(max_height)}", "-c:v", "h264_vaapi",
                 "-qp", q, *rate, "-profile:v", "high"])
    if name == "h264_qsv":
        return [], ["-vf", f"{scale},format=nv12", "-c:v", "h264_qsv", "-global_quality", q, *rate,
                    "-profile:v", "high"]
    if name == SOFTWARE:
        return [], ["-vf", f"{scale},format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", q, *rate,
                    "-profile:v", "high", "-pix_fmt", "yuv420p"]
    raise ValueError(f"unknown encoder {name!r}")


def test_command(name: str) -> list[str]:
    """A three-frame encode of a lavfi colour source to the null muxer, with the real flags."""
    pre, args = build_video_args(name, 240, int(settings.video_preview_crf), 1000)
    return ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *pre,
            "-f", "lavfi", "-i", "color=size=320x240:rate=25", "-frames:v", str(TEST_FRAMES), *args,
            "-an", "-f", "null", "-"]


def reason_from_stderr(text: str, name: str) -> str:
    """The line of ffmpeg's stderr that says why `name` failed: the first line tagged with the
    encoder or the device ("Cannot load libnvidia-encode.so.1", "No VA display found", "Error
    creating a MFX session"), else the first non-empty line, cut short."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "no output"
    tagged = [ln for ln in lines if f"[{name}" in ln or "[VAAPI" in ln or "[AVHWDeviceContext" in ln]
    line = tagged[0] if tagged else lines[0]
    line = line.split("] ", 1)[1] if line.startswith("[") and "] " in line else line
    return line[:160]


def _try(name: str) -> str | None:
    """None when the encoder works, else a short reason."""
    try:
        res = subprocess.run(test_command(name), capture_output=True, timeout=TEST_TIMEOUT_S,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return f"{name}: test encode timed out"
    except OSError as e:
        return f"{name}: could not run ffmpeg ({e})"
    if res.returncode == 0:
        return None
    err = res.stderr.decode("utf-8", "replace")
    return f"{name}: {reason_from_stderr(err, name) if err.strip() else f'exit {res.returncode}'}"


def _device_for(name: str) -> str | None:
    if name == "h264_nvenc":
        try:
            from app.ml import runtime

            return runtime.gpu_name() or "NVIDIA GPU"
        except Exception:  # noqa: BLE001 - the name is cosmetic
            return "NVIDIA GPU"
    if name == "h264_vaapi":
        return settings.vaapi_device
    if name == "h264_qsv":
        return "Intel Quick Sync"
    return None


def detect() -> EncoderInfo:
    """The encoder to use, probing the candidates once per process."""
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
        reasons: list[str] = []
        info: EncoderInfo | None = None
        for name in candidates():
            if name in _failed:
                reasons.append(f"{name}: {_failed[name]}")
                continue
            why = _try(name)
            if why is None:
                info = EncoderInfo(name=name, hw=name != SOFTWARE, device=_device_for(name),
                                   reason="; ".join(reasons) or None, checked_at=now_iso())
                break
            reasons.append(why)
        if info is None:
            # nothing worked (ffmpeg missing or broken): the transcode itself reports the failure
            info = EncoderInfo(name=SOFTWARE, hw=False, device=None, reason="; ".join(reasons) or None,
                               checked_at=now_iso())
        _cached = info
        log.info("video encoder: %s%s%s", info.name, f" on {info.device}" if info.device else "",
                 f" ({info.reason})" if info.reason else "")
        return info


def detected() -> EncoderInfo | None:
    """The cached result without probing (None until the first detect())."""
    return _cached


def is_device_error(reason: str) -> bool:
    """True when a failure reason says the device or driver is unusable rather than one file being odd."""
    text = (reason or "").lower()
    return any(marker in text for marker in DEVICE_ERROR_MARKERS)


def mark_failed(name: str, reason: str) -> None:
    """Skip `name` until redetect(), whatever the reason."""
    global _cached
    with _lock:
        _failed[name] = reason[:200]
        _strikes.pop(name, None)
        _cached = None


def note_failure(name: str, reason: str, *, device_error: bool = False) -> bool:
    """A real transcode with `name` failed. Returns True when the encoder is now skipped: at once for
    a device or driver problem, else after STRIKES_TO_MARK failures in a row (note_success resets)."""
    global _cached
    with _lock:
        strikes = _strikes.get(name, 0) + 1
        _strikes[name] = strikes
        if not device_error and strikes < STRIKES_TO_MARK:
            return False
        _failed[name] = reason[:200]
        _strikes.pop(name, None)
        _cached = None
        return True


def note_success(name: str) -> None:
    """A real transcode with `name` worked: the count of failures in a row starts again."""
    with _lock:
        _strikes.pop(name, None)


def redetect() -> None:
    """Forget the cached encoder, the failure marks and the strikes; the next detect() probes again."""
    global _cached
    with _lock:
        _cached = None
        _failed.clear()
        _strikes.clear()


def status() -> dict[str, Any]:
    info = detected()
    d = asdict(info) if info is not None else {}
    return {
        "encoder": d.get("name"), "hw": bool(d.get("hw", False)), "device": d.get("device"),
        "reason": d.get("reason"), "checked_at": d.get("checked_at"), "candidates": candidates(),
    }
