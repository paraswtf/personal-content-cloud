"""Video helpers built on the system ffmpeg/ffprobe binaries (subprocess, no shell).

CONTRACT. ffmpeg applies rotation metadata automatically when decoding (autorotate is on by
default), so frames come out in display orientation and their size matches MediaInfo.width/height.
- iter_frames(path, info, interval_s, max_frames, max_side) -> Iterator[(time_s, BGR ndarray)]
- extract_frame(path, t, max_side=None) -> BGR uint8 ndarray | None (extract_frame_image: PIL RGB)
- frames_at(path, info, times, max_side=256) -> [PIL RGB | None] (fingerprints)
- poster_time(info), stream_friendly(codec, ext, w, h, bitrate), needs_web_preview(info, ext),
  transcode_preview(src, dst, info, max_height) -> encoder name, make_video_derivatives(...)

All subprocess calls need a timeout and must not leave temp files behind on failure.
"""
from __future__ import annotations

import io
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from app.config import settings
from app.media import encoders, images
from app.media.probe import MediaInfo, ToolFailure

log = logging.getLogger(__name__)

# Codecs/containers every major browser plays. An original streams as is only when it uses these
# AND is already small (see stream_friendly); every other video gets a compressed H.264 copy.
WEB_SAFE_CODECS = {"h264"}
WEB_SAFE_EXTS = {".mp4", ".m4v", ".mov"}
KEYFRAME_INTERVAL_S = 2  # keyframe spacing of the streaming copy, so seeking stays quick
NICE_LEVEL = 10  # transcodes run at low CPU priority so the web server stays responsive

FRAME_TIMEOUT_S = 60
# Longer clips are sampled by seeking to each timestamp instead of decoding the whole stream.
SEEK_SAMPLING_MIN_S = 600.0
_FF = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
# Frame sampling reads showinfo's per-frame lines, which ffmpeg logs at the info level.
_FF_INFO = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]
_SHOWINFO_TB = re.compile(r"config in time_base:\s*(\d+)/(\d+)")
_SHOWINFO_PTS = re.compile(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[0-9.]+)")


def poster_time(info: MediaInfo) -> float:
    """Time used for the poster/thumbnail frame: 10% into the video, at most 3 s, 0 for very short clips."""
    d = info.duration_s or 0.0
    if d < 1.0:
        return 0.0
    return round(min(3.0, d * 0.1), 3)


def _limit_filter(max_side: int) -> str:
    """scale filter so the longer side <= max_side (never upscales, even dimensions)."""
    m = int(max_side)
    return (f"scale=w='if(gte(iw,ih),min(iw,{m}),-2)':h='if(gte(iw,ih),-2,min(ih,{m}))'"
            ":flags=bicubic")


def extract_frame(path: Path, t: float, max_side: int | None = None, *, exact_seek: bool = False,
                  gentle: bool = False) -> np.ndarray | None:
    """extract_frame_image as a contiguous HxWx3 uint8 BGR array (OpenCV / the ML engines)."""
    img = extract_frame_image(path, t, max_side, exact_seek=exact_seek, gentle=gentle)
    return images.to_bgr(img) if img is not None else None


def extract_frame_image(path: Path, t: float, max_side: int | None = None, *, exact_seek: bool = False,
                        gentle: bool = False) -> Image.Image | None:
    """One RGB frame at time t (seconds), display-oriented, optionally downscaled: the first frame at
    or after t. None if the file does not decode there; ToolFailure if ffmpeg could not run (timeout,
    spawn error, signal). exact_seek: seek coarsely to 3 s before t, then decode up to t (MPEG-TS input
    seeks can land after t). gentle: low CPU priority and 2 decoder threads (requests from visitors)."""
    t = max(0.0, float(t))
    if exact_seek:
        pre = max(0.0, t - 3.0)
        seek = ["-ss", f"{pre:.3f}", *(["-threads", "2"] if gentle else []), "-i", str(path),
                "-ss", f"{t - pre:.3f}"]
    else:
        seek = ["-ss", f"{t:.3f}", *(["-threads", "2"] if gentle else []), "-i", str(path)]
    cmd = [*_FF, *seek, "-map", "0:V:0", "-an", "-sn", "-dn", "-frames:v", "1"]
    if max_side:
        cmd += ["-vf", _limit_filter(max_side)]
    cmd += ["-c:v", "bmp", "-f", "image2pipe", "pipe:1"]
    if gentle:
        cmd = _low_priority(cmd)
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=FRAME_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("frame extraction failed for %s at %.2fs: %s", path, t, e)
        raise ToolFailure(f"ffmpeg frame extraction failed: {e}") from e
    if res.returncode < 0:
        raise ToolFailure(f"ffmpeg was killed by signal {-res.returncode}")
    if res.returncode != 0 or not res.stdout:
        return None
    try:
        with Image.open(io.BytesIO(res.stdout)) as im:
            im.load()
            return im.convert("RGB")
    except Exception as e:  # corrupt pipe output
        log.warning("could not decode extracted frame of %s: %s", path, e)
        return None


def shown_frame_time(path: Path, t: float) -> float | None:
    """Time of the frame on screen at t, on the timeline extract_frame seeks on (the file's start
    at 0): the last video frame that starts at or before t, since a frame stays up until the next
    one. A player's currentTime usually sits inside a frame, where extract_frame alone would return
    the next frame. Reads packet times only (no decoding). None when ffprobe fails or finds no frame
    near t (the caller then uses t)."""
    t = max(0.0, float(t))
    # -read_intervals takes the file's own timestamps, which start at format start_time.
    first = _timeline_start(path)
    origin = float(first) if first is not None else 0.0
    cmd = ["ffprobe", "-v", "error", "-select_streams", "V:0", "-read_intervals",
           f"{max(0.0, t - 3.0) + origin:.3f}%{t + 2.0 + origin:.3f}", "-show_entries",
           "packet=pts_time:format=start_time", "-of", "json", str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=FRAME_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("frame times of %s near %.2fs: %s", path, t, e)
        return None
    if res.returncode != 0:
        return None
    try:
        data = json.loads(res.stdout or b"{}")
        start = float((data.get("format") or {}).get("start_time") or 0.0)
        times = [float(p["pts_time"]) - start for p in data.get("packets") or []
                 if p.get("pts_time") not in (None, "N/A")]
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    # 1 ms of slack: callers send times floored to the millisecond, so a frame starting up to 1 ms
    # after t is the one meant (frames are far more than 1 ms apart).
    best = max((x for x in times if x <= t + 0.0011), default=None)
    return max(0.0, best) if best is not None else None


def frames_at(path: Path, info: MediaInfo, times: list[float], max_side: int = 256) -> list[Image.Image | None]:
    """RGB frames at the given times (used for fingerprints). Entries are None where decoding failed."""
    d = info.duration_s
    # Seeking to within one frame of the end yields nothing: stay a frame and a half clear of it.
    margin = max(0.1, 1.5 / info.fps) if info.fps else 0.1
    out: list[Image.Image | None] = []
    for t in times:
        t = max(0.0, float(t))
        if d:
            t = min(t, max(0.0, d - margin))
        frame = extract_frame_image(path, t, max_side)
        if frame is None and t > 0:
            frame = extract_frame_image(path, max(0.0, t - 1.0), max_side)
        out.append(frame)
    return out


def _even_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    s = min(1.0, max_side / max(width, height, 1))
    w = max(2, int(round(width * s / 2)) * 2)
    h = max(2, int(round(height * s / 2)) * 2)
    return w, h


def _sample_filter(interval: float, w: int, h: int) -> str:
    """Keep the first frame, then the first frame at least `interval` after the last kept one
    (tolerance 0.1% so a 29.97 fps clip still keeps every 30th frame at a 1 s interval), scale it,
    and log it with showinfo. Unlike the fps filter, select passes the decoded frames through
    with their own timestamps, so each frame is paired with the time it really has."""
    return (f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval * 0.999:.6f})',"
            f"scale={w}:{h}:flags=bicubic,showinfo")


def _pts_reader(stream, out: "queue.Queue[Fraction | None]") -> None:
    """Read ffmpeg's stderr and put the exact presentation time (seconds, as a Fraction from the
    integer pts and the filter's time base) of every frame showinfo logs. None marks the end."""
    tb: Fraction | None = None
    try:
        for raw in stream:
            line = raw.decode("utf-8", "replace")
            if "showinfo" not in line:
                continue
            m = _SHOWINFO_TB.search(line)
            if m:
                num, den = int(m.group(1)), int(m.group(2))
                tb = Fraction(num, den) if num > 0 and den > 0 else None
                continue
            m = _SHOWINFO_PTS.search(line)
            if m:
                out.put(int(m.group(1)) * tb if tb is not None else Fraction(m.group(2)))
    except (OSError, ValueError):
        pass
    finally:
        out.put(None)


def _timeline_start(path: Path) -> Fraction | None:
    """Start of the file's presentation timeline in its own timestamps (ffprobe format start_time):
    earlier than the first video frame when another stream, usually audio, starts first."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
        )
        value = res.stdout.strip()
        return Fraction(value) if value and value != "N/A" else None
    except (OSError, subprocess.TimeoutExpired, ValueError, ZeroDivisionError):
        return None


def _first_pts(stderr: bytes) -> Fraction | None:
    """The exact time of the first frame showinfo logged in a finished ffmpeg run's stderr."""
    q: queue.Queue[Fraction | None] = queue.Queue()
    _pts_reader(io.BytesIO(stderr), q)
    return q.get_nowait()


def iter_frames(
    path: Path, info: MediaInfo, interval_s: float, max_frames: int, max_side: int
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time_s, BGR uint8 array) sampled about every interval_s seconds (interval grows so
    that at most max_frames frames are produced), scaled so the longer side <= max_side.

    time_s is the exact presentation time of the frame whose pixels are yielded, on the file's own
    timeline: without -copyts ffmpeg shifts the input so the file starts at 0, which is also the 0 a
    browser plays from (for the original and for the streaming copy, which keeps that offset). So
    when audio starts before the video, the first frame correctly comes out after 0 rather than at
    it. The frames come from ONE ffmpeg process: select keeps the first frame
    and then the first frame at least interval_s after the last kept one, showinfo logs each kept
    frame's pts on stderr (read by a thread), and the scaled pixels arrive on stdout as rawvideo
    bgr24, W*H*3 bytes per frame. ffmpeg runs at low CPU priority and is killed if the consumer
    stops early.

    The whole run is limited to min(max(120 s, 4x duration), settings.video_faces_max_s): the
    duration comes from the uploaded file and must not hold the main queue for hours. When the
    limit is hit the frames read so far are all that come out. Clips longer than
    SEEK_SAMPLING_MIN_S are sampled by seeking to each timestamp (no full decode).
    """
    if max_frames <= 0:
        return
    interval = max(float(interval_s), 0.01)
    duration = info.duration_s or 0.0
    if duration > 0 and duration / interval > max_frames:
        interval = duration / max_frames
    w, h = _even_size(int(info.width), int(info.height), max_side)
    frame_bytes = w * h * 3
    timeout = min(max(120.0, 4 * duration), float(settings.video_faces_max_s))
    if duration > SEEK_SAMPLING_MIN_S:
        yield from _seek_frames(path, duration, interval, max_frames, w, h, timeout)
        return
    cmd = [*_FF_INFO, "-i", str(path), "-map", "0:V:0", "-an", "-sn", "-dn",
           "-vf", _sample_filter(interval, w, h), "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    try:
        proc = subprocess.Popen(_low_priority(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except OSError as e:
        raise ToolFailure(f"could not run ffmpeg: {e}") from e
    timer = threading.Timer(timeout, proc.kill)
    timer.daemon = True
    timer.start()
    deadline = time.monotonic() + timeout
    times: queue.Queue[Fraction | None] = queue.Queue()
    reader = threading.Thread(target=_pts_reader, args=(proc.stderr, times), daemon=True)
    reader.start()
    count = 0
    try:
        assert proc.stdout is not None
        while count < max_frames:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            try:
                # showinfo logs a frame before it is written to stdout, so its time is (nearly) there.
                pts = times.get(timeout=max(1.0, min(30.0, deadline - time.monotonic())))
            except queue.Empty:
                pts = None
            if pts is None:
                log.warning("frame sampling of %s stopped after %d frames: no frame time from ffmpeg", path, count)
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
            yield round(max(0.0, float(pts)), 6), arr
            count += 1
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()
        reader.join(timeout=5)
        if proc.stderr is not None:
            proc.stderr.close()
    if count == 0:
        # A stream that does not decode from the start may still give a frame this way.
        frame = extract_frame_image(path, 0.0)
        if frame is not None:
            if frame.size != (w, h):
                frame = frame.resize((w, h), Image.Resampling.BILINEAR)
            yield 0.0, images.to_bgr(frame)


def _seek_frames(path: Path, duration: float, interval: float, max_frames: int, w: int, h: int,
                 budget_s: float) -> Iterator[tuple[float, np.ndarray]]:
    """iter_frames for long clips: one short ffmpeg run per timestamp (accurate input seek, so only
    the frames from the preceding keyframe are decoded). -copyts keeps the file's own timestamps
    and showinfo reports the pts of the frame that comes out, so each time is that frame's exact
    pts minus the start of the file's timeline (ffprobe format start_time, the same 0 iter_frames
    and the browser use; the first frame's pts if ffprobe cannot tell). A frame that does not lie
    after the previous one (a container whose seeking is imprecise) is skipped. Stops when budget_s
    is used up."""
    deadline = time.monotonic() + budget_s
    frame_bytes = w * h * 3
    t, count = 0.0, 0
    base = _timeline_start(path)
    first: Fraction | None = None
    last: Fraction | None = None
    while count < max_frames and t < duration:
        left = deadline - time.monotonic()
        if left <= 0:
            log.warning("frame sampling of %s stopped at %.0fs: time limit reached", path, t)
            return
        seek = ["-ss", f"{t:.3f}"] if count > 0 else []
        cmd = [*_FF_INFO, "-copyts", *seek, "-i", str(path), "-map", "0:V:0", "-an", "-sn", "-dn",
               "-frames:v", "1", "-vf", f"scale={w}:{h}:flags=bicubic,showinfo", "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        try:
            res = subprocess.run(_low_priority(cmd), capture_output=True, timeout=min(FRAME_TIMEOUT_S, left),
                                 stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            log.warning("frame sampling of %s stopped at %.0fs: ffmpeg timed out", path, t)
            return
        except OSError as e:
            raise ToolFailure(f"could not run ffmpeg: {e}") from e
        pts = _first_pts(res.stderr)
        if count == 0:
            first = pts if base is None or pts is None else min(base, pts)
        if res.returncode == 0 and len(res.stdout) >= frame_bytes:
            if pts is not None and first is not None:
                if last is None or pts > last:
                    last = pts
                    arr = np.frombuffer(res.stdout[:frame_bytes], dtype=np.uint8).reshape(h, w, 3).copy()
                    yield round(max(0.0, float(pts - first)), 6), arr
            else:
                # No usable frame time (the start does not decode): fall back to the seek target.
                arr = np.frombuffer(res.stdout[:frame_bytes], dtype=np.uint8).reshape(h, w, 3).copy()
                yield round(t, 3), arr
        count += 1
        t = count * interval


def make_video_derivatives(path: Path, info: MediaInfo, thumb_path: Path, preview_path: Path,
                           preview_small_path: Path) -> None:
    """Poster frame -> grid thumbnail + large and small preview WebP (same sizes as photos).
    Raises ValueError if no frame can be decoded."""
    t = poster_time(info)
    frame = extract_frame_image(path, t, settings.preview_size)
    if frame is None and t > 0:
        frame = extract_frame_image(path, 0.0, settings.preview_size)
    if frame is None:
        raise ValueError("no video frame could be decoded")
    images.make_image_derivatives(frame, thumb_path, preview_path, preview_small_path)


def stream_friendly(codec: str | None, ext: str | None, width: int | None, height: int | None,
                    bitrate: int | None) -> bool:
    """True if the original can be streamed to viewers as it is: H.264 in mp4/m4v/mov, short side
    at most settings.video_preview_max_height * 1.1 (so 720p in either orientation), and a known
    bitrate of at most settings.video_stream_max_kbps. Anything heavier or less compatible gets a
    compressed streaming copy, so viewers never pull a full camera original just to watch."""
    if (codec or "").lower() not in WEB_SAFE_CODECS:
        return False
    if (ext or "").lower() not in WEB_SAFE_EXTS:
        return False
    short = min(int(width or 0), int(height or 0))
    if not short or short > settings.video_preview_max_height * 1.1:
        return False
    return bool(bitrate) and int(bitrate) <= settings.video_stream_max_kbps * 1000


def needs_web_preview(info: MediaInfo, ext: str) -> bool:
    """True unless the original is stream_friendly: every other video gets a compressed copy."""
    return not stream_friendly(info.video_codec, ext, info.width, info.height, info.bitrate)


def _low_priority(cmd: list[str]) -> list[str]:
    """Prefix cmd with `nice` (when available) so ffmpeg yields the CPU to the web server. The nice
    binary is used instead of preexec_fn=os.nice, which is unsafe in a process with threads."""
    nice = shutil.which("nice")
    return [nice, "-n", str(NICE_LEVEL), *cmd] if nice else cmd


def _transcode_cmd(src: Path, tmp: Path, encoder: str, max_height: int) -> list[str]:
    pre, video_args = encoders.build_video_args(encoder, int(max_height), int(settings.video_preview_crf),
                                                int(settings.video_preview_maxrate_kbps))
    return [*_FF, "-y", *pre, "-i", str(src), "-map", "0:V:0", "-map", "0:a:0?", "-sn", "-dn",
            "-map_metadata", "-1", *video_args, "-fps_mode", "passthrough",
            "-force_key_frames", f"expr:gte(t,n_forced*{KEYFRAME_INTERVAL_S})",
            "-c:a", "aac", "-b:a", f"{int(settings.video_preview_audio_kbps)}k", "-ac", "2",
            "-threads", str(max(1, int(settings.video_transcode_threads))),
            "-movflags", "+faststart", "-f", "mp4", str(tmp)]


class TranscodeTimeout(RuntimeError):
    """ffmpeg did not finish within the budget (a hung device counts as a device failure)."""


PROBE_TIMEOUT_S = 60
FALLBACK_FLOOR_S = 120.0


def _output_problem(path: Path) -> str | None:
    """None when `path` is a readable file with one video stream and a positive duration (ffprobe),
    else why not. An encoder that exits 0 and leaves a broken file must not be served."""
    try:
        res = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                             capture_output=True, timeout=PROBE_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"ffprobe could not read the copy: {e}"
    if res.returncode != 0:
        return "ffprobe could not read the copy: " + (res.stderr.decode("utf-8", "replace").strip()[-200:] or "no output")
    try:
        data = json.loads(res.stdout.decode("utf-8", "replace") or "{}")
    except ValueError:
        return "ffprobe gave no readable output for the copy"
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1:
        return f"the copy has {len(streams)} video streams"
    try:
        duration = float(data.get("format", {}).get("duration") or streams[0].get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        return "the copy has no duration"
    return None


def _fallback_budget(deadline: float, now: float, total: float) -> float:
    """Seconds left for the libx264 fallback: what remains of the job's budget, with a floor of
    FALLBACK_FLOOR_S (or the whole budget when that is shorter) so a short clip still gets its copy."""
    return max(deadline - now, min(FALLBACK_FLOOR_S, total))


def _run_transcode(cmd: list[str], tmp: Path, timeout: float) -> None:
    """Run one ffmpeg command and check its output; RuntimeError (with the last stderr lines) when
    it fails, TranscodeTimeout when it overruns `timeout`."""
    try:
        res = subprocess.run(_low_priority(cmd), capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise TranscodeTimeout(f"transcode timed out after {timeout:.0f}s") from e
    except OSError as e:
        raise RuntimeError(f"could not run ffmpeg: {e}") from e
    if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        msg = res.stderr.decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"transcode failed: {msg or 'no output'}")
    problem = _output_problem(tmp)
    if problem is not None:
        raise RuntimeError(f"transcode produced an unusable file: {problem}")


def transcode_preview(src: Path, dst: Path, info: MediaInfo, max_height: int) -> str:
    """Compressed web copy: H.264 from the detected encoder (encoders.detect: NVENC, VAAPI, Quick
    Sync or libx264 per settings.video_encoder), quality at the CRF equivalent of
    settings.video_preview_crf capped at settings.video_preview_maxrate_kbps with a buffer of twice
    that, high profile, yuv420p, even dims, a keyframe every KEYFRAME_INTERVAL_S seconds for seeking,
    SHORT side <= max_height (a portrait phone clip becomes 1080x1920 rather than 608x1080), never
    upscaled, plus AAC settings.video_preview_audio_kbps stereo (if audio), mp4 with +faststart.
    A hardware encoder that fails is noted (encoders.note_failure: skipped from then on when the
    device or driver is the problem or after two failures in a row) and the same call runs libx264
    instead with what is left of the time budget, so one bad driver or one odd file never costs a
    job attempt; a libx264 failure raises RuntimeError. The finished file is probed before it is
    moved into place.
    ffmpeg runs at low CPU priority (nice) with settings.video_transcode_threads threads. Writes a
    temp file and os.replace()s it into dst. Timeout scales with duration, up to
    settings.video_transcode_max_s. -fps_mode passthrough keeps every frame with its own timestamp,
    so a face's frame_time_s shows the same frame in the copy as in the original. Returns the name
    of the encoder that produced the copy."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp.mp4")
    total = min(max(120.0, 4 * (info.duration_s or 0.0)), float(settings.video_transcode_max_s))
    deadline = time.monotonic() + total
    first = encoders.detect().name
    plan = [first] if first == encoders.SOFTWARE else [first, encoders.SOFTWARE]
    budget = total
    try:
        for encoder in plan:
            try:
                _run_transcode(_transcode_cmd(src, tmp, encoder, max_height), tmp, budget)
            except RuntimeError as e:
                tmp.unlink(missing_ok=True)
                if encoder == encoders.SOFTWARE:
                    raise
                why = encoders.reason_from_stderr(str(e).replace("transcode failed: ", "", 1), encoder)
                device_error = isinstance(e, TranscodeTimeout) or encoders.is_device_error(why)
                skipped = encoders.note_failure(encoder, why, device_error=device_error)
                log.warning("web copy: %s failed (%s); libx264 makes this copy%s", encoder, why,
                            "; later copies skip it until the encoders are tested again" if skipped else "")
                budget = _fallback_budget(deadline, time.monotonic(), total)
                continue
            encoders.note_success(encoder)
            os.replace(tmp, dst)
            return encoder
        raise RuntimeError("transcode failed: no encoder ran")  # not reached
    finally:
        tmp.unlink(missing_ok=True)
