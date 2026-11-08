"""Video encoders (docs/ARCHITECTURE.md section 27): candidate order, detection with a fake ffmpeg,
the argument builders, the libx264 fallback inside transcode_preview, and the system routes."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import jobs
from app.config import reload_settings, settings
from app.media import encoders, video
from app.media.probe import MediaInfo
from tests.helpers import make_item


@pytest.fixture(autouse=True)
def _fresh_encoders():
    encoders.redetect()
    yield
    encoders.redetect()


class FakeRun:
    """Stands in for subprocess.run: `fail` names the encoders whose runs exit 1; every call is kept."""

    def __init__(self, fail=(), write_output=True, stderr=None):
        self.fail = set(fail)
        self.write_output = write_output
        self.stderr = stderr  # the failing run's stderr; the default names a driver problem
        self.calls: list[list[str]] = []

    def encoder_of(self, cmd):
        return cmd[cmd.index("-c:v") + 1] if "-c:v" in cmd else None

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        enc = self.encoder_of(cmd)
        if enc in self.fail:
            msg = self.stderr or f"[{enc} @ 0x1] Cannot load driver\n"
            return subprocess.CompletedProcess(cmd, 1, b"", msg.encode())
        out = Path(cmd[-1])
        if self.write_output and out.suffix == ".mp4":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00" * 64)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


def test_candidates_follow_the_setting():
    reload_settings(video_encoder="auto")
    assert encoders.candidates() == ["h264_nvenc", "h264_vaapi", "h264_qsv", "libx264"]
    reload_settings(video_encoder="h264_vaapi")
    assert encoders.candidates() == ["h264_vaapi", "libx264"]
    reload_settings(video_encoder="libx264")
    assert encoders.candidates() == ["libx264"]
    reload_settings(video_encoder="nvenc")  # a typo falls back to the auto order
    assert encoders.candidates() == ["h264_nvenc", "h264_vaapi", "h264_qsv", "libx264"]
    reload_settings()


def test_detect_takes_the_first_working_candidate(monkeypatch):
    reload_settings(video_encoder="auto")
    fake = FakeRun(fail={"h264_nvenc"})
    monkeypatch.setattr(encoders.subprocess, "run", fake)
    monkeypatch.setattr(encoders, "_device_for", lambda name: "dev-" + name)
    info = encoders.detect()
    assert info.name == "h264_vaapi" and info.hw and info.device == "dev-h264_vaapi"
    assert info.reason and info.reason.startswith("h264_nvenc:") and "Cannot load driver" in info.reason
    assert [fake.encoder_of(c) for c in fake.calls] == ["h264_nvenc", "h264_vaapi"]
    # cached: no more probing
    assert encoders.detect() is info and len(fake.calls) == 2
    assert encoders.status()["encoder"] == "h264_vaapi" and encoders.status()["hw"] is True
    reload_settings()


def test_detect_falls_back_to_software_and_says_why(monkeypatch):
    reload_settings(video_encoder="auto")
    fake = FakeRun(fail={"h264_nvenc", "h264_vaapi", "h264_qsv"})
    monkeypatch.setattr(encoders.subprocess, "run", fake)
    info = encoders.detect()
    assert info.name == "libx264" and not info.hw and info.device is None
    assert info.reason.count("Cannot load driver") == 3
    # every test command has the real flags of its encoder and ends in the null muxer
    for cmd in fake.calls:
        assert cmd[-3:] == ["-an", "-f", "null"][:3] or cmd[-2:] == ["null", "-"]
        assert "-frames:v" in cmd and "lavfi" in cmd
    reload_settings()


def test_detect_when_nothing_runs(monkeypatch):
    def boom(cmd, **kw):
        raise OSError("ffmpeg: not found")
    monkeypatch.setattr(encoders.subprocess, "run", boom)
    info = encoders.detect()
    assert info.name == "libx264" and "not found" in info.reason


def test_status_before_detection():
    s = encoders.status()
    assert s["encoder"] is None and s["hw"] is False and s["checked_at"] is None and s["candidates"]


def test_build_video_args_per_encoder():
    pre, args = encoders.build_video_args("libx264", 1080, 23, 6000)
    assert pre == [] and args[args.index("-c:v") + 1] == "libx264"
    assert "-crf" in args and args[args.index("-crf") + 1] == "23"
    assert args[args.index("-maxrate") + 1] == "6000k" and args[args.index("-bufsize") + 1] == "12000k"
    assert "min(ih,1080)" in args[args.index("-vf") + 1] and "yuv420p" in args[args.index("-vf") + 1]
    assert args[args.index("-profile:v") + 1] == "high"

    pre, args = encoders.build_video_args("h264_nvenc", 720, 20, 4000)
    assert pre == [] and args[args.index("-c:v") + 1] == "h264_nvenc"
    assert args[args.index("-cq") + 1] == "20" and args[args.index("-rc") + 1] == "vbr" and args[args.index("-b:v") + 1] == "0"
    assert args[args.index("-preset") + 1] == "p4" and args[args.index("-maxrate") + 1] == "4000k"
    assert "min(ih,720)" in args[args.index("-vf") + 1]

    reload_settings(vaapi_device="/dev/dri/renderD129")
    pre, args = encoders.build_video_args("h264_vaapi", 1080, 23, 6000)
    assert pre == ["-vaapi_device", "/dev/dri/renderD129"]
    vf = args[args.index("-vf") + 1]
    assert vf.startswith("format=nv12,hwupload,scale_vaapi=") and "min(ih,1080)" in vf
    assert args[args.index("-c:v") + 1] == "h264_vaapi" and args[args.index("-qp") + 1] == "23"
    reload_settings()

    pre, args = encoders.build_video_args("h264_qsv", 1080, 23, 6000)
    assert pre == [] and args[args.index("-c:v") + 1] == "h264_qsv" and args[args.index("-global_quality") + 1] == "23"
    assert "format=nv12" in args[args.index("-vf") + 1]

    with pytest.raises(ValueError):
        encoders.build_video_args("h264_amf", 1080, 23, 6000)


def _info(duration=3.0):
    return MediaInfo(kind="video", mime="video/mp4", ext=".mp4", width=1920, height=1080, duration_s=duration,
                     bitrate=12_000_000, video_codec="hevc", has_audio=True)


@pytest.fixture()
def sound_output(monkeypatch):
    """The fake runs write 64 bytes, not a video: stand in for the ffprobe check of the finished file."""
    monkeypatch.setattr(video, "_output_problem", lambda path: None)


def test_transcode_falls_back_to_libx264_in_the_same_call(tmp_path, monkeypatch, caplog, sound_output):
    reload_settings(video_encoder="h264_nvenc")
    fake = FakeRun()
    monkeypatch.setattr(encoders.subprocess, "run", fake)  # detection: nvenc works
    assert encoders.detect().name == "h264_nvenc"
    real = FakeRun(fail={"h264_nvenc"})  # the real run: nvenc fails, libx264 writes the file
    monkeypatch.setattr(video.subprocess, "run", real)
    monkeypatch.setattr(video.shutil, "which", lambda name: None)  # no nice: argv starts with ffmpeg
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    dst = tmp_path / "videos" / "out.mp4"
    used = video.transcode_preview(src, dst, _info(), 1080)
    assert used == "libx264" and dst.exists() and dst.stat().st_size == 64
    assert [real.encoder_of(c) for c in real.calls] == ["h264_nvenc", "libx264"]
    assert real.calls[0][0] == "ffmpeg" and "+faststart" in real.calls[1]
    assert not list((tmp_path / "videos").glob(".*.tmp.mp4"))
    assert any("libx264 makes this copy; later copies skip it" in r.message for r in caplog.records)
    # a driver problem ("Cannot load") skips the encoder from now on: detection moves on without probing it again
    # (encoders and video share the subprocess module, so one fake sees both the probe and the run)
    monkeypatch.setattr(encoders.subprocess, "run", fake)
    assert encoders.detect().name == "libx264" and "h264_nvenc" in (encoders.detect().reason or "")
    assert video.transcode_preview(src, dst, _info(), 1080) == "libx264"
    # (the fake also sees nvidia-smi from _device_for, which has no -c:v: dropped here)
    assert [e for e in (fake.encoder_of(c) for c in fake.calls) if e] == ["h264_nvenc", "libx264", "libx264"]
    reload_settings()


def test_transcode_software_failure_raises(tmp_path, monkeypatch, sound_output):
    reload_settings(video_encoder="libx264")
    monkeypatch.setattr(encoders.subprocess, "run", FakeRun())
    monkeypatch.setattr(video.subprocess, "run", FakeRun(fail={"libx264"}))
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Cannot load driver"):
        video.transcode_preview(src, tmp_path / "out.mp4", _info(), 1080)
    assert not list(tmp_path.glob(".*.tmp.mp4"))
    reload_settings()


def test_transcode_command_shape(tmp_path, monkeypatch, sound_output):
    reload_settings(video_encoder="libx264", video_preview_crf=21, video_preview_maxrate_kbps=5000)
    fake = FakeRun()
    monkeypatch.setattr(encoders.subprocess, "run", fake)
    monkeypatch.setattr(video.subprocess, "run", fake)
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    video.transcode_preview(src, tmp_path / "out.mp4", _info(), 720)
    cmd = fake.calls[-1]
    assert cmd[cmd.index("-i") + 1] == str(src) and cmd[cmd.index("-crf") + 1] == "21"
    assert cmd[cmd.index("-maxrate") + 1] == "5000k" and "min(ih,720)" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-c:a") + 1] == "aac" and cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert cmd[-2:] == ["mp4", str(cmd[-1])] and cmd[-1].endswith(".tmp.mp4")
    reload_settings()


def test_one_odd_file_does_not_retire_the_encoder(tmp_path, monkeypatch, caplog, sound_output):
    """A per-file failure (not a driver problem) costs that copy only; two in a row retire the encoder,
    and a success in between starts the count again."""
    reload_settings(video_encoder="h264_nvenc")
    monkeypatch.setattr(encoders.subprocess, "run", FakeRun())
    assert encoders.detect().name == "h264_nvenc"
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    odd = FakeRun(fail={"h264_nvenc"}, stderr="[h264_nvenc @ 0x1] Invalid picture size 7681x4321\n")
    monkeypatch.setattr(video.subprocess, "run", odd)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    assert video.transcode_preview(src, tmp_path / "a.mp4", _info(), 1080) == "libx264"
    assert encoders.detected() is not None and encoders.detected().name == "h264_nvenc"  # still the encoder
    assert "h264_nvenc" not in encoders._failed and encoders._strikes["h264_nvenc"] == 1
    assert any("libx264 makes this copy" in r.message and "later copies" not in r.message for r in caplog.records)
    # a good run clears the strike
    monkeypatch.setattr(video.subprocess, "run", FakeRun())
    assert video.transcode_preview(src, tmp_path / "b.mp4", _info(), 1080) == "h264_nvenc"
    assert "h264_nvenc" not in encoders._strikes
    # two odd files in a row retire it
    monkeypatch.setattr(video.subprocess, "run", odd)
    assert video.transcode_preview(src, tmp_path / "c.mp4", _info(), 1080) == "libx264"
    assert "h264_nvenc" not in encoders._failed
    assert video.transcode_preview(src, tmp_path / "d.mp4", _info(), 1080) == "libx264"
    assert "h264_nvenc" in encoders._failed and encoders.detected() is None
    reload_settings()


def test_device_errors_are_recognised():
    assert encoders.is_device_error("Cannot load libnvidia-encode.so.1")
    assert encoders.is_device_error("No VA display found for device /dev/dri/renderD128")
    assert encoders.is_device_error("Error creating a MFX session: -9")
    assert encoders.is_device_error("transcode timed out after 400s")
    assert not encoders.is_device_error("Invalid picture size 7681x4321")
    assert not encoders.is_device_error("")


def test_nvenc_asks_for_idr_keyframes():
    _, args = encoders.build_video_args("h264_nvenc", 1080, 23, 6000)
    assert args[args.index("-forced-idr") + 1] == "1" and args.index("-forced-idr") < args.index("-profile:v")
    for name in ("h264_vaapi", "h264_qsv", "libx264"):
        assert "-forced-idr" not in encoders.build_video_args(name, 1080, 23, 6000)[1]


def test_a_broken_output_file_is_a_failure(tmp_path, monkeypatch):
    """exit 0 plus an unreadable file: refused, never moved into place; a hardware run falls back."""
    reload_settings(video_encoder="libx264")
    monkeypatch.setattr(encoders.subprocess, "run", FakeRun())
    monkeypatch.setattr(video.subprocess, "run", FakeRun())
    monkeypatch.setattr(video, "_output_problem", lambda path: "the copy has 0 video streams")
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="unusable file: the copy has 0 video streams"):
        video.transcode_preview(src, tmp_path / "out.mp4", _info(), 1080)
    assert not (tmp_path / "out.mp4").exists() and not list(tmp_path.glob(".*.tmp.mp4"))
    # a hardware encoder that writes a broken file: libx264 makes the copy instead
    encoders.redetect()
    reload_settings(video_encoder="h264_nvenc")
    assert encoders.detect().name == "h264_nvenc"
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        return "the copy has no duration" if calls["n"] == 1 else None

    monkeypatch.setattr(video, "_output_problem", flaky)
    assert video.transcode_preview(src, tmp_path / "out2.mp4", _info(), 1080) == "libx264"
    assert (tmp_path / "out2.mp4").exists() and calls["n"] == 2
    reload_settings()


@pytest.mark.skipif(__import__("shutil").which("ffprobe") is None, reason="ffprobe is not installed")
def test_output_probe_with_the_real_ffprobe(tmp_path):
    junk = tmp_path / "junk.mp4"; junk.write_bytes(b"\x00" * 4096)
    problem = video._output_problem(junk)
    assert problem is not None and "copy" in problem


def test_fallback_budget():
    assert video._fallback_budget(deadline=400.0, now=100.0, total=400.0) == 300.0  # what is left
    assert video._fallback_budget(deadline=400.0, now=390.0, total=400.0) == 120.0  # the floor
    assert video._fallback_budget(deadline=90.0, now=80.0, total=90.0) == 90.0      # a short job: its whole budget


def test_fallback_runs_with_the_remaining_time(tmp_path, monkeypatch):
    """A hardware run that hangs for 300 s of a 400 s budget leaves libx264 120 s (the floor), not 400 s
    more; the hang retires the encoder."""
    reload_settings(video_encoder="h264_nvenc")
    monkeypatch.setattr(encoders.subprocess, "run", FakeRun())
    assert encoders.detect().name == "h264_nvenc"
    clock = iter([0.0, 300.0, 300.0, 300.0, 300.0])
    monkeypatch.setattr(video.time, "monotonic", lambda: next(clock, 300.0))
    seen: list[tuple[str, float]] = []

    def fake_run(cmd, tmp, timeout):
        enc = cmd[cmd.index("-c:v") + 1]
        seen.append((enc, timeout))
        if enc == "h264_nvenc":
            raise video.TranscodeTimeout(f"transcode timed out after {timeout:.0f}s")
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp).write_bytes(b"\x00" * 64)

    monkeypatch.setattr(video, "_run_transcode", fake_run)
    src = tmp_path / "in.mp4"; src.write_bytes(b"x")
    assert video.transcode_preview(src, tmp_path / "out.mp4", _info(duration=100.0), 1080) == "libx264"
    assert seen == [("h264_nvenc", 400.0), ("libx264", 120.0)]
    assert "h264_nvenc" in encoders._failed  # a hang counts as a device problem
    reload_settings()


def test_new_defaults_change_what_streams_as_is():
    # a 1080p H.264 original under 8 Mbit/s now plays as it is; a 4K one or a HEVC one gets a copy
    assert video.stream_friendly("h264", ".mp4", 1920, 1080, 7_500_000)
    assert not video.stream_friendly("h264", ".mp4", 3840, 2160, 7_500_000)
    assert not video.stream_friendly("h264", ".mp4", 1920, 1080, 9_000_000)
    assert not video.stream_friendly("hevc", ".mp4", 1280, 720, 2_000_000)
    assert settings.video_preview_max_height == 1080 and settings.video_preview_crf == 23
    assert settings.video_preview_maxrate_kbps == 6000 and settings.video_stream_max_kbps == 8000


def test_system_video_status_and_routes(client, conn, monkeypatch):
    done = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", video_codec="hevc", bitrate=12_000_000,
                     width=1920, height=1080, preview_status="done", write_files=False)
    # a Live Photo's clip is a companion: hidden everywhere, so not counted and not requeued
    make_item(conn, kind="video", ext=".mov", mime="video/quicktime", video_codec="hevc", bitrate=12_000_000,
              width=1920, height=1080, preview_status="done", companion_of=done["id"], write_files=False)
    needs = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", video_codec="hevc", bitrate=12_000_000,
                      width=1920, height=1080, preview_status="none", write_files=False)
    bad = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", video_codec="hevc", bitrate=12_000_000,
                    width=1920, height=1080, preview_status="error", write_files=False)
    make_item(conn, kind="video", ext=".mp4", mime="video/mp4", video_codec="h264", bitrate=3_000_000,
              width=1280, height=720, preview_status="none", write_files=False)
    make_item(conn, kind="video", ext=".mp4", mime="video/mp4", video_codec="hevc", bitrate=12_000_000,
              width=1920, height=1080, preview_status="pending", write_files=False)
    jobs.enqueue(conn, "video_preview", {"item_id": 999_999, "version": 1})
    v = client.get("/api/system").json()["video"]
    assert v["encoder"] is None and v["checked_at"] is None
    assert v == {**v, "queued": 1, "running": 0, "failed": 1, "done": 1, "pending": 1}

    monkeypatch.setattr(encoders.subprocess, "run", FakeRun(fail={"h264_nvenc", "h264_vaapi"}))
    monkeypatch.setattr(encoders, "_device_for", lambda name: "Intel Quick Sync")
    v = client.post("/api/system/video/detect").json()
    assert v["encoder"] == "h264_qsv" and v["hw"] is True and v["device"] == "Intel Quick Sync" and v["checked_at"]
    assert client.get("/api/system").json()["video"]["encoder"] == "h264_qsv"

    r = client.post("/api/system/video/requeue", json={"scope": "failed"})
    assert r.json() == {"requeued": 1}
    assert conn.execute("SELECT preview_status FROM items WHERE id = ?", (bad["id"],)).fetchone()[0] == "pending"
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE type = 'video_preview' AND status = 'queued'").fetchone()[0] == 2
    # all: the rows without a copy that need one (the 'none' HEVC one and the 'pending' one without a
    # job); the finished copy and the stream-friendly original are left alone, and the row that
    # already has a queued job is not queued twice, so two new jobs
    r = client.post("/api/system/video/requeue", json={"scope": "all"})
    assert r.json() == {"requeued": 2}
    assert conn.execute("SELECT preview_status FROM items WHERE id = ?", (done["id"],)).fetchone()[0] == "done"
    assert conn.execute("SELECT preview_status FROM items WHERE id = ?", (needs["id"],)).fetchone()[0] == "pending"
    assert conn.execute("SELECT COUNT(*) FROM items WHERE kind = 'video' AND companion_of IS NULL AND preview_status = 'pending'").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE type = 'video_preview' AND status = 'queued'").fetchone()[0] == 4
    assert client.post("/api/system/video/requeue", json={"scope": "later"}).status_code == 422
