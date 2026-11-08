"""Video faces followed through the whole clip (section 10, 2026-11-08): the tracker links a face by motion
and look, keeps two faces seen together apart, joins a person who leaves and comes back, and knows a
person by their clear views. Pure tests with made-up embeddings; analyze_video runs on a fake engine."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import faces as faces_mod
from app.analysis.faces import FaceTrack, FaceTracker, VideoDetection, identity, video_plan, visible_spans
from app.config import settings

DIM = 512


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _noisy(base: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.standard_normal(DIM)
    n -= (n @ base) * base
    return _unit(base + sigma * _unit(n))


def _det(t: float, box, emb, det=0.9, size=200, clarity=23.0, sharpness=40.0) -> VideoDetection:
    return VideoDetection(time_s=round(t, 3), box=box, det_score=det, size_px=size, embedding=emb,
                          clarity=clarity, sharpness=sharpness)


def _finish(tracker: FaceTracker, min_count: int = 4, min_tracklet: int = 2) -> list[FaceTrack]:
    return tracker.finish(settings.face_good_min_size_px, settings.face_good_det_score, min_count, min_tracklet)


def test_motion_keeps_a_face_whose_look_alone_would_split_it():
    """Every view looks like the person at about 0.55 but like the other views at only about 0.3, below the
    0.5 a look-only link needs: a smoothly moving box keeps them one track; the same views jumping around
    the frame are only flickers."""
    rng = np.random.default_rng(1)
    who = _unit(rng.standard_normal(DIM))
    views = [_noisy(who, 1.53, rng) for _ in range(10)]
    assert max(float(views[i] @ views[i + 1]) for i in range(9)) < settings.video_track_threshold

    moving = FaceTracker(settings.video_track_threshold, 1.0)
    for k, e in enumerate(views):
        moving.add_frame(0.2 * k, [_det(0.2 * k, (0.2 + 0.01 * k, 0.3, 0.15, 0.2), e)])
    (track,) = _finish(moving)
    assert track.count == 10

    jumping = FaceTracker(settings.video_track_threshold, 1.0)
    for k, e in enumerate(views):
        x = 0.05 if k % 2 == 0 else 0.75
        jumping.add_frame(0.2 * k, [_det(0.2 * k, (x, 0.1 + 0.07 * (k % 5), 0.15, 0.2), e)])
    assert len(jumping.tracks) > 1
    assert all(t.count < 10 for t in _finish(jumping, min_tracklet=1))


def test_two_look_alikes_in_one_frame_stay_two_people():
    rng = np.random.default_rng(2)
    who = _unit(rng.standard_normal(DIM))
    tracker = FaceTracker(settings.video_track_threshold, 1.0)
    for k in range(8):
        t = 0.2 * k
        tracker.add_frame(t, [_det(t, (0.1, 0.3, 0.15, 0.2), _noisy(who, 0.2, rng)),
                              _det(t, (0.7, 0.3, 0.15, 0.2), _noisy(who, 0.2, rng))])
    tracks = _finish(tracker)
    assert len(tracks) == 2
    assert sorted(t.count for t in tracks) == [8, 8]
    assert all(len({d.box[0] for d in t.dets}) == 1 for t in tracks)


def test_a_person_who_leaves_and_comes_back_is_one_track_with_two_spans():
    rng = np.random.default_rng(3)
    who = _unit(rng.standard_normal(DIM))
    tracker = FaceTracker(settings.video_track_threshold, 1.0)
    samples = [round(0.2 * k, 3) for k in range(61)]  # 0 .. 12 s
    for t in samples:
        if t <= 2.0:
            tracker.add_frame(t, [_det(t, (0.1, 0.3, 0.15, 0.2), _noisy(who, 0.3, rng))])
        elif t >= 10.0:
            tracker.add_frame(t, [_det(t, (0.7, 0.4, 0.15, 0.2), _noisy(who, 0.3, rng))])
        else:
            tracker.add_frame(t, [])
    assert len(tracker.tracks) == 2  # two tracklets ...
    (track,) = _finish(tracker)  # ... one person
    rec = faces_mod.track_to_record(track, samples, 12.0, 1.0)
    assert rec.spans == [(0.0, 2.1), (9.9, 12.0)]


def test_a_crossing_keeps_each_face_on_its_own_track():
    rng = np.random.default_rng(4)
    a, b = _unit(rng.standard_normal(DIM)), _unit(rng.standard_normal(DIM))
    tracker = FaceTracker(settings.video_track_threshold, 1.0)
    for k in range(11):
        t = 0.2 * k
        xa, xb = 0.1 + 0.06 * k, 0.7 - 0.06 * k  # they pass each other at k = 5
        tracker.add_frame(t, [_det(t, (xa, 0.3, 0.15, 0.2), _noisy(a, 0.4, rng)),
                              _det(t, (xb, 0.32, 0.15, 0.2), _noisy(b, 0.4, rng))])
    tracks = _finish(tracker)
    assert len(tracks) == 2
    for tr in tracks:
        xs = [d.box[0] for d in tr.dets]
        assert xs == sorted(xs) or xs == sorted(xs, reverse=True)  # each track keeps one direction


def test_identity_counts_views_by_how_clear_they_are():
    rng = np.random.default_rng(5)
    who, other = _unit(rng.standard_normal(DIM)), _unit(rng.standard_normal(DIM))
    dets = [_det(0.5 * k, (0.1, 0.1, 0.2, 0.2), _noisy(who, 0.3, rng)) for k in range(3)]
    # Sharper-looking by detector score and size, but behind a helmet: the face model barely sees a face.
    dets += [_det(2 + 0.5 * k, (0.1, 0.1, 0.2, 0.2), _noisy(other, 0.1, rng), det=0.95, size=400, clarity=9.0)
             for k in range(3)]
    e = identity(dets)
    assert float(e @ who) > 0.9 and float(e @ other) < 0.2
    # Only unclear views: still an identity (by quality), never an error.
    e2 = identity(dets[3:])
    assert abs(float(np.linalg.norm(e2)) - 1.0) < 1e-4


def test_the_best_view_is_the_clearest_not_the_largest():
    rng = np.random.default_rng(6)
    who = _unit(rng.standard_normal(DIM))
    track = FaceTrack()
    assert track.add(_det(0.0, (0.1, 0.1, 0.3, 0.3), _noisy(who, 0.2, rng), det=0.95, size=500, sharpness=6.0))
    assert track.add(_det(0.2, (0.1, 0.1, 0.2, 0.2), _noisy(who, 0.2, rng), det=0.85, size=150))
    assert track.best is not None and track.best.time_s == 0.2


def test_flickers_go_on_dense_sampling_but_a_good_face_stays_on_sparse_sampling():
    rng = np.random.default_rng(7)
    tracker = FaceTracker(settings.video_track_threshold, 1.0)
    tracker.add_frame(0.0, [_det(0.0, (0.1, 0.1, 0.2, 0.2), _unit(rng.standard_normal(DIM)))])
    assert _finish(tracker, min_count=4, min_tracklet=2) == []
    assert len(_finish(tracker, min_count=2, min_tracklet=1)) == 1  # one good face, as at one frame a second


@pytest.mark.parametrize("duration", [0.0, 2.0, 10.0, 151.0, 299.0, 535.0, 601.0, 3600.0])
def test_the_frames_for_siglip_keep_the_plain_cadence(duration):
    plan = video_plan(duration)
    plain = settings.video_sample_interval_s
    assert plan.keep_max == settings.video_max_samples
    assert plan.keep_step == pytest.approx(max(plain, duration / settings.video_max_samples) if duration else plain)
    if duration > 600:  # seeking clips keep one ffmpeg run per frame for as few frames as before
        assert (plan.interval, plan.max_frames) == (plain, settings.video_max_samples)
    else:
        assert plan.interval == settings.video_face_interval_s and plan.max_frames == settings.video_face_max_frames
    assert plan.lost_s >= 1.0 and plan.lost_s >= 2.5 * plan.step - 1e-9
    assert plan.min_tracklet == (2 if plan.step < 0.3 else 1)


def test_spans_close_short_gaps():
    samples = [round(0.2 * k, 3) for k in range(11)]
    seen = [0.0, 0.2, 0.4, 1.0, 1.2, 2.0]
    assert visible_spans(seen, samples, 2.0) == [(0.0, 0.5), (0.9, 1.3), (1.9, 2.0)]
    assert visible_spans(seen, samples, 2.0, bridge_s=1.0) == [(0.0, 2.0)]


class _FakeEngine:
    """Detects one face per frame, moving right, as app.ml.faces.FaceEngine.analyze would."""

    def __init__(self, who: np.ndarray, rng: np.random.Generator):
        self.who, self.rng, self.calls = who, rng, 0

    def analyze(self, bgr, sizes, threshold, min_size):
        from app.ml.faces import DetectedFace

        h, w = bgr.shape[:2]
        k = self.calls
        self.calls += 1
        x1 = 0.2 * w + k
        box = np.array([x1, 0.3 * h, x1 + 0.2 * w, 0.3 * h + 0.2 * w], dtype=np.float32)
        kps = np.zeros((5, 2), dtype=np.float32)
        f = DetectedFace(bbox=box, kps=kps, det_score=0.9, embedding=_noisy(self.who, 0.3, self.rng), norm=23.0,
                         sharpness=40.0)
        return [f] if f.size_px >= min_size else []


def test_analyze_video_follows_the_face_through_every_frame_and_feeds_siglip_once_a_second(monkeypatch):
    rng = np.random.default_rng(8)
    who = _unit(rng.standard_normal(DIM))
    engine = _FakeEngine(who, rng)
    asked = {}

    def fake_frames(path, info, interval, max_frames, max_side):
        asked.update(interval=interval, max_frames=max_frames)
        for k in range(31):  # 0 .. 6 s
            yield round(0.2 * k, 3), np.zeros((72, 128, 3), dtype=np.uint8)

    monkeypatch.setattr("app.media.video.iter_frames", fake_frames)
    monkeypatch.setattr("app.ml.faces.get_engine", lambda: engine)
    seen: list[float] = []

    class Info:
        duration_s, width, height = 6.0, 1280, 720

    records = faces_mod.analyze_video("clip.mp4", Info(), on_frame=lambda t, bgr: seen.append(t))
    assert asked == {"interval": settings.video_face_interval_s, "max_frames": settings.video_face_max_frames}
    assert engine.calls == 31  # every sampled frame was looked at
    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # SigLIP, NSFW and pets keep about one a second
    (rec,) = records
    assert rec.spans == [(0.0, 6.0)]
    assert rec.clarity == 23.0 and rec.sharpness == 40.0
    assert float(np.asarray(rec.embedding) @ who) > 0.9
