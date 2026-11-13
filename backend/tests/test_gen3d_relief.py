"""The relief of the face from a depth model (gen3d/relief.py, docs/ARCHITECTURE.md section 36), with made-up
heads and stand-in models: no model file, no PyTorch, no photo.

The made-up head is a sheet over the picture: its depth is a function of the pixel, so the truth is known at every
vertex. The "mesh" is a smooth blob half as deep as the true head and without its nose and brow; the stand-in depth
model answers with the true depth, scaled, shifted and turned round (larger means nearer), and the stand-in normal
model with the true surface's tilt. A second sheet behind the first is what the camera does not see."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gen3d import relief  # noqa: E402

SIDE = 512                                   # the square the generator was given
DIST = 1.9
FOCAL = 0.5 * SIDE / math.tan(math.radians(20.0))
SRC = 768                                    # the look's picture: 1.5 of its pixels to one of the square's
FRAMING = 2.6
CAM = np.array([0.0, 0.0, DIST])
CU, CV = 0.5 * SIDE, (0.5 - relief.FACE_ABOVE) * SIDE     # the face centre, where the app puts it
RU, RV = 0.26 * SIDE, 0.30 * SIDE
DEEP = 0.08                                  # how far the true head's middle stands in front of its rim
NOSE, BROW = 0.02, 0.008
READ_SHIFT, READ_SCALE = 0.48, 0.03          # the stand-in reading: READ_SHIFT - READ_SCALE * distance
TRUE_FACTOR = -1.0 / READ_SCALE


def to_source(xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64) * (SRC / SIDE)


def project(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = DIST - v[:, 2]
    return np.stack([SIDE / 2.0 + FOCAL * v[:, 0] / depth, SIDE / 2.0 - FOCAL * v[:, 1] / depth], axis=1), depth


def unproject(u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.stack([(u - SIDE / 2.0) * depth / FOCAL, -(v - SIDE / 2.0) * depth / FOCAL, DIST - depth], axis=-1)


def rim(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return ((u - CU) / RU) ** 2 + ((v - CV) / RV) ** 2


def blob_depth(u: np.ndarray, v: np.ndarray, deep: float) -> np.ndarray:
    return DIST - deep * np.clip(1.0 - rim(u, v), 0.0, None)


def true_depth(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The detailed head: the blob at full depth with a nose and a brow; a wall 0.3 behind where the head is not."""
    nose = NOSE * np.exp(-((u - CU) / 18.0) ** 2 - ((v - (CV + 20.0)) / 30.0) ** 2)
    brow = BROW * np.exp(-((v - (CV - 45.0)) / 9.0) ** 2) * np.exp(-((u - CU) / (0.6 * RU)) ** 2)
    return np.where(rim(u, v) < 1.0, blob_depth(u, v, DEEP) - nose - brow, DIST + 0.3)


def sheet(depth_of, inside: float, step: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A sheet of triangles over the pixels whose rim value is under `inside`: (vertices, faces, u, v)."""
    us = np.arange(CU - RU, CU + RU + step, step)
    vs = np.arange(CV - RV, CV + RV + step, step)
    gu, gv = np.meshgrid(us, vs)
    ok = rim(gu, gv) < inside
    index = -np.ones(gu.shape, dtype=np.int64)
    index[ok] = np.arange(int(ok.sum()))
    u, v = gu[ok], gv[ok]
    quad = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    a, b, c, d = index[:-1, :-1][quad], index[:-1, 1:][quad], index[1:, :-1][quad], index[1:, 1:][quad]
    faces = np.concatenate([np.stack([a, c, b], axis=1), np.stack([b, c, d], axis=1)])
    return unproject(u, v, depth_of(u, v)), faces, u, v


class Head:
    """The blob the generator gave (half as deep as the truth), a hidden sheet behind it, and its picture."""

    def __init__(self) -> None:
        front, f_front, self.u, self.v = sheet(lambda u, v: blob_depth(u, v, DEEP / 2.0), 0.999)
        back, f_back, _, _ = sheet(lambda u, v: blob_depth(u, v, DEEP / 2.0) + 0.4, 0.5)
        self.n_front = front.shape[0]
        self.vertices = np.concatenate([front, back])
        self.faces = np.concatenate([f_front, f_back[:, ::-1] + self.n_front])
        self.source = np.full((SRC, SRC, 3), 128, dtype=np.uint8)
        cx, cy = SRC // 2, round((0.5 - relief.FACE_ABOVE) * SRC)
        self.source[cy - 6:cy + 6, cx - 6:cx + 6] = (255, 0, 0)                 # red on the face centre
        self.source[cy - 6:cy + 6, cx + 94:cx + 106] = (0, 0, 255)              # blue 100 pixels to its right
        yy, xx = np.mgrid[0:SRC, 0:SRC]
        self.alpha = np.where(rim((xx + 0.5) * SIDE / SRC, (yy + 0.5) * SIDE / SRC) < 1.0, 255, 0).astype(np.uint8)

    def models(self, inside=None):
        """The stand-in depth and normal models for the crop the stage will cut."""
        x0, y0, m = relief.face_crop(SRC, SRC, FRAMING, inside)
        j, i = np.mgrid[0:512, 0:512]
        u = (x0 + (i + 0.5) * m / 512.0) * SIDE / SRC
        v = (y0 + (j + 0.5) * m / 512.0) * SIDE / SRC
        depth = true_depth(u, v)
        reading = (READ_SHIFT - READ_SCALE * depth).astype(np.float32)
        p = unproject(u, v, depth)
        normal = -np.cross(np.gradient(p, axis=1), np.gradient(p, axis=0))      # right x down points away: turn it
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        normals = np.ascontiguousarray(normal.transpose(2, 0, 1)).astype(np.float32)
        return (lambda rgb: reading), (lambda rgb: normals)

    def refine(self, inside=None, vertices=None, **changes):
        run_depth, run_normals = self.models(inside)
        args = dict(project=project, camera_position=CAM, focal=FOCAL, side=SIDE, source_rgb=self.source,
                    alpha=self.alpha, square_to_source=to_source, framing=FRAMING, inside=inside,
                    run_depth=run_depth, run_normals=run_normals)
        args.update(changes)
        return relief.refine(self.vertices if vertices is None else vertices, self.faces, **args)

    def error(self, vertices: np.ndarray, keep: np.ndarray | None = None) -> float:
        """How far the front sheet's depth is from the true head's, root mean square."""
        e = (DIST - vertices[:self.n_front, 2]) - true_depth(self.u, self.v)
        return float(np.sqrt((e[keep] ** 2).mean() if keep is not None else (e ** 2).mean()))


@pytest.fixture(scope="module")
def head() -> Head:
    return Head()


@pytest.fixture(scope="module")
def refined(head: Head):
    before = head.vertices.copy()
    out, info = head.refine()
    assert np.array_equal(head.vertices, before)  # the input is not written to
    return out, info


# ------------------------------------------------------------------ the stage
def test_the_seen_front_comes_closer_to_the_true_head(head: Head, refined):
    """The blob is half as deep as the true head. The edge of the region is held to the blob (so the head keeps its
    outline), which keeps part of that flatness: the front comes about half of the way overall, two thirds in the
    middle of the face."""
    out, info = refined
    assert info["relief"] is True and info["note"] == "" and out is not head.vertices
    middle = rim(head.u, head.v) < 0.8     # away from the rim, where the sheet is held to what it was
    before, after = head.error(head.vertices, middle), head.error(out, middle)
    assert before > 0.02                   # the blob is far off: half the depth, and no nose
    assert after < 0.55 * before           # measured: 0.46 of it
    assert head.error(out) < 0.55 * head.error(head.vertices)                   # measured: 0.46 over the whole front
    centre = int(np.argmin((head.u - CU) ** 2 + (head.v - (CV + 20.0)) ** 2))
    truth = float(true_depth(head.u[centre], head.v[centre]))
    gap, left = (DIST - head.vertices[centre, 2]) - truth, (DIST - out[centre, 2]) - truth
    assert gap > 0.05 and 0.0 < left < 0.4 * gap                                # measured: 0.32 of the gap is left


def test_the_nose_stands_out_nearly_as_far_as_it_truly_does(head: Head, refined):
    """What the stage is for: the relief. The nose's tip against a ring around it."""
    out, _ = refined
    away = np.hypot(head.u - CU, head.v - (CV + 20.0))
    tip, ring = int(np.argmin(away)), (away > 55) & (away < 65)

    def nose(depth: np.ndarray) -> float:
        return float(depth[ring].mean() - depth[tip])

    true = nose(true_depth(head.u, head.v))
    assert nose(DIST - head.vertices[:head.n_front, 2]) < 0.25 * true           # the blob: only its own roundness
    assert 0.8 * true < nose(DIST - out[:head.n_front, 2]) < 1.05 * true        # measured: 0.88 of the true nose


def test_vertices_move_along_their_rays_and_keep_their_place_in_the_picture(head: Head, refined):
    out, _ = refined
    uv_before, _ = project(head.vertices)
    uv_after, _ = project(out)
    assert float(np.abs(uv_after - uv_before).max()) < 1e-6


def test_what_the_camera_does_not_see_stays_where_it_was(head: Head, refined):
    out, info = refined
    assert np.array_equal(out[head.n_front:], head.vertices[head.n_front:])     # the hidden sheet, bit for bit
    assert 0 < info["moved"] <= head.n_front


def test_the_edge_of_the_region_stays_where_it_was(head: Head, refined):
    import cv2

    out, _ = refined
    move = np.abs(out[:head.n_front, 2] - head.vertices[:head.n_front, 2])
    edge = rim(head.u, head.v) > 0.97      # the outermost few pixels of the head
    assert float(move.max()) > 0.04        # the middle moved by about half the head's depth
    assert float(move[edge].max()) < 0.1 * float(move.max())
    # the region ends 4 pixels inside the cut-out: what lies within 2 pixels of the cut-out's edge does not move
    # (measured: under 1.2e-6 of a scene unit; 4e-5 when the region runs to the edge)
    px = np.arange(SIDE) + 0.5
    inside_by = cv2.distanceTransform((rim(*np.meshgrid(px, px)) < 1.0).astype(np.uint8), cv2.DIST_L2, 5)
    outermost = inside_by[head.v.astype(int), head.u.astype(int)] < 2.0
    assert int(outermost.sum()) > 200 and float(move[outermost].max()) < 5e-6


def test_how_deep_comes_from_the_normals_not_from_the_mesh(head: Head, refined):
    """The mesh is half as deep as the truth, so fitting the reading to the mesh's depth gives half the factor.
    The normals give the true one."""
    _, info = refined
    assert abs(info["factor"] / TRUE_FACTOR - 1.0) < 0.10
    assert abs(info["fit_factor"] / TRUE_FACTOR) < 0.7
    assert info["correlation"] > 0.9


def test_the_normals_are_not_followed_beyond_what_the_mesh_allows(head: Head):
    """A normal model that reads a relief five times too deep is held to 2.2 times the plain fit; one whose tilts
    agree with the reading only faintly (the true tilts under five times as much noise) is not followed at all."""
    run_depth, run_normals = head.models()
    true = run_normals(None)
    tilted = true.copy()
    tilted[:2] *= 5.0
    tilted /= np.linalg.norm(tilted, axis=0, keepdims=True)
    _, info = head.refine(run_normals=lambda rgb: tilted)
    assert info["relief"] and info["factor"] == pytest.approx(relief.FACTOR_HIGH * info["fit_factor"], rel=1e-3)
    tilt = true[:2] / true[2] + np.random.default_rng(5).standard_normal((2, 512, 512))
    faint = np.concatenate([tilt, np.ones((1, 512, 512))]).astype(np.float32)
    faint /= np.linalg.norm(faint, axis=0, keepdims=True)
    _, info = head.refine(run_normals=lambda rgb: faint)
    assert info["relief"] and info["factor"] == info["fit_factor"]
    assert 0.05 < info["correlation"] < relief.MIN_CORRELATION             # measured: 0.14, and of the right sign


def test_only_the_part_of_the_picture_that_is_the_photo_is_read_and_moved(head: Head):
    """The look's picture below a line is filler (the photo's edge repeated). The models get a crop cut to the
    photo's part, and no vertex whose place in the picture is below the line moves."""
    line = round(CV * SRC / SIDE)                       # through the middle of the face, in picture pixels: 323
    inside = (0, 0, SRC, line)
    assert relief.face_crop(SRC, SRC, FRAMING, inside) == (222, 0, 323)
    filler = head.source.copy()
    filler[line:] = (0, 255, 0)                         # green where the picture is not the photo
    given: list[np.ndarray] = []
    run_depth, run_normals = head.models(inside)
    out, info = head.refine(inside, source_rgb=filler, run_depth=lambda rgb: (given.append(rgb), run_depth(rgb))[1])
    assert info["relief"] is True and len(given) == 1 and given[0].shape == (512, 512, 3)
    assert not ((given[0][..., 1] > 200) & (given[0][..., 0] < 100)).any()      # no green reached the models
    below = to_source(np.stack([head.u, head.v], axis=1))[:, 1] >= line
    front_before, front_after = head.vertices[:head.n_front], out[:head.n_front]
    assert below.any() and np.array_equal(front_after[below], front_before[below])
    assert float(np.abs(front_after[~below] - front_before[~below]).max()) > 0.01
    # without the line the same vertices do move
    full, _ = head.refine()
    assert float(np.abs(full[:head.n_front][below] - front_before[below]).max()) > 0.01
    # and the head stays in one piece: towards the line the move dies away (measured: 0.006 of the largest)
    move = np.abs(front_after[:, 2] - front_before[:, 2])
    last = ~below & (to_source(np.stack([head.u, head.v], axis=1))[:, 1] > line - 3.0 * SRC / SIDE)
    assert int(last.sum()) > 100 and float(move[last].max()) < 0.02 * float(move.max())


def test_the_region_ends_where_the_photo_ends_whatever_the_crop(head: Head, monkeypatch):
    """Should the crop ever reach past the photo's part, the region still ends at the line and is held there."""
    line = round(CV * SRC / SIDE)
    whole = relief.face_crop(SRC, SRC, FRAMING)
    monkeypatch.setattr(relief, "face_crop", lambda *args, **more: whole)
    out, info = head.refine((0, 0, SRC, line))
    assert info["relief"] is True
    y = to_source(np.stack([head.u, head.v], axis=1))[:, 1]
    move = np.abs(out[:head.n_front, 2] - head.vertices[:head.n_front, 2])
    last = (y < line) & (y > line - 3.0 * SRC / SIDE)
    assert float(move.max()) > 0.01 and not move[y >= line].any()
    assert int(last.sum()) > 100 and float(move[last].max()) < 0.02 * float(move.max())


@pytest.mark.parametrize("case", ["no models", "turned off", "nothing seen", "runner fails", "wrong shape",
                                  "not a number", "no match", "does not settle", "face too small"])
def test_whatever_goes_wrong_the_head_comes_back_as_it_was(head: Head, monkeypatch, tmp_path, case):
    run_depth, run_normals = head.models()
    changes: dict = {}
    words = ""
    if case == "no models":
        changes, words = dict(run_depth=None, run_normals=None, options={"weights_dir": str(tmp_path)}), "not installed"
    elif case == "turned off":
        changes, words = dict(options={"relief": 0}), "turned off"
    elif case == "nothing seen":
        def never(rgb):
            raise AssertionError("the models are not run when there is nothing to read them for")
        changes, words = dict(alpha=np.zeros_like(head.alpha), run_depth=never, run_normals=never), "too little"
    elif case == "runner fails":
        def broken(rgb):
            raise RuntimeError("no memory")
        changes, words = dict(run_normals=broken), "RuntimeError: no memory"
    elif case == "wrong shape":
        changes, words = dict(run_depth=lambda rgb: np.zeros((256, 256), np.float32)), "another shape"
    elif case == "not a number":
        changes, words = dict(run_depth=lambda rgb: np.full((512, 512), np.nan, np.float32)), "not a number"
    elif case == "no match":
        noise = np.random.default_rng(1).standard_normal((512, 512)).astype(np.float32)
        changes, words = dict(run_depth=lambda rgb: noise), "does not match"
    elif case == "does not settle":
        real = relief.solve
        monkeypatch.setattr(relief, "solve", lambda *a, **k: real(*a, **{**k, "max_rounds": 3}))
        words = "did not settle in 3 rounds"
    elif case == "face too small":
        changes, words = dict(inside=(0, 0, 40, 40)), "too small"
    out, info = head.refine(**changes)
    assert out is head.vertices and info["relief"] is False and words in info["note"] and info["moved"] == 0


def test_files_that_are_not_models_leave_the_head_as_it_was(head: Head, tmp_path):
    """The real onnxruntime on two files that are there but are no models (a download cut short)."""
    pytest.importorskip("onnxruntime")
    for path in relief.model_paths(tmp_path):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"not a model")
    assert relief.available(tmp_path)
    out, info = head.refine(run_depth=None, run_normals=None, options={"weights_dir": str(tmp_path)})
    assert out is head.vertices and info["relief"] is False and info["moved"] == 0
    assert info["note"] and "not installed" not in info["note"]


@pytest.mark.parametrize("kind", [Exception, BaseException])
def test_a_stop_request_is_not_swallowed(head: Head, kind):
    """The job's stop request (gen3d.__main__.Stopped, raised from a signal handler wherever the job is) passes
    through, whether it is an Exception or not."""
    Stopped = type("Stopped", (kind,), {})

    def stop(rgb):
        raise Stopped()

    with pytest.raises(Stopped):
        head.refine(run_depth=stop)


def test_the_mesh_may_stand_in_another_frame_than_the_scene(head: Head, refined):
    """A fitted camera draws the mesh point p as if it stood at scale * p + shift. The same head given in such a
    frame moves to the same places: the limits are sizes of the scene, not of the frame."""
    out, info = refined
    scale, shift = 2.5, np.array([0.05, -0.12, 0.3])
    got, got_info = head.refine(vertices=(head.vertices - shift) / scale, project=lambda p: project(scale * p + shift),
                                camera_position=(CAM - shift) / scale)
    assert got_info == info
    assert float(np.abs((scale * got + shift) - out).max()) < 1e-9


def test_no_vertex_moves_further_than_the_limit(head: Head, monkeypatch):
    """The blob's middle is 0.04 and more from the truth; with the limit at 0.02 of the head's size it moves that."""
    monkeypatch.setattr(relief, "LIMIT", 0.02)
    out, info = head.refine()
    uv, depth = project(head.vertices)
    moved = np.abs(project(out)[1] - depth)
    assert info["relief"] is True
    assert float(moved.max()) == pytest.approx(0.02 * relief.head_size(uv, depth, FOCAL, SIDE), rel=1e-4)


def test_the_models_are_given_the_face_the_right_way_round(head: Head):
    """The crop: 1.9 face sizes around the face centre, as RGB, at 512."""
    given: list[np.ndarray] = []
    run_depth, run_normals = head.models()
    head.refine(run_depth=lambda rgb: (given.append(rgb), run_depth(rgb))[1],
                run_normals=lambda rgb: (given.append(rgb), run_normals(rgb))[1])
    assert len(given) == 2 and given[0].dtype == np.uint8 and np.array_equal(given[0], given[1])
    crop = relief.face_crop(SRC, SRC, FRAMING)[2]                               # 561 pixels of the picture
    assert given[0][256, 256].tolist() == [255, 0, 0]                           # red in the middle
    assert given[0][256, 256 + round(100 * 512 / crop)].tolist() == [0, 0, 255]  # blue to its right
    assert given[0][100, 100].tolist() == [128, 128, 128]


def test_a_depth_buffer_of_the_callers_is_used(head: Head, refined):
    out, info = refined
    calls: list[tuple] = []

    def own(uv, depth, faces, h, w):
        calls.append((h, w))
        return relief.raster(uv, depth, faces, h, w)

    got, got_info = head.refine(raster=own)
    assert calls == [(relief.GRID, relief.GRID)] and got_info == info and np.array_equal(got, out)


# ------------------------------------------------------------------ the pieces
def test_the_head_size_is_measured_in_the_scene():
    """From where the corners fall in the picture and how far they are, not from the numbers they were given as."""
    def box(wide: float, tall: float, deep: float) -> float:
        corners = np.array([[x, y, z] for x in (-0.1, wide - 0.1) for y in (-0.3, tall - 0.3)
                            for z in (-0.2, deep - 0.2)])
        return relief.head_size(*project(corners), FOCAL, SIDE)

    assert box(0.3, 0.5, 0.35) == pytest.approx(0.5)
    assert box(0.6, 0.5, 0.35) == pytest.approx(0.6)
    assert box(0.3, 0.5, 0.7) == pytest.approx(0.7)
    assert relief.head_size(np.zeros((0, 2)), np.zeros(0), FOCAL, SIDE) == 0.0
    assert relief.head_size(np.full((3, 2), np.nan), np.ones(3), FOCAL, SIDE) == 0.0


def _dense(target, base, omega, pin, steep):
    """The same system built pixel by pixel and solved directly."""
    h, w = target.shape
    cells = [(y, x) for y in range(h) for x in range(w) if omega[y, x]]
    at = {c: n for n, c in enumerate(cells)}
    a = np.zeros((len(cells), len(cells)))
    b = np.zeros(len(cells))
    for (y, x), n in at.items():
        a[n, n] += pin[y, x]
        b[n] += pin[y, x] * base[y, x]
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            other = (y + dy, x + dx)
            if other not in at:
                continue
            slope = target[other] - target[y, x]
            if steep > 0 and abs(slope) > steep:
                slope = base[other] - base[y, x]
            a[n, n] += 1.0
            a[n, at[other]] -= 1.0
            b[n] -= slope
    out = base.astype(np.float64).copy()
    solved = np.linalg.solve(a, b)
    for c, n in at.items():
        out[c] = solved[n]
    return out


def _small_case():
    rng = np.random.default_rng(7)
    h, w = 22, 18
    yy, xx = np.mgrid[0:h, 0:w]
    omega = (((xx - 8.5) / 8.0) ** 2 + ((yy - 10.5) / 10.0) ** 2) < 1.0
    omega &= ~((np.abs(xx - 11) <= 1) & (np.abs(yy - 6) <= 1))          # a hole
    base = 2.0 + 0.002 * (xx - 8.5) ** 2 + 0.001 * (yy - 10.5) ** 2
    bump = 0.05 * np.exp(-(((xx - 8.5) / 3.0) ** 2 + ((yy - 12.0) / 3.0) ** 2))
    target = base + 0.7 - bump + 0.002 * rng.standard_normal((h, w))
    target[15:, :] += 0.5                                               # a jump across the lower part
    pin = rng.uniform(0.01, 2.0, size=(h, w))
    return target, base, omega, pin


@pytest.mark.parametrize("steep", [0.0, 0.1])
def test_the_solver_agrees_with_a_direct_solve(steep):
    target, base, omega, pin = _small_case()
    want = _dense(target, base, omega, pin, steep)
    got, rounds, converged = relief.solve(target, base, omega, pin, steep=steep, tol=1e-13)
    assert converged and 0 < rounds < 400
    assert float(np.abs(got - want).max()) < 1e-9
    assert np.array_equal(got[~omega], base[~omega])


def test_a_jump_in_the_reading_takes_the_base_slope():
    target, base, omega, pin = _small_case()
    plain, _, _ = relief.solve(target, base, omega, pin, steep=0.0, tol=1e-13)
    held, _, _ = relief.solve(target, base, omega, pin, steep=0.1, tol=1e-13)
    across = (held[15, 4:14] - held[14, 4:14]), (plain[15, 4:14] - plain[14, 4:14])
    assert float(np.abs(across[0]).max()) < 0.1 and float(across[1].min()) > 0.2      # the 0.5 step is not built


def test_the_solver_gives_up_in_the_open():
    target, base, omega, pin = _small_case()
    _, rounds, converged = relief.solve(target, base, omega, pin, tol=1e-13, max_rounds=2)
    assert rounds == 2 and not converged
    # a pixel of the region that nothing holds and nothing joins keeps the base
    alone = np.zeros_like(omega)
    alone[3, 3] = True
    got, rounds, converged = relief.solve(target, base, alone, np.zeros_like(pin))
    assert converged and rounds == 0 and got[3, 3] == base[3, 3]


def test_the_rasteriser():
    def tri(*points):
        return np.array(points, dtype=np.float64)

    uv = np.concatenate([tri((1, 1), (9, 1), (1, 9)),            # near, small
                         tri((0, 0), (12, 0), (0, 12)),          # behind it, larger than the batch box of 10
                         tri((20, 20), (26, 20), (20, 26)),      # one corner behind the camera: left out
                         tri((30, 2), (38, 2), (30, 10))])       # depth that changes across it
    depth = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, -0.5, 1.0, 1.0, 3.0, 1.0])
    faces = np.arange(12).reshape(4, 3)
    z = relief.raster(uv, depth, faces, 32, 40)
    assert z.shape == (32, 40) and z.dtype == np.float32
    assert z[2, 2] == 1.0 and z[1, 1] == 1.0          # pixel centres (2.5, 2.5) and (1.5, 1.5): the near one wins
    assert z[8, 2] == 2.0 and z[1, 9] == 2.0          # only the far one covers these
    assert np.isinf(z[11, 11]) and np.isinf(z[18:28, 18:28]).all()
    # the plane of the last triangle: depth 1 at x = 30, 3 at x = 38
    assert z[3, 32] == pytest.approx(1.0 + 2.0 * (32.5 - 30.0) / 8.0)
    # pixel centres inside: 36 in the near triangle (8 + 7 + ... + 1), 78 in the far one (12 + 11 + ... + 1), which
    # holds the near one; 36 in the last
    assert int((z == 1.0).sum()) == 36 and int((z == 2.0).sum()) == 78 - 36
    assert int(np.isfinite(z).sum()) == 78 + 36


def test_the_face_crop():
    assert relief.face_crop(1024, 1024, 2.6) == (138, 56, 748)                    # 1.9 / 2.6 of the side, face centred
    x0, y0, m = relief.face_crop(1024, 1024, 2.6)
    assert (x0 + m / 2.0, y0 + m / 2.0) == pytest.approx((512.0, (0.5 - 0.08) * 1024.0), abs=0.5)
    assert relief.face_crop(1024, 1024, 1.8) == (0, 0, 1024)                      # wider than the picture: all of it
    assert relief.face_crop(1024, 1024, 2.6, (0, 0, 1024, 700)) == (162, 0, 700)  # made smaller, still on the face
    assert relief.face_crop(1024, 1024, 2.6, (200, 0, 1024, 1024)) == (200, 56, 748)
    assert relief.face_crop(1024, 1024, 2.6, (0, 0, 0, 0)) == (0, 0, 0)
    assert relief.photo_part(100, 200, None) == (0.0, 0.0, 200.0, 100.0)
    assert relief.photo_part(100, 200, (-5, 10, 400, 90)) == (0.0, 10.0, 200.0, 90.0)


def test_which_vertices_the_camera_sees():
    near = np.full((6, 8), 1e9, dtype=np.float32)        # nothing is drawn, but
    near[2, 5] = 2.0                                     # a surface at depth 2 around the pixel x 5, y 2
    pix = np.array([[5.5, 2.5], [5.9, 2.1], [5.5, 2.5], [2.5, 5.5], [0.2, 0.2], [-3.0, 2.5], [99.0, 99.0]])
    depth = np.array([2.0, 2.04, 2.06, 2.06, 5.0, 5.0, 5.0])
    # on it; 0.04 behind it with a slack of 0.05; too far behind; the last four have nothing in front of them
    assert relief.visible(pix, depth, near, 0.05).tolist() == [True, True, False, True, True, True, True]
    near[:, 0], near[5, 7] = 1.0, 1.0                    # points off the grid are judged by its edge
    assert relief.visible(pix, depth, near, 0.05).tolist() == [True, True, False, True, False, False, False]


def test_the_shift_is_blurred_a_little_and_limited():
    one = np.zeros((21, 21), dtype=np.float64)
    one[10, 10] = 1.0
    out = relief.soften(one, 10.0)
    assert out.dtype == np.float32 and float(out.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(out[10, 10]) == pytest.approx(1.0 / (2.0 * math.pi * relief.BLUR_SIGMA ** 2), rel=0.05)    # 0.11
    assert float(out[10, 13] / out[10, 10]) == pytest.approx(math.exp(-9.0 / (2.0 * relief.BLUR_SIGMA ** 2)), rel=0.05)
    assert np.allclose(out, out.T) and np.allclose(out, out[::-1, ::-1])
    assert float(relief.soften(5.0 * one, 0.2).max()) == pytest.approx(0.2)
    assert float(relief.soften(-5.0 * one, 0.2).min()) == pytest.approx(-0.2)


def test_the_factor_from_the_normals(monkeypatch):
    """A smooth bump seen straight on: z towards the camera known, the reading 0.4 - 0.05 * distance."""
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float64)
    step = 0.002
    z = 0.12 * np.exp(-(((xx - 256) / 90.0) ** 2 + ((yy - 256) / 110.0) ** 2))
    reading = 0.4 - 0.05 * (2.0 - z)
    zx, zy = np.gradient(z, axis=1) / step, -np.gradient(z, axis=0) / step      # per scene unit, y up
    normal = np.stack([-zx, -zy, np.ones_like(z)])
    normal /= np.linalg.norm(normal, axis=0, keepdims=True)
    inside = np.ones((512, 512), dtype=bool)
    factor, correlation, kept = relief.factor_from_normals(reading, normal, inside, step)
    assert factor == pytest.approx(-1.0 / 0.05, rel=0.02) and correlation > 0.99 and kept > 100_000
    assert relief.factor_from_normals(reading, normal, np.zeros_like(inside), step) == (0.0, 0.0, 0)
    # a patch on the bump's flank where the normal model is wildly wrong (tilts three times too steep, as on a
    # glasses lens): the trimming drops it (measured: left in, it pulls the factor to -23.0)
    wild = np.stack([-zx, -zy, np.ones_like(z)])
    wild[:2, 200:300, 150:230] *= 3.0
    wild /= np.linalg.norm(wild, axis=0, keepdims=True)
    factor, correlation, _ = relief.factor_from_normals(reading, wild, inside, step)
    assert factor == pytest.approx(-1.0 / 0.05, rel=0.02) and correlation > 0.99
    # pixels that face away from the camera do not count, also where the trimming would not catch them (measured,
    # with neither: +13.9)
    monkeypatch.setattr(relief, "TRIM", 1e30)
    away = np.stack([-zx, -zy, np.ones_like(z)])
    away[:2, 200:300, 150:230] = 5.0
    away /= np.linalg.norm(away, axis=0, keepdims=True)
    assert float(away[2, 250, 190]) < relief.NORMAL_FACING
    assert relief.factor_from_normals(reading, away, inside, step)[0] == pytest.approx(-1.0 / 0.05, rel=0.02)


def test_the_fit_and_the_sampler():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, size=(40, 40))
    y = -3.0 * x + 5.0 + 0.001 * rng.standard_normal((40, 40))
    y[:3, :3] += 50.0                                                           # a few wild ones
    a, b, share = relief.affine_fit(x, y, np.ones((40, 40), dtype=bool))
    assert a == pytest.approx(-3.0, abs=0.01) and b == pytest.approx(5.0, abs=0.01) and share > 0.99
    assert relief.affine_fit(x, y, np.zeros((40, 40), dtype=bool)) == (0.0, 0.0, 0.0)
    # a line that fewer than 16 values agree on says nothing: 14 on a line and 6 far off it, on both sides
    few_x = np.arange(20, dtype=np.float64)
    few_y = 2.0 * few_x + 1.0
    few_y[[1, 4, 8, 11, 15, 18]] += [40.0, -40.0, 40.0, -40.0, 40.0, -40.0]
    assert relief.affine_fit(few_x, few_y, np.ones(20, dtype=bool)) == (0.0, 0.0, 0.0)
    a, b, share = relief.affine_fit(few_x, 2.0 * few_x + 1.0, np.ones(20, dtype=bool))
    assert (a, b) == pytest.approx((2.0, 1.0)) and share == pytest.approx(1.0)
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = relief.sample(image, np.array([[0.5, 0.5], [1.0, 0.5], [3.5, 2.5], [2.5, 1.0], [4.2, 1.0], [-0.1, 1.0]]))
    assert got.tolist() == pytest.approx([0.0, 0.5, 11.0, 4.0, 0.0, 0.0])


def test_the_model_files_and_how_a_picture_is_given_to_them(tmp_path, monkeypatch):
    assert not relief.available(None) and not relief.available(tmp_path)
    depth, normals = relief.model_paths(tmp_path)
    assert (depth.parent.name, depth.name, normals.name) == ("david", "depth-model-vitb16_384.onnx",
                                                             "normal-model-vitb16_384.onnx")
    depth.parent.mkdir()
    depth.write_bytes(b"x")
    assert not relief.available(tmp_path)                 # one file is not enough
    normals.write_bytes(b"x")
    assert relief.available(tmp_path)

    calls: list[dict] = []
    events: list[str] = []
    alive: list[int] = [0, 0]                             # sessions now, and the most there were at once

    class Session:
        def __init__(self, path, options, providers):
            alive[0] += 1
            alive[1] = max(alive[1], alive[0])
            events.append("load " + Path(path).name.split("-")[0])
            calls.append({"path": path, "threads": options.intra_op_num_threads, "providers": providers,
                          "arena": options.enable_cpu_mem_arena})

        def __del__(self):
            alive[0] -= 1

        def get_inputs(self):
            return [type("I", (), {"name": "input"})()]

        def run(self, names, feed):
            events.append("run")
            calls[-1]["feed"] = feed["input"]
            return [np.full((1, 512, 512), 7.0, dtype=np.float32)]

    fake = type(sys)("onnxruntime")
    fake.SessionOptions = lambda: type("O", (), {"intra_op_num_threads": 0, "enable_cpu_mem_arena": True})()
    fake.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setenv("GEN3D_THREADS", "3")
    relief._give_back()                                   # the real one: whatever the system, it does not raise

    def collect(*args):
        events.append("collect" if alive[0] == 0 else "collect with a model still loaded")
        return 0

    monkeypatch.setattr(relief.gc, "collect", collect)
    monkeypatch.setattr(relief, "_give_back", lambda: events.append("give back"))
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 255, 51, 0    # red full, a fifth of green, no blue
    run_depth, run_normals = relief.runners(tmp_path)
    assert events == []                                   # nothing is loaded until a picture is read
    out = run_depth(rgb)
    assert out.shape == (512, 512) and out.dtype == np.float32 and float(out[0, 0]) == 7.0
    call = calls[0]
    assert call["path"] == str(depth) and call["threads"] == 3 and call["providers"] == ["CPUExecutionProvider"]
    assert call["arena"] is False                         # onnxruntime's memory arena is turned off
    feed = call["feed"]
    assert feed.shape == (1, 3, 512, 512) and feed.dtype == np.float32
    assert feed[0, :, 0, 0].tolist() == pytest.approx([0.0, 0.2, 1.0])        # blue, green, red, each 0 to 1
    run_normals(rgb)
    assert calls[1]["path"] == str(normals)
    # each model is loaded, run, let go, collected and its memory handed back before the next one is loaded
    one = ["run", "collect", "give back"]
    assert events == ["load depth", *one, "load normal", *one] and alive == [0, 1]

    # a model that fails while it runs, or while it loads, is let go and its memory handed back all the same; the
    # failure comes out as it was, also the job's stop request
    Stopped = type("Stopped", (Exception,), {})

    class Broken(Session):
        def run(self, names, feed):
            raise Stopped() if "normal" in calls[-1]["path"] else RuntimeError("no memory")

    class Unreadable:
        def __init__(self, path, options, providers):
            raise ValueError("not a model file")

    fake.InferenceSession = Broken
    with pytest.raises(RuntimeError, match="no memory"):
        run_depth(rgb)
    assert events[-3:] == ["load depth", "collect", "give back"] and alive == [0, 1]
    with pytest.raises(Stopped):
        run_normals(rgb)
    assert events[-3:] == ["load normal", "collect", "give back"] and alive == [0, 1]
    fake.InferenceSession = Unreadable
    with pytest.raises(ValueError, match="not a model file"):
        run_depth(rgb)
    assert events[-3:] == ["give back", "collect", "give back"] and alive == [0, 1]


# ------------------------------------------------------------------ what is installed
def test_the_check_says_whether_the_relief_models_are_there(tmp_path):
    def check() -> dict:
        done = subprocess.run([sys.executable, "-m", "gen3d", "--check", "--weights-dir", str(tmp_path)], cwd=str(REPO),
                              capture_output=True, text=True, timeout=120, check=False)
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    assert check()["relief"] is False
    depth, normals = relief.model_paths(tmp_path)
    depth.parent.mkdir()
    depth.write_bytes(b"x")
    normals.write_bytes(b"x")
    info = check()
    assert info["relief"] is True and info["backends"]["fake"] is True and info["torch"] is None


@pytest.mark.skipif(not all(shutil.which(tool) for tool in ("bash", "curl", "sha256sum")),
                    reason="needs bash, curl and sha256sum")
def test_the_installer_goes_on_without_an_optional_file(tmp_path):
    """scripts/setup_3d.sh fetches the two relief models with fetch_optional: a file that cannot be had, or is not
    the pinned one, is said and left out, and the script goes on. Run here on files of this folder, not the web."""
    script = (REPO / "scripts" / "setup_3d.sh").read_text(encoding="utf-8")
    say = re.search(r"^say\(\) \{.*\}$", script, re.M).group(0)
    fetch = re.search(r"^fetch_optional\(\) \{.*?^\}$", script, re.M | re.S).group(0)
    program = "\n".join(["set -euo pipefail", say, fetch, "relief=yes", 'fetch_optional "$1" "$2" "$3" || relief=no',
                         'echo "relief=$relief"'])
    source = tmp_path / "there.bin"
    source.write_bytes(b"a depth model")
    right = hashlib.sha256(b"a depth model").hexdigest()

    def run(url: str, sha: str, name: str) -> tuple[str, Path]:
        target = tmp_path / "weights" / "david" / name
        done = subprocess.run(["bash", "-c", program, "bash", url, str(target), sha], capture_output=True, text=True,
                              timeout=60, check=False)
        assert done.returncode == 0, done.stderr                 # whatever happened to the file, the script went on
        assert not target.with_name(name + ".part").exists()
        return done.stdout, target

    out, target = run(source.as_uri(), right, "a.onnx")
    assert "relief=yes" in out and target.read_bytes() == b"a depth model"
    out, target = run((tmp_path / "missing.bin").as_uri(), right, "a.onnx")       # already there: nothing is fetched
    assert "relief=yes" in out and "downloading" not in out
    out, target = run(source.as_uri(), "0" * 64, "b.onnx")
    assert "relief=no" in out and "checksum mismatch" in out and not target.exists()
    out, target = run((tmp_path / "missing.bin").as_uri(), right, "c.onnx")
    assert "relief=no" in out and "could not download" in out and not target.exists()
    # the two files the script asks for are the two the stage reads, with the pinned checksums
    for name, sha in ((relief.DEPTH_FILE, "c8dc49821c95f7ab56a30150e58eba989b1d67ce988d11f083dc7350c54b883b"),
                      (relief.NORMAL_FILE, "3d37ecc812b6b50a09c514f3c8c8ad95da5cd6ae9b68c425bc52f1c9cd78c344")):
        assert f'="{name}"' in script and f'_SHA="{sha}"' in script
    for name in ("$DAVID_DEPTH", "$DAVID_NORMAL"):
        assert f'"$WEIGHTS/{relief.FOLDER}/{name}"' in script
