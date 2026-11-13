"""The generator's finishing stage (gen3d.finishing with gen3d.sparse, gen3d.raster and gen3d.prep's picture record;
docs/ARCHITECTURE.md section 36): the smoothing, the camera fit, the photo's weights, the carried colours, the photo
layer and its texture, and the file with two parts. Made-up meshes and made-up pictures only, in the server's own
Python (numpy, OpenCV, Pillow): no model runs here."""
from __future__ import annotations

import functools
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gen3d import finishing as F  # noqa: E402
from gen3d import glb, prep, raster, sparse  # noqa: E402
from gen3d.finishing import Fit, View  # noqa: E402
from gen3d.mesh import Camera, Head, Mesh, clean, finish, signed_volume, vertex_normals  # noqa: E402
from gen3d.prep import Prepared  # noqa: E402

SQUARE = 1024

# a head and shoulders turned on a lathe: height, half width, half depth of each ring, from the crown down to the base
PROFILE = np.array([
    (0.52, 0.00, 0.00), (0.50, 0.08, 0.09), (0.44, 0.15, 0.17), (0.36, 0.19, 0.22), (0.26, 0.20, 0.23),
    (0.16, 0.18, 0.21), (0.08, 0.14, 0.17), (0.02, 0.10, 0.12), (-0.04, 0.09, 0.10), (-0.10, 0.14, 0.12),
    (-0.16, 0.30, 0.17), (-0.22, 0.42, 0.20), (-0.34, 0.46, 0.22), (-0.45, 0.46, 0.22), (-0.45, 0.00, 0.00)])


def _lathe(profile: np.ndarray, lat: int, lon: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed surface: `lat` rings of `lon` vertices along the profile, a vertex at each end. Faces wound
    outwards."""
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(profile[:, :2], axis=0), axis=1))])
    t = np.linspace(0.0, along[-1], lat + 2)[1:-1]
    y, rx, rz = (np.interp(t, along, profile[:, k]) for k in range(3))
    around = 2.0 * np.pi * np.arange(lon) / lon
    rings = np.stack([rx[:, None] * np.sin(around)[None, :], np.repeat(y[:, None], lon, axis=1),
                      rz[:, None] * np.cos(around)[None, :]], axis=2).reshape(-1, 3)
    v = np.vstack([[(0.0, profile[0, 0], 0.0)], rings, [(0.0, profile[-1, 0], 0.0)]])
    j = np.arange(lon)
    nxt = (j + 1) % lon
    ring = lambda i: 1 + i * lon  # noqa: E731 - the first vertex of ring i
    last = v.shape[0] - 1
    faces = [np.stack([np.zeros(lon, int), ring(0) + j, ring(0) + nxt], axis=1),
             np.stack([np.full(lon, last), ring(lat - 1) + nxt, ring(lat - 1) + j], axis=1)]
    for i in range(lat - 1):
        a, b, c, d = ring(i) + j, ring(i) + nxt, ring(i + 1) + j, ring(i + 1) + nxt
        faces += [np.stack([a, c, d], axis=1), np.stack([a, d, b], axis=1)]
    return v, np.vstack(faces).astype(np.int64)


@functools.lru_cache(maxsize=None)
def _bust_arrays(lat: int, lon: int, lumps: float) -> tuple[np.ndarray, np.ndarray]:
    v, f = _lathe(PROFILE, lat, lon)
    if lumps:
        rng = np.random.default_rng(7)
        v = v + lumps * rng.standard_normal(v.shape)
    v.setflags(write=False)
    f.setflags(write=False)
    return v, f


def _bust(lat: int = 150, lon: int = 160, lumps: float = 0.0, colours: bool = True, camera: Camera | None = None) -> Mesh:
    """The made-up head: 24,002 vertices by default (enough for the quick outline), its face along +Z, skin
    coloured in front and at the sides, dark on top and at the back."""
    v, f = _bust_arrays(lat, lon, lumps)
    c = None
    if colours:
        hair = np.clip((np.maximum(v[:, 1] - 0.36, -v[:, 2] - 0.05)) / 0.06, 0.0, 1.0)[:, None]
        c = (np.array([190.0, 140.0, 110.0]) * (1.0 - hair) + np.array([40.0, 30.0, 25.0]) * hair).astype(np.uint8)
    return Mesh(vertices=v.astype(np.float32), faces=f.astype(np.uint32), colors=c, camera=camera)


def _gradient(size: int) -> np.ndarray:
    """A picture whose colour says where a pixel is: red rises to the right, green downwards."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    return np.stack([40.0 + 180.0 * xx / size, 60.0 + 150.0 * yy / size, np.full_like(xx, 120.0)], axis=2).astype(np.uint8)


def _gradient_at(xy: np.ndarray, size: int) -> np.ndarray:
    """The gradient's colour at source points (pixel centres at +0.5)."""
    return np.stack([40.0 + 180.0 * (xy[:, 0] - 0.5) / size, 60.0 + 150.0 * (xy[:, 1] - 0.5) / size,
                     np.full(xy.shape[0], 120.0)], axis=1)


def _picture(vertices: np.ndarray, faces: np.ndarray, view: View, size: int = 1536, cut_out: bool = True,
             inside=None) -> Prepared:
    """The made-up picture of a mesh seen from `view`: the gradient, and the cut-out where the mesh is drawn."""
    alpha = np.full((size, size), 255, dtype=np.uint8)
    if cut_out:
        uv, depth = view.sized(size).project(vertices)
        alpha = np.where(raster.outline(uv, depth, faces, size, size), 255, 0).astype(np.uint8)
    return Prepared(rgba=None, source=_gradient(size), alpha=alpha, side=SQUARE, x0=0.0, y0=0.0, sx=SQUARE / size,
                    sy=SQUARE / size, ox=0.0, oy=0.0, inside=inside)


def _decode(jpeg: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.float64)


SURE = Camera(1.9, 40.0, fit=False)     # a camera the backend is sure of: nothing is searched
GUESS = Camera(1.9, 40.0, fit=True)


@functools.lru_cache(maxsize=None)
def _laid() -> tuple[Head, Prepared, Mesh]:
    """The made-up head finished with its photo, once for the tests that only read the result."""
    mesh = _bust(camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE))
    return F.finish_head(mesh, picture, {"smoothing": 2, "texture": 2}), picture, mesh


# ------------------------------------------------------------------ the sparse helper
def test_rows_times_matches_dense_with_empty_rows():
    rng = np.random.default_rng(1)
    dense = (rng.random((7, 7)) < 0.4) * rng.random((7, 7))
    dense[2] = 0.0   # an empty row in the middle
    dense[6] = 0.0   # and the last row empty: np.add.reduceat gets both wrong
    r, c = np.nonzero(dense)
    rows = sparse.Rows(7, r, c, dense[r, c])
    x, many = rng.random(7), rng.random((7, 3))
    assert np.allclose(rows.times(x), dense @ x) and np.allclose(rows.times(many), dense @ many)
    assert rows.times(x)[2] == 0.0 and rows.times(x)[6] == 0.0 and rows.times(many).shape == (7, 3)
    plain = sparse.Rows(7, r, c)                       # without values: the sum over the row's columns
    ones = (dense != 0).astype(float)
    assert np.allclose(plain.times(x), ones @ x) and np.array_equal(plain.count, ones.sum(axis=1).astype(int))
    # the mean over a row's columns; a row without any keeps its own value
    mean = sparse.average(plain, x)
    assert mean[2] == x[2] and mean[6] == x[6]
    full = ones.sum(axis=1) > 0
    assert np.allclose(mean[full], (ones @ x)[full] / ones.sum(axis=1)[full])
    assert np.allclose(sparse.average(plain, many)[full], (ones @ many)[full] / ones.sum(axis=1)[full][:, None])
    # the smallest value around every row, itself included
    least = np.array([min([x[i]] + [x[k] for k in np.nonzero(ones[i])[0]]) for i in range(7)])
    assert np.allclose(plain.least(x), least)
    # a part of the matrix: some rows, some columns, renumbered
    keep_r, keep_c = np.array([1, 0, 1, 1, 0, 1, 1], bool), np.array([1, 1, 0, 1, 1, 0, 1], bool)
    part = rows.part(keep_r, keep_c)
    assert part.n == 5 and np.allclose(part.times(x[keep_c]), dense[keep_r][:, keep_c] @ x[keep_c])
    none = rows.part(keep_r, np.zeros(7, bool))        # no column at all: every sum is 0
    assert np.array_equal(none.times(np.zeros((0, 3))), np.zeros((5, 3)))


def test_neighbours_are_the_mesh_edges_each_once():
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    g = sparse.neighbours(f, 5)                        # vertex 4 belongs to no triangle
    assert np.array_equal(g.count, [3, 3, 3, 3, 0]) and g.rows.size == 12
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert np.allclose(g.times(x), [14.0, 13.0, 11.0, 7.0, 0.0])
    assert np.allclose(sparse.average(g, x), [14 / 3, 13 / 3, 11 / 3, 7 / 3, 16.0])


def test_solve_matches_a_direct_solve_also_when_rows_weigh_very_differently():
    rng = np.random.default_rng(2)
    n = 40
    a = rng.random((n, n)) * (rng.random((n, n)) < 0.2)
    a = np.triu(a, 1)
    a = a + a.T
    for weights in (np.ones(n), np.where(np.arange(n) % 5 == 0, 1e6, 1.0)):   # some rows held a million times harder
        m = np.diag(a.sum(axis=1) + weights) - a           # symmetric, positive definite
        b = rng.random((n, 3)) * weights[:, None]
        x, used = sparse.solve(lambda v: m @ v, b, np.diag(m), None, 1e-9, 500)
        assert used > 0 and np.allclose(x, np.linalg.solve(m, b), atol=1e-6)
    one, _ = sparse.solve(lambda v: m @ v, b[:, 0], np.diag(m), None, 1e-9, 500)   # one right-hand side, flat
    assert one.shape == (n,) and np.allclose(one, np.linalg.solve(m, b[:, 0]), atol=1e-6)
    exact = np.linalg.solve(m, b)
    again, used = sparse.solve(lambda v: m @ v, b, np.diag(m), exact, 1e-6, 500)   # started at the answer: no round
    assert used == 0 and np.allclose(again, exact)


# ------------------------------------------------------------------ the rasteriser
def _slow_depth(uv: np.ndarray, depth: np.ndarray, faces: np.ndarray, h: int, w: int) -> np.ndarray:
    """Every triangle against every pixel centre, one triangle at a time."""
    out = np.full((h, w), np.inf)
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    for a, b, c in faces:
        (ax, ay), (bx, by), (cx, cy) = uv[a], uv[b], uv[c]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        w0 = ((bx - xs) * (cy - ys) - (by - ys) * (cx - xs)) / area
        w1 = ((cx - xs) * (ay - ys) - (cy - ys) * (ax - xs)) / area
        w2 = 1.0 - w0 - w1
        z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
        out = np.where((w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (z < out), z, out)
    return out


def test_depth_buffer_of_a_hand_made_scene():
    # one triangle over the lower left half of an 8 x 8 picture, its depth rising to the right
    uv = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    z = raster.depth_buffer(uv, np.array([1.0, 2.0, 1.0]), np.array([[0, 1, 2]]), 8, 8)
    covered = np.isfinite(z)
    assert covered.sum() == 36 and all(covered[i, j] == (i + j <= 7) for i in range(8) for j in range(8))
    assert np.isclose(z[0, 0], 1.0 + 0.5 / 8) and np.isclose(z[3, 4], 1.0 + 4.5 / 8)   # flat across the triangle
    assert np.array_equal(raster.outline(uv, np.ones(3), np.array([[0, 1, 2]]), 8, 8), covered)
    # a nearer, smaller triangle in front of it wins where it lies; elsewhere the first one still shows
    uv2 = np.vstack([uv, [[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]]])
    both = raster.depth_buffer(uv2, np.array([1.0, 2.0, 1.0, 0.5, 0.5, 0.5]), np.array([[0, 1, 2], [3, 4, 5]]), 8, 8)
    assert both[1, 1] == 0.5 and both[1, 2] == 0.5 and np.isclose(both[0, 0], z[0, 0]) and np.isclose(both[5, 1], z[5, 1])
    # nothing to draw, a triangle with no area, one outside the picture, one with a depth that is no number
    assert not np.isfinite(raster.depth_buffer(uv, np.ones(3), np.zeros((0, 3), int), 8, 8)).any()
    odd = np.array([[1.0, 1.0], [5.0, 5.0], [3.0, 3.0], [20.0, 20.0], [30.0, 20.0], [20.0, 30.0], [0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    depth = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0])
    assert not np.isfinite(raster.depth_buffer(odd, depth, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), 8, 8)).any()


def test_depth_buffer_matches_drawing_every_triangle_alone():
    rng = np.random.default_rng(3)
    small = rng.random((300, 1, 2)) * 64 + (rng.random((300, 3, 2)) - 0.5) * 6      # a few pixels each
    large = rng.random((4, 3, 2)) * 90 - 13                                         # tens of pixels, partly outside
    uv = np.vstack([small.reshape(-1, 2), large.reshape(-1, 2)])
    faces = np.arange(uv.shape[0]).reshape(-1, 3)
    depth = 1.0 + rng.random(uv.shape[0])
    fast, slow = raster.depth_buffer(uv, depth, faces, 64, 64), _slow_depth(uv, depth, faces, 64, 64)
    assert np.array_equal(np.isfinite(fast), np.isfinite(slow)) and np.isfinite(fast).mean() > 0.5
    assert np.allclose(fast[np.isfinite(fast)], slow[np.isfinite(slow)], atol=1e-5)
    assert fast.dtype == np.float32 and fast.shape == (64, 64)


def test_quick_outline_is_the_exact_outline_of_a_dense_mesh():
    m = clean(_bust(220, 240))
    view = View(SURE, Fit(1.1, 0.02, -0.05), 256)
    uv, depth = view.project(m.vertices)
    exact = raster.outline(uv, depth, m.faces, 256, 256)
    quick = raster.quick_outline(uv, 256, 256, gap=F.long_edge(m.vertices, m.faces) * view.focal / float(np.median(depth)))
    assert F.overlap(quick, exact) > 0.985
    assert abs(float(quick.sum()) / float(exact.sum()) - 1.0) < 0.01     # not fatter, not thinner


# ------------------------------------------------------------------ the camera
def test_a_view_draws_as_the_camera_the_backend_names():
    f = 0.5 * SQUARE / np.tan(np.radians(20.0))
    plain = View(SURE, Fit(), SQUARE)
    p = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    uv, depth = plain.project(p)
    assert np.allclose(uv[0], [512.0, 512.0]) and np.allclose(depth, [1.9, 1.6]) and np.isclose(plain.focal, f)
    assert np.allclose(uv[1], [512.0 + f * 0.1 / 1.6, 512.0 - f * 0.2 / 1.6])
    assert np.allclose(plain.place, [0.0, 0.0, 1.9])
    # with a fit: the point is drawn as if it stood at scale * p + (mx, my, 0), and the depth is in the mesh's units
    e, mx, my = 1.2, 0.05, -0.03
    fitted = View(SURE, Fit(e, mx, my), SQUARE)
    uv, depth = fitted.project(p)
    assert np.allclose(uv[1], [512.0 + f * (e * 0.1 + mx) / (1.9 - e * 0.3), 512.0 - f * (e * 0.2 + my) / (1.9 - e * 0.3)])
    assert np.allclose(depth * e, [1.9, 1.9 - e * 0.3]) and np.allclose(fitted.place, [-mx / e, -my / e, 1.9 / e])
    half, _ = fitted.sized(512).project(p)
    assert np.allclose(half, uv / 2.0)                                   # the same camera for a smaller picture


def test_the_camera_fit_recovers_a_known_scale_and_shift(monkeypatch):
    quick = []
    real = raster.quick_outline
    monkeypatch.setattr(raster, "quick_outline", lambda *a, **k: quick.append(1) or real(*a, **k))
    m = clean(_bust(220, 240))                                           # 52,802 vertices close together
    true = Fit(1.12, 0.03, -0.04)
    picture = _picture(m.vertices, m.faces, View(GUESS, true, SQUARE), size=1024)
    view, covered, covered_head = F.find_camera(m.vertices, m.faces, GUESS, picture)
    assert 20 < len(quick) <= F.MAX_TRIES                                # searched with the quick outline
    unit = SQUARE * 0.5 / np.tan(np.radians(20.0)) / 1.9                 # square pixels per unit of the scene
    assert abs(view.fit.scale / true.scale - 1.0) < 0.006
    assert abs(view.fit.mx - true.mx) * unit < 2.0 and abs(view.fit.my - true.my) * unit < 2.0
    assert covered > 0.98 and covered_head > 0.98
    # where each vertex lands: within two pixels of where the true camera puts it, over the whole head
    got, _ = view.project(m.vertices)
    want, _ = View(GUESS, true, SQUARE).project(m.vertices)
    assert np.abs(got - want).max() < 2.5
    # a coarse mesh is searched with the triangle rasteriser itself
    del quick[:]
    coarse = clean(_bust(40, 48))
    picture = _picture(coarse.vertices, coarse.faces, View(GUESS, true, SQUARE), size=1024)
    view, covered, _ = F.find_camera(coarse.vertices, coarse.faces, GUESS, picture)
    assert not quick
    assert abs(view.fit.scale / true.scale - 1.0) < 0.012 and covered > 0.97
    assert abs(view.fit.mx - true.mx) * unit < 4.0 and abs(view.fit.my - true.my) * unit < 4.0
    # a camera the backend is sure of is taken as it is
    view, covered, covered_head = F.find_camera(coarse.vertices, coarse.faces, SURE, picture)
    assert view.fit == Fit() and covered is None and covered_head is None


def test_the_fit_is_not_pulled_by_where_the_picture_ends(monkeypatch):
    """The shoulders run out of a look's picture, so the cut-out ends in a straight line; the mesh ends in a base of
    its own, a little short of it. Compared over those rows, a larger head moved down would cover the cut-out better
    and the face would slide off the photo."""
    m = clean(_bust(220, 240))
    true = Fit(1.1, 0.0, 0.02)
    whole = _picture(m.vertices, m.faces, View(GUESS, true, SQUARE), size=1024)
    rows = np.nonzero(whole.alpha.any(axis=1))[0]
    cut = int(rows[0] + 0.86 * (rows[-1] - rows[0]))                     # the picture ends here
    whole.alpha[cut:] = 0
    # the mesh the model made of it: flat a little above that line (0.82 of the way down)
    level = float(m.vertices[:, 1].max() - 0.82 * (m.vertices[:, 1].max() - m.vertices[:, 1].min()))
    short = F.press_base(m.vertices, level)
    unit = SQUARE * 0.5 / np.tan(np.radians(20.0)) / 1.9

    def error() -> tuple[float, float]:
        view, _, _ = F.find_camera(short, m.faces, GUESS, whole)
        return abs(view.fit.scale / true.scale - 1.0), abs(view.fit.my - true.my) * unit

    scale_off, shift_off = error()
    assert scale_off < 0.008 and shift_off < 3.0
    monkeypatch.setattr(F, "BOTTOM_BAND", 0.0)                           # compared down to the last row: pulled
    monkeypatch.setattr(F, "MIN_OVERLAP", 0.0)
    worse_scale, worse_shift = error()
    assert worse_scale > 2 * scale_off + 0.004 or worse_shift > 2 * shift_off + 2.0


def test_what_is_compared_leaves_out_the_bottom_touched_sides_and_filler():
    size = 512
    alpha = np.zeros((size, size), np.uint8)
    alpha[100:400, 0:300] = 255                                          # the person touches the left side
    def record(inside=None):
        return Prepared(rgba=None, source=np.zeros((size, size, 3), np.uint8), alpha=alpha, side=SQUARE, x0=0.0, y0=0.0,
                        sx=2.0, sy=2.0, ox=0.0, oy=0.0, inside=inside)
    target, counted, head = F.fit_masks(record(), 256)
    assert target[50:200, 0:150].all() and not target[:50].any() and not target[200:].any() and not target[:, 150:].any()
    assert not counted[:, :2].any() and counted[100, 2] and counted[100, 255]          # 8 square pixels are 2 here
    assert counted[0].any()                                              # the top is not touched: compared
    assert counted[190, 100] and not counted[191:].any()                 # the lowest 6 percent of 150 rows, and below
    assert head[50:117, 100].all() and not head[118:, 100].any() and not (head & ~counted).any()   # the upper 45 percent
    # the photo's own part: nothing outside it is compared, and a side the person touches inside it loses its strip
    target, counted, _ = F.fit_masks(record(inside=[0, 0, 200, 512]), 256)
    assert not target[:, 100:].any() and not counted[:, 98:].any() and counted[100, 97]
    target, counted, head = F.fit_masks(record(inside=[0, 0, 0, 0]), 256)             # the square missed the photo
    assert not target.any() and not counted.any() and not head.any()


def test_a_photo_that_does_not_line_up_gives_one_part_and_says_so():
    mesh = _bust(40, 48, camera=GUESS)
    m = clean(mesh)
    size = 512
    alpha = np.zeros((size, size), np.uint8)
    alpha[60:200, 300:480] = 255                                          # a box in a corner: no head has this outline
    picture = Prepared(rgba=None, source=_gradient(size), alpha=alpha, side=SQUARE, x0=0.0, y0=0.0, sx=2.0, sy=2.0,
                       ox=0.0, oy=0.0)
    head = F.finish_head(mesh, picture, {"smoothing": 2, "texture": 2})
    assert head.photo is None and head.texture is None and head.texture_size is None
    assert "did not line up" in head.note and head.overlap is not None and head.overlap < F.MIN_OVERLAP
    assert head.overlap_head is not None
    plain = F.finish_head(mesh, picture, {"smoothing": 2, "texture": 0})
    assert plain.note is None and plain.overlap is None
    assert np.array_equal(head.base.vertices, plain.base.vertices) and np.array_equal(head.base.colors, plain.base.colors)
    assert glb.encode(head) == glb.encode(head.base)                      # written as a head always was
    # outlines that agree overall but not at the head are turned down too: the face is what has to line up
    good = _picture(m.vertices, m.faces, View(GUESS, Fit(), SQUARE), size=512)
    rows = np.nonzero(good.alpha.any(axis=1))[0]
    good.alpha[rows[0]:rows[0] + int(0.10 * rows.size)] = 0               # the crown is missing in the cut-out
    head = F.finish_head(mesh, good, {"smoothing": 0, "texture": 2})
    assert head.photo is None and "did not line up" in head.note
    assert head.overlap >= F.MIN_OVERLAP > head.overlap_head


# ------------------------------------------------------------------ the weights
def _weights(m: Mesh, picture: Prepared, view: View) -> tuple[np.ndarray, np.ndarray, sparse.Rows]:
    graph = sparse.neighbours(m.faces, m.vertices.shape[0])
    calm = F.taubin(m.vertices, graph, 10)
    weight, xy = F.photo_weights(m.vertices, m.faces, graph, vertex_normals(calm.astype(np.float32), m.faces).astype(np.float64),
                                 vertex_normals(m.vertices.astype(np.float32), m.faces).astype(np.float64), view, picture)
    return weight, xy, graph


def test_weights_seen_facing_and_never_a_step():
    m = clean(_bust())
    view = View(SURE, Fit(), SQUARE)
    weight, xy, graph = _weights(m, _picture(m.vertices, m.faces, view), view)
    v = m.vertices
    assert weight.min() >= 0.0 and weight.max() == 1.0
    front = (v[:, 2] > 0.15) & (np.abs(v[:, 0]) < 0.08) & (v[:, 1] > 0.15) & (v[:, 1] < 0.4)   # the middle of the face
    assert front.sum() > 100 and (weight[front] == 1.0).all()
    assert (weight[v[:, 2] < -0.02] == 0.0).all()                        # the back: turned away, and hidden
    chest = (v[:, 2] > 0.1) & (np.abs(v[:, 0]) < 0.2) & (v[:, 1] < -0.2) & (v[:, 1] > -0.35)
    assert chest.sum() > 100 and (weight[chest] == 1.0).all()
    side = (np.abs(v[:, 0]) > 0.19) & (np.abs(v[:, 2]) < 0.03) & (v[:, 1] > 0.2) & (v[:, 1] < 0.32)   # the temples, edge on
    assert side.sum() > 10 and weight[side].max() < 0.5
    assert 0.2 < float((weight > 0.5).mean()) < 0.5
    # where each vertex lands in the picture: the camera's projection, at the source's size
    want, _ = view.sized(1536).project(v)
    assert np.allclose(xy, want)
    # the photo ends nowhere at a line: no two neighbours differ by more than a quarter, anywhere on the head
    assert F.largest_step(weight, graph) <= F.WEIGHT_STEP + 1e-9 < 0.25
    partly = (weight > 0.05) & (weight < 0.95)
    assert partly.sum() > 500                                            # and there is a band where it fades


def test_weights_fade_with_the_facing_of_the_calm_surface_and_of_the_surface_itself():
    m = clean(_bust())
    view = View(SURE, Fit(), SQUARE)
    everything = _picture(m.vertices, m.faces, view, cut_out=False)       # no cut-out: only seeing and facing decide
    graph = sparse.neighbours(m.faces, m.vertices.shape[0])
    normals = vertex_normals(m.vertices.astype(np.float32), m.faces).astype(np.float64)
    to_camera = view.place[None, :] - m.vertices
    to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True)
    facing = (normals * to_camera).sum(axis=1)
    weight, _ = F.photo_weights(m.vertices, m.faces, graph, normals, normals, view, everything)
    assert (weight[facing < F.FACING_FADE] == 0.0).all() and (weight[facing < 0.0] == 0.0).all()
    head_on = (facing > 0.6) & (m.vertices[:, 1] > -0.3) & (m.vertices[:, 1] < 0.45)
    assert head_on.sum() > 1000 and (weight[head_on] == 1.0).all()
    between = (facing > 0.12) & (facing < 0.25) & (m.vertices[:, 1] > 0.15) & (m.vertices[:, 1] < 0.35)
    assert between.sum() > 20 and 0.0 < weight[between].max() < 1.0      # the fade itself
    # the calm copy says "all of it" (a lump smoothed away): the surface itself still shuts out what is turned away
    forward = np.repeat([[0.0, 0.0, 1.0]], m.vertices.shape[0], axis=0)
    weight, _ = F.photo_weights(m.vertices, m.faces, graph, forward, normals, view, everything)
    assert (weight[facing < F.REAL_FADE] == 0.0).all()
    just_past = (facing > F.REAL_FADE - 0.2) & (facing < F.REAL_FADE)
    assert just_past.sum() > 100
    weight, _ = F.photo_weights(m.vertices, m.faces, graph, forward, forward, view, everything)   # both say "all of it"
    assert weight[just_past].max() > 0.0                                  # those were seen: only the facing shut them out


def test_limit_step_only_lowers_and_bounds_the_step():
    # a row of seven vertices: 0 - 1 - 2 - 3 - 4 - 5 - 6
    f = np.array([[k, k + 1, k + 1] for k in range(6)])
    graph = sparse.neighbours(f, 7)
    got = F.limit_step(np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), graph, 0.2)
    assert np.allclose(got, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0])
    got = F.limit_step(np.array([1.0, 1.0, 0.3, 1.0, 1.0, 0.9, 1.0]), graph, 0.25)
    assert np.allclose(got, [0.8, 0.55, 0.3, 0.55, 0.8, 0.9, 1.0])
    smooth = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.array_equal(F.limit_step(smooth, graph, 0.2), smooth)      # already gentle: left alone


def test_weights_keep_away_from_a_jump_in_depth_and_from_what_is_hidden():
    m = clean(_bust())
    # a plate floating in front of the chest: 0.16 wide, a fine grid, facing the camera
    n = 41
    gx, gy = np.meshgrid(np.linspace(-0.08, 0.08, n), np.linspace(-0.36, -0.20, n))
    plate = np.stack([gx.ravel(), gy.ravel(), np.full(n * n, 0.42)], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    plate_faces = np.vstack([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)])
    v = np.vstack([m.vertices, plate])
    faces = np.vstack([m.faces, plate_faces + m.vertices.shape[0]])
    both = Mesh(vertices=v, faces=faces)
    view = View(SURE, Fit(), SQUARE)
    weight, _, _ = _weights(both, _picture(v, faces, view), view)
    uv, _ = view.project(v)
    plate_uv = uv[m.vertices.shape[0]:]
    lo, hi = plate_uv.min(axis=0), plate_uv.max(axis=0)
    bust_uv, bust_w = uv[:m.vertices.shape[0]], weight[:m.vertices.shape[0]]
    in_front = m.vertices[:, 2] > 0.05
    behind = in_front & (bust_uv > lo + 4).all(axis=1) & (bust_uv < hi - 4).all(axis=1)       # hidden by the plate
    assert behind.sum() > 50 and (bust_w[behind] == 0.0).all()
    out = np.maximum(np.maximum(lo - bust_uv, bust_uv - hi), 0.0).max(axis=1)                # pixels outside its outline
    beside = in_front & (out > 0.0) & (out < 1.5)
    assert beside.sum() > 10 and (bust_w[beside] == 0.0).all()                               # no photo next to the jump
    clear = in_front & (out > 60.0) & (np.abs(m.vertices[:, 0]) < 0.2) & (m.vertices[:, 1] < -0.2) & (m.vertices[:, 1] > -0.35)
    assert clear.sum() > 20 and (bust_w[clear] == 1.0).all()                                 # the chest further off: all of it
    plate_w = weight[m.vertices.shape[0]:].reshape(n, n)
    assert (plate_w[0] == 0.0).all() and (plate_w[:, -1] == 0.0).all() and plate_w[n // 2, n // 2] == 1.0


def test_weights_stop_inside_the_cut_out_and_inside_the_photo_itself():
    m = clean(_bust())
    view = View(SURE, Fit(), SQUARE)
    half = _picture(m.vertices, m.faces, view)
    half.alpha[:, 768:] = 0                                               # the cut-out ends at the picture's middle
    weight, xy, graph = _weights(m, half, view)
    front = (m.vertices[:, 2] > 0.1) & (m.vertices[:, 1] > -0.4)
    assert (weight[xy[:, 0] > 768 - 4 * 1.5 + 1] == 0.0).all()            # nothing within 4 square pixels of the edge
    deep = (front & (xy[:, 0] < 768 - 100 * 1.5) & (xy[:, 0] > 768 - 220 * 1.5) & (m.vertices[:, 1] < -0.2)
            & (m.vertices[:, 1] > -0.35))
    assert deep.sum() > 20 and (weight[deep] == 1.0).all()
    assert F.largest_step(weight, graph) <= F.WEIGHT_STEP + 1e-9
    # the photo's own part: the picture's lowest quarter is filler, repeated from the photo's last row
    filler = _picture(m.vertices, m.faces, view, inside=[0, 0, 1536, 1152])
    weight, xy, _ = _weights(m, filler, view)
    assert (weight[xy[:, 1] > 1152 - 4 * 1.5 + 1] == 0.0).all() and (weight[front & (xy[:, 1] < 900) & (m.vertices[:, 1] < -0.1)
                                                                        & (np.abs(m.vertices[:, 0]) < 0.2)] == 1.0).all()
    head = F.finish_head(_bust(camera=SURE), filler, {"smoothing": 1, "texture": 3})
    assert head.photo is not None and head.texture_size[1] <= 1152        # the texture never reaches into the filler
    lowest = head.photo.uv[:, 1].max() * head.texture_size[1]
    assert lowest <= head.texture_size[1] and head.texture_size[0] < 1536
    # a picture that shows nothing of the photo: no layer, and the head says why
    nothing = F.finish_head(_bust(40, 48, camera=SURE), _picture(m.vertices, m.faces, view, size=256, inside=[0, 0, 0, 0]), {})
    assert nothing.photo is None and "nothing of the photo" in nothing.note


# ------------------------------------------------------------------ the colours
def _cylinder(rings: int, around: int, radius: float = 1.0, height: float = 2.0, at=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """An open tube: `rings` rings of `around` vertices."""
    angle = 2.0 * np.pi * np.arange(around) / around
    y = np.linspace(-height / 2, height / 2, rings)
    v = np.stack([radius * np.sin(angle)[None, :].repeat(rings, 0), y[:, None].repeat(around, 1),
                  radius * np.cos(angle)[None, :].repeat(rings, 0)], axis=2).reshape(-1, 3) + np.asarray(at)
    j = np.arange(around)
    faces = []
    for i in range(rings - 1):
        a, b, c, d = i * around + j, i * around + (j + 1) % around, (i + 1) * around + j, (i + 1) * around + (j + 1) % around
        faces += [np.stack([a, c, d], axis=1), np.stack([a, d, b], axis=1)]
    return v, np.vstack(faces)


def test_carried_colours_go_round_at_their_height(monkeypatch):
    rings, around = 40, 72
    v, f = _cylinder(rings, around)
    graph = sparse.neighbours(f, v.shape[0])
    ring = np.arange(v.shape[0]) // around
    known = v[:, 2] > 0.5                                                 # the front third is the photo's
    red, blue, grey = np.array([220.0, 40.0, 40.0]), np.array([40.0, 40.0, 220.0]), np.array([128.0, 128.0, 128.0])
    colours = np.where((ring >= rings // 2)[:, None], red, blue) * known[:, None]   # the upper half red, the lower blue
    prior = np.repeat(grey[None, :], v.shape[0], axis=0)
    back = (v[:, 2] < -0.99)                                              # straight behind, 24 vertices from the photo
    upper, lower = ring >= rings - 10, ring < 10                          # well above and well below the line between them
    # with next to no pull towards the prior, a ring's colour ends up on the far side at the same height
    monkeypatch.setattr(F, "PRIOR", 1e-7)
    carried, used = F.carry_colours(v, graph, colours, known, prior)
    assert used > 0 and np.array_equal(carried[known], colours[known])   # the known ones are held
    assert np.abs(carried[back & upper] - red).max() < 12 and np.abs(carried[back & lower] - blue).max() < 12
    assert np.abs(carried[~known & (v[:, 2] > 0.3) & (ring == rings // 2 + 1)] - red).max() < 30   # the line stays a line
    monkeypatch.undo()
    # as shipped: next to the photo the carried colour, far from it the prior, and never black
    carried, _ = F.carry_colours(v, graph, colours, known, prior)
    beside = ~known & (v[:, 2] > 0.3)                                     # a few vertices past the photo's edge
    assert np.abs(carried[beside & upper] - red).max() < 30 and np.abs(carried[beside & lower] - blue).max() < 30
    assert np.abs(carried[back & upper] - grey).max() < 40 and carried[back & upper, 0].min() > 140   # grey, with some red
    assert np.abs(carried[back & lower] - grey).max() < 40 and carried[back & lower, 2].min() > 140   # grey, with some blue
    assert carried.min() >= 39.0 and carried.max() <= 221.0
    # a part the photo never reaches is the prior's colour, not black, and the solve still ends
    v2, f2 = _cylinder(6, 12, radius=0.2, height=0.4, at=(5.0, 0.0, 0.0))
    both_v, both_f = np.vstack([v, v2]), np.vstack([f, f2 + v.shape[0]])
    graph2 = sparse.neighbours(both_f, both_v.shape[0])
    known2 = np.concatenate([known, np.zeros(v2.shape[0], bool)])
    prior2 = np.vstack([prior, np.repeat([[90.0, 60.0, 30.0]], v2.shape[0], axis=0)])
    carried2, used = F.carry_colours(both_v, graph2, np.vstack([colours, np.zeros_like(v2)]), known2, prior2)
    assert used < F.CARRY_ROUNDS and np.allclose(carried2[v.shape[0]:], [90.0, 60.0, 30.0], atol=0.5)
    assert np.allclose(carried2[:v.shape[0]], carried, atol=0.5)
    # nothing known at all: everything is the prior; everything known: nothing changes
    alone, used = F.carry_colours(v, graph, colours, np.zeros(v.shape[0], bool), prior)
    assert used == 0 and np.array_equal(alone, prior)
    same, used = F.carry_colours(v, graph, colours, np.ones(v.shape[0], bool), prior)
    assert used == 0 and np.array_equal(same, colours)


def test_the_models_colours_are_matched_to_the_photo():
    rng = np.random.default_rng(4)
    photo = rng.random((500, 3)) * 200 + 20
    known = np.arange(500) < 300
    own = 0.6 * photo + 30.0                                              # the same colours, flatter and shifted
    matched = F.match_colours(own, photo, known)
    assert np.allclose(matched, photo, atol=1e-6)                         # also where the photo is not known
    # colours that have nothing to do with the photo's are moved to its level and not stretched beyond the limits
    flat = np.full((500, 3), 77.0)
    assert np.allclose(F.match_colours(flat, photo, known), photo[known].mean(axis=0))
    against = 255.0 - photo                                               # dark where the photo is light
    turned = F.match_colours(against, photo, known)
    k = (turned[1] - turned[0]) / (against[1] - against[0])
    assert np.allclose(k, F.GAIN_LOW) and np.allclose(turned[known].mean(axis=0), photo[known].mean(axis=0))
    wild = F.match_colours(0.1 * photo, photo, known)                     # a gain of 10 would be needed: it stays at the limit
    assert np.allclose((wild[1] - wild[0]) / (0.1 * (photo[1] - photo[0])), F.GAIN_HIGH)
    assert turned.min() >= 0.0 and turned.max() <= 255.0
    assert np.array_equal(F.match_colours(own, photo, np.zeros(500, bool)), own)   # nothing known: left as they are


def test_the_soft_copy_blurs_over_the_vertex_spacing_and_keeps_the_background_out():
    size = 800
    rng = np.random.default_rng(5)
    source = np.zeros((size, size, 3), np.uint8)
    source[:, :400] = np.clip(150 + 40 * rng.standard_normal((size, 400, 1)), 0, 255)   # the person: grey with grain
    source[:, 400:] = (0, 255, 0)                                                        # a loud background
    alpha = np.zeros((size, size), np.uint8)
    alpha[:, :400] = 255
    picture = Prepared(rgba=None, source=source, alpha=alpha, side=SQUARE, x0=0.0, y0=0.0, sx=0.5, sy=0.5, ox=0.0, oy=0.0)
    soft, (qx, qy) = F.soft_photo(picture, spacing=6.0)
    assert soft.shape == (400, 400, 3) and (qx, qy) == (0.5, 0.5)        # at the scale of the model's square
    inside = soft[20:380, 20:195]
    assert abs(float(inside.mean()) - 150.0) < 2.0 and float(inside.std()) < 4.0   # the grain of 40 is gone
    assert np.abs(soft[:, 196:200] - 150.0).max() < 12.0                 # right at the edge: still the person's grey
    assert np.abs(soft[:, 196:200, 1] - soft[:, 196:200, 0]).max() < 1e-3          # no green has leaked in
    sharp, _ = F.soft_photo(picture, spacing=0.0)                        # no spacing: only made smaller
    assert float(sharp[20:380, 20:195].std()) > 10.0


# ------------------------------------------------------------------ the photo layer and its texture
def test_the_photo_layer_carries_the_photo_where_the_camera_put_it():
    head, picture, mesh = _laid()
    m = clean(mesh)
    assert head.note is None and head.overlap is None and head.photo is not None
    layer, base = head.photo, head.base
    assert base.vertices.shape[0] == m.vertices.shape[0] and base.faces.shape[0] == m.faces.shape[0]
    assert layer.vertex_ids.shape[0] == layer.uv.shape[0] == layer.alpha.shape[0] and layer.faces.max() < layer.uv.shape[0]
    assert layer.uv.min() >= 0.0 and layer.uv.max() <= 1.0 and layer.alpha.min() >= 0.0 and layer.alpha.max() == 1.0
    assert (layer.alpha < 0.01).sum() > 100 and ((layer.alpha > 0.1) & (layer.alpha < 0.9)).sum() > 500   # it fades out
    assert 0.15 * m.faces.shape[0] < layer.faces.shape[0] < 0.6 * m.faces.shape[0]
    # the layer's triangles are the base's own triangles, wound the same way
    tri = np.sort(layer.vertex_ids[layer.faces.astype(np.int64)], axis=1)
    assert {tuple(t) for t in tri[:200]} <= {tuple(t) for t in np.sort(m.faces, axis=1)}
    out = layer.vertex_ids[layer.faces.astype(np.int64)]
    normal = np.cross(base.vertices[out[:, 1]] - base.vertices[out[:, 0]], base.vertices[out[:, 2]] - base.vertices[out[:, 0]])
    assert (normal[:, 2] > 0).mean() > 0.95                               # they face the camera
    # every triangle that shows any photo is in the layer, and the layer's rim shows none
    shown = np.zeros(m.vertices.shape[0])
    shown[layer.vertex_ids] = layer.alpha
    rim = np.setdiff1d(np.arange(m.vertices.shape[0]), layer.vertex_ids)
    graph = sparse.neighbours(m.faces, m.vertices.shape[0])
    assert (graph.times((shown > F.LAYER_FROM).astype(float))[rim] == 0).all()
    # the texture is the JPEG of the photo's used part; a corner's place in it is its place in the photo
    texture = _decode(head.texture)
    assert (texture.shape[1], texture.shape[0]) == head.texture_size and max(head.texture_size) <= 2048
    full = layer.alpha > 0.999
    at = layer.uv[full] * np.array(head.texture_size)                     # in pixels of the texture
    got = F.sample(texture, at).astype(np.float64)
    graph_v = F.taubin(m.vertices, graph, F.rounds_for(F.TAUBIN_ROUNDS[2], F.round_scale(m.vertices, graph)))
    graph_v = F.press_base(graph_v, F.base_level(graph_v))
    want_xy, _ = View(SURE, Fit(), 1536).project(graph_v[layer.vertex_ids[full]])
    want = _gradient_at(want_xy, 1536)
    assert full.sum() > 2000 and np.abs(got - want).max() < 6.0           # the gradient says where a pixel came from
    # underneath, the base's own colours are the photo's too where the photo is all there is
    under = base.colors[layer.vertex_ids[full]].astype(np.float64)
    assert np.abs(under - want).max() < 6.0
    # and where no photo reaches, the head's own colours matched to it: nothing is black, the back is not the front
    back = m.vertices[:, 2] < -0.1
    assert base.colors[back].min() > 5 and np.abs(base.colors[back].astype(float).mean(axis=0) - want.mean(axis=0)).max() > 15


def test_the_texture_is_the_photos_own_pixels_up_to_the_limit():
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=3000, cut_out=False)
    sizes = {level: F.finish_head(mesh, picture, {"smoothing": 0, "texture": level}).texture_size for level in (1, 2, 3)}
    assert max(sizes[1]) == 1024 and max(sizes[2]) == 2048                # shrunk to the limit
    assert 2048 < max(sizes[3]) <= 3000                                   # 3: the photo's own size, as it is
    for size in sizes.values():
        assert abs(size[0] / size[1] - sizes[3][0] / sizes[3][1]) < 0.01  # the same part of the picture each time
    small = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=600, cut_out=False)
    assert max(F.finish_head(mesh, small, {"texture": 3}).texture_size) <= 600   # never made larger than the photo


def test_the_background_is_painted_over_with_the_persons_colours():
    size = 240
    yy, xx = np.mgrid[0:size, 0:size]
    person = (xx - 120) ** 2 + (yy - 120) ** 2 < 70 ** 2
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[...] = (0, 255, 0)                                                # a loud background
    rgb[person] = (180, 120, 90)
    filled = F.spread(rgb, person)
    assert np.array_equal(filled[person], rgb[person])                    # the person is untouched
    assert np.abs(filled[~person].astype(int) - [180, 120, 90]).max() <= 2
    assert np.array_equal(F.spread(rgb, np.ones_like(person)), rgb) and np.array_equal(F.spread(rgb, np.zeros_like(person)), rgb)
    # in a head's texture the cut-out's surroundings are the person's colours as well: none of a loud background
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=768)
    picture.source[picture.alpha == 0] = (255, 0, 255)
    head = F.finish_head(mesh, picture, {"smoothing": 0, "texture": 2})
    texture = _decode(head.texture)
    assert head.photo is not None and texture[:, :, 2].max() < 150 and texture[:, :, 0].max() < 235   # blue is 120 in the photo
    assert abs(float(texture[:, :, 2].mean()) - 120.0) < 3.0


def test_texture_0_or_no_camera_or_no_picture_gives_one_part_smoothed_with_a_flat_base():
    lumpy = _bust(lumps=0.002, camera=SURE)
    m = clean(lumpy)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE))
    plain = F.finish_head(lumpy, picture, {"smoothing": 2, "texture": 0})
    assert plain.photo is None and plain.texture is None and plain.note is None and plain.overlap is None
    for other in (F.finish_head(_bust(lumps=0.002, camera=None), picture, {"smoothing": 2, "texture": 2}),
                  F.finish_head(lumpy, None, {"smoothing": 2, "texture": 2})):
        assert other.photo is None and other.note is None and np.array_equal(other.base.vertices, plain.base.vertices)
    base = plain.base
    assert base.vertices.dtype == np.float32 and base.faces.dtype == np.uint32 and base.colors.dtype == np.uint8
    assert np.isclose(float((base.vertices.max(axis=0) - base.vertices.min(axis=0)).max()), 1.0, atol=1e-6)   # the unit box
    assert np.allclose(np.linalg.norm(base.normals, axis=1), 1.0, atol=1e-4)
    low = base.vertices[:, 1] <= base.vertices[:, 1].min() + 1e-6
    assert low.sum() > 300                                                # the base: one level, not a ragged rim
    graph = sparse.neighbours(m.faces, m.vertices.shape[0])
    rough = lambda x: float(np.linalg.norm(sparse.average(graph, x) - x, axis=1)[~low].mean())  # noqa: E731
    off = F.finish_head(lumpy, picture, {"smoothing": 0, "texture": 0}).base
    assert rough(base.vertices.astype(float)) < 0.5 * rough(off.vertices.astype(float))      # the lumps are smoothed
    calm = F.taubin(m.vertices, graph, 30)                                # and the smoothing does not shrink the head
    assert abs(signed_volume(calm, m.faces) / signed_volume(m.vertices, m.faces) - 1.0) < 0.01
    assert np.array_equal(F.taubin(m.vertices, graph, 0), m.vertices)
    strong = F.finish_head(lumpy, picture, {"smoothing": 3, "texture": 0}).base
    assert rough(strong.vertices.astype(float)) < rough(base.vertices.astype(float))
    # options outside their range are moved to the nearest end; none at all means normal smoothing and the photo
    assert F.finish_head(lumpy, picture, {"smoothing": 9, "texture": -3}).photo is None
    assert np.array_equal(F.finish_head(lumpy, None, None).base.vertices, plain.base.vertices)


def test_a_failure_in_the_photo_stage_never_fails_the_head(monkeypatch):
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=512)
    good = F.finish_head(mesh, picture, {"texture": 2})
    plain = F.finish_head(mesh, picture, {"texture": 0})
    assert good.photo is not None

    def broken(*_a, **_k):
        raise RuntimeError("boom")

    for name in ("photo_weights", "carry_colours", "make_texture", "calm_unseen", "soft_photo", "map_agrees"):
        with monkeypatch.context() as patch:
            patch.setattr(F, name, broken)
            head = F.finish_head(mesh, picture, {"texture": 2})
        assert head.photo is None and head.texture is None and "RuntimeError: boom" in head.note, name
        assert np.array_equal(head.base.vertices, plain.base.vertices) and np.array_equal(head.base.colors, plain.base.colors)
    # nothing of the head shows in the photo: said in words, not as an error
    with monkeypatch.context() as patch:
        patch.setattr(F, "photo_weights", lambda v, *_a, **_k: (np.zeros(v.shape[0]), np.zeros((v.shape[0], 2))))
        head = F.finish_head(mesh, picture, {"texture": 2})
    assert head.photo is None and "too little of the head" in head.note
    with pytest.raises(F.NoPhoto):
        F.crop_box(picture, np.zeros((0, 2)))
    # an order to stop is not a failure of the photo: it passes through

    class Stop(BaseException):
        pass

    def stop(*_a, **_k):
        raise Stop()

    monkeypatch.setattr(F, "photo_weights", stop)
    with pytest.raises(Stop):
        F.finish_head(mesh, picture, {"texture": 2})


def test_what_the_photo_does_not_show_is_pulled_smooth_and_the_rest_stays():
    m = clean(_bust(lumps=0.002))
    graph = sparse.neighbours(m.faces, m.vertices.shape[0])
    v = m.vertices
    weight = F.limit_step((v[:, 2] > 0.05).astype(float), graph, 0.2)     # the front is the photo's
    moved, used = F.calm_unseen(v, graph, weight, 60.0)
    assert used > 0 and np.array_equal(moved[weight == 1.0], v[weight == 1.0])   # under the photo: not a hair
    rough = lambda x, part: float(np.linalg.norm(sparse.average(graph, x) - x, axis=1)[part].mean())  # noqa: E731
    unseen = weight == 0.0
    assert rough(moved, unseen) < 0.25 * rough(v, unseen)
    assert abs(signed_volume(moved, m.faces) / signed_volume(v, m.faces) - 1.0) < 0.03   # the shape is kept
    wide = lambda x: float(x[:, 0].max() - x[:, 0].min())  # noqa: E731
    assert wide(moved) / wide(v) > 0.97
    still, used = F.calm_unseen(v, graph, weight, 0.0)                    # smoothing off
    assert used == 0 and np.array_equal(still, v)
    all_photo, used = F.calm_unseen(v, graph, np.ones(v.shape[0]), 60.0)
    assert used == 0 and np.array_equal(all_photo, v)
    # in a finished head the base is flat again afterwards, on the level it had before the photo came into it
    head, _, _ = _laid()
    low = head.base.vertices[:, 1] <= head.base.vertices[:, 1].min() + 1e-6
    assert low.sum() > 300


def _push(calls: list | None = None):
    """A stage that moves the front of the head along the camera's rays, towards the camera."""
    def push(vertices, faces, view, prepared):
        if calls is not None:
            calls.append((vertices.shape, faces.shape, view, prepared))
        out = vertices.copy()
        at = view.place
        front = vertices[:, 2] > 0.1
        out[front] = at + (vertices[front] - at) * 0.97
        return out
    return push


def test_another_stage_may_move_the_vertices_once_the_camera_is_known():
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=512)
    plain = F.finish_head(mesh, picture, {"texture": 2})
    assert plain.reshaped is False and plain.reshape_note is None          # no stage was given
    calls = []
    pushed = F.finish_head(mesh, picture, {"texture": 2}, reshape=_push(calls))
    assert len(calls) == 1 and calls[0][0] == m.vertices.shape and calls[0][1] == m.faces.shape
    view = calls[0][2]
    assert isinstance(view, View) and view.side == SQUARE and calls[0][3] is picture
    assert np.allclose(view.project(np.zeros((1, 3)))[0], [[512.0, 512.0]])
    assert pushed.photo is not None and not np.allclose(pushed.base.vertices, plain.base.vertices, atol=1e-3)
    assert pushed.reshaped is True and pushed.reshape_note is None and pushed.note is None
    low = pushed.base.vertices[:, 1] <= pushed.base.vertices[:, 1].min() + 1e-6
    assert low.sum() > 50                                                 # the base is flat again after the move
    for same in (lambda v, f, view, prepared: None, lambda v, f, view, prepared: v):    # nothing moved
        left = F.finish_head(mesh, picture, {"texture": 2}, reshape=same)
        assert np.array_equal(left.base.vertices, plain.base.vertices) and left.texture == plain.texture
        assert left.reshaped is False and left.reshape_note is None      # it ran: why it did nothing is its own to say


def test_a_stage_that_fails_costs_the_head_nothing_else():
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=512)
    plain = F.finish_head(mesh, picture, {"texture": 2})

    def broken(*_a):
        raise ValueError("no")

    failed = F.finish_head(mesh, picture, {"texture": 2}, reshape=broken)
    assert failed.photo is not None and failed.note is None              # the photo is laid all the same
    assert failed.reshaped is False and failed.reshape_note == "ValueError: no"
    assert np.array_equal(failed.base.vertices, plain.base.vertices) and failed.texture == plain.texture
    # what is not this surface is not taken: another shape, or a vertex that is not a number
    wrong = lambda v, f, view, prepared: v[:-1].copy()  # noqa: E731

    def nan(v, f, view, prepared):
        out = v.copy()
        out[3, 1] = np.nan
        return out

    word = lambda v, f, view, prepared: "moved"  # noqa: E731
    ragged = lambda v, f, view, prepared: [[1.0, 2.0, 3.0], [4.0]]  # noqa: E731
    thing = lambda v, f, view, prepared: object()  # noqa: E731
    for stage in (wrong, nan, word, ragged, thing):
        head = F.finish_head(mesh, picture, {"texture": 2}, reshape=stage)
        assert head.reshaped is False and "not this surface" in head.reshape_note and head.photo is not None
        assert np.array_equal(head.base.vertices, plain.base.vertices)

    # an order to stop is not a failure of the stage: it passes through
    class Stop(BaseException):
        pass

    def stop(*_a):
        raise Stop()

    with pytest.raises(Stop):
        F.finish_head(mesh, picture, {"texture": 2}, reshape=stop)


def test_the_stage_also_runs_for_a_head_without_the_photo():
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=512)
    plain = F.finish_head(mesh, picture, {"texture": 0})
    assert plain.overlap is None                                          # nothing asked for the camera
    calls = []
    head = F.finish_head(mesh, picture, {"texture": 0}, reshape=_push(calls))
    assert len(calls) == 1 and head.reshaped is True and head.reshape_note is None
    assert head.photo is None and head.texture is None and head.note is None    # no layer was asked for
    assert not np.allclose(head.base.vertices, plain.base.vertices, atol=1e-3)
    assert np.array_equal(head.base.colors, plain.base.colors)           # the model's own colours, as without it
    assert glb.encode(head) == glb.encode(head.base)                      # one part
    # the same surface as the head with the photo has under its layer, before the unseen part is pulled smooth
    with_photo = F.finish_head(mesh, picture, {"texture": 2, "smoothing": 0}, reshape=_push())
    without = F.finish_head(mesh, picture, {"texture": 0, "smoothing": 0}, reshape=_push())
    assert np.allclose(with_photo.base.vertices, without.base.vertices, atol=1e-6)


def test_what_the_stage_did_is_kept_when_the_photo_fails_after_it(monkeypatch):
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(SURE, Fit(), SQUARE), size=512)
    alone = F.finish_head(mesh, picture, {"texture": 0}, reshape=_push())

    def broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(F, "photo_weights", broken)
    head = F.finish_head(mesh, picture, {"texture": 2}, reshape=_push())
    assert head.photo is None and "RuntimeError: boom" in head.note
    assert head.reshaped is True and head.reshape_note is None
    assert np.array_equal(head.base.vertices, alone.base.vertices)        # the moved surface, without the layer


def test_without_the_camera_the_stage_does_not_run_and_the_head_says_why():
    called = []
    stage = lambda v, f, view, prepared: called.append(1)  # noqa: E731
    mesh = _bust(40, 48, camera=GUESS)
    size = 512
    alpha = np.zeros((size, size), np.uint8)
    alpha[60:200, 300:480] = 255                                          # a box in a corner: no head has this outline
    off = Prepared(rgba=None, source=_gradient(size), alpha=alpha, side=SQUARE, x0=0.0, y0=0.0, sx=2.0, sy=2.0,
                   ox=0.0, oy=0.0)
    head = F.finish_head(mesh, off, {"texture": 2}, reshape=stage)
    assert not called and head.reshaped is False
    assert "did not line up" in head.note and head.reshape_note == head.note
    # without the photo there is nothing to say about a layer, but the stage's reason stays, with the overlaps
    head = F.finish_head(mesh, off, {"texture": 0}, reshape=stage)
    assert not called and head.note is None and "did not line up" in head.reshape_note
    assert head.overlap is not None and head.overlap < F.MIN_OVERLAP
    # a backend that names no camera, a look without a picture
    m = clean(mesh)
    picture = _picture(m.vertices, m.faces, View(GUESS, Fit(), SQUARE), size=512)
    # the camera step itself breaks: said as the camera's failure, which is the stage's reason with or without a photo
    with pytest.MonkeyPatch.context() as patch:
        def broken(*_a, **_k):
            raise RuntimeError("boom")

        patch.setattr(F, "map_agrees", broken)
        head = F.finish_head(mesh, picture, {"texture": 2}, reshape=stage)
        assert head.note == "the picture's camera could not be found (RuntimeError: boom)" and head.reshape_note == head.note
        head = F.finish_head(mesh, picture, {"texture": 0}, reshape=stage)
        assert head.note is None and head.reshape_note == "the picture's camera could not be found (RuntimeError: boom)"
        assert head.photo is None and not called
    for texture in (0, 2):
        head = F.finish_head(_bust(40, 48, camera=None), picture, {"texture": texture}, reshape=stage)
        assert head.reshape_note == "the backend names no camera" and head.note is None
        head = F.finish_head(mesh, None, {"texture": texture}, reshape=stage)
        assert head.reshape_note == "the look has no picture" and head.note is None
    assert not called


# ------------------------------------------------------------------ the file
def test_glb_round_trip_with_the_photo_layer(tmp_path):
    head, _, _ = _laid()
    size = glb.write(tmp_path / "h.glb", head)
    data = (tmp_path / "h.glb").read_bytes()
    assert size == len(data) and len(data) % 4 == 0 and data == glb.encode(head)
    back = glb.read(tmp_path / "h.glb")
    doc, layer, base = back["json"], head.photo, head.base
    assert np.array_equal(back["positions"], base.vertices) and np.array_equal(back["indices"].reshape(-1, 3), base.faces)
    assert np.array_equal(back["colors"][:, :3], base.colors) and np.allclose(back["normals"], base.normals)
    photo = back["photo"]
    assert np.array_equal(photo["positions"], base.vertices[layer.vertex_ids])       # exactly on the surface
    assert np.array_equal(photo["normals"], base.normals[layer.vertex_ids])
    assert np.array_equal(photo["uv"], layer.uv) and np.array_equal(photo["indices"].reshape(-1, 3), layer.faces)
    assert np.abs(photo["alpha"] - layer.alpha).max() <= 0.5 / 65535 + 1e-9
    assert back["texture"] == head.texture and back["texture"][:3] == b"\xff\xd8\xff"   # the JPEG, byte for byte
    skin, over = doc["meshes"][0]["primitives"]
    assert set(skin["attributes"]) == {"POSITION", "NORMAL", "COLOR_0"}
    assert set(over["attributes"]) == {"POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0"}
    assert [mat["name"] for mat in doc["materials"]] == ["skin", "photo"] and (skin["material"], over["material"]) == (0, 1)
    assert doc["materials"][0]["doubleSided"] is True and "alphaMode" not in doc["materials"][0]
    assert doc["materials"][1]["alphaMode"] == "BLEND" and doc["materials"][1]["doubleSided"] is False
    assert doc["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"] == {"index": 0}
    assert doc["textures"] == [{"source": 0, "sampler": 0}]
    assert doc["samplers"] == [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    # the picture: its own buffer view, which is no vertex data (no target), and its media type
    assert doc["images"] == [{"bufferView": len(doc["bufferViews"]) - 1, "mimeType": "image/jpeg"}]
    image_view = doc["bufferViews"][-1]
    assert "target" not in image_view and image_view["byteLength"] == len(head.texture)
    assert all("target" in view and view["byteOffset"] % 4 == 0 for view in doc["bufferViews"][:-1])
    uv = doc["accessors"][over["attributes"]["TEXCOORD_0"]]
    assert (uv["componentType"], uv["type"], uv["count"]) == (5126, "VEC2", layer.uv.shape[0])
    alpha = doc["accessors"][over["attributes"]["COLOR_0"]]
    assert (alpha["componentType"], alpha["type"], alpha["normalized"]) == (5123, "VEC4", True)
    assert doc["buffers"][0]["byteLength"] == image_view["byteOffset"] + len(head.texture) + (-len(head.texture)) % 4


def test_a_head_without_a_layer_is_the_file_it_always_was(tmp_path):
    m = finish(_bust(40, 48))
    assert glb.encode(Head(base=m)) == glb.encode(m)
    assert glb.encode(Head(base=m, note="no photo", overlap=0.5)) == glb.encode(m)
    glb.write(tmp_path / "old.glb", m)
    back = glb.read(tmp_path / "old.glb")
    assert back["photo"] is None and back["texture"] is None and len(back["json"]["meshes"][0]["primitives"]) == 1
    assert "images" not in back["json"] and "textures" not in back["json"] and len(back["json"]["materials"]) == 1


# ------------------------------------------------------------------ the picture record
def test_the_map_from_the_square_back_to_the_picture(monkeypatch):
    """gen3d.prep with a made-up cut-out in place of the model's: a marked dot of the square maps back to the dot in
    the picture. A slip here would lay the photo on the head shifted, and nothing else would notice."""
    w, h = 2500, 2300
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - 1500) / 520.0) ** 2 + ((yy - 950) / 700.0) ** 2 < 1.0   # off centre, taller than wide
    rgb = np.full((h, w, 3), 90, np.uint8)
    rgb[blob] = (60, 110, 160)
    dot = (1322, 1207)                                                    # x, y of the dot's middle pixel
    rgb[dot[1] - 4:dot[1] + 5, dot[0] - 4:dot[0] + 5] = (255, 0, 0)
    monkeypatch.setattr(prep, "matte", lambda image: np.where(blob, 255, 0).astype(np.uint8))
    got = prep.prepare(Image.fromarray(rgb), {})
    assert got.rgba.size == (SQUARE, SQUARE) and got.source.shape == (h, w, 3) and got.alpha.shape == (h, w)
    square = np.asarray(got.rgba).astype(int)
    assert 0.84 * SQUARE <= np.ptp(np.nonzero(square[..., 3] > 127)[0]) + 1 <= 0.87 * SQUARE   # the longer side fills 0.86
    red = (square[..., 0] > 200) & (square[..., 1] < 60) & (square[..., 2] < 60) & (square[..., 3] > 200)
    ys, xs = np.nonzero(red)
    assert ys.size >= 6
    found = got.to_source(np.array([[xs.mean() + 0.5, ys.mean() + 0.5]]))[0]       # the dot's middle, as a point
    assert abs(found[0] - (dot[0] + 0.5)) < 1.0 and abs(found[1] - (dot[1] + 0.5)) < 1.0
    there = got.to_square(np.array([[dot[0] + 0.5, dot[1] + 0.5]]))[0]
    assert abs(there[0] - (xs.mean() + 0.5)) < 0.5 and abs(there[1] - (ys.mean() + 0.5)) < 0.5
    pts = np.array([[10.0, 20.0], [900.5, 333.25]])
    assert np.allclose(got.to_square(got.to_source(pts)), pts)
    # the run-time check of the same thing: the cut-out laid into the square by the map covers the model's own
    assert F.map_agrees(got) > 0.99
    slipped = Prepared(**{**got.__dict__, "ox": got.ox + 40.0})
    assert F.map_agrees(slipped) < F.MAP_OVERLAP
    head = F.finish_head(_bust(40, 48, camera=GUESS), slipped, {"texture": 2})
    assert head.photo is None and "does not sit where the model saw it" in head.note
    # the stand-in's record: the square is the picture, all of it the person
    record = prep.whole(Image.fromarray(rgb[:600, :800]))
    assert record.rgba is None and record.alpha.all() and (record.sx, record.sy) == (SQUARE / 800, SQUARE / 600)
    assert np.allclose(record.to_source(np.array([[512.0, 512.0]])), [[400.0, 300.0]]) and F.map_agrees(record) == 1.0


def test_the_photos_own_part_of_the_picture():
    image = Image.new("RGB", (200, 100), (10, 20, 30))
    assert prep.whole(image).part() == (0, 0, 200, 100) and prep.whole(image).inside is None
    assert prep.whole(image, {"inside": [10.2, 0, 150.7, 60]}).part() == (11, 0, 150, 60)     # rounded inwards
    assert prep.whole(image, {"inside": [-5, -5, 900, 900]}).part() == (0, 0, 200, 100)       # kept in the picture
    left, top, right, bottom = prep.whole(image, {"inside": [0, 0, 0, 0]}).part()
    assert right <= left and bottom <= top
    record = prep.whole(image, {"inside": [20, 10, 120, 90]})
    person = F.person(record)
    assert person[10:90, 20:120].all() and person.sum() == 80 * 100
    assert F.depth_inside(record)[50, 20] < F.depth_inside(record)[50, 70]                    # it fades towards its edge
    # the texture's crop takes a margin round what the layer uses, and stops at the photo's own part
    used = np.array([[25.0, 15.0], [118.0, 88.0]])
    assert F.crop_box(prep.whole(image), used) == (1, 0, 143, 100) and F.crop_box(record, used) == (20, 10, 120, 90)
