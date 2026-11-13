"""What a head gets between the backend and the file: a smooth surface, a flat base, and the photo itself laid
over the front as its texture (docs/ARCHITECTURE.md section 36). numpy and OpenCV only.

An image-to-3D model returns a lumpy surface with one blurred colour per vertex: on a head of 60,000 vertices the
face has about a hundred colours across. The picture it was made from has twenty times that. So the picture goes
back onto the mesh:

1. the surface is smoothed (Taubin, which does not shrink it) and everything under a level near the bottom is
   pressed flat: a clean base instead of a ragged one;
2. the picture's camera is found: the backend names the camera it assumes, and a search over one scale and two
   shifts makes the mesh's outline cover the person's outline in the picture. The outlines are compared without
   the cut-out's lowest rows and without what lies next to a side of the picture the person touches: there the
   picture ends, and that line is no outline of the person. The photo is used only when the outlines agree, over
   everything that is compared and over the head alone;
3. every vertex gets a weight, how much of the photo it may show. It must be seen from that camera, lie inside
   the cut-out and inside the part of the picture that is the photo itself, keep away from a jump in depth (an ear
   in front of hair), and face the camera: read from a much smoother copy of the surface, so a lump does not punch
   a hole, and from the surface itself, so the underside of a nose does not take the pixels of its tip. Each of
   these fades, and no two neighbouring vertices differ by more than a fifth: the photo ends nowhere at a line;
4. where the weight is zero the surface is only the model's guess: it is pulled smooth, by a solve that reaches
   the long ripples plain smoothing rounds leave, and a second one that gives back most of the broad shape the
   first took along with the lumps;
5. the vertex colours become the photo's where the weight is high (read from a copy of the photo blurred to the
   vertices' spacing, so one hair or one glint does not decide a vertex), and elsewhere the photo's colours carried
   around the head at the same height: neighbours at one height count far more than those above and below, so
   hair stays hair and the shirt stays the shirt. The carried colours are held weakly to the model's own, which are
   matched to the photo first: far from the photo the back keeps the model's guess, and a part the photo never
   reaches is not left black;
6. the triangles that show any photo are written a second time as a layer with the photo as its texture and the
   weight as its transparency, so the photo fades into the colours underneath.

Between 2 and 3 another stage may move the surface (`reshape`: the face's relief, read from the photo by a depth
model, gen3d.relief). It needs the camera and nothing else of the photo's, so it also runs for a head that is asked
for without the photo (`texture` 0), and what it did is kept when a later step of the photo fails.

A head whose photo does not line up, or whose backend names no camera, keeps one part with a colour per vertex.
Nothing here fails a look: when the photo stage raises, the head is written without the layer and says why.

Lengths are counted for a mesh whose edges are REF_EDGE of its longest side (a marching-cubes grid of 256 on a head)
and scaled for finer and coarser meshes, so every detail level is smoothed over the same distance, and the
stand-in's coarse egg is left as it is.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from gen3d import raster, sparse
from gen3d.mesh import Camera, Head, Layer, Mesh, box, clean, normalise, vertex_normals

TAUBIN_ROUNDS = (0, 12, 30, 60)          # by the `smoothing` option: off, light, normal, strong
UNSEEN_PULL = (0.0, 100.0, 300.0, 500.0) # by `smoothing`: how hard what the photo does not show is pulled smooth
TEXTURE_SIDE = (0, 1024, 2048, 4096)     # by `texture`: no photo; its longest side at most this (3: the photo's own size)
LAMBDA, MU = 0.5, -0.53                  # Taubin's two steps
CALM_ROUNDS = 60                         # the much smoother copy the facing is read from
COLOUR_ROUNDS = 2
RESHAPED_ROUNDS = 4                      # after another stage moved the vertices (the face's relief)
REF_EDGE = 0.0067                        # mean edge over longest side of the meshes the numbers were tuned on
MAX_ROUND_SCALE = 4.0
BASE_CUT = 0.035                         # the flat base: this share of the height above the lowest point
MAP_OVERLAP = 0.95                       # the cut-out, laid into the square by prep's map, against the model's own
SEARCH_SIDE = 256                        # the camera search compares outlines at this size
QUICK_FROM = 20_000                      # the search draws the quick outline from this many vertices on ...
QUICK_GAP = 3.2                          # ... when neighbours land no further apart than this, in its pixels
MAX_TRIES = 240
SCALE_LOW, SCALE_HIGH = 0.5, 2.0         # the search's scale stays within these
SCALE_STEP = 0.03                        # the search's first step in scale ...
SHIFT_STEP, LAST_STEP = 8.0, 0.25        # ... and in shift, in pixels of the model's square; it stops under LAST_STEP
BOTTOM_BAND = 0.06                       # of the cut-out's height: its lowest rows are where the picture ends
SIDE_BAND = 8.0                          # pixels of the model's square left out along a side the person touches
HEAD_SHARE = 0.45                        # of the cut-out's height, from its top: the head
MIN_OVERLAP = 0.92                       # under this, overall or over the head, the photo did not line up
DEPTH_SIDE = 1024                        # the depth buffer's size
SEEN_SLACK = 0.02                        # of the mesh's size: how far behind the nearest surface still counts as seen
JUMP = 0.035                             # of the mesh's size: a depth range over 5 x 5 pixels that is a jump
JUMP_NONE, JUMP_FULL = 2.0, 8.0          # pixels from a jump: no photo, and all of it
ALPHA_SURE = 230
INSIDE_NONE, INSIDE_FULL = 4.0, 16.0     # pixels of the model's square inside the cut-out's edge: no photo, all of it
FACING_FADE, FACING_FULL = 0.06, 0.32    # facing of the smoother copy
REAL_FADE, REAL_FULL = -0.05, 0.15       # facing of the surface itself
WEIGHT_STEP = 0.2                        # the weight differs by no more than this between two neighbouring vertices
KNOWN = 0.5                              # weight from which a vertex's colour is the photo's
LEVEL = 8.0                              # how much less a neighbour above or below counts when colours are carried
PRIOR = 0.003                            # how much the model's own colour counts next to a vertex's neighbours
GAIN_LOW, GAIN_HIGH = 0.5, 2.0           # the model's colours are stretched no further when matched to the photo
SOFT_SPACING, SOFT_MAX = 0.75, 8.0       # the blurred copy: sigma in vertex spacings, and at most in square pixels
LAYER_FROM = 0.004                       # a triangle joins the photo layer when one corner's weight is over this
CROP_MARGIN = 24
JPEG_QUALITY = 90
MIN_KNOWN = 30                           # fewer vertices than this with the photo's colour: no layer
CARRY_TOLERANCE, CARRY_ROUNDS = 1e-4, 2000
UNSEEN_TOLERANCE, UNSEEN_ROUNDS = 1e-4, 600
FREE_FROM = 1e-4                         # a vertex pulled less than this stays where it is


LAST: dict[str, float] = {}   # the last finish_head in numbers: seconds per stage, solver rounds, the largest
                              # weight step between neighbours (read by measurements, never by code)


class _Clock:
    def __init__(self) -> None:
        import time

        self._now = time.perf_counter
        self._at = self._now()
        LAST.clear()

    def mark(self, stage: str) -> None:
        now = self._now()
        LAST[stage] = LAST.get(stage, 0.0) + now - self._at
        self._at = now


class NoPhoto(Exception):
    """The photo cannot be laid over this head; the message is the note written with it."""

    def __init__(self, note: str, overlap: float | None = None, overlap_head: float | None = None) -> None:
        super().__init__(note)
        self.overlap = overlap
        self.overlap_head = overlap_head


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mean_edge(vertices: np.ndarray, graph: sparse.Rows) -> float:
    if not graph.rows.size:
        return 0.0
    return float(np.linalg.norm(vertices[graph.rows] - vertices[graph.cols], axis=1).mean())


def round_scale(vertices: np.ndarray, graph: sparse.Rows) -> float:
    """How many times the tuned number of rounds this mesh needs: (REF_EDGE / its own edge share) squared, since
    a round moves a point by about one edge and distances add up as the square root of the rounds."""
    edge = mean_edge(vertices, graph)
    if edge <= 0:
        return 0.0
    _, extent = box(vertices)
    return min(MAX_ROUND_SCALE, (REF_EDGE / max(edge / extent, 1e-9)) ** 2)


def rounds_for(tuned: int, scale: float) -> int:
    return int(round(tuned * scale))


def taubin(vertices: np.ndarray, graph: sparse.Rows, rounds: int) -> np.ndarray:
    """Taubin smoothing: a step towards the neighbours' mean and a slightly larger step back, so lumps go and
    the volume stays."""
    x = np.asarray(vertices, dtype=np.float64)
    for _ in range(max(0, int(rounds))):
        x = x + LAMBDA * (sparse.average(graph, x) - x)
        x = x + MU * (sparse.average(graph, x) - x)
    return x


def press_base(vertices: np.ndarray, level: float) -> np.ndarray:
    out = np.array(vertices, dtype=np.float64)
    out[:, 1] = np.maximum(out[:, 1], level)
    return out


def base_level(vertices: np.ndarray) -> float:
    low, high = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    return low + BASE_CUT * (high - low)


def calm_unseen(vertices: np.ndarray, graph: sparse.Rows, weight: np.ndarray, pull: float) -> tuple[np.ndarray, int]:
    """What the photo does not show, pulled smooth. Every vertex is put where it balances between its own place
    and the mean of its neighbours' new places; the mean counts `pull` times its own place where the weight is 0
    and not at all where it is 1, so the part under the photo stays as it is and the rest gives way smoothly from
    its edge. This is the system (I + T (I - mean of neighbours)) x = v, written symmetric by multiplying each row
    by its number of neighbours over its pull, so conjugate gradient solves it. Plain smoothing rounds would do the
    same for short waves only and leave the long ripples of a marching-cubes back.

    One such solve also takes some of the surface's broad shape along with the lumps: the outline moves inwards
    from where the camera was fitted to it, and ears and shoulders melt. So it is run a second time, on what the
    first took away, and the smooth part of that is given back. Measured on three heads at pull 300, this is as
    smooth as a single solve at pull 120 and loses about two thirds as much of the outline's overlap with the
    photo (2 to 4 points of it, where the single solve lost 4 to 5) and under half as much of the volume (2 to 3
    percent). Returns (the vertices, the solver's rounds)."""
    v = np.asarray(vertices, dtype=np.float64)
    hold = float(pull) * (1.0 - smoothstep(0.0, 1.0, weight))
    free = hold > FREE_FROM
    if not free.any():
        return v, 0
    degree = graph.count.astype(np.float64)[free]
    keep = degree / hold[free]
    inner = graph.part(free, free)
    diagonal = keep + degree

    def apply(x: np.ndarray) -> np.ndarray:
        return diagonal[:, None] * x - inner.times(x)

    held = graph.part(free, ~free).times(v[~free])        # the neighbours that stay where they are
    first, used = sparse.solve(apply, keep[:, None] * v[free] + held, diagonal, v[free], UNSEEN_TOLERANCE, UNSEEN_ROUNDS)
    taken = v[free] - first
    back, more = sparse.solve(apply, keep[:, None] * taken, diagonal, taken, UNSEEN_TOLERANCE, UNSEEN_ROUNDS)
    out = v.copy()
    out[free] = first + back
    return out, used + more


# ------------------------------------------------------------------ the camera
@dataclass(frozen=True)
class Fit:
    """The camera's correction: the mesh point p is drawn as if it stood at scale * p + (mx, my, 0). A shift along
    the camera's axis is not searched: it gives the same picture as another scale."""
    scale: float = 1.0
    mx: float = 0.0
    my: float = 0.0

    def array(self) -> np.ndarray:
        return np.array([self.scale, self.mx, self.my], dtype=np.float64)


def focal(camera: Camera, side: float) -> float:
    return 0.5 * float(side) / math.tan(math.radians(camera.fov_y_deg) / 2.0)


@dataclass(frozen=True)
class View:
    """The picture's camera in the mesh's own frame, the fit taken in: a pinhole at `place` looking down -z, for a
    square picture of `side` pixels. Every later stage goes between the mesh and the picture through this: `project`
    for a vertex's place in the square, gen3d.raster.depth_buffer to draw the mesh from it, and the picture record's
    `to_source` from the square on to the photo."""
    camera: Camera
    fit: Fit
    side: float

    @property
    def focal(self) -> float:
        """In pixels of the square."""
        return focal(self.camera, self.side)

    @property
    def place(self) -> np.ndarray:
        """Where the camera stands."""
        return np.array([-self.fit.mx, -self.fit.my, self.camera.distance], dtype=np.float64) / self.fit.scale

    def project(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pixel coordinates (n, 2) in the square, pixel centres at +0.5; the distance along the camera's axis
        (n,), in the mesh's own units)."""
        v = np.asarray(vertices, dtype=np.float64)
        at = self.place
        depth = np.maximum(at[2] - v[:, 2], 1e-9)
        f, half = self.focal, self.side / 2.0
        return np.stack([half + f * (v[:, 0] - at[0]) / depth, half - f * (v[:, 1] - at[1]) / depth], axis=1), depth

    def sized(self, side: float) -> "View":
        """The same camera for the picture at another size."""
        return View(self.camera, self.fit, float(side))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two masks."""
    union = float((a | b).sum())
    return float((a & b).sum()) / union if union > 0 else 0.0


def person(prepared: Any, level: int = 127) -> np.ndarray:
    """Where the source picture shows the person: the cut-out above `level`, inside the photo's own part."""
    mask = prepared.alpha > level
    left, top, right, bottom = prepared.part()
    h, w = mask.shape
    if (left, top, right, bottom) == (0, 0, w, h):
        return mask
    kept = np.zeros_like(mask)
    if right > left and bottom > top:
        kept[top:bottom, left:right] = mask[top:bottom, left:right]
    return kept


def square_alpha(prepared: Any, side: int, mask: np.ndarray | None = None) -> np.ndarray:
    """The person (`mask`, or person(prepared)) as the model saw it, in a square of side x side: True where at
    least half a pixel is person. The mask is laid into the model's square first, by the record's own map, and
    shrunk from there by whole areas, so a thin edge does not flicker."""
    import cv2

    full = int(prepared.side)
    # pixel i of the source (its centre at i + 0.5) lands at square pixel sx * i + (0.5 - x0) * sx + ox - 0.5
    m = np.array([[prepared.sx, 0.0, (0.5 - prepared.x0) * prepared.sx + prepared.ox - 0.5],
                  [0.0, prepared.sy, (0.5 - prepared.y0) * prepared.sy + prepared.oy - 0.5]], dtype=np.float64)
    inside = (person(prepared) if mask is None else mask).astype(np.float32)
    if min(prepared.sx, prepared.sy) < 0.5:   # shrinking by more than two: average first, or the edge aliases
        k = int(math.ceil(0.5 / min(prepared.sx, prepared.sy)))
        inside = cv2.blur(inside, (k, k))
    laid = cv2.warpAffine(inside, m, (full, full), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if side != full:
        laid = cv2.resize(laid, (int(side), int(side)), interpolation=cv2.INTER_AREA if side < full else cv2.INTER_LINEAR)
    return laid >= 0.5


def map_agrees(prepared: Any) -> float:
    """How well the record's map puts the source's cut-out where the model saw it: the overlap of the two cut-outs
    in the model's square (1.0 when the model was given the picture itself). A slip in the map would lay the photo
    on the head shifted, and the outlines, which are compared in the square alone, would not show it."""
    if prepared.rgba is None:
        return 1.0
    seen = np.asarray(prepared.rgba)[..., 3] > 127
    return overlap(seen, square_alpha(prepared, int(prepared.side), prepared.alpha > 127))


def fit_masks(prepared: Any, side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For comparing outlines in a square of side x side: (the person, which pixels are compared, which of those
    are the head). Not compared: what lies outside the photo's own part; the cut-out's lowest rows (BOTTOM_BAND of
    its height) and everything under them, because the shoulders run out of the picture there and the mesh ends in
    a base of its own; and a strip along the left, right or top of the photo's part when the person touches that
    side, for the same reason. The head is the upper HEAD_SHARE of the cut-out's height."""
    mask = person(prepared)
    target = square_alpha(prepared, side, mask)
    none = np.zeros_like(target)
    rows = np.nonzero(target.any(axis=1))[0]
    left, top, right, bottom = prepared.part()
    if not rows.size or right <= left or bottom <= top:
        return target, none, none
    k = float(side) / float(prepared.side)
    corner = prepared.to_square(np.array([[left, top], [right, bottom]], dtype=np.float64)) * k
    (x_lo, y_lo), (x_hi, y_hi) = corner
    near = max(1, int(round(2.0 / max(min(prepared.sx, prepared.sy), 1e-6))))   # two square pixels, in source pixels
    band = SIDE_BAND * k
    if mask[top:bottom, left:left + near].any():
        x_lo += band
    if mask[top:bottom, max(left, right - near):right].any():
        x_hi -= band
    if mask[top:top + near, left:right].any():
        y_lo += band
    first, last = int(rows[0]), int(rows[-1])
    tall = last - first + 1
    y_hi = min(y_hi, last + 1 - BOTTOM_BAND * tall)
    centre = np.arange(int(side)) + 0.5
    counted = ((centre >= y_lo) & (centre < y_hi))[:, None] & ((centre >= x_lo) & (centre < x_hi))[None, :]
    head = counted & (centre < first + HEAD_SHARE * tall)[:, None]
    return target, counted, head


def bounding_start(vertices: np.ndarray, target: np.ndarray, camera: Camera) -> Fit:
    """A start for the search from the two bounding boxes: the scale and shift that would lay the mesh's box over
    the outline's box if the mesh were flat."""
    ys, xs = np.nonzero(target)
    if not ys.size:
        return Fit()
    side = target.shape[0]
    f = focal(camera, side)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    size = np.maximum(high - low, 1e-9)
    middle = (low + high) / 2.0
    tall, wide = float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1)
    scale = 0.5 * (tall * camera.distance / (f * size[1]) + wide * camera.distance / (f * size[0]))
    mx = ((xs.max() + xs.min() + 1) / 2.0 - side / 2.0) * camera.distance / f - scale * middle[0]
    my = -((ys.max() + ys.min() + 1) / 2.0 - side / 2.0) * camera.distance / f - scale * middle[1]
    return Fit(float(scale), float(mx), float(my))


def drawn_outline(vertices: np.ndarray, faces: np.ndarray, view: View) -> np.ndarray:
    """Where the mesh covers the view's picture, by the triangle rasteriser."""
    uv, depth = view.project(vertices)
    side = int(view.side)
    return raster.outline(uv, depth, faces, side, side)


def long_edge(vertices: np.ndarray, faces: np.ndarray) -> float:
    """How far apart neighbouring vertices are at most, the few longest edges aside (the 98th percentile)."""
    f = np.asarray(faces, dtype=np.int64)
    v = np.asarray(vertices, dtype=np.float64)
    return float(np.percentile(np.concatenate([np.linalg.norm(v[f[:, a]] - v[f[:, b]], axis=1)
                                               for a, b in ((0, 1), (1, 2), (2, 0))]), 98))


def fit_camera(vertices: np.ndarray, faces: np.ndarray, camera: Camera, target: np.ndarray,
               counted: np.ndarray | None = None, square: float = 1024.0) -> tuple[Fit, int]:
    """The scale and shift that make the mesh's outline cover `target` (a square mask) best where `counted` is
    set, and how many outlines were drawn for it. A pattern search: each of the three is tried a step up and a
    step down, a better try is kept, and when none is better the steps halve. It starts from the camera as the
    backend named it or from the two bounding boxes, whichever covers better, with steps of SCALE_STEP in scale
    and SHIFT_STEP pixels of the model's square (`square`: its side) in shift, and ends when the shift's step is
    under LAST_STEP pixels. A mesh with many vertices close together is drawn with the quick outline while
    searching; any other with the triangle rasteriser."""
    side = int(target.shape[0])
    counted = np.ones_like(target) if counted is None else counted
    aim = target & counted
    reach = long_edge(vertices, faces)
    quick = False

    def gap(view: View, depth: np.ndarray) -> float:
        return reach * view.focal / max(float(np.median(depth)), 1e-9)   # pixels between neighbouring vertices

    def score(x: np.ndarray) -> float:
        view = View(camera, Fit(float(x[0]), float(x[1]), float(x[2])), side)
        if quick:
            uv, depth = view.project(vertices)
            drawn = raster.quick_outline(uv, side, side, gap=gap(view, depth))
        else:
            drawn = drawn_outline(vertices, faces, view)
        return overlap(drawn & counted, aim)

    # the limits keep the mesh in front of the camera and the search out of nonsense
    furthest = 0.8 * camera.distance
    nearest = float(vertices[:, 2].max())
    low = np.array([SCALE_LOW, -furthest, -furthest])
    high = np.array([min(SCALE_HIGH, 0.9 * camera.distance / nearest) if nearest > 0 else SCALE_HIGH, furthest, furthest])
    high[0] = max(high[0], low[0])
    starts = [np.clip(s.array(), low, high) for s in (Fit(), bounding_start(vertices, target, camera))]
    widest = max(gap(view, view.project(vertices)[1]) for view in (View(camera, Fit(*s), side) for s in starts))
    quick = vertices.shape[0] >= QUICK_FROM and widest <= QUICK_GAP
    scored = [score(s) for s in starts]
    first = int(np.argmax(scored))
    x, best, tries = starts[first], scored[first], len(starts)
    step = np.array([SCALE_STEP * x[0], 1.0, 1.0])
    step[1:] = SHIFT_STEP * camera.distance / focal(camera, square)   # that many square pixels, in the scene's units
    share = 1.0
    while share >= LAST_STEP / SHIFT_STEP and tries < MAX_TRIES:
        better = False
        for k in range(3):
            for sign in (1.0, -1.0):
                y = x.copy()
                y[k] = min(high[k], max(low[k], y[k] + sign * share * step[k]))
                if y[k] == x[k]:
                    continue
                got = score(y)
                tries += 1
                if got > best + 1e-6:
                    x, best, better = y, got, True
                    break
        if not better:
            share /= 2.0
    return Fit(float(x[0]), float(x[1]), float(x[2])), tries


# ------------------------------------------------------------------ the weights
def sample(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of an image (h, w) or (h, w, k) at pixel coordinates (n, 2), centres at +0.5; the edge
    repeats. float32, (n,) or (n, k)."""
    h, w = image.shape[:2]
    x = np.clip(np.asarray(xy[:, 0], dtype=np.float64) - 0.5, 0.0, w - 1.0)
    y = np.clip(np.asarray(xy[:, 1], dtype=np.float64) - 0.5, 0.0, h - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), max(w - 2, 0))
    y0 = np.minimum(np.floor(y).astype(np.int64), max(h - 2, 0))
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    if image.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    top = image[y0, x0].astype(np.float32) * (1 - fx) + image[y0, x1].astype(np.float32) * fx
    bottom = image[y1, x0].astype(np.float32) * (1 - fx) + image[y1, x1].astype(np.float32) * fx
    return top * (1 - fy) + bottom * fy


def depth_inside(prepared: Any) -> np.ndarray:
    """For every source pixel, how far inside the person it lies: the distance, in pixels of the model's square, to
    the nearest pixel that is not surely the person. The picture's own border and the border of the photo's part
    count as such pixels, so the photo fades towards them as well."""
    import cv2

    sure = np.zeros((prepared.alpha.shape[0] + 2, prepared.alpha.shape[1] + 2), dtype=np.uint8)
    sure[1:-1, 1:-1] = person(prepared, ALPHA_SURE)
    far = cv2.distanceTransform(sure, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return far * np.float32(math.sqrt(prepared.sx * prepared.sy))


def photo_weights(vertices: np.ndarray, faces: np.ndarray, graph: sparse.Rows, calm_normals: np.ndarray,
                  real_normals: np.ndarray, view: View, prepared: Any) -> tuple[np.ndarray, np.ndarray]:
    """(weight per vertex in 0..1, where each vertex lands in the source picture)."""
    import cv2

    _, extent = box(vertices)
    uv, depth = view.project(vertices)
    k = DEPTH_SIDE / float(view.side)
    at = uv * k
    zbuf = raster.depth_buffer(at, depth, faces, DEPTH_SIDE, DEPTH_SIDE)
    px = np.clip(np.floor(at).astype(np.int64), 0, DEPTH_SIDE - 1)
    drawn = np.isfinite(zbuf)
    kernel3, kernel5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    far = np.where(drawn, zbuf, np.float32(1e9))
    near = cv2.erode(far, kernel3)                      # the nearest depth around each pixel
    weight = (depth <= near[px[:, 1], px[:, 0]] + SEEN_SLACK * extent).astype(np.float64)
    # a jump in depth among drawn pixels (an ear in front of hair, the chin over the neck): no photo next to it
    span = cv2.dilate(np.where(drawn, zbuf, np.float32(-1e9)), kernel5) - cv2.erode(far, kernel5)
    jump = (span > JUMP * extent) & drawn
    if jump.any():
        away = cv2.distanceTransform((~jump).astype(np.uint8), cv2.DIST_L2, 5)
        weight *= smoothstep(JUMP_NONE, JUMP_FULL, sample(away, at))
    xy = prepared.to_source(uv)
    h, w = prepared.alpha.shape
    in_picture = (xy[:, 0] > 0) & (xy[:, 1] > 0) & (xy[:, 0] < w) & (xy[:, 1] < h)
    weight *= in_picture * smoothstep(INSIDE_NONE, INSIDE_FULL, sample(depth_inside(prepared), xy))
    to_camera = view.place[None, :] - vertices
    to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
    weight = weight * smoothstep(FACING_FADE, FACING_FULL, (calm_normals * to_camera).sum(axis=1))
    weight = weight * smoothstep(REAL_FADE, REAL_FULL, (real_normals * to_camera).sum(axis=1))
    return limit_step(weight, graph, WEIGHT_STEP), xy


def limit_step(weight: np.ndarray, graph: sparse.Rows, step: float) -> np.ndarray:
    """The largest weights under `weight` that differ by at most `step` between neighbouring vertices. A weight is
    only ever lowered: a vertex that was shut out stays shut out, and from it the photo comes in over 1 / step
    vertices. The factors of the weight fade in the picture, but over the mesh two neighbours can still land far
    apart in them (a fold, a long edge, a vertex seen next to one that is not); this bounds what smoothing them
    would only reduce."""
    out = np.asarray(weight, dtype=np.float64)
    for _ in range(int(math.ceil(1.0 / step))):
        out = np.minimum(out, graph.least(out) + step)
    return out


def largest_step(weight: np.ndarray, graph: sparse.Rows) -> float:
    """The largest difference in weight between two neighbouring vertices."""
    if not graph.rows.size:
        return 0.0
    return float(np.abs(weight[graph.rows] - weight[graph.cols]).max())


# ------------------------------------------------------------------ the colours
def soft_photo(prepared: Any, spacing: float) -> tuple[np.ndarray, tuple[float, float]]:
    """(a small, blurred copy of the source picture, how many of its pixels one source pixel is across and down).
    Small: no finer than the model's square. Blurred: over about the distance between two vertices (`spacing`, in
    pixels of the square), counting the person's pixels only, so the background does not leak in at the edge. The
    vertices' colours are read from this copy: a vertex stands for the patch around it, not for the one hair or
    glint that happens to sit under it, which the carried colours would otherwise draw round the head as bands."""
    import cv2

    s = math.sqrt(prepared.sx * prepared.sy)       # square pixels per source pixel
    q = min(1.0, s)
    h, w = prepared.alpha.shape
    size = (max(1, int(round(w * q))), max(1, int(round(h * q))))
    shrink = size != (w, h)
    small = cv2.resize(prepared.source, size, interpolation=cv2.INTER_AREA) if shrink else prepared.source
    mask = person(prepared).astype(np.uint8)
    part = (cv2.resize(mask * 255, size, interpolation=cv2.INTER_AREA) > 127) if shrink else mask > 0
    sigma = min(SOFT_MAX, SOFT_SPACING * float(spacing)) * q / s
    out = small.astype(np.float32)
    if sigma >= 0.3 and part.any():
        counts = part.astype(np.float32)
        total = cv2.GaussianBlur(out * counts[..., None], (0, 0), sigma)
        share = cv2.GaussianBlur(counts, (0, 0), sigma)
        out = np.where(share[..., None] > 1e-3, total / np.maximum(share, 1e-3)[..., None], out)
    return out.astype(np.float32), (size[0] / float(w), size[1] / float(h))


def match_colours(own: np.ndarray, photo: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The model's own colours with a gain and an offset per channel, fitted by least squares so that on the known
    vertices they come as near to the photo's as they can. The gain stays within GAIN_LOW to GAIN_HIGH: colours
    that have little to do with the photo's are shifted to its level, not turned over or blown up."""
    out = np.array(own, dtype=np.float64)
    if not known.any():
        return out
    for k in range(out.shape[1]):
        a, b = out[known, k], np.asarray(photo, dtype=np.float64)[known, k]
        spread_a = float(a.var())
        gain = float(((a - a.mean()) * (b - b.mean())).mean() / spread_a) if spread_a > 1.0 else 1.0
        gain = min(GAIN_HIGH, max(GAIN_LOW, gain))
        out[:, k] = gain * (out[:, k] - a.mean()) + b.mean()
    return np.clip(out, 0.0, 255.0)


def edge_weights(vertices: np.ndarray, graph: sparse.Rows) -> np.ndarray:
    """How much each neighbour counts when colours are carried: 1 at the same height, 1 / (1 + LEVEL squared)
    straight above or below."""
    d = vertices[graph.cols] - vertices[graph.rows]
    length = np.linalg.norm(d, axis=1) + 1e-12
    return 1.0 / (1.0 + (LEVEL * np.abs(d[:, 1]) / length) ** 2)


def carry_colours(vertices: np.ndarray, graph: sparse.Rows, colours: np.ndarray, known: np.ndarray,
                  prior: np.ndarray) -> tuple[np.ndarray, int]:
    """The known vertices' colours carried over the rest of the surface. Every other vertex becomes the weighted
    mean of its neighbours (edge_weights) and, with the small weight PRIOR, of its own colour in `prior`: the
    Laplace equation with the known vertices held fixed and a weak pull towards the prior. Next to the photo the
    carried colours win; a dozen edges on the prior does; a part with no known vertex in reach is the prior, which
    is also where the solve starts. PRIOR was set on three real heads: at a third of it the back of the head greyed
    towards what was carried round from the temples, at three times it the model's blotches came back.
    Returns (colours for every vertex, the solver's rounds)."""
    out = np.array(colours, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    unknown = ~known
    if not unknown.any():
        return out, 0
    if not known.any():
        out[unknown] = prior[unknown]
        return out, 0
    weighted = graph.with_values(edge_weights(vertices, graph))
    degree = weighted.times(np.ones(graph.n))[unknown]
    inner = weighted.part(unknown, unknown)
    diagonal = (1.0 + PRIOR) * degree
    b = weighted.part(unknown, known).times(out[known]) + PRIOR * degree[:, None] * prior[unknown]
    solved, used = sparse.solve(lambda x: diagonal[:, None] * x - inner.times(x), b, diagonal, prior[unknown],
                                CARRY_TOLERANCE, CARRY_ROUNDS)
    out[unknown] = solved
    return out, used


# ------------------------------------------------------------------ the texture
def spread(rgb: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """The picture with the person's colours spread outwards over everything else (push and pull over a pyramid
    of half sizes). Nobody sees those pixels, but the graphics card mixes them in when it shrinks the texture, and
    the background's colours would show as a rim."""
    import cv2

    if inside.all() or not inside.any():
        return np.ascontiguousarray(rgb)
    colour = rgb.astype(np.float32) * inside[..., None]
    weight = inside.astype(np.float32)
    levels: list[tuple[np.ndarray, np.ndarray]] = []
    while min(weight.shape) > 1 and not (weight > 0).all():
        levels.append((colour, weight))
        h, w = weight.shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        c = np.zeros((h2 * 2, w2 * 2, 3), dtype=np.float32)
        c[:h, :w] = colour
        k = np.zeros((h2 * 2, w2 * 2), dtype=np.float32)
        k[:h, :w] = weight
        colour = c.reshape(h2, 2, w2, 2, 3).sum(axis=(1, 3))
        weight = k.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
    mean = colour.reshape(-1, 3).sum(axis=0) / max(float(weight.sum()), 1e-6)
    filled = np.where((weight > 0)[..., None], colour / np.maximum(weight, 1e-6)[..., None], mean).astype(np.float32)
    for colour, weight in reversed(levels):
        h, w = weight.shape
        up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        filled = np.where((weight > 0)[..., None], colour / np.maximum(weight, 1e-6)[..., None], up).astype(np.float32)
    return np.clip(np.rint(filled), 0, 255).astype(np.uint8)


def crop_box(prepared: Any, used: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the part of the source picture these points use, with a margin, inside the photo's own
    part of the picture."""
    left, top, right, bottom = prepared.part()
    if not used.size:
        raise NoPhoto("the photo shows too little of the head")
    x0 = max(left, int(math.floor(float(used[:, 0].min()))) - CROP_MARGIN)
    y0 = max(top, int(math.floor(float(used[:, 1].min()))) - CROP_MARGIN)
    x1 = min(right, int(math.ceil(float(used[:, 0].max()))) + CROP_MARGIN + 1)
    y1 = min(bottom, int(math.ceil(float(used[:, 1].max()))) + CROP_MARGIN + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise NoPhoto("the photo shows too little of the head")
    return x0, y0, x1, y1


def texture_coordinates(part: tuple[int, int, int, int], xy: np.ndarray) -> np.ndarray:
    """Where source points sit in the texture cut by `part`: (0, 0) its top left corner, (1, 1) its bottom right.
    A point outside (a corner of the layer that shows nothing) is put on the nearest edge."""
    x0, y0, x1, y1 = part
    uv = np.stack([(xy[:, 0] - x0) / float(x1 - x0), (xy[:, 1] - y0) / float(y1 - y0)], axis=1)
    return np.clip(uv, 0.0, 1.0).astype(np.float32)


def make_texture(prepared: Any, part: tuple[int, int, int, int], limit: int) -> tuple[bytes, tuple[int, int]]:
    """(the JPEG, its (width, height)): that part of the source picture, the person's colours spread over the
    background, shrunk when its longest side is over `limit`. It is never enlarged: the texture is the photo's own
    pixels."""
    import cv2
    from PIL import Image

    x0, y0, x1, y1 = part
    picture = spread(prepared.source[y0:y1, x0:x1], person(prepared, 200)[y0:y1, x0:x1])
    shrink = min(1.0, float(limit) / max(picture.shape[:2]))
    if shrink < 1.0:
        picture = cv2.resize(picture, (max(1, round(picture.shape[1] * shrink)), max(1, round(picture.shape[0] * shrink))),
                             interpolation=cv2.INTER_AREA)
    held = io.BytesIO()
    Image.fromarray(picture).save(held, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    return held.getvalue(), (int(picture.shape[1]), int(picture.shape[0]))


# ------------------------------------------------------------------ the whole stage
Reshape = Callable[[np.ndarray, np.ndarray, View, Any], "np.ndarray | None"]


def _own_colours(mesh: Mesh, graph: sparse.Rows, smoothing: int) -> np.ndarray | None:
    if mesh.colors is None:
        return None
    colours = mesh.colors.astype(np.float64)
    if smoothing > 0:
        for _ in range(COLOUR_ROUNDS):
            colours = 0.5 * colours + 0.5 * sparse.average(graph, colours)
    return colours


def find_camera(vertices: np.ndarray, faces: np.ndarray, camera: Camera, prepared: Any) -> tuple[View, float | None, float | None]:
    """(the picture's camera, how well the outlines overlap where they are compared, and over the head alone).
    A camera the backend is sure of is taken as it is, with no overlaps. NoPhoto when the outlines do not agree."""
    side = float(prepared.side)
    if not camera.fit:
        return View(camera, Fit(), side), None, None
    target, counted, head = fit_masks(prepared, SEARCH_SIDE)
    if not (target & counted).any():
        raise NoPhoto("no person was found in the picture")
    fit, _ = fit_camera(vertices, faces, camera, target, counted, side)
    view = View(camera, fit, side)
    drawn = drawn_outline(vertices, faces, view.sized(SEARCH_SIDE))
    covered = overlap(drawn & counted, target & counted)
    covered_head = overlap(drawn & head, target & head)
    if covered < MIN_OVERLAP or covered_head < MIN_OVERLAP:
        raise NoPhoto(f"the photo did not line up with the head (outlines overlap {covered:.2f}, over the head "
                      f"{covered_head:.2f})", covered, covered_head)
    return view, covered, covered_head


def picture_camera(vertices: np.ndarray, faces: np.ndarray, camera: Camera,
                   prepared: Any) -> tuple[View, float | None, float | None]:
    """Step 2 of the module's description, with the two checks of the picture that come before it: (the picture's
    camera, how well the outlines overlap, and over the head alone). Raises NoPhoto when the picture cannot be
    used."""
    left, top, right, bottom = prepared.part()
    if right - left < 2 or bottom - top < 2:
        raise NoPhoto("the picture shows nothing of the photo itself")
    agreed = map_agrees(prepared)
    if agreed < MAP_OVERLAP:
        raise NoPhoto(f"the picture's cut-out does not sit where the model saw it (overlap {agreed:.2f})")
    return find_camera(vertices, faces, camera, prepared)


def _lay_photo(vertices: np.ndarray, faces: np.ndarray, graph: sparse.Rows, colours: np.ndarray | None, view: View,
               prepared: Any, level: float, scale: float, smoothing: int, limit: int, clock: _Clock) -> dict[str, Any]:
    """Steps 3 to 6 of the module's description, from the camera `view`. Raises NoPhoto when the photo cannot be
    used."""
    calm = taubin(vertices, graph, rounds_for(CALM_ROUNDS, scale))
    clock.mark("calm copy")
    weight, xy = photo_weights(vertices, faces, graph, vertex_normals(calm.astype(np.float32), faces).astype(np.float64),
                               vertex_normals(vertices.astype(np.float32), faces).astype(np.float64), view, prepared)
    del calm
    LAST["weight step"] = largest_step(weight, graph)
    clock.mark("weights")
    known = weight > KNOWN
    if int(known.sum()) < MIN_KNOWN:
        raise NoPhoto("the photo shows too little of the head")
    shows = (weight[faces] > LAYER_FROM).any(axis=1)
    if not shows.any():
        raise NoPhoto("the photo shows too little of the head")
    # what the photo does not show is the model's guess: pull it smooth, then the base flat again, on the level
    # found before the photo came into it. The photo's place on every vertex (xy) stays the one the camera was
    # fitted to
    moved, used = calm_unseen(vertices, graph, weight, UNSEEN_PULL[smoothing] * scale)
    LAST["unseen rounds"] = float(used)
    moved = press_base(moved, level)
    clock.mark("unseen part")
    _, depth = view.project(vertices)
    spacing = mean_edge(vertices, graph) * view.focal / max(float(np.median(depth)), 1e-9)
    soft, (qx, qy) = soft_photo(prepared, spacing)
    photo = sample(soft, xy * np.array([qx, qy])).astype(np.float64)
    del soft
    if colours is None:
        prior = np.repeat(photo[known].mean(axis=0)[None, :], vertices.shape[0], axis=0)
    else:
        prior = match_colours(colours, photo, known)
    carried, used = carry_colours(moved, graph, np.where(known[:, None], photo, prior), known, prior)
    LAST["carry rounds"] = float(used)
    clock.mark("colours")
    final = weight[:, None] * photo + (1.0 - weight[:, None]) * carried
    ids = np.unique(faces[shows])
    renumber = np.full(vertices.shape[0], -1, dtype=np.int64)
    renumber[ids] = np.arange(ids.size)
    part = crop_box(prepared, xy[ids[weight[ids] > LAYER_FROM]])   # the corners that show something decide the crop
    jpeg, size = make_texture(prepared, part, limit)
    layer = Layer(vertex_ids=ids, faces=renumber[faces[shows]].astype(np.uint32), uv=texture_coordinates(part, xy[ids]),
                  alpha=weight[ids].astype(np.float32))
    clock.mark("texture")
    return {"vertices": moved, "colours": np.clip(np.rint(final), 0, 255).astype(np.uint8), "layer": layer,
            "texture": jpeg, "size": size}


def finish_head(mesh: Mesh, prepared: Any | None, options: dict[str, Any] | None = None,
                reshape: Reshape | None = None) -> Head:
    """The head that is written for a backend's mesh. `prepared` is the look's picture (gen3d.prep.Prepared) or
    None; options: `smoothing` and `texture`, each 0 to 3 (see the module's description).

    `reshape`, when given, is called once the picture's camera is known, as reshape(vertices, faces, view,
    prepared), and may return the vertices moved (a stage that corrects the face's shape from the photo). They are
    smoothed a few rounds, the base is pressed flat again, and the weights, colours and texture are worked out on
    them. The camera is found for it also when no photo is asked for (`texture` 0). The head says whether the
    stage's surface was kept (`reshaped`) and, when the stage could not run, why (`reshape_note`: the camera's
    reason, or what the stage raised; a stage that ran and returned nothing knows its own reason). A stage that
    fails costs the head nothing else: the photo is laid on the surface as it was."""
    options = options or {}
    smoothing = min(3, max(0, int(options.get("smoothing", 2))))
    texture = min(3, max(0, int(options.get("texture", 2))))
    clock = _Clock()
    m = clean(mesh)
    faces = m.faces
    graph = sparse.neighbours(faces, m.vertices.shape[0])
    clock.mark("clean")
    scale = round_scale(m.vertices, graph)
    vertices = taubin(m.vertices, graph, rounds_for(TAUBIN_ROUNDS[smoothing], scale))
    level = base_level(vertices)
    vertices = press_base(vertices, level)
    colours = _own_colours(m, graph, smoothing)
    clock.mark("smoothing")
    layer, jpeg, size, covered, covered_head, note = None, None, None, None, None, None
    view, reshaped, reshape_note = None, False, None
    if (texture > 0 or reshape is not None) and m.camera is not None and prepared is not None:
        try:
            view, covered, covered_head = picture_camera(vertices, faces, m.camera, prepared)
        except NoPhoto as e:
            note, covered, covered_head = str(e), e.overlap, e.overlap_head
        except Exception as e:  # noqa: BLE001 - the head is still written, with its colours and no photo
            note = f"the photo could not be laid over the head ({type(e).__name__}: {e})"[:300]
        clock.mark("camera")
    if reshape is not None and view is None:
        # without the camera the stage has nothing to go by. The reason is the photo's, when there is one
        reshape_note = note or ("the backend names no camera" if m.camera is None else "the look has no picture")
    if texture == 0:
        note = None                 # no photo layer was asked for, so there is nothing to explain about it
    if view is not None and reshape is not None:
        try:
            moved = reshape(vertices, faces, view, prepared)
        except Exception as e:  # noqa: BLE001 - the head keeps the surface it had, and still gets its photo
            moved, reshape_note = None, f"{type(e).__name__}: {e}"[:300]
        if moved is not None and moved is not vertices:
            moved = np.asarray(moved, dtype=np.float64)
            if moved.shape == vertices.shape and np.isfinite(moved).all():
                vertices = press_base(taubin(moved, graph, rounds_for(RESHAPED_ROUNDS, scale)), level)
                reshaped = True
            else:
                reshape_note = "the stage returned something that is not this surface"
        clock.mark("reshape")
    if view is not None and texture > 0:
        try:
            laid = _lay_photo(vertices, faces, graph, colours, view, prepared, level, scale, smoothing,
                              TEXTURE_SIDE[texture], clock)
        except NoPhoto as e:
            note = str(e)
        except Exception as e:  # noqa: BLE001 - the head is still written, with its colours and no photo
            note = f"the photo could not be laid over the head ({type(e).__name__}: {e})"[:300]
        else:
            vertices, layer, jpeg, size = laid["vertices"], laid["layer"], laid["texture"], laid["size"]
            colours = laid["colours"]
    v32 = normalise(vertices)
    out_colours = None if colours is None else np.clip(np.rint(colours), 0, 255).astype(np.uint8)
    base = Mesh(vertices=v32, faces=faces.astype(np.uint32), colors=out_colours, normals=vertex_normals(v32, faces))
    clock.mark("normals")
    return Head(base=base, photo=layer, texture=jpeg, texture_size=size, overlap=covered, overlap_head=covered_head,
                note=note, reshaped=reshaped, reshape_note=reshape_note)
