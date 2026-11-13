"""The relief of the face, read from the photo by a depth model (numpy and OpenCV; onnxruntime only in the default
model runners).

The image-to-3D generator gives a face that is a blob with a bump for the nose. DAViD (Microsoft, ICCV 2025, MIT
licence: "DAViD models and runtime code are licensed under the MIT License") is a pair of models made for people: one
reads relative depth from a photo, the other the tilt of the surface. `refine` uses them to correct the part of the
head the photo sees: the seen vertices move along the camera's rays, so each keeps its place in the photo and a
texture projected from that photo stays where it was.

The files (scripts/setup_3d.sh fetches them, optional): WEIGHTS/david/depth-model-vitb16_384.onnx and
WEIGHTS/david/normal-model-vitb16_384.onnx. Input 1 x 3 x 512 x 512, BGR, 0 to 1. Depth output 1 x 512 x 512,
relative, larger means nearer. Normals output 1 x 3 x 512 x 512, unit length, x to the right, y up, z towards the
camera.

The recipe, on a 512 grid over the square picture the generator was given:
1. the face crop: a square 1.9 face sizes wide around the face, cut from the look's picture at full resolution;
2. the mesh's own depth: a depth buffer from the fitted camera;
3. the region: has mesh depth, inside the cut-out eroded by 4 px, inside the crop;
4. how deep the relief is: from the normals, not from the mesh (which is too flat or too deep by up to a factor of
   two): the distance from the camera is `factor` times the depth model's reading, with the factor the robust slope
   of the tilt the normals give against the depth model's own pixel slopes, kept within 0.6 to 2.2 times what a plain
   fit of the reading to the mesh's depth says;
5. the new depth: the reading's slopes between neighbouring pixels (the mesh's own slope where the reading jumps: the
   chin over the neck, a glasses rim), held to the mesh's depth firmly at the region's edge and barely inside: one
   symmetric positive definite system, solved by conjugate gradient;
6. the shift (new depth minus the mesh's), blurred a little and limited, moves each seen vertex along its ray.

Where the look's square leaves the photo, the app fills the picture by repeating the photo's edge (up to 15 percent
of the area). A depth model reads those streaks as surface, so the caller says which part of the picture is the
photo itself (`inside`): the models are run on a crop cut to it, and nothing outside it moves.

Sizes ("of the head's box") are measured in the scene, from where the vertices fall in the picture and how far they
are, so the mesh may be handed over in its own frame, scaled and shifted against the camera's.

Memory: the generator shares its memory with the app, so the two models are never loaded together. Each is loaded,
run once and let go, and what it held is handed back to the system. Measured on real heads with four threads: one
call takes 5 to 7 s (2 s a model), and the process holds 1.0 to 1.1 GB more than before for the seconds a model
runs (0.7 GB loaded, the rest while it computes), then no more than before. onnxruntime's memory arena is off: with
it the peak is 0.5 GB higher and a run no faster.

Nothing here fails a head: whatever goes wrong, `refine` returns the vertices as they came with `relief: False`
and a note (only the job's own stop request passes through). Smoothing after the move (a few Taubin rounds) is the
caller's.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

GRID = 512                 # the working grid over the generator's square picture
MODEL_SIDE = 512           # the models' input and output side
FOLDER = "david"
DEPTH_FILE = "depth-model-vitb16_384.onnx"
NORMAL_FILE = "normal-model-vitb16_384.onnx"
FACE_ABOVE = 0.08          # the face centre sits this share of the look's picture above its middle (app.analysis.looks)
CROP_FACES = 1.9           # the crop's side in face sizes
MIN_CROP = 64              # a crop under this many pixels is not read
ALPHA_INSIDE = 230         # the cut-out counts as the person above this
ERODE = 9                  # the kernel that takes 4 px off the cut-out's edge
CALM = 0.05                # of the head's box: a 3 x 3 patch whose mesh depth spreads more is a jump, left out of fits
MIN_REGION = 2000          # grid pixels: with less of the face seen, nothing is done
MIN_FIT = 0.3              # the plain fit must explain this share of the mesh depth's spread, or nothing is done
NORMAL_FACING = 0.55       # only pixels whose normal faces the camera this much count for the factor
TRIM, TRIM_ROUNDS = 2.5, 4
MIN_FIT_POINTS = 16        # a line through fewer values says nothing
MIN_CORRELATION = 0.3
FACTOR_LOW, FACTOR_HIGH = 0.6, 2.2
EDGE_PX, PIN_EDGE = 3.0, 4.0
PIN_INSIDE, PIN_BAND, PIN_FALL = 1e-4, 0.5, 5.0
STEEP = 2.0                # scene units of depth per scene unit across: steeper is a jump, not a surface
BLUR_SIGMA = 1.2
LIMIT = 0.09               # of the head's box: no vertex moves further
SEEN = 0.03                # of the head's box: a vertex this close behind the depth buffer counts as seen
CG_TOL, CG_ROUNDS = 1e-6, 4000

Runner = Callable[[np.ndarray], np.ndarray]


# ------------------------------------------------------------------ the models
def model_paths(weights_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    base = Path(weights_dir) / FOLDER
    return base / DEPTH_FILE, base / NORMAL_FILE


def available(weights_dir: str | os.PathLike[str] | None) -> bool:
    """Both model files are there."""
    if not weights_dir:
        return False
    try:
        depth, normals = model_paths(weights_dir)
        return depth.is_file() and normals.is_file()
    except (OSError, TypeError, ValueError):
        return False


def _give_back() -> None:
    """Hand the memory a model has let go of back to the system. The C library keeps it otherwise (measured: about
    750 MB stays with the process after one model), and it counts against the memory the generator may hold."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - another C library, another system: nothing to hand back this way
        pass


def _run_model(path: Path, rgb: np.ndarray) -> np.ndarray:
    """One model on one 512 x 512 RGB picture, on the CPU. The model is loaded for the call and let go after it: the
    two models are never in memory together, neither stays beside the generator, and no session is kept anywhere."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get("GEN3D_THREADS", "4"))
    options.enable_cpu_mem_arena = False
    session, out, failure = None, None, None
    try:
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        bgr = np.ascontiguousarray(rgb[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        out = np.array(session.run(None, {session.get_inputs()[0].name: bgr})[0][0], dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - raised again below, once the model is gone
        failure = e.with_traceback(None)   # its traceback holds the model, through the calls the failure came up
    del session
    gc.collect()
    _give_back()
    if failure is not None:
        raise failure
    return out


def runners(weights_dir: str | os.PathLike[str]) -> tuple[Runner, Runner]:
    """(run_depth, run_normals) on the installed files."""
    depth, normals = model_paths(weights_dir)
    return (lambda rgb: _run_model(depth, rgb)), (lambda rgb: _run_model(normals, rgb))


# ------------------------------------------------------------------ pieces
def photo_part(height: int, width: int,
               inside: tuple[float, float, float, float] | None) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of the part of a height x width picture that is the photo itself: `inside` cut to
    the picture, or the whole picture when `inside` is None."""
    if inside is None:
        return 0.0, 0.0, float(width), float(height)
    left, top, right, bottom = (float(x) for x in inside)
    return max(0.0, left), max(0.0, top), min(float(width), right), min(float(height), bottom)


def face_crop(height: int, width: int, framing: float,
              inside: tuple[float, float, float, float] | None = None) -> tuple[int, int, int]:
    """(left, top, side) in pixels of the square read by the models: CROP_FACES face sizes wide around the face of a
    look's picture, which is `framing` face sizes wide with the face centre FACE_ABOVE of its side above the middle.
    The square stays inside the picture and inside `inside` (the part that is the photo itself): it is made smaller
    when it has to be, and moved as little off the face as that takes. Side 0 when there is no such square."""
    left, top, right, bottom = photo_part(height, width, inside)
    left, top = int(np.ceil(left)), int(np.ceil(top))
    right, bottom = int(np.floor(right)), int(np.floor(bottom))
    if right <= left or bottom <= top:
        return 0, 0, 0
    side = CROP_FACES * float(width) / max(float(framing), 1e-6)
    m = int(min(round(side), right - left, bottom - top))
    cx, cy = 0.5 * width, (0.5 - FACE_ABOVE) * height
    x0 = int(min(max(round(cx - m / 2.0), left), right - m))
    y0 = int(min(max(round(cy - m / 2.0), top), bottom - m))
    return x0, y0, m


def raster(uv: np.ndarray, depth: np.ndarray, faces: np.ndarray, h: int, w: int, max_box: int = 10) -> np.ndarray:
    """A depth buffer (inf where nothing is) of triangles sampled at pixel centres: uv (n, 2) in pixels with centres
    at +0.5, depth (n,). Small triangles are done together, one pixel offset of their boxes at a time; the few large
    ones one by one. A triangle with a corner at or behind the camera is left out."""
    zbuf = np.full(h * w, np.inf, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    p = np.asarray(uv, dtype=np.float64)[f]
    zz = np.asarray(depth, dtype=np.float64)[f]
    lo = np.ceil(p.min(axis=1) - 0.5).astype(np.int64)
    hi = np.floor(p.max(axis=1) - 0.5).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi[:, 0] = np.minimum(hi[:, 0], w - 1)
    hi[:, 1] = np.minimum(hi[:, 1], h - 1)
    size = hi - lo + 1
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok = (size > 0).all(axis=1) & (np.abs(area) > 1e-12) & (zz > 1e-9).all(axis=1) & np.isfinite(p).all(axis=(1, 2))

    def cover(sel: np.ndarray, px: np.ndarray, py: np.ndarray) -> None:
        aa, bb, cc, ar = a[sel], b[sel], c[sel], area[sel]
        w0 = ((bb[:, 0] - px) * (cc[:, 1] - py) - (bb[:, 1] - py) * (cc[:, 0] - px)) / ar
        w1 = ((cc[:, 0] - px) * (aa[:, 1] - py) - (cc[:, 1] - py) * (aa[:, 0] - px)) / ar
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            return
        s = sel[inside]
        z = (w0[inside] * zz[s, 0] + w1[inside] * zz[s, 1] + w2[inside] * zz[s, 2]).astype(np.float32)
        lin = (py[inside] - 0.5).astype(np.int64) * w + (px[inside] - 0.5).astype(np.int64)
        np.minimum.at(zbuf, lin, z)

    small = np.nonzero(ok & (size.max(axis=1) <= max_box))[0]
    for dy in range(max_box):
        rows = small[size[small, 1] > dy]
        if not rows.size:
            break
        for dx in range(max_box):
            sel = rows[size[rows, 0] > dx]
            if not sel.size:
                break
            cover(sel, lo[sel, 0] + dx + 0.5, lo[sel, 1] + dy + 0.5)
    for t in np.nonzero(ok & (size.max(axis=1) > max_box))[0]:
        xs, ys = np.meshgrid(np.arange(lo[t, 0], hi[t, 0] + 1) + 0.5, np.arange(lo[t, 1], hi[t, 1] + 1) + 0.5)
        cover(np.full(xs.size, t), xs.ravel(), ys.ravel())
    return zbuf.reshape(h, w)


_own_raster = raster


def solve(target: np.ndarray, base: np.ndarray, omega: np.ndarray, pin: np.ndarray, steep: float = 0.0,
          tol: float = CG_TOL, max_rounds: int = CG_ROUNDS) -> tuple[np.ndarray, int, bool]:
    """Z over `omega` with (L + diag(pin)) Z = div(slopes) + pin * base: L joins neighbouring pixels of omega (the
    five-point stencil), the slopes between neighbours are the target's, and the base's own where the target's is
    steeper than `steep` (0: never). `pin` holds Z to the base, pixel by pixel. Conjugate gradient with a diagonal
    preconditioner, started at the base. Returns (Z, equal to the base outside omega; the rounds used; whether the
    residual fell under `tol` times the right-hand side)."""
    t = np.asarray(target, dtype=np.float64)
    bz = np.asarray(base, dtype=np.float64)
    om = np.asarray(omega, dtype=bool)
    right = om[:, :-1] & om[:, 1:]
    down = om[:-1, :] & om[1:, :]
    gx = t[:, 1:] - t[:, :-1]
    gy = t[1:, :] - t[:-1, :]
    if steep > 0:
        gx = np.where(np.abs(gx) > steep, bz[:, 1:] - bz[:, :-1], gx)
        gy = np.where(np.abs(gy) > steep, bz[1:, :] - bz[:-1, :], gy)
    gx = np.where(right, gx, 0.0)
    gy = np.where(down, gy, 0.0)
    p = np.where(om, np.asarray(pin, dtype=np.float64), 0.0)
    diag = p.copy()
    diag[:, :-1] += right
    diag[:, 1:] += right
    diag[:-1, :] += down
    diag[1:, :] += down
    rhs = p * np.where(om, bz, 0.0)
    rhs[:, :-1] -= gx
    rhs[:, 1:] += gx
    rhs[:-1, :] -= gy
    rhs[1:, :] += gy
    rhs = np.where(om, rhs, 0.0)

    def apply(z: np.ndarray) -> np.ndarray:
        out = diag * z
        out[:, :-1] -= np.where(right, z[:, 1:], 0.0)
        out[:, 1:] -= np.where(right, z[:, :-1], 0.0)
        out[:-1, :] -= np.where(down, z[1:, :], 0.0)
        out[1:, :] -= np.where(down, z[:-1, :], 0.0)
        return np.where(om, out, 0.0)

    # a pixel of omega with no neighbour in it and no hold has nothing to say: it keeps the base
    free = om & (diag > 0)
    inv = np.where(free, 1.0 / np.maximum(diag, 1e-300), 0.0)
    z = np.where(om, bz, 0.0)
    r = np.where(free, rhs - apply(z), 0.0)
    norm = float(np.sqrt((rhs * rhs).sum()))
    rounds = 0
    converged = norm == 0.0 or float(np.sqrt((r * r).sum())) <= tol * norm
    if not converged:
        s = inv * r
        d = s.copy()
        rs = float((r * s).sum())
        for rounds in range(1, int(max_rounds) + 1):
            ad = np.where(free, apply(d), 0.0)
            dad = float((d * ad).sum())
            if not np.isfinite(dad) or dad <= 0.0:
                break
            step = rs / dad
            z += step * d
            r -= step * ad
            if float(np.sqrt((r * r).sum())) <= tol * norm:
                converged = True
                break
            s = inv * r
            rs_new = float((r * s).sum())
            d = s + (rs_new / rs) * d
            rs = rs_new
    out = bz.copy()
    out[om] = z[om]
    return out, rounds, converged


def affine_fit(reading: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """reference ~ a * reading + b over the mask, trimmed: each round keeps what lies within TRIM deviations of the
    line, the deviation measured about the line itself (so a crowd of wild values on one side cannot empty the
    fit). Returns (a, b, the share of the spread explained); zeros when there is too little to fit."""
    x = np.asarray(reading, dtype=np.float64)[mask]
    y = np.asarray(reference, dtype=np.float64)[mask]
    if x.size < MIN_FIT_POINTS:
        return 0.0, 0.0, 0.0
    keep = np.ones(x.size, dtype=bool)
    a = b = 0.0
    for _ in range(TRIM_ROUNDS):
        xa, ya = x[keep], y[keep]
        mx, my = xa.mean(), ya.mean()
        var = float(((xa - mx) ** 2).sum())
        if var <= 0:
            return 0.0, 0.0, 0.0
        a = float(((xa - mx) * (ya - my)).sum() / var)
        b = float(my - a * mx)
        r = y - (a * x + b)
        sd = 1.4826 * float(np.median(np.abs(r[keep]))) + 1e-12
        keep = np.abs(r) < TRIM * sd
        if int(keep.sum()) < MIN_FIT_POINTS:
            return 0.0, 0.0, 0.0
    r = y[keep] - (a * x[keep] + b)
    spread = float(y[keep].var())
    return a, b, (1.0 - float(r.var()) / spread) if spread > 0 else 0.0


def factor_from_normals(reading: np.ndarray, normals: np.ndarray, inside: np.ndarray,
                        step: float) -> tuple[float, float, int]:
    """The factor that turns the depth model's reading into distance from the camera, from the tilt the normal
    model gives: z towards the camera = alpha * reading, where across one pixel (spanning `step` scene units) z
    changes by -(nx / nz) * step to the right and +(ny / nz) * step downwards. alpha is the trimmed slope of those
    against the reading's own pixel slopes; the factor on distance is -alpha. Returns (factor, the correlation of
    the kept slopes, how many were kept)."""
    d = np.asarray(reading, dtype=np.float64)
    nx, ny, nz = (np.asarray(normals[k], dtype=np.float64) for k in range(3))
    gx = np.zeros_like(d)
    gy = np.zeros_like(d)
    gx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / 2.0
    gy[1:-1, :] = (d[2:, :] - d[:-2, :]) / 2.0
    ok = np.asarray(inside, dtype=bool) & (nz > NORMAL_FACING)
    ok[:, [0, -1]] = False
    ok[[0, -1], :] = False
    if int(ok.sum()) < 64:
        return 0.0, 0.0, 0
    x = np.concatenate([gx[ok], gy[ok]])
    y = np.concatenate([-(nx[ok] / nz[ok]) * step, (ny[ok] / nz[ok]) * step])
    keep = np.ones(x.size, dtype=bool)
    alpha = 0.0
    for _ in range(TRIM_ROUNDS):
        xx = float((x[keep] * x[keep]).sum())
        if xx <= 0:
            return 0.0, 0.0, 0
        alpha = float((x[keep] * y[keep]).sum()) / xx
        r = y - alpha * x
        sd = 1.4826 * float(np.median(np.abs(r[keep]))) + 1e-18
        keep = np.abs(r) < TRIM * sd
    if int(keep.sum()) < 64 or float(x[keep].std()) <= 0 or float(y[keep].std()) <= 0:
        return 0.0, 0.0, 0
    return -alpha, float(np.corrcoef(x[keep], y[keep])[0, 1]), int(keep.sum())


def sample(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of a (h, w) image at pixel coordinates (n, 2), pixel centres at +0.5; zero outside."""
    h, w = image.shape[:2]
    x = np.asarray(xy[:, 0], dtype=np.float64) - 0.5
    y = np.asarray(xy[:, 1], dtype=np.float64) - 0.5
    inside = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0 = np.minimum(np.floor(x).astype(np.int64), max(w - 2, 0))
    y0 = np.minimum(np.floor(y).astype(np.int64), max(h - 2, 0))
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    im = np.asarray(image, dtype=np.float64)
    top = im[y0, x0] * (1.0 - fx) + im[y0, x1] * fx
    bottom = im[y1, x0] * (1.0 - fx) + im[y1, x1] * fx
    return np.where(inside, top * (1.0 - fy) + bottom * fy, 0.0)


def soften(shift: np.ndarray, limit: float) -> np.ndarray:
    """The shift as it is laid on the vertices: blurred a little (BLUR_SIGMA pixels), and nowhere over `limit`."""
    import cv2

    return np.clip(cv2.GaussianBlur(np.asarray(shift, dtype=np.float32), (0, 0), BLUR_SIGMA), -limit, limit)


def head_size(uv: np.ndarray, depth: np.ndarray, focal: float, side: float) -> float:
    """The longest side of the head's box in the units of `depth`, from where its vertices fall in the picture: so
    the sizes this module compares with depths do not hang on the frame the vertices themselves are given in."""
    ok = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-9)
    if not ok.any():
        return 0.0
    across = (uv[ok, 0] - side / 2.0) * depth[ok] / focal
    down = (uv[ok, 1] - side / 2.0) * depth[ok] / focal
    return float(max(np.ptp(across), np.ptp(down), np.ptp(depth[ok])))


def _info(relief: bool = False, note: str = "", **more: Any) -> dict[str, Any]:
    return {"relief": relief, "factor": None, "fit_factor": None, "correlation": None, "moved": 0, "note": note,
            **more}


# ------------------------------------------------------------------ the stage
def refine(vertices: np.ndarray, faces: np.ndarray, *,
           project: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
           camera_position: np.ndarray, focal: float, side: float,
           source_rgb: np.ndarray, alpha: np.ndarray,
           square_to_source: Callable[[np.ndarray], np.ndarray], framing: float,
           inside: tuple[float, float, float, float] | None = None,
           raster: Callable[[np.ndarray, np.ndarray, np.ndarray, int, int], np.ndarray] | None = None,
           run_depth: Runner | None = None, run_normals: Runner | None = None,
           options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Move the seen front of a head along the camera's rays to the relief the depth models read from the photo.

    vertices (n, 3) and faces (m, 3): the head, in the mesh's own frame.
    project(vertices) -> (uv, depth): where the fitted camera sees them. uv (n, 2) in pixels of the square picture
        the generator was given (side `side`, pixel centres at +0.5, the camera's axis through the middle); depth
        (n,) the distance along the camera's axis in scene units: a point x to the right of the axis falls
        focal * x / depth pixels right of the middle. The mesh's frame may be scaled and shifted against the
        scene (a fitted camera that draws p as scale * p + shift): every size here is taken from uv and depth.
    camera_position (3,): the camera's centre in the frame of the vertices. focal: in pixels of that square.
    source_rgb (H, W, 3) uint8: the look's picture as the app cut it: `framing` face sizes wide, the face centre
        FACE_ABOVE of its side above the middle. alpha (H, W) uint8: the person's cut-out at the same size.
    square_to_source(xy) -> xy: points (k, 2) of the square as points of the source picture (same pixel convention).
    inside: (left, top, right, bottom) in pixels of the source picture, the part that comes from the photo itself
        (the rest is filler where the look's square left the photo); None when all of it does. The models read a
        crop cut to it, the region lies inside it, and no vertex whose place in the picture is outside it moves.
    raster(uv, depth, faces, h, w) -> depth buffer, inf where empty (this module's own when None).
    run_depth(rgb) -> (512, 512), larger means nearer; run_normals(rgb) -> (3, 512, 512), x right, y up, z towards
        the camera; both take a 512 x 512 RGB uint8 picture. When None, the installed files under
        options["weights_dir"] are used.
    options: {"relief": 0 turns the stage off, "weights_dir": where the model files are}.

    Returns (vertices, info): the vertices moved (a new array; unseen ones are bit for bit what they were), or the
    very array that came in when nothing was done; info = {"relief", "factor" (distance = factor * reading),
    "fit_factor" (what fitting the reading to the mesh's depth alone says), "correlation" (of the normals' tilts
    with the reading's slopes), "moved" (vertices), "note" (why not, when relief is False)}."""
    options = options or {}
    try:
        if not int(options.get("relief", 1)):
            return vertices, _info(note="turned off")
        if run_depth is None or run_normals is None:
            weights = options.get("weights_dir")
            if not available(weights):
                return vertices, _info(note="the depth models are not installed")
            default_depth, default_normals = runners(weights)
            run_depth = run_depth or default_depth
            run_normals = run_normals or default_normals
        return _refine(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), project,
                       np.asarray(camera_position, dtype=np.float64).reshape(3), float(focal), float(side),
                       np.asarray(source_rgb), np.asarray(alpha), square_to_source, float(framing), inside,
                       raster or _own_raster, run_depth, run_normals, vertices)
    except Exception as e:  # noqa: BLE001 - the relief never fails a head
        if type(e).__name__ == "Stopped":  # the job's own stop request is not this stage's to swallow
            raise
        return vertices, _info(note=f"{type(e).__name__}: {e}"[:200])


def _refine(v: np.ndarray, faces: np.ndarray, project: Callable, cam: np.ndarray, focal: float, side: float,
            source: np.ndarray, alpha: np.ndarray, square_to_source: Callable, framing: float,
            inside: tuple[float, float, float, float] | None, raster_fn: Callable,
            run_depth: Runner, run_normals: Runner, given: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3 or faces.ndim != 2 or faces.shape[1] != 3 or not faces.size:
        return given, _info(note="not a mesh")
    if source.ndim != 3 or source.shape[2] < 3 or alpha.ndim != 2 or alpha.shape != source.shape[:2]:
        return given, _info(note="the picture and its cut-out do not match")
    height, width = source.shape[:2]
    uv, depth = project(v)
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    box = head_size(uv, depth, focal, side)
    if not np.isfinite(box) or box <= 0:
        return given, _info(note="the mesh has no size")

    # 1. the face crop: cut to the part of the picture that is the photo itself
    part = photo_part(height, width, inside)
    x0, y0, m = face_crop(height, width, framing, inside)
    if m < MIN_CROP:
        return given, _info(note="the face is too small in the picture")

    # 2. the mesh's own depth on the grid
    k = GRID / side
    zbuf = np.asarray(raster_fn(uv * k, depth, faces, GRID, GRID), dtype=np.float32)
    have = np.isfinite(zbuf)

    # where each grid pixel lies in the source picture, and in the crop's 512 map
    gy, gx = np.mgrid[0:GRID, 0:GRID]
    square = np.stack([(gx.ravel() + 0.5) / k, (gy.ravel() + 0.5) / k], axis=1)
    src = np.asarray(square_to_source(square), dtype=np.float64).reshape(GRID, GRID, 2)
    sx, sy = src[..., 0], src[..., 1]
    mapx = ((sx - x0) / m * MODEL_SIDE - 0.5).astype(np.float32)
    mapy = ((sy - y0) / m * MODEL_SIDE - 0.5).astype(np.float32)
    a_grid = cv2.remap(np.ascontiguousarray(alpha), (sx - 0.5).astype(np.float32), (sy - 0.5).astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. the region, as far as it is known before the models are run: with too little of it, they are not
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE, ERODE))
    inner = cv2.erode((a_grid > ALPHA_INSIDE).astype(np.uint8), kernel) > 0
    in_photo = (sx >= part[0]) & (sy >= part[1]) & (sx < part[2]) & (sy < part[3])
    in_crop = (mapx >= 0) & (mapy >= 0) & (mapx <= MODEL_SIDE - 1) & (mapy <= MODEL_SIDE - 1)
    three = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(np.where(have, zbuf, -1e9).astype(np.float32), three)
    near = cv2.erode(np.where(have, zbuf, 1e9).astype(np.float32), three)      # the nearest depth around each pixel
    calm = (zmax - near) < CALM * box
    if int((have & inner & in_photo & in_crop & calm).sum()) < MIN_REGION:
        return given, _info(note="too little of the face is seen")

    # the two models read the crop, one after the other
    crop = np.ascontiguousarray(source[y0:y0 + m, x0:x0 + m, :3])
    crop512 = cv2.resize(crop, (MODEL_SIDE, MODEL_SIDE),
                         interpolation=cv2.INTER_AREA if m >= MODEL_SIDE else cv2.INTER_CUBIC)
    reading = np.asarray(run_depth(crop512), dtype=np.float32)
    normals = np.asarray(run_normals(crop512), dtype=np.float32)
    if reading.shape != (MODEL_SIDE, MODEL_SIDE) or normals.shape != (3, MODEL_SIDE, MODEL_SIDE):
        return given, _info(note="the depth models answered in another shape")
    if not np.isfinite(reading).all() or not np.isfinite(normals).all():
        return given, _info(note="the depth models answered with something that is not a number")
    d = cv2.remap(reading, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                  borderValue=float("nan"))
    omega = have & inner & in_photo & in_crop & np.isfinite(d)
    fit_mask = omega & calm
    if int(fit_mask.sum()) < MIN_REGION:
        return given, _info(note="too little of the face is seen")
    d0 = np.where(np.isfinite(d), d, 0.0)

    # 4. how deep: the plain fit as the anchor, the normals as the measure
    fit_factor, _, share = affine_fit(d0, zbuf, fit_mask)
    if share < MIN_FIT or fit_factor == 0.0:
        return given, _info(note="the depth reading does not match the head", fit_factor=round(fit_factor, 4))
    depth_mid = float(np.median(zbuf[fit_mask]))
    corner = np.asarray(square_to_source(np.array([[0.0, 0.0], [side, 0.0], [0.0, side]])), dtype=np.float64)
    per_source = 2.0 * side / max(float(np.linalg.norm(corner[1] - corner[0]) + np.linalg.norm(corner[2] - corner[0])),
                                  1e-9)           # square pixels to one source pixel
    step = (m * per_source / MODEL_SIDE) * depth_mid / focal      # scene units one pixel of the crop's map spans
    a_crop = cv2.resize(np.ascontiguousarray(alpha[y0:y0 + m, x0:x0 + m]), (MODEL_SIDE, MODEL_SIDE),
                        interpolation=cv2.INTER_AREA if m >= MODEL_SIDE else cv2.INTER_LINEAR)
    person = cv2.erode((a_crop > ALPHA_INSIDE).astype(np.uint8), np.ones((ERODE, ERODE), np.uint8)) > 0
    by_normals, correlation, _ = factor_from_normals(reading, normals, person, step)
    if abs(correlation) > MIN_CORRELATION and by_normals * fit_factor > 0:
        factor = float(np.clip(by_normals / fit_factor, FACTOR_LOW, FACTOR_HIGH)) * fit_factor
    else:
        factor = fit_factor

    # 5. the new depth: the reading's slopes, held to the mesh at the region's edge
    target = factor * d0.astype(np.float64)
    dist = cv2.distanceTransform(omega.astype(np.uint8), cv2.DIST_L2, 5)
    pin = np.where(dist < EDGE_PX, PIN_EDGE, PIN_INSIDE + PIN_BAND * np.exp(-((dist - EDGE_PX) / PIN_FALL) ** 2))
    span = depth_mid / (focal * k)                                # scene units one grid pixel spans
    base = np.where(have, zbuf, 0.0).astype(np.float64)
    z, rounds, converged = solve(target, base, omega, pin, steep=STEEP * span)
    if not converged:
        return given, _info(note=f"the relief did not settle in {rounds} rounds", fit_factor=round(fit_factor, 4))

    # 6. the shift, and the seen vertices along their rays
    shift_map = soften(np.where(omega, z - base, 0.0), LIMIT * box)
    placed = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-9)
    where = np.where(placed[:, None], uv, -1.0)       # a vertex the camera cannot place is put outside: it stays
    pix = where * k
    col = np.clip(np.floor(pix[:, 0]), 0, GRID - 1).astype(np.int64)
    row = np.clip(np.floor(pix[:, 1]), 0, GRID - 1).astype(np.int64)
    seen = placed & (depth <= near[row, col] + SEEN * box)
    # a vertex whose place in the picture is not the photo stays, whatever the blur carried over the line
    at = np.asarray(square_to_source(where), dtype=np.float64)
    seen &= (at[:, 0] >= part[0]) & (at[:, 1] >= part[1]) & (at[:, 0] < part[2]) & (at[:, 1] < part[3])
    shift = np.where(seen, sample(shift_map, pix), 0.0)
    moved = shift != 0.0
    if not moved.any():
        return given, _info(note="nothing to move", fit_factor=round(fit_factor, 4))
    out = v.copy()
    ratio = (depth[moved] + shift[moved]) / depth[moved]
    out[moved] = cam[None, :] + (v[moved] - cam[None, :]) * ratio[:, None]
    if not np.isfinite(out[moved]).all():
        return given, _info(note="the relief gave something that is not a number")
    return out, _info(True, factor=round(factor, 4), fit_factor=round(fit_factor, 4),
                      correlation=round(correlation, 3), moved=int(moved.sum()))
