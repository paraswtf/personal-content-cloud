"""Drawing a mesh's depth from a camera, numpy only (OpenCV for one morphology step in the quick outline).

A marching-cubes head has a hundred thousand and more triangles, most of them a pixel or two across. They are not
drawn one by one: all triangles whose box is at most a few pixels wide are sampled together, one pixel of their
boxes at a time. The few large ones (a coarse mesh, the stand-in's egg) are drawn one by one.
"""
from __future__ import annotations

import math

import numpy as np

CLASSES = (2, 4, 8, 16, 32)   # box sides (in pixels) drawn together, class by class; larger triangles one by one
EDGE = 1e-6                   # a pixel centre this far outside an edge still counts as on it
FINE = 4                      # the quick outline marks vertices in a picture this many times as fine


def depth_buffer(uv: np.ndarray, depth: np.ndarray, faces: np.ndarray, height: int, width: int) -> np.ndarray:
    """The nearest depth at every pixel centre, (height, width) float32, inf where no triangle is.
    uv: (n, 2) pixel coordinates (x, y), the centre of pixel (i, j) at (i + 0.5, j + 0.5); depth: (n,), smaller is
    nearer; faces: (m, 3). Depth is interpolated flat across a triangle in the picture, which is exact enough for
    a head seen from one and a half heads away."""
    zbuf = np.full(int(height) * int(width), np.inf, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0:
        return zbuf.reshape(height, width)
    p = np.asarray(uv, dtype=np.float64)[f]           # (m, 3, 2)
    zz = np.asarray(depth, dtype=np.float64)[f]       # (m, 3)
    lo = np.ceil(p.min(axis=1) - 0.5).astype(np.int64)
    hi = np.floor(p.max(axis=1) - 0.5).astype(np.int64)
    np.maximum(lo, 0, out=lo)
    hi[:, 0] = np.minimum(hi[:, 0], width - 1)
    hi[:, 1] = np.minimum(hi[:, 1], height - 1)
    size = hi - lo + 1
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok = (size > 0).all(axis=1) & (np.abs(area) > 1e-12) & np.isfinite(zz).all(axis=1) & np.isfinite(p).all(axis=(1, 2))
    side = size.max(axis=1)

    def cover(sel: np.ndarray, px: np.ndarray, py: np.ndarray) -> None:
        aa, bb, cc, ar = a[sel], b[sel], c[sel], area[sel]
        w0 = ((bb[:, 0] - px) * (cc[:, 1] - py) - (bb[:, 1] - py) * (cc[:, 0] - px)) / ar
        w1 = ((cc[:, 0] - px) * (aa[:, 1] - py) - (cc[:, 1] - py) * (aa[:, 0] - px)) / ar
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -EDGE) & (w1 >= -EDGE) & (w2 >= -EDGE)
        if not inside.any():
            return
        s = sel[inside]
        z = (w0[inside] * zz[s, 0] + w1[inside] * zz[s, 1] + w2[inside] * zz[s, 2]).astype(np.float32)
        at = (py[inside] - 0.5).astype(np.int64) * int(width) + (px[inside] - 0.5).astype(np.int64)
        np.minimum.at(zbuf, at, z)

    below = 0
    for bound in CLASSES:
        group = np.nonzero(ok & (side > below) & (side <= bound))[0]
        below = bound
        for dy in range(bound):
            tall = group[size[group, 1] > dy]
            if not tall.size:
                break
            for dx in range(bound):
                sel = tall[size[tall, 0] > dx]
                if not sel.size:
                    break
                cover(sel, lo[sel, 0] + dx + 0.5, lo[sel, 1] + dy + 0.5)
    for t in np.nonzero(ok & (side > CLASSES[-1]))[0]:
        xs, ys = np.meshgrid(np.arange(lo[t, 0], hi[t, 0] + 1) + 0.5, np.arange(lo[t, 1], hi[t, 1] + 1) + 0.5)
        cover(np.full(xs.size, t, dtype=np.int64), xs.ravel(), ys.ravel())
    return zbuf.reshape(height, width)


def outline(uv: np.ndarray, depth: np.ndarray, faces: np.ndarray, height: int, width: int) -> np.ndarray:
    """Where the mesh covers a pixel centre: (height, width) bool."""
    return np.isfinite(depth_buffer(uv, depth, faces, height, width))


def quick_outline(uv: np.ndarray, height: int, width: int, gap: float = 1.0) -> np.ndarray:
    """A fast stand-in for `outline` for a dense mesh. Every vertex marks its pixel in a picture FINE times as
    fine, a closing fills the gaps between neighbours (`gap`: about how far apart they land, in pixels of the
    answer), and a pixel of the answer counts when at least half of it is marked. Marking the answer's own pixels
    would make the outline about half a pixel too fat all round, and a search that uses it would shrink the mesh to
    make up for it. Used inside the camera search, where the outline is drawn hundreds of times; the answer is
    checked with the real one."""
    import cv2

    fine = FINE
    h, w = int(height) * fine, int(width) * fine
    p = np.floor(np.asarray(uv, dtype=np.float64) * fine).astype(np.int64)
    ok = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < w) & (p[:, 1] < h)
    mask = np.zeros(h * w, dtype=np.uint8)
    mask[p[ok, 1] * w + p[ok, 0]] = 1
    k = int(min(4 * fine + 1, max(3, 2 * math.ceil(float(gap) * fine / 2.0) + 1)))
    closed = cv2.morphologyEx(mask.reshape(h, w), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return cv2.resize(closed.astype(np.float32), (int(width), int(height)), interpolation=cv2.INTER_AREA) >= 0.5
