"""The mesh a backend returns and what every mesh gets before it is written (numpy only).

Axes, for every backend: +Y up, +Z towards the viewer (the face looks along +Z), right-handed, triangles wound
counter-clockwise seen from outside. `finish` does not turn a mesh: a backend that works in other axes converts
before it returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOOSE_SHARE = 0.02  # a loose part with fewer triangles than this share of the largest part is dropped


@dataclass
class Mesh:
    vertices: np.ndarray               # (n, 3) float32
    faces: np.ndarray                  # (m, 3) uint32
    colors: np.ndarray | None = None   # (n, 3) uint8 per vertex, sRGB as in the picture (gen3d.glb stores them linear)
    normals: np.ndarray | None = None  # (n, 3) float32, unit length; set by finish


class BadMesh(ValueError):
    """The backend returned something that is not a usable triangle mesh."""


def _checked(mesh: Mesh) -> Mesh:
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
        raise BadMesh("the mesh has no vertices")
    if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 1:
        raise BadMesh("the mesh has no triangles")
    f = f.astype(np.int64)
    if int(f.min()) < 0 or int(f.max()) >= v.shape[0]:
        raise BadMesh("a triangle points at a vertex that is not there")
    if not np.isfinite(v).all():
        raise BadMesh("the mesh has a vertex that is not a number")
    c = None
    if mesh.colors is not None:
        c = np.asarray(mesh.colors)
        if c.ndim != 2 or c.shape[0] != v.shape[0] or c.shape[1] < 3:
            raise BadMesh("the colours do not match the vertices")
        if c.dtype != np.uint8:  # floats in 0..1 are taken as such
            c = np.clip(np.asarray(c, dtype=np.float32) * (255.0 if float(c.max(initial=0.0)) <= 1.0 else 1.0), 0, 255)
        c = np.ascontiguousarray(c[:, :3]).astype(np.uint8)
    return Mesh(vertices=v, faces=f, colors=c)


def components(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """A label per vertex: vertices joined by triangles share one (the smallest vertex index of their part)."""
    labels = np.arange(n_vertices, dtype=np.int64)
    f = np.asarray(faces, dtype=np.int64)
    while True:
        low = labels[f].min(axis=1)
        before = labels.copy()
        for k in range(3):
            np.minimum.at(labels, f[:, k], low)
        while True:  # follow the pointers to their ends
            nxt = labels[labels]
            if np.array_equal(nxt, labels):
                break
            labels = nxt
        if np.array_equal(labels, before):
            return labels


def drop_loose(mesh: Mesh, share: float = LOOSE_SHARE) -> Mesh:
    """Without the parts that have fewer triangles than `share` of the largest part, and without unused vertices."""
    labels = components(mesh.faces, mesh.vertices.shape[0])
    face_label = labels[mesh.faces[:, 0]]
    names, counts = np.unique(face_label, return_counts=True)
    keep_names = names[counts >= max(1.0, share * float(counts.max()))]
    keep_face = np.isin(face_label, keep_names)
    faces = mesh.faces[keep_face]
    used = np.zeros(mesh.vertices.shape[0], dtype=bool)
    used[faces.reshape(-1)] = True
    remap = np.full(mesh.vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return Mesh(vertices=mesh.vertices[used], faces=remap[faces],
                colors=None if mesh.colors is None else mesh.colors[used])


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = (vertices[faces[:, k]].astype(np.float64) for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normals per vertex, each triangle counting by its area."""
    v = vertices.astype(np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    length = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.where(length > 1e-12, out / np.maximum(length, 1e-12), np.array([0.0, 0.0, 1.0]))
    return out.astype(np.float32)


def finish(mesh: Mesh) -> Mesh:
    """Check the mesh, drop loose bits, centre it, scale it into a unit box, turn inside-out triangles the right
    way round and compute the normals. The axes stay as the backend gave them."""
    m = drop_loose(_checked(mesh))
    v = m.vertices.astype(np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise BadMesh("the mesh is flat")
    v = (v - (lo + hi) / 2.0) / extent
    faces = m.faces
    if signed_volume(v, faces) < 0:
        faces = faces[:, [0, 2, 1]]
    v32 = v.astype(np.float32)
    return Mesh(vertices=v32, faces=faces.astype(np.uint32), colors=m.colors, normals=vertex_normals(v32, faces))
