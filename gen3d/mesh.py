"""The mesh a backend returns, the head that is written, and what every mesh gets on the way (numpy only).

Axes, for every backend: +Y up, +Z towards the viewer (the face looks along +Z), right-handed, triangles wound
counter-clockwise seen from outside. Nothing here turns a mesh: a backend that works in other axes converts before
it returns.

A backend that knows where its picture was taken from names that camera on the mesh and leaves the mesh in its own
frame, not centred and not scaled: gen3d.finishing needs the two to line up to lay the photo over the front. The
centring and scaling into the unit box happen once, at the very end (`normalise`). `finish` is the short way for a
mesh without a photo: clean, normalise, normals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOOSE_SHARE = 0.02  # a loose part with fewer triangles than this share of the largest part is dropped


@dataclass(frozen=True)
class Camera:
    """Where the picture is assumed to come from, in the mesh's frame: at (0, 0, distance), looking down -z, x to
    the right and y up. A point (x, y, z) lands in a square picture of side S at
    u = S/2 + f x / (distance - z), v = S/2 - f y / (distance - z), f = 0.5 S / tan(fov_y / 2), pixel centres at
    +0.5. `fit`: the backend only guesses the camera, so the finishing searches the scale and shift that make the
    mesh's outline cover the person's outline in the picture."""
    distance: float = 1.9
    fov_y_deg: float = 40.0
    fit: bool = True


@dataclass
class Mesh:
    vertices: np.ndarray               # (n, 3) float32
    faces: np.ndarray                  # (m, 3) uint32
    colors: np.ndarray | None = None   # (n, 3) uint8 per vertex, sRGB as in the picture (gen3d.glb stores them linear)
    normals: np.ndarray | None = None  # (n, 3) float32, unit length; set by finish
    camera: Camera | None = None       # the picture's camera, when the backend knows it (the mesh is in its frame)


@dataclass
class Layer:
    """The photo laid over the front of a head: some of the base's triangles again, with a place in the texture
    for every corner and how much of the photo shows there (0 none, 1 all of it)."""
    vertex_ids: np.ndarray             # (k,) which of the base's vertices these are
    faces: np.ndarray                  # (t, 3) uint32, indices into vertex_ids
    uv: np.ndarray                     # (k, 2) float32, (0, 0) the texture's top left corner
    alpha: np.ndarray                  # (k,) float32 in 0..1


@dataclass
class Head:
    """What is written for a look: the whole surface with a colour per vertex, and, when the photo lined up, the
    photo as a texture on a layer over the front."""
    base: Mesh
    photo: Layer | None = None
    texture: bytes | None = None                    # the JPEG the layer shows
    texture_size: tuple[int, int] | None = None     # (width, height)
    overlap: float | None = None                    # how well the mesh's outline covered the person's (0..1)
    overlap_head: float | None = None               # the same over the head alone, the upper part of the person
    note: str | None = None                         # why there is no photo layer, when one was asked for


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
    return Mesh(vertices=v, faces=f, colors=c, camera=mesh.camera)


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
                colors=None if mesh.colors is None else mesh.colors[used], camera=mesh.camera)


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


def box(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """(the middle of the vertices' box, its longest side). BadMesh when the mesh is flat."""
    v = np.asarray(vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise BadMesh("the mesh is flat")
    return (lo + hi) / 2.0, extent


def clean(mesh: Mesh) -> Mesh:
    """Check the mesh, drop loose bits and turn inside-out triangles the right way round. The mesh stays where it
    is (its frame is the camera's): vertices as float64, faces as int64, no normals yet."""
    m = drop_loose(_checked(mesh))
    v = m.vertices.astype(np.float64)
    middle, extent = box(v)
    faces = m.faces
    if signed_volume((v - middle) / extent, faces) < 0:
        faces = faces[:, [0, 2, 1]]
    return Mesh(vertices=v, faces=faces, colors=m.colors, camera=m.camera)


def normalise(vertices: np.ndarray) -> np.ndarray:
    """The vertices centred and scaled into a unit box, as float32."""
    middle, extent = box(vertices)
    return ((np.asarray(vertices, dtype=np.float64) - middle) / extent).astype(np.float32)


def finish(mesh: Mesh) -> Mesh:
    """Check the mesh, drop loose bits, centre it, scale it into a unit box, turn inside-out triangles the right
    way round and compute the normals. The axes stay as the backend gave them."""
    m = clean(mesh)
    v32 = normalise(m.vertices)
    return Mesh(vertices=v32, faces=m.faces.astype(np.uint32), colors=m.colors, normals=vertex_normals(v32, m.faces))
