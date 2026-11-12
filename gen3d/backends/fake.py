"""A stand-in backend that needs no model: an egg-shaped head on a short neck, coloured from the picture.

It exists so the whole path (job file, child process, mesh clean-up, the GLB, the viewer) can be tested and shown
without PyTorch. It is deterministic: the same picture gives the same mesh, byte for byte.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gen3d.mesh import Mesh

LAT, LON = 24, 40          # the head: 2 * LON * (LAT - 1) = 1,840 triangles
NECK_SIDES = 20            # the neck: 2 * NECK_SIDES side triangles and NECK_SIDES at the bottom
HEAD_RADII = (0.38, 0.50, 0.42)
HEAD_Y = 0.16
NECK_RADIUS, NECK_TOP, NECK_BOTTOM = 0.16, -0.22, -0.52


def _region_colour(rgb: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    part = rgb[int(y0 * h):max(int(y0 * h) + 1, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 1, int(x1 * w))]
    return part.reshape(-1, 3).astype(np.float64).mean(axis=0)


def _head() -> tuple[np.ndarray, np.ndarray]:
    """A UV sphere: one vertex at each pole, LAT - 1 rings of LON between them."""
    verts = [(0.0, 1.0, 0.0)]
    for i in range(1, LAT):
        polar = np.pi * i / LAT
        for j in range(LON):
            around = 2.0 * np.pi * j / LON
            verts.append((np.sin(polar) * np.sin(around), np.cos(polar), np.sin(polar) * np.cos(around)))
    verts.append((0.0, -1.0, 0.0))
    v = np.asarray(verts, dtype=np.float64)
    faces = []
    ring = lambda i, j: 1 + (i - 1) * LON + (j % LON)  # noqa: E731
    for j in range(LON):
        faces.append((0, ring(1, j), ring(1, j + 1)))
        faces.append((len(verts) - 1, ring(LAT - 1, j + 1), ring(LAT - 1, j)))
    for i in range(1, LAT - 1):
        for j in range(LON):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)
            faces.append((a, c, d))
            faces.append((a, d, b))
    return v, np.asarray(faces, dtype=np.int64)


def _neck() -> tuple[np.ndarray, np.ndarray]:
    top = [(NECK_RADIUS * np.sin(2 * np.pi * j / NECK_SIDES), NECK_TOP, NECK_RADIUS * np.cos(2 * np.pi * j / NECK_SIDES))
           for j in range(NECK_SIDES)]
    bottom = [(x, NECK_BOTTOM, z) for x, _y, z in top]
    v = np.asarray(top + bottom + [(0.0, NECK_BOTTOM, 0.0)], dtype=np.float64)
    faces = []
    centre = 2 * NECK_SIDES
    for j in range(NECK_SIDES):
        a, b = j, (j + 1) % NECK_SIDES
        c, d = NECK_SIDES + j, NECK_SIDES + (j + 1) % NECK_SIDES
        faces.append((a, c, d))
        faces.append((a, d, b))
        faces.append((centre, d, c))
    return v, np.asarray(faces, dtype=np.int64)


class FakeBackend:
    name = "fake"
    version = "1"

    def available(self, weights_dir: Path) -> bool:  # noqa: ARG002 - it needs no weights
        return True

    def load(self, weights_dir: Path, device: str, options: dict[str, Any]) -> None:  # noqa: ARG002
        return None

    def generate(self, image: Any, options: dict[str, Any]) -> Mesh:  # noqa: ARG002
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        skin = _region_colour(rgb, 0.35, 0.35, 0.65, 0.65)   # the face sits in the middle of the picture
        hair = _region_colour(rgb, 0.35, 0.05, 0.65, 0.22)   # and the hair above it
        low = _region_colour(rgb, 0.30, 0.80, 0.70, 0.98)    # the neck and shoulders below
        unit, head_faces = _head()
        head = unit * np.asarray(HEAD_RADII) + np.asarray([0.0, HEAD_Y, 0.0])
        # hair on the top and the back of the head, skin on the face: a soft edge between them
        up, back = unit[:, 1], -unit[:, 2]
        hairness = np.clip((np.maximum(up, back * 0.9 + up * 0.5) - 0.35) / 0.25, 0.0, 1.0)[:, None]
        head_colours = skin * (1.0 - hairness) + hair * hairness
        neck, neck_faces = _neck()
        neck_colours = np.repeat((0.7 * skin + 0.3 * low)[None, :], neck.shape[0], axis=0)
        vertices = np.vstack([head, neck]).astype(np.float32)
        faces = np.vstack([head_faces, neck_faces + head.shape[0]]).astype(np.uint32)
        colours = np.clip(np.vstack([head_colours, neck_colours]), 0, 255).astype(np.uint8)
        return Mesh(vertices=vertices, faces=faces, colors=colours)

    def close(self) -> None:
        return None


BACKEND = FakeBackend()
