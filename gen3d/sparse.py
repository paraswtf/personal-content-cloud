"""Sums over a mesh's edges and a small linear solver, numpy only.

The finishing stage needs "for every vertex, the sum over its neighbours" many times (smoothing, carrying colours)
and one solve of a symmetric system. The server's own Python, where the stand-in generator and the tests run, has
no scipy, so this file is the sparse matrix: entries sorted by row, summed per row with np.bincount (measured twice
as fast as np.add.reduceat on a head of 68,000 vertices: 7 ms against 15 ms for one product).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class Rows:
    """A sparse matrix by rows: entry k sits in row `rows[k]`, column `cols[k]`, the rows in rising order."""

    def __init__(self, n: int, rows: np.ndarray, cols: np.ndarray, values: np.ndarray | None = None) -> None:
        self.n = int(n)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.values = None if values is None else np.asarray(values, dtype=np.float64)
        self.count = np.bincount(self.rows, minlength=self.n)[:self.n]   # entries per row

    def times(self, x: np.ndarray) -> np.ndarray:
        """The matrix times x: per row, the sum of x at the row's columns (times the values, when there are any).
        x has one entry per column, or one row of entries per column. The answer is float64."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            picked = x[self.cols]
            return np.bincount(self.rows, weights=picked if self.values is None else picked * self.values,
                               minlength=self.n)
        out = np.empty((self.n, x.shape[1]), dtype=np.float64)
        for k in range(x.shape[1]):
            picked = x[:, k][self.cols]
            out[:, k] = np.bincount(self.rows, weights=picked if self.values is None else picked * self.values,
                                    minlength=self.n)
        return out

    def least(self, x: np.ndarray) -> np.ndarray:
        """Per row, the smallest of x at the row's columns and at the row itself (x: one entry per row; the
        matrix is square). On a mesh's neighbours: every vertex takes the smallest value around it."""
        x = np.asarray(x, dtype=np.float64)
        out = x.copy()
        np.minimum.at(out, self.rows, x[self.cols])
        return out

    def with_values(self, values: np.ndarray) -> "Rows":
        return Rows(self.n, self.rows, self.cols, values)

    def part(self, rows_in: np.ndarray, cols_in: np.ndarray) -> "Rows":
        """The entries whose row is marked in `rows_in` and whose column is marked in `cols_in` (boolean, one per
        row and per column), renumbered among the marked ones."""
        keep = rows_in[self.rows] & cols_in[self.cols]
        new_row = np.cumsum(rows_in) - 1
        new_col = np.cumsum(cols_in) - 1
        return Rows(int(rows_in.sum()), new_row[self.rows[keep]], new_col[self.cols[keep]],
                    None if self.values is None else self.values[keep])


def neighbours(faces: np.ndarray, n: int) -> Rows:
    """Every vertex's neighbours along the triangles' edges, each once: Rows.times(x) is the sum of x over them."""
    f = np.asarray(faces, dtype=np.int64)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    key = np.unique(i[i != j] * int(n) + j[i != j])   # sorted by row, then column
    return Rows(n, key // int(n), key % int(n))


def average(graph: Rows, x: np.ndarray) -> np.ndarray:
    """For every vertex the mean of x over its neighbours; a vertex without any keeps its own value."""
    total = graph.times(x)
    count = graph.count if total.ndim == 1 else graph.count[:, None]
    return np.where(count > 0, total / np.maximum(count, 1), x)


def solve(apply: Callable[[np.ndarray], np.ndarray], b: np.ndarray, diagonal: np.ndarray, start: np.ndarray | None = None,
          tolerance: float = 1e-4, rounds: int = 2000) -> tuple[np.ndarray, int]:
    """Conjugate gradient for a symmetric positive definite system, preconditioned by its diagonal, for several
    right-hand sides at once (the columns of b). `apply(x)` is the matrix times x. Stops when every column's
    residual, each row divided by its diagonal entry, is under `tolerance` times the right-hand side taken the same
    way: a row with a large diagonal (a vertex held hard to its place) then counts like any other, where the plain
    residual would be all about such rows. Returns (x, rounds used)."""
    b = np.asarray(b, dtype=np.float64)
    flat = b.ndim == 1
    if flat:
        b = b[:, None]
    inverse = 1.0 / np.maximum(np.asarray(diagonal, dtype=np.float64), 1e-300)[:, None]
    wrap = (lambda v: apply(v[:, 0])[:, None]) if flat else apply
    x = np.zeros_like(b) if start is None else np.array(start, dtype=np.float64).reshape(b.shape)
    r = b - wrap(x)
    scaled = b * inverse
    limit = tolerance * np.maximum(np.sqrt((scaled * scaled).sum(axis=0)), 1e-30)
    z = r * inverse
    used = 0
    if (np.sqrt((z * z).sum(axis=0)) > limit).any():
        p = z.copy()
        rz = (r * z).sum(axis=0)
        for used in range(1, int(rounds) + 1):
            ap = wrap(p)
            pap = (p * ap).sum(axis=0)
            step = np.where(pap > 0, rz / np.where(pap > 0, pap, 1.0), 0.0)
            x += step * p
            r -= step * ap
            z = r * inverse
            if (np.sqrt((z * z).sum(axis=0)) <= limit).all():
                break
            rz_next = (r * z).sum(axis=0)
            p = z + np.where(rz > 0, rz_next / np.where(rz > 0, rz, 1.0), 0.0) * p
            rz = rz_next
    return (x[:, 0] if flat else x), used
