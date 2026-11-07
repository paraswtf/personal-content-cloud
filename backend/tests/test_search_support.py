"""Shared helpers and fixtures for the search tests (no tests of its own).

The real SigLIP model is never loaded: `no_engine` (requested by every search test, through
pytestmark or another fixture) makes app.search.service see no engine, and `fake_engine` swaps in FakeEngine, which embeds text as a
sum of fixed random concept vectors and uses the real SigLIP relevance curve."""
from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from app.db import bump_search_version, tx
from app.util import now_iso

DIM = 768
MODEL = "ViT-B-16-SigLIP__webli"
CSRF = {"X-Requested-With": "pcc"}

_CONCEPT_WORDS = ("dog", "cat", "car", "beach", "shirt", "mountain", "red", "orange", "yellow", "green", "blue",
                  "purple", "pink", "brown", "black", "white", "grey")


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seeded(text: str, lo: int, hi: int) -> np.ndarray:
    """A unit vector that is non-zero only in dims lo..hi-1, seeded by the text."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)
    v = np.zeros(DIM)
    v[lo:hi] = np.random.default_rng(seed).standard_normal(hi - lo)
    return unit(v)


# Disjoint dimensions keep the scores exact: concepts are one-hot in dims 0..31, text the fake does
# not know lives in dims 32..63, image noise in dims 64..767. So a concept query has cosine 0 with
# an image that lacks the concept, and unknown text has cosine 0 with every image.
CONCEPTS = {w: unit(np.eye(DIM)[i]) for i, w in enumerate(_CONCEPT_WORDS)}


def hash_vec(text: str) -> np.ndarray:
    return _seeded(text, 32, 64)


def image_vec(*concepts: str, seed: str, weight: float = 0.6) -> np.ndarray:
    """An 'image' embedding: noise plus the named concepts (cosine with a concept query ~0.4..0.6)."""
    v = _seeded(f"noise:{seed}", 64, DIM).astype(np.float64) * 0.8
    for c in concepts:
        v = v + weight * CONCEPTS[c]
    return unit(v)


class FakeEngine:
    def __init__(self) -> None:
        self.fail = False
        self.texts: list[str] = []

    def embed_text(self, text: str, prompt: bool = True) -> np.ndarray:
        if self.fail:
            raise RuntimeError("engine broke")
        self.texts.append(text)
        known = [CONCEPTS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in CONCEPTS]
        if not known:
            return hash_vec(f"text:{text}")
        return unit(np.sum(known, axis=0))

    @staticmethod
    def relevance(cos):
        return 1.0 / (1.0 + np.exp(-(np.asarray(cos, dtype=np.float64) * 117.33076 - 12.932437)))


@pytest.fixture()
def no_engine(monkeypatch):
    from app.config import settings
    from app.search import service

    monkeypatch.setattr(settings, "search_enabled", True)
    monkeypatch.setattr(service, "_siglip_module", lambda: SimpleNamespace(
        available=lambda: False, get_engine=_no_get_engine, MODEL_ID=MODEL))


def _no_get_engine():
    raise AssertionError("the engine must not be loaded when available() is False")


@pytest.fixture()
def fake_engine(monkeypatch, no_engine):
    from app.search import service

    engine = FakeEngine()
    monkeypatch.setattr(service, "_siglip_module", lambda: SimpleNamespace(
        available=lambda: True, get_engine=lambda: engine, MODEL_ID=MODEL))
    return engine


# ------------------------------------------------------------------ rows
def add_fts(conn, item_id: int, *, filename: str = "", text: str = "", doc: str = "", people: str = "",
            labels: str = "") -> None:
    with tx(conn):
        conn.execute("DELETE FROM items_fts WHERE rowid = ?", (item_id,))
        conn.execute("INSERT INTO items_fts(rowid, filename, text, doc, people, labels) VALUES (?, ?, ?, ?, ?, ?)",
                     (item_id, filename, text, doc, people, labels))


def add_embedding(conn, item_id: int, vec: np.ndarray, *, frame_index: int = 0, frame_time_s: float | None = None,
                  version: int = 1, model: str = MODEL) -> None:
    with tx(conn):
        conn.execute(
            "INSERT OR REPLACE INTO item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (item_id, version, model, frame_index, frame_time_s, unit(vec).tobytes(), now_iso()),
        )
        bump_search_version(conn)


def item(conn, **kw):
    """helpers.make_item without files (search never reads them)."""
    from tests.helpers import make_item

    kw.setdefault("write_files", False)
    return make_item(conn, **kw)


def pad(conn, n: int = 12) -> list[int]:
    """Items with plain noise embeddings, so the contrast gate has a realistic median."""
    ids = []
    for k in range(n):
        row = item(conn, filename=f"pad{k}.jpg")
        add_embedding(conn, row["id"], image_vec(seed=f"pad{k}"))
        ids.append(int(row["id"]))
    return ids


def hit_ids(result) -> list[int]:
    return [int(h["id"]) for h in result.hits]
