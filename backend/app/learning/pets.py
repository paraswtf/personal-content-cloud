"""Pet naming learner (sections 18 and 22): one model per species, trained on the owner's named pets.

Why. Pet crops were matched to the centroid of each pet's crops (SigLIP) at a threshold. Two cats of one home, and
the street cats that look like them, sit about as close to one another as one pet's own photos do (on the owner's
library: median 0.79 between the two named cats, 0.81 within one), and every crop the owner took away raised the
pet's threshold (learning.people), until only 24% of the owner's confirmed crops would still match their pet (each
photo held out); the rest joined or started unnamed pets.

What. For every species with a named pet that has at least MIN_CONFIRMED crops the owner assigned, a softmax
regression on [SigLIP of the padded crop, DINOv2 CLS of the stored crop] (each L2-normalised, then standardised over
the training rows), with one class per such pet and one class "another animal":
- a pet's examples: the crops the owner assigned to it;
- "another animal", trusted: the owner's crops they took away from pets (guess_rejections kind 'animal', weight
  HARD_WEIGHT; not the automatic look-alikes that followed them) when taken away from every pet of the model or
  unlike the others (OTHER_PET_SIM), unless the owner has put them on one of the pets since; and the crops the
  owner assigned to any other pet of the species;
- "another animal", far: automatic crops of the species whose top-3 DINOv2 similarity to the pets' examples is
  under FAR_SIM (they look like none of the owner's photos of the pets), at most FAR_LIMIT, newest first.
Classes are balanced by weight; the fit is deterministic. The check folds the examples by photo (FOLDS) and stores
the share of the owner's crops named right at CLAIM_P, those named as another of the pets, and the trusted
negatives that got a pet's name, overall and per pet. A model whose check names under MIN_RIGHT of the owner's
crops right, or more than MAX_WRONG of them wrongly, is not kept (the old matching goes on for that species); in a
kept model, a pet whose own crops are named right under MIN_RIGHT keeps its class but gets no names from it
(pets_of leaves it out, so the old matching goes on for that pet).
On the owner's library (2026-11-19, two named cats): 94% right, none named as the other cat, 1 of 31 crops the
owner had taken away named.

Use (app.analysis.pets): probabilities(species, rows) and pets_of(species). A crop gets a pet's name at
p >= CLAIM_P; a crop the app already put on another named pet moves only at p >= SWITCH_P. With learning switched
off, or before a model exists, current() is None and the old matching runs.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

from app.learning import store
from app.util import now_iso

log = logging.getLogger(__name__)

MODEL = "pets"
RULES_REV = 1
SIG_DIM = 768
IDENT_DIM = 384
MIN_CONFIRMED = 8     # crops the owner assigned to a named pet before it gets a class
MIN_NEGATIVES = 5     # "another animal" examples a model needs
HARD_WEIGHT = 5.0     # a crop the owner took away from a pet counts this much
FAR_SIM = 0.5         # top-3 DINOv2 similarity to the pets' examples under which an automatic crop is another animal
FAR_LIMIT = 2000
OTHER_PET_SIM = 0.6   # a crop taken away from one pet that comes this close to another pet's crops may be that pet
CLAIM_P = 0.85        # a crop gets the pet's name at this probability or more
SWITCH_P = 0.95       # a crop the app put on another named pet moves at this probability or more
MIN_RIGHT = 0.6
MAX_WRONG = 0.05
FOLDS = 8
_FIT = dict(l2=0.01, steps=600, lr=0.1)

_cache: tuple[str, dict[str, dict[str, Any]]] | None = None


def _unit(blob: bytes | None, dim: int) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    if v.shape[0] != dim:
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float64) if n > 0 else None


def features(embedding: bytes | None, ident: bytes | None) -> np.ndarray | None:
    """[SigLIP, DINOv2], each L2-normalised, or None without both."""
    s = _unit(embedding, SIG_DIM)
    d = _unit(ident, IDENT_DIM)
    if s is None or d is None:
        return None
    return np.concatenate([s, d])


# ------------------------------------------------------------------ fitting
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(mu, sd, W, b): standardisation and a class-balanced, weighted, L2-regularised softmax regression."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    xs = (x - mu) / sd
    n, d = xs.shape
    per_class = np.array([w[y == c].sum() for c in range(k)])
    sw = w * (w.sum() / (k * np.maximum(per_class, 1e-9)))[y]
    onehot = np.eye(k)[y]
    W = np.zeros((d, k))
    b = np.zeros(k)
    for _ in range(int(_FIT["steps"])):
        g = (_softmax(xs @ W + b) - onehot) * sw[:, None] / n
        W -= _FIT["lr"] * (xs.T @ g + _FIT["l2"] * W)
        b -= _FIT["lr"] * g.sum(axis=0)
    return mu, sd, W, b


def _predict(m: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return _softmax(((x - m["mu"]) / m["sd"]) @ m["W"] + m["b"])


def _top3(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape[0])
    k = min(3, ref.shape[0])
    for start in range(0, x.shape[0], 2048):
        s = x[start:start + 2048] @ ref.T
        out[start:start + 2048] = -np.sort(-s, axis=1)[:, :k].mean(axis=1)
    return out


def _fold(item_id: int) -> int:
    return (int(item_id) * 2654435761 % 2 ** 32) % FOLDS


# ------------------------------------------------------------------ training
def _species_examples(conn: sqlite3.Connection, species: str) -> dict[str, Any] | None:
    named = {int(r[0]) for r in conn.execute("SELECT id FROM persons WHERE kind = 'pet' AND name IS NOT NULL")}
    rows = conn.execute(
        "SELECT id, item_id, person_id, assigned_by, embedding, ident FROM animals "
        "WHERE species = ? AND embedding IS NOT NULL AND ident IS NOT NULL ORDER BY id", (species,)
    ).fetchall()
    feats: dict[int, np.ndarray] = {}
    for r in rows:
        f = features(r["embedding"], r["ident"])
        if f is not None:
            feats[int(r["id"])] = f
    rows = [r for r in rows if int(r["id"]) in feats]
    by_pet: dict[int, list[int]] = {}
    for r in rows:
        if r["assigned_by"] == "user" and r["person_id"] is not None and int(r["person_id"]) in named:
            by_pet.setdefault(int(r["person_id"]), []).append(int(r["id"]))
    pets = sorted(p for p, ids in by_pet.items() if len(ids) >= MIN_CONFIRMED)
    if not pets:
        return None
    cls_of = {p: i for i, p in enumerate(pets)}
    other = len(pets)
    label: dict[int, int] = {}
    weight: dict[int, float] = {}
    trusted: set[int] = set()
    for r in rows:
        aid = int(r["id"])
        if r["assigned_by"] != "user" or r["person_id"] is None:
            continue
        pid = int(r["person_id"])
        label[aid] = cls_of.get(pid, other)
        weight[aid] = 1.0
        trusted.add(aid)
    pos_ids = [aid for aid, c in label.items() if c < other]
    ref = np.stack([feats[a][SIG_DIM:] for a in pos_ids])
    # Crops the owner took away from pets (their own crops only: the automatic look-alikes that followed them were
    # picked by the look embedding and may be the pet itself). "Not this pet" says nothing about the other pets:
    # such a crop is another animal only when it was taken away from every pet of the model or looks like none of
    # the others (top-3 identity similarity to each other pet's crops under OTHER_PET_SIM).
    marks = ",".join("?" * len(pets))
    taken: dict[int, set[int]] = {}
    for r in conn.execute(
        f"SELECT g.ref_id, g.person_id FROM guess_rejections g JOIN animals a ON a.id = g.ref_id "
        f"WHERE g.kind = 'animal' AND g.followed = 0 AND g.person_id IN ({marks})", pets
    ):
        taken.setdefault(int(r[0]), set()).add(int(r[1]))
    for aid, frm in taken.items():
        if aid not in feats or label.get(aid, other) != other:
            continue
        if aid in label:  # the owner put it on another pet since: another animal for sure, and a hard one
            weight[aid] = HARD_WEIGHT
            continue
        rest = [p for p in pets if p not in frm]
        if any(float(_top3(feats[aid][None, SIG_DIM:],
                           np.stack([feats[a][SIG_DIM:] for a in by_pet[p]]))[0]) >= OTHER_PET_SIM for p in rest):
            label.pop(aid, None)
            weight.pop(aid, None)
            trusted.discard(aid)
            continue
        label[aid] = other
        weight[aid] = HARD_WEIGHT
        trusted.add(aid)
    auto = [int(r["id"]) for r in rows if r["assigned_by"] == "auto" and int(r["id"]) not in label]
    if auto:
        sims = _top3(np.stack([feats[a][SIG_DIM:] for a in auto]), ref)
        far = [a for a, s in zip(auto, sims) if s < FAR_SIM]
        for a in sorted(far, reverse=True)[:FAR_LIMIT]:
            label[a] = other
            weight[a] = 1.0
    if sum(1 for c in label.values() if c == other) < MIN_NEGATIVES:
        return None
    ids = sorted(label)
    item_of = {int(r["id"]): int(r["item_id"]) for r in rows}
    return {
        "pets": pets, "ids": ids, "x": np.stack([feats[a] for a in ids]),
        "y": np.array([label[a] for a in ids]), "w": np.array([weight[a] for a in ids]),
        "items": np.array([item_of[a] for a in ids]), "trusted": np.array([a in trusted for a in ids]),
        "counts": {str(p): len(by_pet[p]) for p in pets},
        "negatives": {"taken_away_or_other_pets": int(sum(1 for a in trusted if label[a] == other)),
                      "far": int(sum(1 for a in ids if label[a] == other and a not in trusted))},
    }


def _check(ex: dict[str, Any]) -> dict[str, Any]:
    """Folds by photo: each fold is named by a model fitted on the others (the examples of its photos left out)."""
    k = len(ex["pets"]) + 1
    other = k - 1
    folds = np.array([_fold(i) for i in ex["items"]])
    pred = np.full(len(ex["ids"]), other)
    conf = np.zeros(len(ex["ids"]))
    for f in range(FOLDS):
        test = folds == f
        train = ~test
        if not test.any() or len(set(ex["y"][train].tolist())) < k:
            continue
        mu, sd, W, b = _fit(ex["x"][train], ex["y"][train], ex["w"][train], k)
        p = _predict({"mu": mu, "sd": sd, "W": W, "b": b}, ex["x"][test])
        pred[test] = p.argmax(axis=1)
        conf[test] = p.max(axis=1)
    named = (pred < other) & (conf >= CLAIM_P)
    pos = ex["y"] < other
    neg = (ex["y"] == other) & ex["trusted"]
    n_pos = int(pos.sum())
    per_pet = {str(p): round(float((named & (ex["y"] == i) & (pred == i)).sum()) / max(1, int((ex["y"] == i).sum())), 4)
               for i, p in enumerate(ex["pets"])}
    # crops of the other pets that the check gave this pet's name, as a share of those crops
    taken_by = {str(p): round(float((named & pos & (ex["y"] != i) & (pred == i)).sum())
                              / max(1, int((pos & (ex["y"] != i)).sum())), 4)
                for i, p in enumerate(ex["pets"])}
    return {
        "per_pet": per_pet,
        "per_pet_taken": taken_by,
        "examples": n_pos,
        "right": round(float((named & pos & (pred == ex["y"])).sum()) / max(1, n_pos), 4),
        "wrong": round(float((named & pos & (pred != ex["y"])).sum()) / max(1, n_pos), 4),
        "trusted_negatives": int(neg.sum()),
        "negatives_named": int((named & neg).sum()),
        "folds": FOLDS,
    }


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    species = [str(r[0]) for r in conn.execute(
        "SELECT DISTINCT a.species FROM animals a JOIN persons p ON p.id = a.person_id "
        "WHERE p.kind = 'pet' AND p.name IS NOT NULL AND a.assigned_by = 'user' ORDER BY 1")]
    kept: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for sp in species:
        ex = _species_examples(conn, sp)
        if ex is None:
            continue
        check = _check(ex)
        k = len(ex["pets"]) + 1
        used = check["right"] >= MIN_RIGHT and check["wrong"] <= MAX_WRONG
        summary[sp] = {"pets": len(ex["pets"]), "check": check, "used": used, **ex["negatives"]}
        if not used:
            log.info("pets: the %s model is not used (check %s)", sp, check)
            continue
        mu, sd, W, b = _fit(ex["x"], ex["y"], ex["w"], k)
        # every pet keeps its class, but only a pet whose own crops the check named right often enough gets names
        # from the model (and loses the old matching); the others go on as before
        trusted = [p for p in ex["pets"] if check["per_pet"][str(p)] >= MIN_RIGHT
                   and check["per_pet_taken"][str(p)] <= MAX_WRONG]
        kept[sp] = {"pets": ex["pets"], "trusted": trusted, "counts": ex["counts"], "check": check,
                    "mu": mu.tolist(), "sd": sd.tolist(), "W": W.tolist(), "b": b.tolist()}
    store.save(MODEL, {"rev": RULES_REV, "trained_at": now_iso(), "species": kept})
    return {"species": summary}


# ------------------------------------------------------------------ use
def current() -> dict[str, dict[str, Any]] | None:
    """{species: model with numpy arrays}, or None with learning off or no model."""
    global _cache
    if not store.enabled():
        return None
    raw = store.load(MODEL)
    if not raw or int(raw.get("rev", 0)) != RULES_REV or not raw.get("species"):
        return None
    stamp = str(raw.get("trained_at"))
    if _cache is None or _cache[0] != stamp:
        models = {}
        for sp, m in raw["species"].items():
            models[sp] = {"pets": [int(p) for p in m["pets"]], "trusted": [int(p) for p in m.get("trusted", m["pets"])],
                          "mu": np.asarray(m["mu"]), "sd": np.asarray(m["sd"]),
                          "W": np.asarray(m["W"]), "b": np.asarray(m["b"])}
        _cache = (stamp, models)
    return _cache[1]


def trained_at() -> str | None:
    raw = store.load(MODEL) or {}
    return raw.get("trained_at")


def pets_of(species: str) -> list[int]:
    """The pets the model names (those that passed their own check)."""
    m = (current() or {}).get(species)
    return list(m["trusted"]) if m else []


def probabilities(species: str, x: np.ndarray) -> tuple[list[int], np.ndarray] | None:
    """(the species' pets, (n, pets + 1) probabilities: one column per pet, the last for another animal)."""
    m = (current() or {}).get(species)
    if m is None or x.size == 0:
        return None
    return list(m["pets"]), _predict(m, np.atleast_2d(x))


def status() -> dict[str, Any]:
    raw = store.load(MODEL) or {}
    return {"trained_at": raw.get("trained_at"), "min_confirmed": MIN_CONFIRMED,
            "species": {sp: {"pets": len(m.get("pets", [])), "named_by_model": len(m.get("trusted", [])),
                             "check": m.get("check")}
                        for sp, m in (raw.get("species") or {}).items()}}
