"""Mis-click threshold learner (section 18). For each reason the cut on its quality score that best
separates the photos the owner kept from the ones they confirmed as mis-clicks (balanced accuracy),
searched within [0.25x, 4x] of the default and used only with at least MIN_SIDE examples on each side.
Kept: misclick_source 'user' with misclick 0. Confirmed: misclick_source 'user' with misclick 1, or
trashed while flagged automatically. Which reason an example speaks for is read from its scores
against the default cuts, so it does not depend on the stored reasons (cleared when an item is kept).
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from app.config import settings
from app.learning import store
from app.util import json_loads, now_iso

MODEL = "misclick"
MIN_SIDE = 5
LOW, HIGH = 0.25, 4.0
# reason -> (quality score, threshold key, 'below' flags when score < cut, 'above' when score >= cut)
REASONS: dict[str, tuple[str, str, str]] = {
    "blurry": ("blur_var", "blur_var_min", "below"),
    "dark": ("mean_lum", "dark_mean", "below"),
    "blown_out": ("clipped_frac", "clipped_frac", "above"),
    "pocket": ("accident_p", "accident_min", "above"),
}


def defaults() -> dict[str, float]:
    from app.analysis import quality

    return {
        "blur_var_min": float(settings.blur_var_min),
        "dark_mean": float(quality.DARK_MEAN),
        "clipped_frac": float(quality.CLIPPED_FRAC),
        "accident_min": float(quality.ACCIDENT_MIN),
    }


def _flagged_by(reason: str, scores: dict[str, Any], cuts: dict[str, float]) -> bool:
    score_key, cut_key, side = REASONS[reason]
    v = scores.get(score_key)
    if not isinstance(v, (int, float)):
        return False
    return float(v) < cuts[cut_key] if side == "below" else float(v) >= cuts[cut_key]


def examples(conn: sqlite3.Connection) -> dict[str, tuple[list[float], list[float]]]:
    """reason -> (kept values, confirmed values) of the quality score that reason looks at."""
    rows = conn.execute(
        "SELECT quality_scores, misclick, misclick_source, deleted_at FROM items WHERE kind = 'image' "
        "AND quality_scores IS NOT NULL AND (misclick_source = 'user' OR (deleted_at IS NOT NULL AND misclick = 1))"
    ).fetchall()
    cuts = defaults()
    out: dict[str, tuple[list[float], list[float]]] = {r: ([], []) for r in REASONS}
    for r in rows:
        scores = json_loads(r["quality_scores"], {}) or {}
        kept = r["misclick_source"] == "user" and not r["misclick"]
        confirmed = (r["misclick_source"] == "user" and bool(r["misclick"])) or (
            r["deleted_at"] is not None and bool(r["misclick"]))
        if not (kept or confirmed):
            continue
        for reason, (score_key, _cut_key, _side) in REASONS.items():
            v = scores.get(score_key)
            if not isinstance(v, (int, float)):
                continue
            if kept and _flagged_by(reason, scores, cuts):
                out[reason][0].append(float(v))      # flagged by the defaults, but the owner kept it
            elif confirmed and _flagged_by(reason, scores, cuts):
                out[reason][1].append(float(v))      # flagged and confirmed
    return out


def best_cut(kept: list[float], confirmed: list[float], default: float, side: str) -> float | None:
    """The cut with the best balanced accuracy within [LOW x default, HIGH x default]; ties go to the
    value closest to the default. None below MIN_SIDE examples on either side."""
    if len(kept) < MIN_SIDE or len(confirmed) < MIN_SIDE:
        return None
    lo, hi = default * LOW, default * HIGH
    pos = np.asarray(confirmed, dtype=np.float64)
    neg = np.asarray(kept, dtype=np.float64)
    candidates = sorted({float(v) for v in [*pos, *neg, default] if lo <= v <= hi} | {lo, hi})
    best_t, best_acc = None, -1.0
    for t in candidates:
        if side == "below":
            acc = 0.5 * (float((pos < t).mean()) + float((neg >= t).mean()))
        else:
            acc = 0.5 * (float((pos >= t).mean()) + float((neg < t).mean()))
        if acc > best_acc + 1e-9 or (abs(acc - best_acc) <= 1e-9 and best_t is not None and abs(t - default) < abs(best_t - default)):
            best_t, best_acc = t, acc
    return None if best_t is None else round(best_t, 4)


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    ex = examples(conn)
    cuts = defaults()
    learned: dict[str, float] = {}
    per_reason: dict[str, dict[str, int]] = {}
    for reason, (score_key, cut_key, side) in REASONS.items():
        kept, confirmed = ex[reason]
        per_reason[reason] = {"kept": len(kept), "confirmed": len(confirmed)}
        t = best_cut(kept, confirmed, cuts[cut_key], side)
        if t is not None and abs(t - cuts[cut_key]) > 1e-9:
            learned[cut_key] = t
    total = sum(len(k) + len(c) for k, c in ex.values())
    store.save(MODEL, {"thresholds": learned, "examples": total, "per_reason": per_reason, "trained_at": now_iso()})
    return status()


def thresholds() -> dict[str, float]:
    """The learned cuts with the defaults filled in for reasons that have nothing learned."""
    cuts = defaults()
    if not store.enabled():
        return cuts
    model = store.load(MODEL)
    if model:
        for k, v in (model.get("thresholds") or {}).items():
            if k in cuts and isinstance(v, (int, float)):
                cuts[k] = float(v)
    return cuts


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    return {"examples": int(model.get("examples", 0)), "thresholds": dict(model.get("thresholds") or {}),
            "per_reason": model.get("per_reason", {}), "defaults": defaults()}
