"""Body traits learner (sections 18 and 34): the matching parameters for tattoos, mehndi and nails.

CONTRACT.
- TraitParams: the numbers app.analysis.trait_match reads. A frozen dataclass, so two params compare equal
  after a JSON round trip (to_dict / from_dict) and train can tell when they changed.
  - tattoo_tiers / mehndi_tiers: ((min_inliers, llr), ...) in ascending min_inliers. A verified pair takes the
    llr of the highest tier whose min_inliers it reaches (tier_llr), 0 below the first. Tiers are not fitted in
    this version: they always hold the defaults (TATTOO_TIERS, MEHNDI_TIERS).
  - tattoo_min_inliers: SIFT inliers a tattoo pair needs to count at all (settings.tattoo_min_inliers, raised by
    train when the library's own hard negatives verify at that count). A pair under it counts 0 whatever the
    tiers say, so a raise above 12 also empties the lowest tier.
  - tattoo_single_min: the inliers one verified pair needs to admit a tattoo-only guess by itself (rule 2 of the
    admission); single_pair_ok False means a tattoo-only guess always needs two reference items.
  - nail_color_de: the colour distance under which two polished hands count as the same colour (only tightens).
  - admission_p: kept for the owner-outcome report; not an admission condition for tattoos.
  - source: "default", or "adjusted" when a raise or a tightening applied.
- params() -> TraitParams: what matching uses now: the defaults built from the current settings, with the stored
  adjustments of the last train() applied (only ever stricter than the settings). The defaults while learning is
  off (store.enabled()).
- train(conn) -> dict, status() -> dict: the learner interface of app.learning.train. train reads only kv
  trait_calib (app.analysis.trait_match.calibration_sample, taken on the runner thread) and the owner's outcomes of
  trait-only guesses; it verifies nothing. Three adjustments (amendments 3, 4 and 8; tiers are not fitted):
  - hard-negative raise: when HARD_NEG_MIN or more pairs of different named people in different photos verify at
    settings.tattoo_min_inliers or more, tattoo_min_inliers becomes one above the largest such count (at most
    TATTOO_MIN_INLIERS_CAP). Pairs inside one photo are reported but do not count: they are mostly one arm inside
    two boxes, which the same-photo rule of matching already sets to 0;
  - nails: nail_color_de = clamp(min(q90 of same-person dE, q10 of hard-negative dE), 6, NAIL_COLOR_DE) with
    NAIL_MIN_PAIRS of each; it only tightens;
  - owner outcomes: with OUTCOME_MIN or more confirmed or rejected trait-only guesses, precision under 0.95 sets
    tattoo_single_min to 30 (and admission_p 0.95), under 0.85 single_pair_ok False (and admission_p 0.98).
  When the params change, the items of the open trait and clothes+trait guesses are requeued for the runner.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from app.config import settings
from app.learning import events, store
from app.util import json_loads, now_iso

log = logging.getLogger(__name__)

MODEL = "traits"

# ln LR per verified tattoo pair, by inliers (native crops, probe7: 12+ same 36/900 vs different 2/1500).
TATTOO_TIERS: tuple[tuple[int, float], ...] = ((12, 2.5), (20, 3.0), (30, 3.5))
TATTOO_CEILINGS: tuple[float, ...] = (3.0, 4.0, 5.0)  # kept for reports; tiers are not fitted in this version
MEHNDI_TIERS: tuple[tuple[int, float], ...] = ((30, 1.0), (60, 1.5))
MEHNDI_CEILINGS: tuple[float, ...] = (2.5, 3.5)
NAIL_COLOR_DE = 10.0
TATTOO_SINGLE_MIN = 20
TATTOO_MIN_INLIERS_CAP = 30  # the hard-negative raise never goes above this
HARD_NEG_MIN = 3             # hard negatives at the floor before it is raised
HARD_NEG_PAIRS = 2           # from at least this many different pairs of people
RULES_REV = 2                # in train.signature: a change to these rules recomputes the stored adjustments
NAIL_MIN_PAIRS = 20
NAIL_DE_FLOOR = 6.0
OUTCOME_MIN = 20
PRECISION_SINGLE = 0.95      # under it: one pair needs 30 inliers
PRECISION_PAIRS = 0.85       # under it: two photos are always needed
STRICT_SINGLE_MIN = 30


def _min_inliers() -> int:
    return int(settings.tattoo_min_inliers)


def _min_p() -> float:
    return float(settings.trait_min_p)


@dataclass(frozen=True)
class TraitParams:
    tattoo_tiers: tuple[tuple[int, float], ...] = TATTOO_TIERS
    mehndi_tiers: tuple[tuple[int, float], ...] = MEHNDI_TIERS
    nail_color_de: float = NAIL_COLOR_DE
    admission_p: float = field(default_factory=_min_p)
    source: str = "default"  # "default" | "adjusted"
    tattoo_single_min: int = TATTOO_SINGLE_MIN
    single_pair_ok: bool = True
    tattoo_min_inliers: int = field(default_factory=_min_inliers)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tattoo_tiers"] = [list(t) for t in self.tattoo_tiers]
        d["mehndi_tiers"] = [list(t) for t in self.mehndi_tiers]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraitParams:
        base = default_params()
        def tiers(v: Any, default: tuple[tuple[int, float], ...]) -> tuple[tuple[int, float], ...]:
            if not v:
                return default
            return tuple((int(a), float(b)) for a, b in v)

        return cls(
            tattoo_tiers=tiers(d.get("tattoo_tiers"), base.tattoo_tiers),
            mehndi_tiers=tiers(d.get("mehndi_tiers"), base.mehndi_tiers),
            nail_color_de=float(d.get("nail_color_de", base.nail_color_de)),
            admission_p=float(d.get("admission_p", base.admission_p)),
            source=str(d.get("source", base.source)),
            tattoo_single_min=int(d.get("tattoo_single_min", base.tattoo_single_min)),
            single_pair_ok=bool(d.get("single_pair_ok", base.single_pair_ok)),
            tattoo_min_inliers=int(d.get("tattoo_min_inliers", base.tattoo_min_inliers)),
        )


def default_params() -> TraitParams:
    """The defaults, built from the settings as they are now (tests change settings with reload_settings)."""
    return TraitParams()


def tier_llr(tiers: tuple[tuple[int, float], ...], inliers: int) -> float:
    """The llr of the highest tier whose min_inliers `inliers` reaches; 0 below the first tier."""
    value = 0.0
    for lo, llr in tiers:
        if inliers >= lo:
            value = float(llr)
    return value


def _apply(base: TraitParams, adj: dict[str, Any]) -> TraitParams:
    """The defaults with stored adjustments, each only ever stricter than the current settings."""
    vals: dict[str, Any] = {}
    if isinstance(adj.get("tattoo_min_inliers"), (int, float)):
        vals["tattoo_min_inliers"] = min(TATTOO_MIN_INLIERS_CAP,
                                         max(base.tattoo_min_inliers, int(adj["tattoo_min_inliers"])))
    if isinstance(adj.get("nail_color_de"), (int, float)):
        vals["nail_color_de"] = min(base.nail_color_de, max(NAIL_DE_FLOOR, float(adj["nail_color_de"])))
    if isinstance(adj.get("tattoo_single_min"), (int, float)):
        vals["tattoo_single_min"] = max(base.tattoo_single_min, int(adj["tattoo_single_min"]))
    if adj.get("single_pair_ok") is False:
        vals["single_pair_ok"] = False
    if isinstance(adj.get("admission_p"), (int, float)):
        vals["admission_p"] = max(base.admission_p, float(adj["admission_p"]))
    out = replace(base, **vals) if vals else base
    if out != base:
        out = replace(out, source="adjusted")
    return out


def params() -> TraitParams:
    """The parameters matching uses now."""
    base = default_params()
    try:
        if not store.enabled():
            return base
        model = store.load(MODEL) or {}
    except Exception:  # noqa: BLE001 - learning never blocks matching
        return base
    adj = model.get("adjustments")
    return _apply(base, adj) if isinstance(adj, dict) else base


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def adjustments(sample: dict[str, Any], outcomes: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(adjustments, report) from a calibration sample and the labels of trait-only guess outcomes."""
    adj: dict[str, Any] = {}
    base = int(settings.tattoo_min_inliers)
    hard = [h for h in (sample.get("hard_negatives") or []) if isinstance(h, dict) and h.get("kind") == "tattoo"]
    real = [h for h in hard if not h.get("same_photo") and h.get("named") and not h.get("weak")
            and int(h.get("inliers") or 0) >= base]
    counted = sorted((int(h.get("inliers") or 0) for h in real), reverse=True)
    pairs = {tuple(h.get("persons") or ()) for h in real}
    if len(counted) >= HARD_NEG_MIN and len(pairs) >= HARD_NEG_PAIRS:
        # one above the HARD_NEG_MIN-th largest: a single outlier pair never sets the floor
        raised = min(TATTOO_MIN_INLIERS_CAP, counted[HARD_NEG_MIN - 1] + 1)
        if raised > base:
            adj["tattoo_min_inliers"] = raised
    nails = sample.get("nails") or {}
    pos = [float(x) for x in nails.get("pos") or [] if isinstance(x, (int, float))]
    neg = [float(x) for x in nails.get("neg") or [] if isinstance(x, (int, float))]
    if len(pos) >= NAIL_MIN_PAIRS and len(neg) >= NAIL_MIN_PAIRS:
        de = max(NAIL_DE_FLOOR, min(NAIL_COLOR_DE, _quantile(pos, 0.9), _quantile(neg, 0.1)))
        if de < NAIL_COLOR_DE:
            adj["nail_color_de"] = round(de, 2)
    precision = None
    if len(outcomes) >= OUTCOME_MIN:
        precision = sum(outcomes) / len(outcomes)
        if precision < PRECISION_SINGLE:
            adj["tattoo_single_min"] = STRICT_SINGLE_MIN
            adj["admission_p"] = max(float(settings.trait_min_p), 0.95)
        if precision < PRECISION_PAIRS:
            adj["single_pair_ok"] = False
            adj["admission_p"] = max(float(settings.trait_min_p), 0.98)
    report = {"hard_negatives": len(hard), "hard_negatives_counted": len(counted), "hard_negative_pairs": len(pairs),
              "hard_negatives_same_photo": sum(1 for h in hard if h.get("same_photo")),
              "nail_pairs": len(pos) + len(neg), "outcomes": len(outcomes),
              "precision": round(precision, 4) if precision is not None else None}
    return adj, report


def _outcomes(conn: sqlite3.Connection) -> list[int]:
    out = []
    for r in events.rows(conn, "guess"):
        if r["label"] is None:
            continue
        f = json_loads(r["features"], {}) or {}
        if isinstance(f, dict) and f.get("basis") == "trait":
            out.append(int(r["label"]))
    return out


def _requeue_trait_guesses(conn: sqlite3.Connection) -> int:
    """Queue the items of open trait and clothes+trait guesses for a re-guess on the traits runner."""
    from app.analysis import traits as atraits

    items: set[int] = set()
    for table in ("faces", "bodies"):
        items.update(int(r[0]) for r in conn.execute(
            f"SELECT item_id FROM {table} WHERE assigned_by = 'guess' AND person_id IS NULL AND guess_evidence IS NOT NULL "
            "AND json_valid(guess_evidence) AND json_extract(guess_evidence, '$.basis') IN ('trait', 'clothes+trait')"))
    if not items:
        return 0
    n = atraits.requeue(conn, sorted(items))
    atraits.kick(conn)
    return n


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    """Adjust the matching parameters from the last calibration sample and the owner's outcomes (no verification)."""
    from app.db import kv_get

    sample = json_loads(kv_get(conn, "trait_calib"), {}) or {}
    if not isinstance(sample, dict):
        sample = {}
    before = params()
    adj, report = adjustments(sample, _outcomes(conn))
    store.save(MODEL, {"adjustments": adj, "report": report, "calib_at": kv_get(conn, "trait_calib_at"),
                       "trained_at": now_iso()})
    after = params()
    if after != before:
        try:
            n = _requeue_trait_guesses(conn)
            log.info("learning: trait params changed (%s); requeued %d item(s)", json.dumps(adj, sort_keys=True), n)
        except Exception:  # noqa: BLE001 - the next sweep or guess picks it up
            log.exception("learning: could not requeue trait guesses")
    return status()


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    report = model.get("report") or {}
    p = params()
    return {"source": p.source, "tattoo_min_inliers": p.tattoo_min_inliers, "nail_color_de": p.nail_color_de,
            "tattoo_single_min": p.tattoo_single_min, "single_pair_ok": p.single_pair_ok,
            "admission_p": p.admission_p, "outcomes": int(report.get("outcomes") or 0),
            "precision": report.get("precision"), "hard_negatives": int(report.get("hard_negatives") or 0),
            "hard_negatives_counted": int(report.get("hard_negatives_counted") or 0),
            "pairs": int(report.get("hard_negatives") or 0) + int(report.get("nail_pairs") or 0),
            "calib_at": model.get("calib_at"), "trained_at": model.get("trained_at")}
