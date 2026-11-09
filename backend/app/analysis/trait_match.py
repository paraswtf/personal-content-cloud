"""Matching body traits against named people, and fusing them with clothes guesses (section 34).

CONTRACT.
- context(conn, item, boxes, cache) -> TraitContext | None: None unless traits are enabled, matching is on, the
  library is ready and an open box of the photo has usable rows (or bare limbs on a current scan, for the tattoo
  absence). guess._compute takes the clothes path, unchanged, on None.
- TraitContext.score_all(face_scores, rejected) -> {(box.kind, box.id): TraitResult}: retrieval and verification
  once per box, within VERIFY_PER_ITEM computed verifications for the whole guess_item call (the budget lives on
  the context; the shared `cache` only holds the reference rows of one run). Rejected and taken people stay in
  the rules (a rejected person's verified tattoo still makes a design shared and still breaks exclusivity) and
  are dropped by TraitResult.without() in guess; people present in the photo and people a face box contradicts
  (guess.FACE_CONTRADICTION) are dropped here.
- TraitResult: per person L(P) (llr), its parts (terms: tattoo, mehndi, nails, neg, plus absence and nails_neg),
  the evidence fragment (traits, against) and the facts admission and tie-breaks read. without(rejected) drops
  people; best() gives (best person, L(best), L(runner-up)).
- fuse(c_win, c_ranked, t, params) -> Candidate | None: the fusion rules of 6.3 as amended (absence veto,
  conflict, agreement, tie-break, trait-only admission). Never changes c_win, c_ranked or t.
- outcome_terms(evidence) -> dict: the features record_outcome keeps for a 'trait' guess.
- trait_targets(conn, person_id, item_id, limit) / targets_for_rows(conn, row_ids, limit) -> item ids with open
  boxes worth a re-guess (cosine only, no verification), nearest first.
- sweep_open(conn) -> items requeued; apply_settings_change(conn) -> guesses withdrawn or requeued;
  after_hide_change(conn, item_ids) -> boxes changed plus items requeued; hide_hits_for_rows / hide_hits_for_hide
  -> rule (c) hits written.
- calibration_sample(conn, force=False) -> dict written to kv trait_calib (hard negatives with their inliers, the
  per-person reference counts the gates admit, the nail dE lists); calibration_due(conn) -> bool.
- hidden_ids(conn, rows, hides=None) -> set of hidden row ids (never verifies). hides_covering(conn, trait_id) ->
  rule (a)/(b) hides in the row's item; hidden_by_lookalike(conn, trait_id) -> rule (c) not overridden by 'shown'.
- person_groups(conn, person_id, *, vault_open) -> {"tattoos", "mehndi", "nails"} in the shape of
  schemas.PersonTraitsOut; trait_count(conn, person_id, vault_open=) -> int.
Imports of app.analysis.guess are for type checking or inside functions, so guess.py can import this module.

Rules (section 5 with the amendments):
- Tattoo: retrieval by DINOv2 cosine (RETRIEVAL_COS, same part family, top RETRIEVAL_TOP, PER_PERSON_TOP per
  person, never the query's photo or a near-duplicate of it), then SIFT verification (trait_sift.passes at
  params.tattoo_min_inliers). A verified pair gives tier_llr(params.tattoo_tiers, inliers) x min(1, kp/KP_FULL),
  0 when either row is 'small'. Same photo: a query row that verifies with another body of its own photo counts
  0; a reference photo where two bodies verify is dropped. Shared design: with several people verifying one
  query row, the strongest keeps 0.75 of its value when its strength (independent verifying photos, +1 with a
  strong reference) is at least twice every other's and no other has a strong reference; otherwise the row counts
  0 for all. Per person: best pair per family, summed, plus support (+SUPPORT_STEP per further verifying photo on
  another date and dup group, up to SUPPORT_MAX); halved without the reference bar (1 strong reference, or weak
  ones from 2 dates); capped at TATTOO_CAP.
- Tattoo absence: a person's track (rows linked by verified pairs) seen on dates 30 or more days apart, with a
  median ink share of 0.15 or more, gives ABSENCE_LLR when the query's scan has both sides of that family bare
  (each segment at least 2 x TRAIT_MIN_LIMB_PX), the query has no tattoo row there and is dated after the track
  was first seen. Rows without an ink share ('ink' or 'ink_frac' in attrs) never give absence.
- Mehndi: the same rules at settings.mehndi_min_inliers within settings.mehndi_max_days (exif dates, not
  forwards), times decay (1 for 3 days, half-life 5 days); cap MEHNDI_CAP.
- Nails: support only, within settings.nails_max_days; colour by rarity and distance, finish and length,
  decay (1 for 2 days, half-life 4 days), further dates 0.25 x decay; cap NAILS_CAP; 0 for all when another named
  person, or another body in the query's or the reference's photo, matches; NAILS_NEG when the person's polish
  within 2 days is clearly another colour (trait-only path only).
- Child band (when ages are read): a body with a child-band face is never a query or reference for nails or
  mehndi, and never a tattoo reference.
- Admission of a trait-only guess (amendment 3): trait-only guesses on; a verified tattoo pair with
  params.tattoo_single_min inliers (when params.single_pair_ok) or verified pairs from 2 or more photos on
  different dates that are not near-duplicates; exclusivity after the same-photo and shared-design rules; the
  reference bar; and a lead of settings.trait_margin_llr. Mehndi alone only with settings.trait_only_mehndi.
  Nails never lead and never admit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from app.analysis import trait_sift
from app.analysis import traits as atraits
from app.config import settings
from app.db import bump_faces_version, kv_get, kv_set, tx
from app.util import json_loads, local_date, now_iso, parse_iso

if TYPE_CHECKING:
    from app.analysis.guess import Box, Candidate
    from app.learning.traits import TraitParams

log = logging.getLogger(__name__)

# Code constants of section 6.8 as amended (settings hold the rest).
RETRIEVAL_COS = 0.45          # DINOv2 cosine a reference needs to be verified at all
RETRIEVAL_TOP = 20            # references verified per query row, at most PER_PERSON_TOP per person
PER_PERSON_TOP = 8             # the probe's 56/90 came from the top 20 regardless of person
VERIFY_PER_ITEM = 120         # verifications per guess_item call, shared by the item's boxes
VERIFY_PER_HIDE = 200
VERIFY_PER_CALIB = 1500
KP_FULL = 300                 # a tattoo term is scaled by min(1, kp / KP_FULL); a 'small' row gives 0
SUPPORT_STEP = 0.5            # each further verifying reference item (other date, other dup group)
SUPPORT_MAX = 1.0
TATTOO_CAP = 6.0
MEHNDI_CAP = 4.0
NAILS_CAP = 1.5
NEG_CAP = -1.4
ABSENCE_LLR = -0.7
NAILS_NEG = -0.5
REGUESS_LIMIT = 50
HIDE_IOU = 0.3
HIDE_BODY_IOU = 0.5
HIDE_COS = RETRIEVAL_COS       # look-alike candidates of a hide: verification decides (same-person native cosines rarely pass 0.7)

SHARED_FACTOR = 0.75          # the strongest claimant of a design several people verify with
SHARED_RATIO = 2.0
PHASH_NEAR = 6                # Hamming distance of two near-duplicate photos
BURST_COS = 0.97              # scene cosine of a burst ...
BURST_S = 120.0               # ... taken this close
_NO_TIME = datetime.min.replace(tzinfo=timezone.utc)
TRACK_ROWS = 60               # rows of one person and family linked into tracks
TRACK_DAYS = 30.0             # a track is persistent when seen on dates this far apart
TRACK_INK = 0.15              # and its median ink share is at least this
MEHNDI_FULL_DAYS, MEHNDI_HALF_DAYS = 3.0, 5.0
NAILS_FULL_DAYS, NAILS_HALF_DAYS = 2.0, 4.0
NAILS_FAR_DE = 25.0           # a clearly different polish colour
NAILS_NEG_DAYS = 2.0
NAILS_FURTHER = 0.25
RARITY_FLOOR = 0.02
RARITY_LOCAL_MIN = 20         # polished rows within the window needed for a local rarity
RARITY_LIBRARY_MIN = 50       # and in the library, else RARITY_FALLBACK
RARITY_FALLBACK = 0.25
FINISH_P = 0.7
FINISH_BONUS = {"art": 0.5, "multi": 0.5, "glitter": 0.4, "french": 0.3}
LENGTH_BONUS = 0.2
MEHNDI_SINGLE_MIN = 60        # a mehndi pair that admits by itself (only with TRAIT_ONLY_MEHNDI)
LEAD_TERM = 2.0               # a trait-only guess needs its leading kind (tattoo, or mehndi) at this term at least
MEHNDI_WINDOW_ROWS = 4000
CALIB_GROWTH = 200            # new rows before another calibration sample
CALIB_EVERY_S = 6 * 3600.0
CALIB_ROWS = 300              # newest usable named rows per kind
CALIB_NAIL_DAYS = 3.0
CALIB_NAIL_NEG_DAYS = 1.0
CALIB_LIST_MAX = 2000
LATELY_DAYS = 60              # person page: mehndi and nails of the last 60 days
LATELY_MAX = 12
PERSON_SOURCE_ROWS = 200      # propagate: the person's newest rows used as sources
HIDE_TARGETS = 10             # after_hide_change: look-alikes requeued per call

SHORT_TERM = ("mehndi", "nails")
MATCHED = ("tattoo", "mehndi")

# The settings that change what matching decides (apply_settings_change hashes them with TRAITS_REV).
RULE_SETTINGS = ("traits_enabled", "trait_matching", "trait_only_guesses", "trait_only_mehndi", "traits_mehndi",
                 "trait_min_body_px",
                 "trait_min_limb_px", "nail_min_hand_px", "tattoo_gate_p", "tattoo_min_inliers", "mehndi_min_inliers",
                 "trait_min_p", "trait_margin_llr", "trait_prior_people", "nails_max_days", "mehndi_max_days")


# ------------------------------------------------------------------ results
@dataclass
class TraitResult:
    """What the traits say about one open box, per candidate person.

    llr: L(P) per person id (tattoo + mehndi + nails + neg, after every rule of section 5).
    terms: per person, the parts of L(P): "tattoo", "mehndi", "nails", "neg" (absence and the nails negative
        together, at least NEG_CAP), and "absence" and "nails_neg" apart.
    evidence: per person, the fragment fuse() merges into a guess's evidence: {"traits": [...], "against": [...]}
        (strongest first, at most 3 positive).
    facts: per person, what admission and tie-breaks read: for k in tattoo and mehndi, f"{k}_term",
        f"{k}_best" (inliers of the best counted pair), f"{k}_items" (independent verifying photos),
        f"{k}_bar" (the reference bar holds), f"{k}_exclusive" (no one else verifies with the box's rows).
    """
    llr: dict[int, float] = field(default_factory=dict)
    terms: dict[int, dict[str, float]] = field(default_factory=dict)
    evidence: dict[int, dict[str, Any]] = field(default_factory=dict)
    facts: dict[int, dict[str, Any]] = field(default_factory=dict)

    def without(self, rejected: set[int]) -> TraitResult:
        """The same result without these people (rejected for the box, or taken by another box of the photo)."""
        keep = lambda d: {p: v for p, v in d.items() if p not in rejected}  # noqa: E731
        return TraitResult(keep(self.llr), keep(self.terms), keep(self.evidence), keep(self.facts))

    def best(self) -> tuple[int | None, float, float]:
        """(best person, L(best), L(runner-up)); (None, 0.0, 0.0) when empty. Ties: the lower person id."""
        ranked = sorted(self.llr.items(), key=lambda kv: (-kv[1], kv[0]))
        if not ranked:
            return None, 0.0, 0.0
        return ranked[0][0], float(ranked[0][1]), float(ranked[1][1]) if len(ranked) > 1 else 0.0


@dataclass
class _Item:
    id: int
    uid: str
    taken_at: str | None
    t: datetime | None
    date: str | None
    short_ok: bool          # exif date, not a forward: usable for short-term kinds
    dup: int | None
    phash: str | None
    content_id: str | None
    companion_of: int | None
    live_video: int | None


@dataclass
class _Row:
    """A usable trait row with what matching reads (never a BLOB)."""
    id: int
    body_id: int
    item_id: int
    kind: str
    part: str
    fam: str
    x: float
    y: float
    w: float
    h: float
    quality: float
    attrs: dict[str, Any]
    vec: np.ndarray | None
    has_detail: bool
    person_id: int | None   # a reference's person (body person, assigned by user or auto); None otherwise
    strong: bool            # the body or its face was assigned by the owner
    child: bool             # the body holds a child-band face (ages read)
    created_at: str = ""

    def __getitem__(self, key: str) -> Any:  # hidden_ids reads rows by column name
        return getattr(self, key)


# ------------------------------------------------------------------ small helpers
def _r2(v: float) -> float:
    return round(float(v), 2)


def _cos(q: np.ndarray, r: np.ndarray) -> float:
    """max(q.crop . r.crop, q.crop . r.mirror): mirror photos and the other arm match too."""
    d = atraits.VEC_DIM
    return float(max(float(q[:d] @ r[:d]), float(q[:d] @ r[d:])))


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _days(a: _Item, b: _Item) -> float | None:
    if a.t is None or b.t is None:
        return None
    return abs((a.t - b.t).total_seconds()) / 86400.0


def _decay(dt: float | None, full: float, half: float, max_days: float) -> float:
    if dt is None or dt > max_days:
        return 0.0
    if dt <= full:
        return 1.0
    return float(0.5 ** ((dt - full) / half))


def mehndi_decay(dt: float | None) -> float:
    return _decay(dt, MEHNDI_FULL_DAYS, MEHNDI_HALF_DAYS, float(settings.mehndi_max_days))


def nails_decay(dt: float | None) -> float:
    return _decay(dt, NAILS_FULL_DAYS, NAILS_HALF_DAYS, float(settings.nails_max_days))


def rarity(f: float) -> float:
    """1.0 for a colour 2% of polished hands wear, 0.35 at the 0.25 fallback, about 0.13 at 60%."""
    return float(math.log(1.0 / max(float(f), RARITY_FLOOR)) / math.log(1.0 / RARITY_FLOOR))


def nail_de(a: Any, b: Any) -> float | None:
    """dE_w = sqrt((dL/2)^2 + da^2 + db^2) between two Lab triples (lightness counts half)."""
    try:
        la, aa, ba = (float(v) for v in a)
        lb, ab, bb = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    return float(math.sqrt(((la - lb) / 2.0) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2))


def _near_colour(a: _Row, b: _Row, de0: float) -> bool:
    d = nail_de(a.attrs.get("lab"), b.attrs.get("lab"))
    return d is not None and d < de0


def posterior(llr: float) -> float:
    """sigmoid(L - ln TRAIT_PRIOR_PEOPLE): a sort key for trait-only guesses, not a calibrated probability."""
    z = float(llr) - math.log(max(1, int(settings.trait_prior_people)))
    return float(1.0 / (1.0 + math.exp(-z)))


def _phash_near(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1") <= PHASH_NEAR
    except ValueError:
        return False


def _ink(attrs: dict[str, Any]) -> float | None:
    v = attrs.get("ink", attrs.get("ink_frac"))
    return float(v) if isinstance(v, (int, float)) else None


def _matching_on() -> bool:
    return bool(settings.traits_enabled and settings.trait_matching)


def _chunks(ids: list[int], n: int = 500):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


# ------------------------------------------------------------------ loading rows
ITEM_COLS = ("i.uid AS i_uid, i.taken_at AS i_taken_at, i.exif_datetime AS i_exif, i.forward AS i_forward, "
             "i.dup_group_id AS i_dup, i.phash AS i_phash, i.content_id AS i_content, i.companion_of AS i_companion, "
             "i.live_video_item_id AS i_live")
ROW_SQL = (
    "SELECT t.id, t.body_id, t.item_id, t.kind, t.part, t.x, t.y, t.w, t.h, t.quality, t.attrs, t.created_at, t.vec, "
    "t.detail IS NOT NULL AS has_detail, b.person_id AS b_person, b.assigned_by AS b_assigned, "
    "f.assigned_by AS f_assigned, f.age_groups AS f_ages, f.det_score AS f_det, f.size_px AS f_size, "
    "f.clarity AS f_clarity, f.sharpness AS f_sharp, " + ITEM_COLS + " "
    "FROM body_traits t JOIN trait_scans s ON s.body_id = t.body_id JOIN bodies b ON b.id = t.body_id "
    "JOIN items i ON i.id = t.item_id LEFT JOIN faces f ON f.id = b.face_id "
)


def _row_sql(usable: str | None = None) -> str:
    """ROW_SQL with the usable-row filter of now (it depends on settings, e.g. TRAITS_MEHNDI), never of import time."""
    return ROW_SQL + "WHERE " + (usable if usable is not None else atraits.usable_sql())


class _Loader:
    """Rows and items of one run, with the child band and the hides applied."""

    def __init__(self, conn: sqlite3.Connection, cache: dict[str, Any] | None):
        self.conn = conn
        self.cache = cache if cache is not None else {}
        self.rev = atraits.ready_rev(conn) or atraits.wanted_rev()
        st = self.cache.setdefault("traits", {})
        if st.get("rev") != self.rev:
            st.clear()
            st["rev"] = self.rev
        self.st = st
        self.items: dict[int, _Item] = st.setdefault("items", {})
        if "ages" not in st:
            try:
                from app.analysis import ages

                st["ages"] = bool(ages.ready(conn))
            except Exception:  # noqa: BLE001 - no ages module: no child band
                st["ages"] = False
        self.ages_ready: bool = st["ages"]

    def _child(self, r: sqlite3.Row) -> bool:
        if not self.ages_ready or r["f_ages"] is None:
            return False
        try:
            from app.analysis import ages, faces

            w = faces.face_weight(r["f_det"], r["f_size"], r["f_clarity"], r["f_sharp"])
            return bool(ages.in_child_band(r["f_ages"], w))
        except Exception:  # noqa: BLE001
            return False

    def _item(self, r: sqlite3.Row) -> _Item:
        iid = int(r["item_id"])
        it = self.items.get(iid)
        if it is None:
            taken = r["i_taken_at"]
            it = _Item(iid, r["i_uid"], taken, parse_iso(taken), local_date(taken, settings.local_timezone) if taken else None,
                       bool(r["i_exif"]) and int(r["i_forward"] or 0) == 0 and bool(taken),
                       r["i_dup"], r["i_phash"], r["i_content"], r["i_companion"], r["i_live"])
            self.items[iid] = it
        return it

    def named(self, person_ids: Iterable[int]) -> set[int]:
        """The ids among these that belong to people with a name (cached per run)."""
        cache: dict[int, bool] = self.st.setdefault("named", {})
        want = sorted({int(p) for p in person_ids if p is not None and int(p) not in cache})
        for chunk in _chunks(want):
            marks = ",".join("?" * len(chunk))
            got = {int(r[0]) for r in self.conn.execute(
                f"SELECT id FROM persons WHERE id IN ({marks}) AND name IS NOT NULL AND TRIM(name) != ''", chunk)}
            for p in chunk:
                cache[p] = p in got
        return {int(p) for p in person_ids if p is not None and cache.get(int(p))}

    def item_info(self, item_id: int) -> _Item | None:
        if int(item_id) not in self.items:
            r = self.conn.execute(f"SELECT id AS item_id, {ITEM_COLS} FROM items i WHERE id = ?", (int(item_id),)).fetchone()
            if r is None:
                return None
            self._item(r)
        return self.items[int(item_id)]

    def rows(self, where: str = "", params: dict[str, Any] | None = None, *, apply_hides: bool = True) -> list[_Row]:
        p = {"ready_rev": self.rev, **(params or {})}
        out: list[_Row] = []
        for r in self.conn.execute(_row_sql() + (f" AND {where}" if where else "") + " ORDER BY t.id", p):
            self._item(r)
            try:
                attrs = json.loads(r["attrs"] or "{}")
                attrs = attrs if isinstance(attrs, dict) else {}
            except (TypeError, ValueError):
                attrs = {}
            named = r["b_person"] is not None and r["b_assigned"] in ("user", "auto")
            out.append(_Row(
                int(r["id"]), int(r["body_id"]), int(r["item_id"]), r["kind"], r["part"], atraits.family(r["part"]),
                float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]), float(r["quality"] or 0.0), attrs,
                atraits.decode_vec(r["vec"]), bool(r["has_detail"]), int(r["b_person"]) if named else None,
                bool(named and (r["b_assigned"] == "user" or r["f_assigned"] == "user")), self._child(r),
                r["created_at"] or ""))
        if apply_hides and out:
            hidden = hidden_ids(self.conn, out)
            out = [x for x in out if x.id not in hidden]
        return out

    def library(self, kind: str) -> list[_Row]:
        """Every usable named row of a kind (references at any date), once per run."""
        key = f"lib:{kind}"
        if key not in self.st:
            self.st[key] = self.rows("t.kind = :kind AND b.person_id IS NOT NULL AND b.assigned_by IN ('user', 'auto')",
                                     {"kind": kind})
        return self.st[key]

    def window(self, kind: str, item: _Item, days: float) -> list[_Row]:
        """Usable rows of a kind (named or not) on photos taken within `days` of the item."""
        if item.t is None:
            return []
        key = f"win:{kind}:{item.id}:{days:g}"
        if key not in self.st:
            from datetime import timedelta

            lo = (item.t - timedelta(days=days)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            hi = (item.t + timedelta(days=days)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            ids = [int(r[0]) for r in self.conn.execute(
                "SELECT id FROM items WHERE +taken_at >= ? AND +taken_at <= ? AND deleted_at IS NULL "
                "AND id IN (SELECT item_id FROM body_traits WHERE kind = ?) ORDER BY taken_at LIMIT ?",
                (lo[:19], hi, kind, MEHNDI_WINDOW_ROWS))]
            rows: list[_Row] = []
            for chunk in _chunks(ids):
                marks = ",".join(str(int(i)) for i in chunk)
                rows.extend(self.rows(f"t.kind = :kind AND t.item_id IN ({marks})", {"kind": kind}))
            self.st[key] = rows
        return self.st[key]

    def bodies_rows(self, body_ids: list[int]) -> list[_Row]:
        if not body_ids:
            return []
        marks = ",".join(str(int(b)) for b in sorted(set(body_ids)))
        return self.rows(f"t.body_id IN ({marks})")

    def item_rows(self, item_id: int) -> list[_Row]:
        key = f"item:{item_id}"
        if key not in self.st:
            self.st[key] = self.rows("t.item_id = :iid", {"iid": int(item_id)})
        return self.st[key]

    def near_dup(self, a: _Item, b: _Item) -> bool:
        """Near-duplicates (5.2) give no evidence: dup group, Live pair, phash, or a burst."""
        if a.id == b.id:
            return True
        if a.dup is not None and a.dup == b.dup:
            return True
        if a.content_id and a.content_id == b.content_id:
            return True
        if a.companion_of == b.id or b.companion_of == a.id or a.live_video == b.id or b.live_video == a.id:
            return True
        if _phash_near(a.phash, b.phash):
            return True
        if a.t is not None and b.t is not None and abs((a.t - b.t).total_seconds()) <= BURST_S:
            from app.analysis.guess import _scene_embeddings

            embs = _scene_embeddings(self.conn, [a.id, b.id], self.cache)
            ea, eb = embs.get(a.id), embs.get(b.id)
            if ea is not None and eb is not None and ea.shape == eb.shape and float(ea @ eb) >= BURST_COS:
                return True
        return False


def _loader_fn(conn: sqlite3.Connection, trait_id: int):
    def load() -> tuple[bytes, float | None]:
        r = conn.execute("SELECT detail, part, attrs FROM body_traits WHERE id = ?", (int(trait_id),)).fetchone()
        if r is None or r["detail"] is None:
            raise LookupError(f"trait {trait_id} has no detail crop")
        try:
            attrs = json.loads(r["attrs"] or "{}")
        except (TypeError, ValueError):
            attrs = {}
        share = attrs.get("share") if isinstance(attrs, dict) and "share" in attrs else atraits.share_for(r["part"])
        return bytes(r["detail"]), share
    return load


class _Verifier:
    """trait_sift.verify through the caches, counting computed verifications against a budget."""

    def __init__(self, conn: sqlite3.Connection, budget: int):
        self.conn = conn
        self.budget = int(budget)
        self.spent = 0

    def features(self, key: tuple[str, int], loader) -> trait_sift.Features | None:
        try:
            return trait_sift.features(key, loader)
        except Exception as e:  # noqa: BLE001 - a broken crop is no evidence
            log.debug("traits: no features for %s: %s", key, e)
            return None

    def verify_keys(self, ka: tuple[str, int], la, kb: tuple[str, int], lb) -> trait_sift.Verification | None:
        hit, v = trait_sift.cached_result(ka, kb)
        if hit:  # a cached pair needs neither its features nor the budget
            return v
        if self.spent >= self.budget:
            return None
        fa = self.features(ka, la)
        fb = self.features(kb, lb)
        if fa is None or fb is None:
            return None
        self.spent += 1
        return trait_sift.verify(fa, fb, key_a=ka, key_b=kb)

    def rows(self, a: _Row, b: _Row) -> trait_sift.Verification | None:
        if not (a.has_detail and b.has_detail):
            return None
        return self.verify_keys(("t", a.id), _loader_fn(self.conn, a.id), ("t", b.id), _loader_fn(self.conn, b.id))

    @property
    def left(self) -> int:
        return max(0, self.budget - self.spent)


# ------------------------------------------------------------------ the context
@dataclass
class TraitContext:
    """Per guess_item call: the item, its open boxes, the shared cache and the params in force."""
    conn: sqlite3.Connection
    item: sqlite3.Row
    boxes: list[Box]
    cache: dict[str, Any] | None
    params: TraitParams
    present: set[int] = field(default_factory=set)
    body_of: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    verifier: _Verifier | None = None

    def score_all(self, face_scores: dict[int, dict[int, float]],
                  rejected: dict[tuple[str, int], set[int]]) -> dict[tuple[str, int], TraitResult]:
        """A TraitResult per open box that has trait evidence."""
        from app.analysis.guess import FACE_CONTRADICTION, FACE_WEAK

        ld = _Loader(self.conn, self.cache)
        if self.verifier is None:
            self.verifier = _Verifier(self.conn, VERIFY_PER_ITEM)
        qitem_row = ld.item_rows(int(self.item["id"]))
        qitem = ld.item_info(int(self.item["id"]))
        if qitem is None:
            return {}
        by_body: dict[int, list[_Row]] = {}
        for r in qitem_row:
            by_body.setdefault(r.body_id, []).append(r)
        queries: dict[tuple[str, int], list[_Row]] = {}
        child_box: dict[tuple[str, int], bool] = {}
        for b in self.boxes:
            key = (b.kind, b.id)
            rows = [r for bid in self.body_of.get(key, []) for r in by_body.get(bid, [])]
            queries[key] = rows
            child_box[key] = any(r.child for r in rows) or _face_child(self.conn, b, ld)
        scorer = _Scorer(ld, self.verifier, self.params, qitem, qitem_row)
        scorer.plan(queries, child_box)
        out: dict[tuple[str, int], TraitResult] = {}
        for b in self.boxes:
            key = (b.kind, b.id)
            res = scorer.result(key, queries[key], child_box[key], self.body_of.get(key, []))
            if b.kind == "face":
                # The face veto, as the clothes path has it, but an unnamed group only vetoes when the face hardly
                # looks like P: a tattooed person's photos split into an unnamed group must not veto that person.
                fs = face_scores.get(b.id) or {}
                named = ld.named([o for o, s in fs.items() if s >= FACE_CONTRADICTION])
                drop = {p for p in res.llr
                        if any(o != p and s >= FACE_CONTRADICTION and (o in named or fs.get(p, 0.0) < FACE_WEAK)
                               for o, s in fs.items())}
                res = res.without(drop)
            res = res.without(self.present)
            if res.llr:
                out[key] = res
        return out


def _face_child(conn: sqlite3.Connection, box: Box, ld: _Loader) -> bool:
    """An open face box in the child band (ages read)."""
    if box.kind != "face" or not ld.ages_ready:
        return False
    r = conn.execute("SELECT age_groups, det_score, size_px, clarity, sharpness FROM faces WHERE id = ?",
                     (box.id,)).fetchone()
    if r is None or r["age_groups"] is None:
        return False
    try:
        from app.analysis import ages, faces

        return bool(ages.in_child_band(r["age_groups"], faces.face_weight(r["det_score"], r["size_px"], r["clarity"],
                                                                           r["sharpness"])))
    except Exception:  # noqa: BLE001
        return False


def _box_bodies(conn: sqlite3.Connection, boxes: list[Box]) -> dict[tuple[str, int], list[int]]:
    out: dict[tuple[str, int], list[int]] = {}
    for b in boxes:
        if b.kind == "body":
            out[(b.kind, b.id)] = [b.id]
        else:
            out[(b.kind, b.id)] = [int(r[0]) for r in conn.execute(
                "SELECT id FROM bodies WHERE face_id = ? ORDER BY id", (b.id,))]
    return out


def context(conn: sqlite3.Connection, item: sqlite3.Row, boxes: list[Box],
            cache: dict[str, Any] | None) -> TraitContext | None:
    if not _matching_on() or not boxes:
        return None
    if item["kind"] != "image" or not item["taken_at"]:
        return None
    rev = atraits.ready_rev(conn)
    if rev is None:
        return None
    body_of = _box_bodies(conn, boxes)
    body_ids = sorted({b for ids in body_of.values() for b in ids})
    if not body_ids:
        return None
    if not _has_rows(conn, body_ids, rev):
        return None
    from app.analysis.guess import _present_people
    from app.learning import traits as ltraits

    return TraitContext(conn, item, boxes, cache, ltraits.params(), present=_present_people(conn, int(item["id"])),
                        body_of=body_of)


def _has_rows(conn: sqlite3.Connection, body_ids: list[int], rev: str) -> bool:
    """One cheap query: a usable row on one of these bodies, or bare limbs on its current scan (tattoo absence)."""
    marks = ",".join(str(int(b)) for b in body_ids)
    hit = conn.execute(
        f"SELECT 1 FROM trait_scans s JOIN bodies b ON b.id = s.body_id JOIN items i ON i.id = s.item_id "
        f"WHERE s.body_id IN ({marks}) AND s.status = 'ok' AND s.rev = :ready_rev AND s.box = {atraits.BOX_SQL} "
        "AND i.deleted_at IS NULL AND +i.nsfw = 0 AND COALESCE(i.category, '') NOT IN ('screenshots', 'documents') "
        "AND (s.bare NOT IN ('', '{}') OR EXISTS (SELECT 1 FROM body_traits t WHERE t.body_id = s.body_id)) LIMIT 1",
        {"ready_rev": rev}).fetchone()
    return hit is not None


def deferred(conn: sqlite3.Connection, item_id: int) -> bool:
    """True when guessing this photo would run trait matching (an open box with usable rows): the serial guess job
    leaves such photos to the traits runner, which verifies within its budget (section 34)."""
    from app.analysis.guess import open_boxes

    if not _matching_on():
        return False
    item = conn.execute("SELECT id, kind, taken_at FROM items WHERE id = ?", (int(item_id),)).fetchone()
    if item is None or item["kind"] != "image" or not item["taken_at"]:
        return False
    rev = atraits.ready_rev(conn)
    boxes = open_boxes(conn, int(item_id))
    if rev is None or not boxes:
        return False
    body_ids = sorted({b for ids in _box_bodies(conn, boxes).values() for b in ids})
    return bool(body_ids) and _has_rows(conn, body_ids, rev)


# ------------------------------------------------------------------ scoring
@dataclass
class _Pair:
    q: _Row
    r: _Row
    cos: float
    same_photo: bool
    v: trait_sift.Verification | None = None
    done: bool = False


class _Scorer:
    """Plans and runs the verifications of one item, then builds each box's TraitResult."""

    def __init__(self, ld: _Loader, verifier: _Verifier, params: TraitParams, qitem: _Item, qitem_rows: list[_Row]):
        self.ld = ld
        self.v = verifier
        self.params = params
        self.qitem = qitem
        self.qitem_rows = qitem_rows
        self.pairs: dict[int, list[_Pair]] = {}   # query row id -> pairs
        self._cos: dict[tuple[int, int], float] = {}

    # ---- retrieval
    def _floor(self, kind: str) -> int:
        return int(self.params.tattoo_min_inliers) if kind == "tattoo" else int(settings.mehndi_min_inliers)

    def _refs_for(self, q: _Row) -> list[_Row]:
        if q.kind == "tattoo":
            return [r for r in self.ld.library("tattoo") if self.ld.items[r.item_id].t is not None]
        rows = self.ld.window("mehndi", self.qitem, float(settings.mehndi_max_days))
        return [r for r in rows if r.person_id is not None and self.ld.items[r.item_id].short_ok]

    def _candidates(self, q: _Row, q_child: bool) -> list[_Pair]:
        if q.vec is None or not q.has_detail:
            return []
        if q.kind == "mehndi" and (q_child or not self.qitem.short_ok):
            return []
        scored: list[tuple[float, _Row]] = []
        for r in self._refs_for(q):
            if r.fam != q.fam or r.item_id == q.item_id or r.vec is None or r.child or r.person_id is None:
                continue
            c = _cos(q.vec, r.vec)
            if c < RETRIEVAL_COS:
                continue
            scored.append((c, r))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        out: list[_Pair] = []
        per: dict[int, int] = {}
        for c, r in scored:
            if len(out) >= RETRIEVAL_TOP:
                break
            if per.get(r.person_id, 0) >= PER_PERSON_TOP:  # type: ignore[arg-type]
                continue
            if self.ld.near_dup(self.qitem, self.ld.items[r.item_id]):
                continue
            per[r.person_id] = per.get(r.person_id, 0) + 1  # type: ignore[index]
            out.append(_Pair(q, r, c, False))
        return out

    def plan(self, queries: dict[tuple[str, int], list[_Row]], child_box: dict[tuple[str, int], bool]) -> None:
        same: list[_Pair] = []
        ref: list[_Pair] = []
        for key, rows in queries.items():
            for q in rows:
                if q.kind not in MATCHED:
                    continue
                cands = self._candidates(q, child_box[key])
                if not cands:
                    self.pairs[q.id] = []
                    continue
                photo = [_Pair(q, o, 1.0, True) for o in self.qitem_rows
                         if o.body_id != q.body_id and o.kind == q.kind and o.fam == q.fam and o.has_detail]
                self.pairs[q.id] = photo + cands
                same.extend(photo)
                ref.extend(cands)
        # The same-photo checks first (they decide whether a row counts at all), then by cosine.
        ref.sort(key=lambda p: (-p.cos, p.q.id, p.r.id))
        for p in [*same, *ref]:
            p.v = self.v.rows(p.q, p.r)
            p.done = True

    # ---- verified kinds (tattoo, mehndi)
    def _pair_value(self, kind: str, q: _Row, r: _Row, v: trait_sift.Verification) -> float:
        if q.attrs.get("small") or r.attrs.get("small"):
            return 0.0
        from app.learning.traits import tier_llr

        if kind == "tattoo":
            kp = q.attrs.get("kp")
            scale = min(1.0, float(kp) / KP_FULL) if isinstance(kp, (int, float)) else 1.0
            return tier_llr(self.params.tattoo_tiers, v.inliers) * scale
        return tier_llr(self.params.mehndi_tiers, v.inliers) * mehndi_decay(_days(self.qitem, self.ld.items[r.item_id]))

    def _independent(self, items: list[_Item]) -> list[_Item]:
        """Photos on different dates and dup groups that are not near-duplicates of each other (in order)."""
        kept: list[_Item] = []
        for it in items:
            if any(it.date == k.date or (it.dup is not None and it.dup == k.dup) or self.ld.near_dup(it, k)
                   for k in kept):
                continue
            kept.append(it)
        return kept

    def _verified(self, kind: str, qrows: list[_Row]) -> dict[str, Any]:
        """Per person: term, entries, facts for one verified kind on one box."""
        floor = self._floor(kind)
        per_row: list[tuple[_Row, dict[int, list[tuple[_Row, trait_sift.Verification, float]]]]] = []
        for q in qrows:
            if q.kind != kind:
                continue
            pairs = self.pairs.get(q.id) or []
            if any(p.same_photo and trait_sift.passes(p.v, floor) for p in pairs):
                continue  # the arm could belong to the other body: whose it is cannot be told
            ok = [p for p in pairs if not p.same_photo and trait_sift.passes(p.v, floor)]
            bodies_by_item: dict[int, set[int]] = {}
            for p in ok:
                bodies_by_item.setdefault(p.r.item_id, set()).add(p.r.body_id)
            ok = [p for p in ok if len(bodies_by_item[p.r.item_id]) < 2]
            by_person: dict[int, list[tuple[_Row, trait_sift.Verification, float]]] = {}
            for p in ok:
                by_person.setdefault(int(p.r.person_id), []).append((p.r, p.v, p.cos))  # type: ignore[arg-type]
            if not by_person:
                continue
            strength: dict[int, int] = {}
            strong: dict[int, bool] = {}
            for pid, lst in by_person.items():
                items = sorted({self.ld.items[r.item_id].id: self.ld.items[r.item_id] for r, _, _ in lst}.values(),
                               key=lambda it: it.id)
                kept: list[_Item] = []
                for it in items:
                    if not any(self.ld.near_dup(it, k) for k in kept):
                        kept.append(it)
                strong[pid] = any(r.strong for r, _, _ in lst)
                strength[pid] = len(kept) + (1 if strong[pid] else 0)
            factor = {pid: 1.0 for pid in by_person}
            if len(by_person) > 1:
                top = sorted(by_person, key=lambda p: (-strength[p], p))[0]
                others = [p for p in by_person if p != top]
                if all(strength[top] >= SHARED_RATIO * strength[o] and not strong[o] for o in others):
                    factor = {p: (SHARED_FACTOR if p == top else 0.0) for p in by_person}
                else:
                    factor = {p: 0.0 for p in by_person}
            for pr in ok:
                self._cos[(q.id, pr.r.id)] = pr.cos
            per_row.append((q, {p: [(r, v, factor[p]) for r, v, _ in lst]
                                for p, lst in by_person.items() if factor[p] > 0}))
        claimants = {p for _, d in per_row for p in d}
        out: dict[int, dict[str, Any]] = {}
        for pid in claimants:
            best_fam: dict[str, tuple[float, dict[str, Any]]] = {}
            items: dict[int, tuple[float, _Item]] = {}
            any_strong = False
            best_inliers = 0
            for q, d in per_row:
                for r, v, fac in d.get(pid, []):
                    val = self._pair_value(kind, q, r, v) * fac
                    if val <= 0:
                        continue  # a small mark or a shared design: no term, no support, no bar
                    it = self.ld.items[r.item_id]
                    best_inliers = max(best_inliers, int(v.inliers))
                    any_strong = any_strong or r.strong
                    if val > items.get(it.id, (-1.0, it))[0]:
                        items[it.id] = (val, it)
                    entry = self._entry(kind, q, r, v, val)
                    cur = best_fam.get(q.fam)
                    if cur is None or (val, v.inliers, -r.id) > (cur[0], cur[1]["inliers"], -cur[1]["ref_trait_id"]):
                        best_fam[q.fam] = (val, entry)
            if not best_fam:
                continue
            ordered = [it for _, it in sorted(items.values(), key=lambda x: (-x[0], x[1].id))]
            kept = self._independent(ordered)
            separate: list[_Item] = []
            for it in ordered:
                if not any(self.ld.near_dup(it, k) for k in separate):
                    separate.append(it)
            n_further = min(2, max(0, len(kept) - 1))
            if kind == "tattoo":
                support = SUPPORT_STEP * n_further
            else:
                support = sum(SUPPORT_STEP * mehndi_decay(_days(self.qitem, it)) for it in kept[1:3])
            support = min(SUPPORT_MAX, support)
            # one owner-confirmed reference, two on different dates, or three separate photos of a named person
            # (one event, one day: a named person's tattoo seen from several sides)
            bar = any_strong or len(kept) >= 2 or (len(separate) >= 3 and pid in self.ld.named([pid]))
            term = sum(v for v, _ in best_fam.values()) + (support if best_fam else 0.0)
            if not bar:
                term *= 0.5
            term = min(TATTOO_CAP if kind == "tattoo" else MEHNDI_CAP, term)
            entries = [e for _, e in sorted(best_fam.values(), key=lambda x: (-x[0], x[1]["trait_id"]))]
            for e in entries:
                e["support"] = n_further
            out[pid] = {"term": term, "entries": entries, "best": best_inliers, "items": len(kept), "bar": bar,
                        "exclusive": claimants == {pid}}
        return out

    def _entry(self, kind: str, q: _Row, r: _Row, v: trait_sift.Verification, val: float) -> dict[str, Any]:
        it = self.ld.items[r.item_id]
        dt = _days(self.qitem, it)
        return {"kind": kind, "part": q.part, "ref_part": r.part, "mirrored": bool(v.mirrored), "trait_id": q.id,
                "ref_trait_id": r.id, "ref_body_id": r.body_id, "ref_item_uid": it.uid, "ref_taken_at": it.taken_at,
                "days_apart": round(dt, 1) if dt is not None else None,
                "ref_before": bool(it.t is not None and self.qitem.t is not None and it.t <= self.qitem.t),
                "cos": round(self._cos.get((q.id, r.id), 0.0), 3), "inliers": int(v.inliers), "support": 0,
                "strong": bool(r.strong), "llr": _r2(val)}

    # ---- nails
    def _nails(self, qrows: list[_Row], q_child: bool, query_bodies: list[int]) -> dict[int, dict[str, Any]]:
        qn = [q for q in qrows if q.kind == "nails" and isinstance(q.attrs.get("lab"), (list, tuple))]
        if not qn or q_child or not self.qitem.short_ok:
            return {}
        days = float(settings.nails_max_days)
        de0 = float(self.params.nail_color_de)
        window = [r for r in self.ld.window("nails", self.qitem, days)
                  if not r.child and isinstance(r.attrs.get("lab"), (list, tuple)) and r.attrs.get("back", True)]
        qb = set(query_bodies)
        blocked = False
        matches: dict[int, list[tuple[float, float, _Row, _Row, float]]] = {}  # pid -> (decayed, terms, q, r, de)
        near_far: dict[int, list[float]] = {}
        for q in qn:
            # another body of the query's photo with the same polish: whose hand it is cannot be told
            for o in window:
                if o.item_id == q.item_id and o.body_id not in qb:
                    d = nail_de(q.attrs["lab"], o.attrs["lab"])
                    if d is not None and d < de0:
                        blocked = True
            for r in window:
                if r.person_id is None or r.item_id == q.item_id or r.body_id in qb:
                    continue
                it = self.ld.items[r.item_id]
                if not it.short_ok or self.ld.near_dup(self.qitem, it):
                    continue
                dt = _days(self.qitem, it)
                d = nail_de(q.attrs["lab"], r.attrs["lab"])
                if d is None or dt is None or dt > days:
                    continue
                if d >= de0:
                    if dt <= NAILS_NEG_DAYS:
                        near_far.setdefault(int(r.person_id), []).append(d)
                    continue
                # another body of the reference's photo with the same polish
                if any(o.item_id == r.item_id and o.body_id != r.body_id and _near_colour(q, o, de0) for o in window):
                    blocked = True
                f = self._nail_share(q, r, window, de0)
                t = rarity(f) * max(0.0, 1.0 - d / de0)
                fin_q, fin_r = q.attrs.get("finish"), r.attrs.get("finish")
                if (fin_q and fin_q == fin_r and float(q.attrs.get("finish_p") or 0) >= FINISH_P
                        and float(r.attrs.get("finish_p") or 0) >= FINISH_P):
                    t += FINISH_BONUS.get(str(fin_q), 0.0)
                if q.attrs.get("length") and q.attrs.get("length") == r.attrs.get("length"):
                    t += LENGTH_BONUS
                matches.setdefault(int(r.person_id), []).append((nails_decay(dt), t, q, r, d))
        out: dict[int, dict[str, Any]] = {}
        if blocked or len(matches) > 1:
            matches = {}
        for pid, lst in matches.items():
            best = max(lst, key=lambda m: (m[0] * m[1], -m[3].id))
            bdate = self.ld.items[best[3].item_id].date
            further: dict[str, float] = {}
            for dec, _, _, r, _ in lst:
                date = self.ld.items[r.item_id].date
                if date and date != bdate:
                    further[date] = max(further.get(date, 0.0), dec)
            total = best[0] * best[1] + sum(NAILS_FURTHER * dec for dec in further.values())
            total = min(NAILS_CAP, total)
            q, r = best[2], best[3]
            it = self.ld.items[r.item_id]
            dt = _days(self.qitem, it)
            entry = {"kind": "nails", "part": q.part, "trait_id": q.id, "ref_trait_id": r.id, "ref_body_id": r.body_id,
                     "ref_item_uid": it.uid, "ref_taken_at": it.taken_at,
                     "days_apart": round(dt, 1) if dt is not None else None,
                     "ref_before": bool(it.t is not None and self.qitem.t is not None and it.t <= self.qitem.t),
                     "color": q.attrs.get("color"), "hex": q.attrs.get("hex"), "finish": q.attrs.get("finish"),
                     "length": q.attrs.get("length"),
                     "de": round(best[4], 1), "decay": round(best[0], 2), "llr": _r2(total)}
            out[pid] = {"term": total, "entries": [entry] if total > 0 else [], "neg": 0.0}
        for pid, ds in near_far.items():
            if pid in matches or not any(d > NAILS_FAR_DE for d in ds):
                continue
            out.setdefault(pid, {"term": 0.0, "entries": [], "neg": 0.0})["neg"] = NAILS_NEG
        return out

    def _nail_share(self, q: _Row, r: _Row, window: list[_Row], de0: float) -> float:
        """f: the share of other polished hands near the query's colour (bodies other than the query and P)."""
        pid = r.person_id
        others = [o for o in window if o.item_id != q.item_id and o.person_id != pid and o.body_id != q.body_id]
        if len(others) < RARITY_LOCAL_MIN:
            lib = self.ld.st.get("nails_lib")
            if lib is None:
                lib = [(o.body_id, o.person_id, o.attrs.get("lab")) for o in self.ld.rows("t.kind = 'nails'")
                       if isinstance(o.attrs.get("lab"), (list, tuple))]
                self.ld.st["nails_lib"] = lib
            labs = [lab for bid, p, lab in lib if bid != q.body_id and p != pid]
            if len(labs) < RARITY_LIBRARY_MIN:
                return RARITY_FALLBACK
        else:
            labs = [o.attrs["lab"] for o in others]
        near = 0
        for lab in labs:
            d = nail_de(q.attrs["lab"], lab)
            near += int(d is not None and d < de0)
        return near / max(1, len(labs))

    # ---- absence
    def _absence(self, qrows: list[_Row], query_bodies: list[int]) -> dict[int, list[dict[str, Any]]]:
        bare: dict[str, Any] = {}
        for bid in query_bodies:
            r = self.ld.conn.execute(
                f"SELECT s.bare FROM trait_scans s JOIN bodies b ON b.id = s.body_id WHERE s.body_id = :bid "
                f"AND s.status = 'ok' AND s.rev = :ready_rev AND s.box = {atraits.BOX_SQL}",
                {"bid": int(bid), "ready_rev": self.ld.rev}).fetchone()
            if r is not None:
                b = json_loads(r["bare"], {}) or {}
                if isinstance(b, dict):
                    bare.update(b)
        if not bare or self.qitem.t is None:
            return {}
        min_len = 2.0 * float(settings.trait_min_limb_px)
        inked = {q.fam for q in qrows if q.kind == "tattoo"}
        fams = []
        for fam in ("forearm", "upper_arm", "calf"):
            sides = [bare.get(f"{fam}_l"), bare.get(f"{fam}_r")]
            if fam in inked or not all(isinstance(s, dict) for s in sides):
                continue
            if all(float(s.get("len") or 0.0) >= min_len for s in sides):  # type: ignore[union-attr]
                fams.append(fam)
        out: dict[int, list[dict[str, Any]]] = {}
        if not fams:
            return out
        lib = self.ld.library("tattoo")
        for fam in fams:
            by_person: dict[int, list[_Row]] = {}
            for r in lib:
                if (r.fam == fam and not r.child and r.vec is not None and r.has_detail
                        and self.ld.items[r.item_id].t is not None):
                    by_person.setdefault(int(r.person_id), []).append(r)  # type: ignore[arg-type]
            for pid in sorted(by_person):
                rows = by_person[pid]
                dated = [self.ld.items[r.item_id].t for r in rows if self.ld.items[r.item_id].t is not None]
                if len(dated) < 2 or (max(dated) - min(dated)).total_seconds() < TRACK_DAYS * 86400:
                    continue
                first = self._track(pid, fam, rows)
                if first is not None and self.qitem.t > first:
                    out.setdefault(pid, []).append({"kind": "tattoo", "part": fam, "llr": ABSENCE_LLR, "why": "bare"})
        return out

    def _track(self, pid: int, fam: str, rows: list[_Row]) -> datetime | None:
        """First date of a persistent track of the person's rows in this family (None when there is none)."""
        key = f"track:{pid}:{fam}"
        if key in self.ld.st:
            return self.ld.st[key]
        rows = sorted(rows, key=lambda r: (self.ld.items[r.item_id].t or _NO_TIME, r.id))
        if len(rows) > TRACK_ROWS:
            step = len(rows) / TRACK_ROWS
            rows = [rows[int(i * step)] for i in range(TRACK_ROWS)]
        parent = {r.id: r.id for r in rows}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        pairs = []
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a.item_id == b.item_id:
                    continue
                c = _cos(a.vec, b.vec)  # type: ignore[arg-type]
                if c >= RETRIEVAL_COS:
                    pairs.append((c, a, b))
        pairs.sort(key=lambda x: (-x[0], x[1].id, x[2].id))
        complete = True
        for _, a, b in pairs:
            if find(a.id) == find(b.id):
                continue
            ia, ib = self.ld.items[a.item_id], self.ld.items[b.item_id]
            if self.ld.near_dup(ia, ib):
                continue
            if self.v.left <= 0 and not trait_sift.cached(("t", a.id), ("t", b.id)):
                complete = False
                break
            if trait_sift.passes(self.v.rows(a, b), int(self.params.tattoo_min_inliers)):
                parent[find(a.id)] = find(b.id)
        comps: dict[int, list[_Row]] = {}
        for r in rows:
            comps.setdefault(find(r.id), []).append(r)
        first: datetime | None = None
        for comp in comps.values():
            ts = [self.ld.items[r.item_id].t for r in comp if self.ld.items[r.item_id].t is not None]
            if len(ts) < 2 or (max(ts) - min(ts)).total_seconds() < TRACK_DAYS * 86400:
                continue
            inks = [x for x in (_ink(r.attrs) for r in comp) if x is not None]
            if not inks or float(np.median(inks)) < TRACK_INK:
                continue
            first = min(ts) if first is None else min(first, min(ts))
        if complete or first is not None:
            self.ld.st[key] = first
        return first

    # ---- one box
    def result(self, key: tuple[str, int], qrows: list[_Row], q_child: bool, query_bodies: list[int]) -> TraitResult:
        res = TraitResult()
        kinds = {k: self._verified(k, qrows) for k in MATCHED}
        nails = self._nails(qrows, q_child, query_bodies)
        absent = self._absence(qrows, query_bodies)
        people = set(kinds["tattoo"]) | set(kinds["mehndi"]) | set(nails) | set(absent)
        for pid in sorted(people):
            tat = kinds["tattoo"].get(pid) or {}
            meh = kinds["mehndi"].get(pid) or {}
            nl = nails.get(pid) or {}
            absence = max(NEG_CAP, sum(a["llr"] for a in absent.get(pid, [])))
            nails_neg = float(nl.get("neg", 0.0))
            neg = max(NEG_CAP, absence + nails_neg)
            terms = {"tattoo": _r2(tat.get("term", 0.0)), "mehndi": _r2(meh.get("term", 0.0)),
                     "nails": _r2(nl.get("term", 0.0)), "neg": _r2(neg), "absence": _r2(absence),
                     "nails_neg": _r2(nails_neg)}
            llr = _r2(terms["tattoo"] + terms["mehndi"] + terms["nails"] + terms["neg"])
            entries = [*tat.get("entries", []), *meh.get("entries", []), *nl.get("entries", [])]
            entries = [e for e in entries if e["llr"] > 0]
            entries.sort(key=lambda e: (-e["llr"], e["kind"], e["trait_id"], e["ref_trait_id"]))
            against = list(absent.get(pid, []))
            if nails_neg < 0:
                against.append({"kind": "nails", "part": "hand", "llr": _r2(nails_neg), "why": "color"})
            res.llr[pid] = llr
            res.terms[pid] = terms
            res.evidence[pid] = {"traits": entries[:3], "against": against}
            facts: dict[str, Any] = {}
            for k, d in (("tattoo", tat), ("mehndi", meh)):
                facts[f"{k}_term"] = _r2(d.get("term", 0.0))
                facts[f"{k}_best"] = int(d.get("best", 0))
                facts[f"{k}_items"] = int(d.get("items", 0))
                facts[f"{k}_bar"] = bool(d.get("bar", False))
                facts[f"{k}_exclusive"] = bool(d.get("exclusive", False))
            res.facts[pid] = facts
        return res


# ------------------------------------------------------------------ fusion
def _agree_llr(t: TraitResult, pid: int) -> float:
    """L(P) without the nails negative, which only counts on the trait-only path."""
    terms = t.terms.get(pid)
    if terms is None:
        return 0.0
    return float(t.llr.get(pid, 0.0)) - float(terms.get("neg", 0.0)) + max(NEG_CAP, float(terms.get("absence", 0.0)))


def _tie_pair(t: TraitResult, pid: int, params: TraitParams) -> bool:
    """A tattoo pair with params.tattoo_single_min (20) inliers, or one at the floor plus a second verifying photo;
    mehndi: any verified pair (its own 30 floor is above 20). Nails never break a tie."""
    f = t.facts.get(pid) or {}
    if f.get("tattoo_term", 0) > 0 and (f.get("tattoo_best", 0) >= int(params.tattoo_single_min)
                                        or f.get("tattoo_items", 0) >= 2):
        return True
    return f.get("mehndi_term", 0) > 0 and f.get("mehndi_best", 0) >= int(settings.mehndi_min_inliers)


def admitted(t: TraitResult, params: TraitParams) -> int | None:
    """The person a trait-only guess may name (amendment 3), or None."""
    if not (_matching_on() and settings.trait_only_guesses):
        return None

    def lead(p: int) -> float:  # L without nails: nails never lead and never make the margin
        return float(t.llr.get(p, 0.0)) - float((t.terms.get(p) or {}).get("nails", 0.0))

    ranked = sorted(t.llr, key=lambda p: (-lead(p), p))
    if not ranked:
        return None
    pid = ranked[0]
    runner = lead(ranked[1]) if len(ranked) > 1 else 0.0
    if lead(pid) - max(0.0, runner) < float(settings.trait_margin_llr):
        return None
    f = t.facts.get(pid) or {}
    nails = float((t.terms.get(pid) or {}).get("nails", 0.0))
    tattoo, mehndi = float(f.get("tattoo_term", 0.0)), float(f.get("mehndi_term", 0.0))
    if (tattoo >= LEAD_TERM and tattoo >= nails and f.get("tattoo_exclusive") and f.get("tattoo_bar")
            and ((params.single_pair_ok and f.get("tattoo_best", 0) >= int(params.tattoo_single_min))
                 or f.get("tattoo_items", 0) >= 2)):
        return pid
    if (settings.trait_only_mehndi and mehndi >= LEAD_TERM and mehndi >= nails and f.get("mehndi_exclusive")
            and f.get("mehndi_bar") and (f.get("mehndi_best", 0) >= MEHNDI_SINGLE_MIN or f.get("mehndi_items", 0) >= 2)):
        return pid
    return None


def _with_traits(c: Candidate, t: TraitResult, pid: int, basis: str) -> Candidate:
    from app.analysis.guess import Candidate as Cand

    ev = copy.deepcopy(c.evidence)
    ev["basis"] = basis
    ev["trait_llr"] = _r2(_agree_llr(t, pid))
    ev["traits"] = copy.deepcopy((t.evidence.get(pid) or {}).get("traits", []))
    ev["score"] = round(float(c.score), 4)
    return Cand(c.person_id, c.score, ev)


def _trait_candidate(t: TraitResult, pid: int) -> Candidate:
    from app.analysis.guess import Candidate as Cand

    llr = float(t.llr[pid])
    score = round(posterior(llr), 4)
    frag = t.evidence.get(pid) or {}
    traits = copy.deepcopy(frag.get("traits", []))
    lead = traits[0] if traits else {}
    terms = t.terms.get(pid) or {}
    ev = {"basis": "trait", "score": score, "trait_llr": _r2(llr), "ref_item_uid": lead.get("ref_item_uid"),
          "ref_body_id": lead.get("ref_body_id"), "days_apart": lead.get("days_apart"), "traits": traits,
          "against": copy.deepcopy(frag.get("against", [])),
          "terms": {k: terms.get(k, 0.0) for k in ("tattoo", "mehndi", "nails", "neg")}}
    return Cand(pid, score, ev)


def fuse(c_win: Candidate | None, c_ranked: list[Candidate], t: TraitResult,
         params: TraitParams) -> Candidate | None:
    """6.3 as amended, first rule that applies: absence veto, conflict, agreement, clothes unchanged, tie-break,
    trait-only admission, else the clothes result."""
    from app.analysis.guess import MARGIN

    if c_win is not None:
        pid = c_win.person_id
        if float((t.terms.get(pid) or {}).get("absence", 0.0)) <= NEG_CAP + 1e-9:
            return None
        other = admitted(t, params)
        if other is not None and other != pid:
            return None
        if _agree_llr(t, pid) > 0:
            return _with_traits(c_win, t, pid, "clothes+trait")
        return c_win
    if c_ranked:
        top = c_ranked[0].score
        tied = [c for c in c_ranked if top - c.score < MARGIN]
        strong = [c for c in tied if _tie_pair(t, c.person_id, params)]
        if len(strong) == 1 and all(_agree_llr(t, c.person_id) <= 0 for c in tied if c is not strong[0]):
            return _with_traits(strong[0], t, strong[0].person_id, "clothes+trait")
    pid = admitted(t, params)
    if pid is not None and (not c_ranked or pid in {c.person_id for c in c_ranked}):
        return _trait_candidate(t, pid)
    return c_win


def outcome_terms(evidence: dict[str, Any]) -> dict[str, Any]:
    """trait_llr, the terms, the best inliers, days apart and the kinds of a 'trait' guess's evidence."""
    ev = evidence if isinstance(evidence, dict) else {}
    terms = ev.get("terms") if isinstance(ev.get("terms"), dict) else {}
    traits = [e for e in (ev.get("traits") or []) if isinstance(e, dict)]
    inl = [int(e["inliers"]) for e in traits if isinstance(e.get("inliers"), (int, float))]
    return {"trait_llr": ev.get("trait_llr"), "tattoo": float(terms.get("tattoo") or 0.0),
            "mehndi": float(terms.get("mehndi") or 0.0), "nails": float(terms.get("nails") or 0.0),
            "neg": float(terms.get("neg") or 0.0), "inliers": max(inl) if inl else None,
            "days_apart": ev.get("days_apart"), "kinds": sorted({str(e.get("kind")) for e in traits if e.get("kind")})}


# ------------------------------------------------------------------ hides
def _hides_for_items(conn: sqlite3.Connection, item_ids: list[int]) -> list[sqlite3.Row]:
    out: list[sqlite3.Row] = []
    for chunk in _chunks(sorted(set(item_ids))):
        marks = ",".join("?" * len(chunk))
        out.extend(conn.execute(f"SELECT * FROM trait_hides WHERE item_id IN ({marks}) ORDER BY id", chunk).fetchall())
    return out


def _region(r: Any) -> tuple[float, float, float, float]:
    return float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])


def _covers(h: Any, r: Any, body_box: tuple[float, float, float, float] | None) -> bool:
    """Rule (a) or (b) for one hide and one row."""
    if h["kind"] != r["kind"] or int(h["item_id"]) != int(r["item_id"]):
        return False
    if _iou(_region(h), _region(r)) < HIDE_IOU:
        return False
    if h["verdict"] == "not_trait":
        return True
    if h["verdict"] == "not_theirs":
        if h["bx"] is None or body_box is None:
            return False
        return _iou((float(h["bx"]), float(h["by"]), float(h["bw"]), float(h["bh"])), body_box) >= HIDE_BODY_IOU
    return False


def _body_boxes(conn: sqlite3.Connection, body_ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    out: dict[int, tuple[float, float, float, float]] = {}
    for chunk in _chunks(sorted(set(body_ids))):
        marks = ",".join("?" * len(chunk))
        for b in conn.execute(f"SELECT id, x, y, w, h FROM bodies WHERE id IN ({marks})", chunk):
            out[int(b["id"])] = _region(b)
    return out


def hidden_ids(conn: sqlite3.Connection, rows: list[sqlite3.Row], hides: list[sqlite3.Row] | None = None) -> set[int]:
    """Ids of the rows the owner's hides take out: (a) a not_trait hide on the region, (b) a not_theirs hide on the
    region and the body, (c) a look-alike hit of a not_trait hide elsewhere, unless a 'shown' hide on the region
    keeps it. Reads the rows' columns, the items' hides and trait_hide_hits only; never verifies."""
    if not rows:
        return set()
    if hides is None:
        hides = _hides_for_items(conn, [int(r["item_id"]) for r in rows])
    by_item: dict[int, list[Any]] = {}
    for h in hides:
        by_item.setdefault(int(h["item_id"]), []).append(h)
    ids = sorted({int(r["id"]) for r in rows})
    hits: set[int] = set()
    for chunk in _chunks(ids):
        marks = ",".join("?" * len(chunk))
        hits.update(int(x[0]) for x in conn.execute(
            f"SELECT DISTINCT trait_id FROM trait_hide_hits WHERE trait_id IN ({marks})", chunk))
    need_body = [int(r["body_id"]) for r in rows
                 if any(h["verdict"] == "not_theirs" for h in by_item.get(int(r["item_id"]), []))]
    boxes = _body_boxes(conn, need_body) if need_body else {}
    out: set[int] = set()
    for r in rows:
        rid = int(r["id"])
        here = by_item.get(int(r["item_id"]), [])
        if any(_covers(h, r, boxes.get(int(r["body_id"]))) for h in here):
            out.add(rid)
            continue
        if rid in hits and not any(h["verdict"] == "shown" and h["kind"] == r["kind"]
                                   and _iou(_region(h), _region(r)) >= HIDE_IOU for h in here):
            out.add(rid)
    return out


def _row(conn: sqlite3.Connection, trait_id: int) -> sqlite3.Row | None:
    return conn.execute(f"SELECT {atraits.ROW_COLUMNS} FROM body_traits WHERE id = ?", (int(trait_id),)).fetchone()


def hides_covering(conn: sqlite3.Connection, trait_id: int) -> list[int]:
    """The rule (a) and (b) hides in the row's own item that cover it (what unhiding there deletes)."""
    r = _row(conn, trait_id)
    if r is None:
        return []
    box = _body_boxes(conn, [int(r["body_id"])]).get(int(r["body_id"]))
    return [int(h["id"]) for h in _hides_for_items(conn, [int(r["item_id"])]) if _covers(h, r, box)]


def hidden_by_lookalike(conn: sqlite3.Connection, trait_id: int) -> bool:
    """Rule (c): a look-alike hit not overridden by a 'shown' hide in the row's item."""
    r = _row(conn, trait_id)
    if r is None:
        return False
    if conn.execute("SELECT 1 FROM trait_hide_hits WHERE trait_id = ? LIMIT 1", (int(trait_id),)).fetchone() is None:
        return False
    return not any(h["verdict"] == "shown" and h["kind"] == r["kind"] and _iou(_region(h), _region(r)) >= HIDE_IOU
                   for h in _hides_for_items(conn, [int(r["item_id"])]))


def _hide_floor(kind: str) -> int:
    if kind == "tattoo":
        from app.learning import traits as ltraits

        return int(ltraits.params().tattoo_min_inliers)
    return int(settings.mehndi_min_inliers)


def _hide_desc(conn: sqlite3.Connection, hide_id: int):
    """(hide row, vec, loader) of a not_trait tattoo or mehndi hide with a descriptor, else None."""
    h = conn.execute("SELECT * FROM trait_hides WHERE id = ?", (int(hide_id),)).fetchone()
    if h is None or h["verdict"] != "not_trait" or h["kind"] not in MATCHED:
        return None
    d = conn.execute("SELECT vec, detail FROM trait_hide_desc WHERE hide_id = ?", (int(hide_id),)).fetchone()
    if d is None:
        return None
    vec = atraits.decode_vec(d["vec"])
    if vec is None:
        return None
    share = None
    if h["src_trait_id"] is not None:
        src = conn.execute("SELECT part, attrs FROM body_traits WHERE id = ?", (h["src_trait_id"],)).fetchone()
        if src is not None:
            a = json_loads(src["attrs"], {}) or {}
            share = a.get("share") if isinstance(a, dict) and "share" in a else atraits.share_for(src["part"])
    detail = bytes(d["detail"])
    return h, vec, (lambda: (detail, share))


def hide_hits_for_hide(conn: sqlite3.Connection, hide_id: int) -> int:
    """Rule (c) for one not_trait hide: usable rows of its kind across the library at DINOv2 cosine HIDE_COS or more
    that verify with its descriptor (VERIFY_PER_HIDE verifications at most). Replaces the hide's hits."""
    got = _hide_desc(conn, hide_id)
    if got is None:
        return 0
    h, hv, load_h = got
    ld = _Loader(conn, None)
    cands = []
    for r in ld.rows("t.kind = :kind", {"kind": h["kind"]}, apply_hides=False):
        if r.vec is None or not r.has_detail:
            continue
        c = _cos(hv, r.vec)
        if c >= HIDE_COS:
            cands.append((c, r))
    cands.sort(key=lambda x: (-x[0], x[1].id))
    ver = _Verifier(conn, VERIFY_PER_HIDE)
    floor = _hide_floor(h["kind"])
    found = [r.id for _, r in cands
             if trait_sift.passes(ver.verify_keys(("h", int(hide_id)), load_h, ("t", r.id), _loader_fn(conn, r.id)),
                                  floor)]
    with tx(conn):
        conn.execute("DELETE FROM trait_hide_hits WHERE hide_id = ?", (int(hide_id),))
        for rid in found:
            conn.execute("INSERT OR IGNORE INTO trait_hide_hits(trait_id, hide_id) VALUES (?, ?)", (rid, int(hide_id)))
    return len(found)


def hide_hits_for_rows(conn: sqlite3.Connection, row_ids: list[int]) -> int:
    """Rule (c) for new rows: each tattoo or mehndi row against every not_trait hide of its kind with a descriptor
    (cosine HIDE_COS or more, then verified). Returns the hits written."""
    ids = sorted({int(i) for i in row_ids})
    if not ids:
        return 0
    hides = [int(r[0]) for r in conn.execute(
        "SELECT h.id FROM trait_hides h JOIN trait_hide_desc d ON d.hide_id = h.id "
        "WHERE h.verdict = 'not_trait' AND h.kind IN ('tattoo', 'mehndi') ORDER BY h.id")]
    if not hides:
        return 0
    rows: list[sqlite3.Row] = []
    for chunk in _chunks(ids):
        marks = ",".join("?" * len(chunk))
        rows.extend(conn.execute(
            f"SELECT id, kind, vec, detail IS NOT NULL AS has_detail FROM body_traits WHERE id IN ({marks}) "
            "AND kind IN ('tattoo', 'mehndi')", chunk).fetchall())
    found: list[tuple[int, int]] = []
    for hid in hides:
        got = _hide_desc(conn, hid)
        if got is None:
            continue
        h, hv, load_h = got
        ver = _Verifier(conn, VERIFY_PER_HIDE)
        floor = _hide_floor(h["kind"])
        for r in rows:
            v = atraits.decode_vec(r["vec"])
            if r["kind"] != h["kind"] or v is None or not r["has_detail"] or _cos(hv, v) < HIDE_COS:
                continue
            if trait_sift.passes(ver.verify_keys(("h", hid), load_h, ("t", int(r["id"])), _loader_fn(conn, int(r["id"]))),
                                 floor):
                found.append((int(r["id"]), hid))
    if found:
        with tx(conn):
            for rid, hid in found:
                conn.execute("INSERT OR IGNORE INTO trait_hide_hits(trait_id, hide_id) VALUES (?, ?)", (rid, hid))
    return len(found)


def _guesses_citing(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    """(item_id, trait row id) for every open guess whose evidence cites a trait row (query or reference)."""
    out: list[tuple[int, int]] = []
    for table in ("faces", "bodies"):
        for r in conn.execute(
                f"SELECT g.item_id, json_extract(j.value, '$.trait_id') AS a, json_extract(j.value, '$.ref_trait_id') AS b "
                f"FROM {table} g, json_each(g.guess_evidence, '$.traits') j WHERE g.assigned_by = 'guess' "
                "AND g.person_id IS NULL AND g.guess_evidence IS NOT NULL "
                "AND json_type(g.guess_evidence, '$.traits') = 'array'"):
            for v in (r["a"], r["b"]):
                if isinstance(v, int):
                    out.append((int(r["item_id"]), int(v)))
    return out


def after_hide_change(conn: sqlite3.Connection, item_ids: Iterable[int]) -> int:
    """After hides in these items were written, removed or replayed by an undo: rule (c) hits of their not_trait
    hides again, a re-guess of the items, and a requeue of open guesses citing rows that are hidden now and of a
    few look-alikes. Returns boxes changed plus items requeued."""
    from app.analysis.guess import guess_item

    items = sorted({int(i) for i in item_ids})
    if not items:
        return 0
    marks = ",".join("?" * len(items))
    hides = conn.execute(
        f"SELECT h.id, h.src_trait_id FROM trait_hides h WHERE h.item_id IN ({marks}) ORDER BY h.id", items).fetchall()
    for h in hides:
        try:
            hide_hits_for_hide(conn, int(h["id"]))
        except Exception:  # noqa: BLE001 - a broken descriptor only loses its look-alikes
            log.exception("traits: look-alikes of hide %s failed", h["id"])
    changed = 0
    for iid in items:
        changed += guess_item(conn, iid)
    requeue: set[int] = set()
    cited = _guesses_citing(conn)
    if cited:
        rows = []
        for chunk in _chunks(sorted({t for _, t in cited})):
            m = ",".join("?" * len(chunk))
            rows.extend(conn.execute(f"SELECT {atraits.ROW_COLUMNS} FROM body_traits WHERE id IN ({m})", chunk).fetchall())
        gone = {t for _, t in cited} - {int(r["id"]) for r in rows}
        hidden = hidden_ids(conn, rows) | gone
        requeue.update(i for i, t in cited if t in hidden and i not in items)
    src = [int(h["src_trait_id"]) for h in hides if h["src_trait_id"] is not None]
    if src and atraits.ready(conn):
        requeue.update(i for i in targets_for_rows(conn, src, HIDE_TARGETS) if i not in items)
    n = 0
    if requeue:
        n = atraits.requeue(conn, sorted(requeue))
        atraits.kick(conn)
    return changed + n


# ------------------------------------------------------------------ targets and the sweep
OPEN_BODY_SQL = ("b.person_id IS NULL AND b.assigned_by IN ('auto', 'guess') AND (b.face_id IS NULL OR EXISTS "
                 "(SELECT 1 FROM faces fo WHERE fo.id = b.face_id AND fo.person_id IS NULL "
                 "AND fo.assigned_by IN ('auto', 'guess')))")


def _targets(conn: sqlite3.Connection, sources: list[_Row], ld: _Loader, limit: int) -> list[int]:
    sources = [s for s in sources if s.kind in MATCHED and s.vec is not None]
    if not sources or limit <= 0:
        return []
    src_items = {s.item_id for s in sources}
    cands = ld.rows(f"t.kind IN ('tattoo', 'mehndi') AND {OPEN_BODY_SQL}")
    mdays = float(settings.mehndi_max_days)
    best: dict[int, float] = {}
    for r in cands:
        if r.item_id in src_items or r.vec is None:
            continue
        ri = ld.items[r.item_id]
        for s in sources:
            if s.kind != r.kind or s.fam != r.fam:
                continue
            if r.kind == "mehndi":
                si = ld.items[s.item_id]
                dt = _days(si, ri)
                if not (si.short_ok and ri.short_ok) or dt is None or dt > mdays:
                    continue
            c = _cos(s.vec, r.vec)
            if c >= RETRIEVAL_COS and c > best.get(r.item_id, -1.0):
                best[r.item_id] = c
    ordered = sorted(best.items(), key=lambda kv: (-kv[1], -kv[0]))
    return [i for i, _ in ordered[:int(limit)]]


def trait_targets(conn: sqlite3.Connection, person_id: int, item_id: int | None, limit: int) -> list[int]:
    """Items whose open boxes carry a tattoo (any date) or mehndi (within its window) close to this person's rows
    from the anchor item, or from the person's newest PERSON_SOURCE_ROWS rows. Cosine only."""
    if not _matching_on() or not atraits.ready(conn):
        return []
    ld = _Loader(conn, None)
    where = "t.kind IN ('tattoo', 'mehndi') AND b.person_id = :pid AND b.assigned_by IN ('user', 'auto')"
    params: dict[str, Any] = {"pid": int(person_id)}
    if item_id is not None:
        where += " AND t.item_id = :iid"
        params["iid"] = int(item_id)
    rows = [r for r in ld.rows(where, params) if not r.child]
    rows.sort(key=lambda r: (ld.items[r.item_id].taken_at or "", r.id), reverse=True)
    return _targets(conn, rows[:PERSON_SOURCE_ROWS], ld, limit)


def targets_for_rows(conn: sqlite3.Connection, row_ids: list[int], limit: int) -> list[int]:
    """The same search from given rows (the runner's new named rows, the hides' rows)."""
    ids = sorted({int(i) for i in row_ids})
    if not ids or not _matching_on() or not atraits.ready(conn):
        return []
    ld = _Loader(conn, None)
    marks = ",".join(str(i) for i in ids)
    return _targets(conn, ld.rows(f"t.id IN ({marks})", apply_hides=False), ld, limit)


def sweep_open(conn: sqlite3.Connection, budget_s: float | None = None) -> int:
    """Once per ready rev, on the runner thread: requeue the items whose open boxes carry a usable tattoo or mehndi
    row. Returns how many items were queued."""
    if not _matching_on() or not atraits.ready(conn):
        return 0
    ids = [int(r[0]) for r in conn.execute(
        "SELECT DISTINCT t.item_id FROM body_traits t JOIN trait_scans s ON s.body_id = t.body_id "
        "JOIN bodies b ON b.id = t.body_id JOIN items i ON i.id = t.item_id "
        f"WHERE t.kind IN ('tattoo', 'mehndi') AND {atraits.usable_sql()} AND {OPEN_BODY_SQL} ORDER BY t.item_id",
        {"ready_rev": atraits.ready_rev(conn)})]
    return atraits.requeue(conn, ids) if ids else 0


# ------------------------------------------------------------------ settings change
def rules_hash() -> str:
    values = {"rev": atraits.TRAITS_REV, **{k: getattr(settings, k) for k in RULE_SETTINGS}}
    return hashlib.sha1(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:16]


def apply_settings_change(conn: sqlite3.Connection) -> int:
    """At startup: record the hash of the trait rules (always, ready or not). When it differs from the one before,
    withdraw the open trait-only guesses if matching or trait-only guesses are off now, and requeue the items of the
    trait and clothes+trait guesses. Returns the guesses withdrawn or requeued."""
    current = rules_hash()
    previous = kv_get(conn, atraits.RULES_KEY)
    if previous == current:
        return 0
    off = not (_matching_on() and settings.trait_only_guesses)
    items: set[int] = set()
    withdrawn = touched = 0
    with tx(conn):
        if previous is not None:
            for table in ("faces", "bodies"):
                for r in conn.execute(
                        f"SELECT id, item_id, json_extract(guess_evidence, '$.basis') AS basis FROM {table} "
                        "WHERE assigned_by = 'guess' AND person_id IS NULL AND guess_evidence IS NOT NULL "
                        "AND json_valid(guess_evidence) "
                        "AND json_extract(guess_evidence, '$.basis') IN ('trait', 'clothes+trait')").fetchall():
                    touched += 1
                    items.add(int(r["item_id"]))
                    if r["basis"] == "trait" and off:
                        conn.execute(f"UPDATE {table} SET assigned_by = 'auto', guess_person_id = NULL, "
                                     "guess_score = NULL, guess_evidence = NULL WHERE id = ? AND person_id IS NULL "
                                     "AND assigned_by = 'guess'", (r["id"],))
                        withdrawn += 1
            if withdrawn:
                bump_faces_version(conn)
        kv_set(conn, atraits.RULES_KEY, current)
    if items:
        atraits.requeue(conn, sorted(items))
        atraits.kick(conn)
        log.info("traits: rules changed; withdrew %d trait guess(es), requeued %d item(s)", withdrawn, len(items))
    return touched


# ------------------------------------------------------------------ calibration
def calibration_due(conn: sqlite3.Connection) -> bool:
    """A sample is due when the library is ready and there is none yet, or CALIB_GROWTH rows were added since the
    last one and it is CALIB_EVERY_S old."""
    if not atraits.ready(conn):
        return False
    prev = json_loads(kv_get(conn, atraits.CALIB_KEY), None)
    if not isinstance(prev, dict):
        return True
    max_id = int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM body_traits").fetchone()[0])
    if max_id - int(prev.get("max_id") or 0) < CALIB_GROWTH:
        return False
    at = parse_iso(kv_get(conn, atraits.CALIB_AT_KEY))
    if at is None:
        return True
    return (parse_iso(now_iso()) - at).total_seconds() >= CALIB_EVERY_S  # type: ignore[operator]


def calibration_sample(conn: sqlite3.Connection, force: bool = False) -> dict[str, Any]:
    """On the runner thread, when due (or forced): hard negatives (different named people whose tattoo or mehndi rows
    verify after every gate; pairs in one photo marked same_photo), the reference bodies the gates admit per person
    (strong and weak), and the nail dE lists. At most VERIFY_PER_CALIB verifications. Written to kv trait_calib and
    trait_calib_at. Returns the sample ({} when not due)."""
    if not (force and atraits.ready(conn)) and not calibration_due(conn):
        return {}
    ld = _Loader(conn, None)
    ver = _Verifier(conn, VERIFY_PER_CALIB)
    hard: list[dict[str, Any]] = []
    refs: dict[str, dict[str, dict[str, int]]] = {}
    for kind in MATCHED:
        lib = [r for r in ld.library(kind) if not r.child and r.vec is not None]
        bodies: dict[int, dict[int, bool]] = {}
        for r in lib:
            bodies.setdefault(int(r.person_id), {})[r.body_id] = (  # type: ignore[arg-type]
                bodies.get(int(r.person_id), {}).get(r.body_id, False) or r.strong)  # type: ignore[arg-type]
        refs[kind] = {str(p): {"strong": sum(1 for s in b.values() if s), "weak": sum(1 for s in b.values() if not s)}
                      for p, b in sorted(bodies.items())}
        floor = int(settings.tattoo_min_inliers) if kind == "tattoo" else int(settings.mehndi_min_inliers)
        newest = sorted(lib, key=lambda r: (ld.items[r.item_id].taken_at or "", r.id), reverse=True)[:CALIB_ROWS]
        for q in newest:
            if not q.has_detail:
                continue
            qi = ld.items[q.item_id]
            cands: list[tuple[float, _Row, bool]] = []
            for r in lib:
                if r.person_id == q.person_id or r.fam != q.fam or not r.has_detail or r.id == q.id:
                    continue
                ri = ld.items[r.item_id]
                same = r.item_id == q.item_id
                if kind == "mehndi" and not same:
                    dt = _days(qi, ri)
                    if not (qi.short_ok and ri.short_ok) or dt is None or dt > float(settings.mehndi_max_days):
                        continue
                cands.append((2.0 if same else _cos(q.vec, r.vec), r, same))  # type: ignore[arg-type]
            cands.sort(key=lambda x: (-x[0], x[1].id))
            for c, r, same in cands[:RETRIEVAL_TOP]:
                if not same and (c < RETRIEVAL_COS or ld.near_dup(qi, ld.items[r.item_id])):
                    continue
                if ver.left <= 0 and not trait_sift.cached(("t", q.id), ("t", r.id)):
                    break
                v = ver.rows(q, r)
                if trait_sift.passes(v, floor):
                    a, b = sorted((q.id, r.id))
                    pq, pr = int(q.person_id), int(r.person_id)  # type: ignore[arg-type]
                    hard.append({"kind": kind, "a": a, "b": b, "inliers": int(v.inliers), "ratio": float(v.ratio),
                                 "same_photo": same, "persons": sorted([pq, pr]),
                                 # only two named people, at least one confirmed by the owner, can be a real false
                                 # match: an unnamed group is usually the same person split in two
                                 "named": ld.named([pq, pr]) == {pq, pr}, "weak": not (q.strong or r.strong)})
    seen: set[tuple[int, int]] = set()
    uniq = []
    for h in sorted(hard, key=lambda h: (-h["inliers"], h["a"], h["b"])):
        if (h["a"], h["b"]) not in seen:
            seen.add((h["a"], h["b"]))
            uniq.append(h)
    nails = [r for r in ld.library("nails") if not r.child and isinstance(r.attrs.get("lab"), (list, tuple))]
    pos: list[float] = []
    neg: list[float] = []
    nails.sort(key=lambda r: (ld.items[r.item_id].t or _NO_TIME, r.id))
    for i, a in enumerate(nails):
        ia = ld.items[a.item_id]
        for b in nails[i + 1:]:
            ib = ld.items[b.item_id]
            dt = _days(ia, ib)
            if dt is None:
                continue
            if dt > CALIB_NAIL_DAYS:
                break
            d = nail_de(a.attrs["lab"], b.attrs["lab"])
            if d is None:
                continue
            if a.person_id == b.person_id and a.item_id != b.item_id and ia.short_ok and ib.short_ok:
                pos.append(round(d, 2))
            elif a.person_id != b.person_id and (a.item_id == b.item_id or dt <= CALIB_NAIL_NEG_DAYS):
                neg.append(round(d, 2))
    max_id = int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM body_traits").fetchone()[0])
    sample = {"taken_at": now_iso(), "max_id": max_id, "verified": ver.spent, "hard_negatives": uniq[:CALIB_LIST_MAX],
              "refs": refs, "nails": {"pos": pos[:CALIB_LIST_MAX], "neg": neg[:CALIB_LIST_MAX]}}
    with tx(conn):
        kv_set(conn, atraits.CALIB_KEY, json.dumps(sample, separators=(",", ":")))
        kv_set(conn, atraits.CALIB_AT_KEY, sample["taken_at"])
    return sample


# ------------------------------------------------------------------ the person page
def _out(r: _Row, ld: _Loader | None = None) -> dict[str, Any]:
    from app import urls

    it = ld.items.get(r.item_id) if ld is not None else None
    return {"id": r.id, "kind": r.kind, "part": r.part, "x": r.x, "y": r.y, "w": r.w, "h": r.h,
            "label": atraits.label(r.kind, r.part, r.attrs), "crop_url": urls.trait_url(r.id), "hidden": False,
            "hex": r.attrs.get("hex") if r.kind == "nails" else None,
            "item_uid": it.uid if it is not None else None, "taken_at": it.taken_at if it is not None else None}


def _person_rows(conn: sqlite3.Connection, person_id: int, vault_open: bool) -> tuple[list[_Row], _Loader]:
    ld = _Loader(conn, None)
    rev = atraits.ready_rev(conn) or atraits.wanted_rev()
    usable = atraits.usable_sql()
    if vault_open:
        usable = usable.replace(" AND +i.nsfw = 0", "")
    sql = _row_sql(usable) + " AND b.person_id = :pid ORDER BY t.id"
    rows: list[_Row] = []
    for r in conn.execute(sql, {"ready_rev": rev, "pid": int(person_id)}):
        ld._item(r)
        attrs = json_loads(r["attrs"], {}) or {}
        rows.append(_Row(int(r["id"]), int(r["body_id"]), int(r["item_id"]), r["kind"], r["part"],
                         atraits.family(r["part"]), float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]),
                         float(r["quality"] or 0.0), attrs if isinstance(attrs, dict) else {}, None,
                         bool(r["has_detail"]), int(person_id), r["b_assigned"] == "user", False, r["created_at"] or ""))
    hidden = hidden_ids(conn, rows)
    return [r for r in rows if r.id not in hidden], ld


def person_groups(conn: sqlite3.Connection, person_id: int, *, vault_open: bool) -> dict[str, Any]:
    """The person page: tattoos grouped by part (photos, first and last seen, the best crop, the photos), and the
    mehndi and nails of the last LATELY_DAYS days (at most LATELY_MAX each, newest first)."""
    rows, ld = _person_rows(conn, person_id, vault_open)
    groups: dict[str, list[_Row]] = {}
    for r in rows:
        if r.kind == "tattoo":
            groups.setdefault(r.part, []).append(r)
    tattoos = []
    for part, lst in groups.items():
        best = max(lst, key=lambda r: (r.quality, r.id))
        items = sorted({r.item_id for r in lst}, key=lambda i: (ld.items[i].taken_at or "", i), reverse=True)
        dates = sorted(ld.items[i].taken_at for i in items if ld.items[i].taken_at)
        tattoos.append({"kind": "tattoo", "part": part, "label": atraits.label("tattoo", part), "photos": len(items),
                        "first_seen": dates[0] if dates else None, "last_seen": dates[-1] if dates else None,
                        "best": _out(best, ld), "item_uids": [ld.items[i].uid for i in items]})
    tattoos.sort(key=lambda g: (-g["photos"], g["part"]))
    since = now_iso(-LATELY_DAYS * 86400.0)

    def lately(kind: str) -> list[dict[str, Any]]:
        lst = [r for r in rows if r.kind == kind and (ld.items[r.item_id].taken_at or "") >= since]
        lst.sort(key=lambda r: (ld.items[r.item_id].taken_at or "", r.id), reverse=True)
        return [_out(r, ld) for r in lst[:LATELY_MAX]]

    return {"tattoos": tattoos, "mehndi": lately("mehndi"), "nails": lately("nails")}


def trait_count(conn: sqlite3.Connection, person_id: int, *, vault_open: bool = False) -> int:
    """Usable, shown trait rows on the person's boxes (vault photos only while the vault is open)."""
    return len(_person_rows(conn, person_id, vault_open)[0])
