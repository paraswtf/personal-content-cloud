"""Following people from one photo to the next (docs/ARCHITECTURE.md section 35).

Arithmetic on the stored links (app.analysis.place) and the photos' current boxes: nothing here reads pixels.

CONTRACT.
- A unit is one person as a photo shows them: a face (with the person box around it, when it has one) or a person
  box without a face. Its head point is the face's centre, or for a faceless box the top centre lowered by
  HEAD_DOWN of the box's width (the median place of the face in the library's person boxes that hold one); its
  width is FACE_WIDTHS face widths, or BOX_WIDTH of the box's width (the two then agree at the median).
- match_pair(link, units_a, units_b, info_a, info_b) -> PairMatch: the head points of the earlier photo are put into
  the later one through the link; two units are the same person standing there when each is the other's nearest,
  the distance over their mean width is at most RESIDUAL_MAX (RESIDUAL_MAX_BOX when one of them is a faceless
  box), the width ratio is within WIDTH_RATIO, the second nearest on either side is clearly worse (the best at
  most SECOND_RATIO of it), their age bands do not differ, and their clothes do not say no (both have a clothes
  vector and its cosine is under CLOTHES_NO, or CLOTHES_NO_MIXED between a face and a faceless box).
  PairMatch.swapped is True when a named person of one photo sits on a different named person of the other: people
  moved around, and the pair gives no evidence. PairMatch.kept counts the known people found in their own place.
- context(conn, item, boxes, cache) -> PlaceContext | None: None unless PLACE_ENABLED, the item is a dated photo
  and a usable link touches it. PlaceContext.score_all(face_scores, rejected) -> {(kind, id): PlaceResult}: for
  each open box, the links are walked both ways, at most HOPS photos away, through boxes nobody is named for, to
  boxes that have a person (assigned_by user or auto, as clothes references). The evidence for a person is their
  best chain: its strength is its weakest link (link_llr: by the gap, how the photos lined up, the clothes and
  the people found in place; a link with a faceless box at either end costs FACELESS_COST and, with a gap over
  FACELESS_NEAR_S, is never stronger than FACELESS_FAR_LLR) minus HOP_COST per further hop. People already in
  the photo are left out, and so is
  anyone the box's own face contradicts (the trait path's veto) or whose age band differs (a child next to a
  parent).
- fuse(win, c_ranked, t) -> Candidate | None, after trait_match.fuse, first rule that applies: a winner that
  position contradicts at admission strength gives no guess; a winner that position agrees with gains the `place`
  fragment (basis 'clothes+place' when it was plain clothes); a clothes tie inside MARGIN that position decides
  for exactly one of the tied goes to that person; with nothing from clothes or traits, a position-only guess
  (basis 'place') under admitted(): PLACE_ONLY_GUESSES on, strength at least ADMIT_LLR and a lead of LEAD_LLR
  over the runner-up. Its score is sigmoid(strength), the measured share of right answers of its tier.
- outcome_terms(evidence) -> what a 'place' guess's outcome records for the learner.
The constants are the ones scripts/place_eval.py measured on the owner's library (section 35 has the numbers).
"""
from __future__ import annotations

import copy
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from app.analysis import place, place_align
from app.config import settings

if TYPE_CHECKING:
    from app.analysis.guess import Box, Candidate

log = logging.getLogger(__name__)

FACE_WIDTHS = 3.2          # a person is about this many face widths wide (faces.TORSO_WIDTH)
HEAD_DOWN = 0.25           # head point of a faceless person box: below its top by this share of its WIDTH (the
#                            median over 21,668 person boxes holding a face; by the width, as the height depends on
#                            how much of the body is in the frame: half of the estimates are within 0.16 widths)
BOX_WIDTH = 0.875          # 3.2 face widths over the width of the person box around the face, the median
RESIDUAL_MAX = 0.25        # two faces: head points at most this far apart, in mean widths
RESIDUAL_MAX_BOX = 0.35    # a faceless box on either side: its head point is only estimated
SECOND_RATIO = 0.75        # the best match is at most this share of the second nearest
WIDTH_RATIO = (0.6, 1.67)
CLOTHES_NO = 0.80          # two clothes vectors of one kind of crop under this cosine are two outfits: no link
CLOTHES_NO_MIXED = 0.60    # a face's head-and-torso crop against a person box's upper part: the crops differ, and
#                            the same person is under 0.70 in 4 pairs of 100 (under 0.60 in 1 of 100)
CLOTHES_YES = 0.85         # the clothes cut of the clothes path: at or above it the clothes back the link
HOPS = 3
HOP_COST = 0.4
ADMIT_LLR = 2.9            # sigmoid(2.9) is about 0.95: the bar for a guess from the position alone
LEAD_LLR = 2.0
TIE_LLR = 1.0              # a link at least this strong may decide a clothes tie
MAX_SCORE = 0.99
# (the longest gap in seconds, the strength): log odds near the lower end of the share of right answers measured
# for that gap, pair by pair on the owner's library (two faces: 0.985 up to 3 s, 0.985 up to 10 s, 0.963 up to
# 60 s, 0.83 up to 5 min). End to end the first three gave 0.978, 0.951 and 0.962 over all boxes, and no wrong
# answer among named people (section 35). The last tier and a burst stay under ADMIT_LLR with every bonus: they
# agree with clothes and break ties, they never name someone alone.
TIERS: tuple[tuple[float, float], ...] = ((3.0, 3.9), (10.0, 3.7), (60.0, 2.9), (300.0, 1.2))
BURST_LLR = 1.2            # photos that did not line up but are a burst (place_align.burst): not measured, so weak
CLOTHES_BONUS = 0.5
KEPT_BONUS = 0.5           # per other known person found in their own place, at most KEPT_MAX
KEPT_MAX = 1.0
FACELESS_COST = 0.5        # a head point estimated from a person box is less exact than a face
FACELESS_NEAR_S = 10.0     # a link with a faceless box at either end and a longer gap than this...
FACELESS_FAR_LLR = 2.7     # ...is never stronger than this, so it never names alone: with support it was right
#                            29 times of 31, under the bar (section 35)


def rules_sig() -> str:
    """The rules a position guess was made under: when it changes (a setting, a constant of a later version), the
    open guesses that rest on position are made again (place.apply_settings_change)."""
    return "|".join(str(v) for v in (
        int(bool(settings.place_enabled)), int(bool(settings.place_only_guesses)), TIERS, BURST_LLR, ADMIT_LLR, LEAD_LLR,
        TIE_LLR, HOPS, HOP_COST, RESIDUAL_MAX, RESIDUAL_MAX_BOX, SECOND_RATIO, WIDTH_RATIO, CLOTHES_NO, CLOTHES_NO_MIXED,
        CLOTHES_YES, CLOTHES_BONUS, KEPT_BONUS, KEPT_MAX, FACELESS_COST, FACELESS_NEAR_S, FACELESS_FAR_LLR,
        HEAD_DOWN, BOX_WIDTH))


@dataclass
class Unit:
    kind: str                      # 'face' | 'body' (a faceless person box)
    id: int
    item_id: int
    hx: float
    hy: float
    width: float
    person_id: int | None          # assigned by the owner or the app; None for an open or guessed box
    torso: np.ndarray | None
    face_id: int | None = None
    body_id: int | None = None
    child: bool | None = None      # the face's age band, when read
    named: bool = False            # the person has a name (an unnamed group may be anyone, the neighbour too)


@dataclass
class Info:
    id: int
    uid: str
    t: float | None
    width: float
    height: float


@dataclass
class PairMatch:
    pairs: dict[int, int] = field(default_factory=dict)        # index in a -> index in b
    residual: dict[int, float] = field(default_factory=dict)   # by index in a
    clothes: dict[int, float | None] = field(default_factory=dict)
    swapped: bool = False
    kept: int = 0
    kept_people: set[int] = field(default_factory=set)


@dataclass
class PlaceResult:
    """What position says about one open box, per candidate person: llr (the chain's strength) and the evidence
    fragment fuse() puts into the guess."""
    llr: dict[int, float] = field(default_factory=dict)
    evidence: dict[int, dict[str, Any]] = field(default_factory=dict)

    def without(self, rejected: set[int]) -> PlaceResult:
        return PlaceResult({p: v for p, v in self.llr.items() if p not in rejected},
                           {p: v for p, v in self.evidence.items() if p not in rejected})

    def best(self) -> tuple[int | None, float, float]:
        ranked = sorted(self.llr.items(), key=lambda kv: (-kv[1], kv[0]))
        if not ranked:
            return None, 0.0, 0.0
        return ranked[0][0], float(ranked[0][1]), float(ranked[1][1]) if len(ranked) > 1 else 0.0


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


# ------------------------------------------------------------------ reads
def load_units(conn: sqlite3.Connection, item_ids: list[int],
               aspect: dict[int, float] | None = None) -> dict[int, list[Unit]]:
    """The units of each photo: faces first (by id), then faceless person boxes (by id). `aspect`: width over
    height of each photo (a faceless box's head point lies HEAD_DOWN box widths below its top, which in normalised
    coordinates depends on the picture's shape; 1.0 when unknown)."""
    from app.analysis import ages
    from app.analysis.faces import face_weight

    ids = sorted({int(i) for i in item_ids})
    out: dict[int, list[Unit]] = {i: [] for i in ids}
    if not ids:
        return out
    young = ages.ready(conn)
    faces: list[sqlite3.Row] = []
    bodies: list[sqlite3.Row] = []
    for start in range(0, len(ids), 400):
        chunk = ids[start:start + 400]
        marks = ",".join("?" * len(chunk))
        faces.extend(conn.execute(
            f"SELECT id, item_id, x, y, w, h, person_id, assigned_by, torso_embedding, age_groups, det_score, size_px, "
            f"clarity, sharpness FROM faces WHERE item_id IN ({marks}) ORDER BY id", chunk).fetchall())
        bodies.extend(conn.execute(
            f"SELECT id, item_id, face_id, x, y, w, h, person_id, assigned_by, torso_embedding FROM bodies "
            f"WHERE item_id IN ({marks}) ORDER BY id", chunk).fetchall())
    faces.sort(key=lambda r: int(r["id"]))
    bodies.sort(key=lambda r: int(r["id"]))
    named = _named(conn, {int(r["person_id"]) for r in (*faces, *bodies) if r["person_id"] is not None})
    holder: dict[int, int] = {}
    for b in bodies:
        if b["face_id"] is not None:
            holder.setdefault(int(b["face_id"]), int(b["id"]))
    for f in faces:
        known = f["person_id"] is not None and f["assigned_by"] in ("user", "auto")
        child = None
        if young and f["age_groups"] is not None:
            child = bool(ages.in_child_band(f["age_groups"], face_weight(f["det_score"], f["size_px"], f["clarity"],
                                                                       f["sharpness"])))
        out[int(f["item_id"])].append(Unit(
            "face", int(f["id"]), int(f["item_id"]), float(f["x"]) + float(f["w"]) / 2.0,
            float(f["y"]) + float(f["h"]) / 2.0, FACE_WIDTHS * float(f["w"]),
            int(f["person_id"]) if known else None, _vec(f["torso_embedding"]), face_id=int(f["id"]),
            body_id=holder.get(int(f["id"])), child=child, named=known and int(f["person_id"]) in named))
    for b in bodies:
        if b["face_id"] is not None:
            continue
        known = b["person_id"] is not None and b["assigned_by"] in ("user", "auto")
        shape = float((aspect or {}).get(int(b["item_id"]), 1.0)) or 1.0
        out[int(b["item_id"])].append(Unit(
            "body", int(b["id"]), int(b["item_id"]), float(b["x"]) + float(b["w"]) / 2.0,
            float(b["y"]) + HEAD_DOWN * float(b["w"]) * shape, BOX_WIDTH * float(b["w"]),
            int(b["person_id"]) if known else None, _vec(b["torso_embedding"]), body_id=int(b["id"]),
            named=known and int(b["person_id"]) in named))
    return out


def load_info(conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, Info]:
    """The live, dated photos among these items (never the vault or the trash)."""
    ids = sorted({int(i) for i in item_ids})
    out: dict[int, Info] = {}
    for start in range(0, len(ids), 400):
        chunk = ids[start:start + 400]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT id, uid, taken_at, exif_subsec, width, height FROM items WHERE id IN ({marks}) AND kind = 'image' "
            "AND deleted_at IS NULL AND nsfw = 0 AND taken_at IS NOT NULL", chunk,
        ):
            out[int(r["id"])] = Info(int(r["id"]), str(r["uid"]), place.capture_time(r), float(r["width"] or 0),
                                     float(r["height"] or 0))
    return out


# ------------------------------------------------------------------ one pair
def match_pair(link: place.Link, ua: list[Unit], ub: list[Unit], ia: Info, ib: Info) -> PairMatch:
    """Which units of the two photos are the same person standing in the same place (see the module contract)."""
    out = PairMatch()
    if not ua or not ub:
        return out
    long_side = max(ib.width, ib.height)
    sx, sy = (ib.width / long_side, ib.height / long_side) if long_side > 0 else (1.0, 1.0)
    m = link.m
    a_pts = []
    for u in ua:
        x, y = place_align.apply(m, u.hx, u.hy)
        a_pts.append((x * sx, y * sy, math.hypot(m[0] * u.width * sx, m[3] * u.width * sy)))
    b_pts = [(u.hx * sx, u.hy * sy, u.width * sx) for u in ub]
    res = np.full((len(ua), len(ub)), np.inf, dtype=np.float64)
    ratio = np.ones((len(ua), len(ub)), dtype=np.float64)
    for i, (ax, ay, aw) in enumerate(a_pts):
        for j, (bx, by, bw) in enumerate(b_pts):
            mean = 0.5 * (aw + bw)
            if mean <= 0 or bw <= 0:
                continue
            res[i, j] = math.hypot(ax - bx, ay - by) / mean
            ratio[i, j] = aw / bw
    for i in range(len(ua)):
        j = int(np.argmin(res[i]))
        r = float(res[i, j])
        bound = RESIDUAL_MAX if ua[i].kind == "face" and ub[j].kind == "face" else RESIDUAL_MAX_BOX
        if not math.isfinite(r) or r > bound or int(np.argmin(res[:, j])) != i:
            continue
        if not WIDTH_RATIO[0] <= float(ratio[i, j]) <= WIDTH_RATIO[1]:
            continue
        others = np.concatenate([np.delete(res[i], j), np.delete(res[:, j], i)])
        second = float(others.min()) if others.size else math.inf
        if math.isfinite(second) and r > SECOND_RATIO * second:
            continue
        a, b = ua[i], ub[j]
        if a.child is not None and b.child is not None and a.child != b.child:
            continue
        cos = None
        if a.torso is not None and b.torso is not None and a.torso.shape == b.torso.shape:
            cos = float(a.torso @ b.torso)
        known_both = a.person_id is not None and b.person_id is not None
        if known_both and a.person_id != b.person_id:
            if a.named and b.named:
                out.swapped = True  # a named person where another named person was: people moved around
            continue  # an unnamed group may be the same person, or not: no evidence either way
        if cos is not None and not known_both and cos < (CLOTHES_NO if a.kind == b.kind else CLOTHES_NO_MIXED):
            continue
        out.pairs[i] = j
        out.residual[i] = r
        out.clothes[i] = cos
        if known_both:
            out.kept += 1
            out.kept_people.add(int(a.person_id))  # type: ignore[arg-type]
    return out


def tier_llr(dt_s: float) -> float | None:
    for longest, llr in TIERS:
        if dt_s <= longest:
            return llr
    return None


def link_llr(link: place.Link, pm: PairMatch, i: int, a: Unit, b: Unit, exclude: int | None = None) -> float | None:
    """The strength of one matched pair of units of a lined-up pair of photos; None when it carries nothing.
    `exclude`: a person whose own place does not count as support (the candidate being scored)."""
    if pm.swapped:
        return None
    base = BURST_LLR if link.kind == "burst" else tier_llr(abs(float(link.dt_s)))
    if base is None:
        return None
    llr = float(base)
    cos = pm.clothes.get(i)
    if cos is not None and cos >= CLOTHES_YES:
        llr += CLOTHES_BONUS
    kept = len(pm.kept_people - ({exclude} if exclude is not None else set()))
    llr += min(KEPT_MAX, KEPT_BONUS * kept)
    if a.kind == "body" or b.kind == "body":
        llr -= FACELESS_COST
        if abs(float(link.dt_s)) > FACELESS_NEAR_S:
            llr = min(llr, FACELESS_FAR_LLR)
    return llr


# ------------------------------------------------------------------ the context
@dataclass
class PlaceContext:
    """Per guess_item call: the photo, its open boxes, and the shared cache of links, units and pair matches."""
    conn: sqlite3.Connection
    item: sqlite3.Row
    boxes: list[Box]
    cache: dict[str, Any] | None
    present: set[int] = field(default_factory=set)

    def _store(self) -> dict[str, Any]:
        if self.cache is None:
            self.cache = {}
        return self.cache.setdefault("place", {"links": {}, "units": {}, "info": {}, "pairs": {}})

    def _links(self, item_ids: list[int]) -> None:
        st = self._store()
        missing = [i for i in item_ids if i not in st["links"]]
        if not missing:
            return
        for i in missing:
            st["links"][i] = []
        for ln in place.links_for(self.conn, missing):
            for i in (ln.a_item, ln.b_item):
                if i in st["links"] and ln not in st["links"][i]:
                    st["links"][i].append(ln)

    def _load(self, item_ids: list[int]) -> None:
        st = self._store()
        missing = [i for i in item_ids if i not in st["units"]]
        if not missing:
            return
        info = load_info(self.conn, missing)
        units = load_units(self.conn, list(info), {i: v.width / v.height for i, v in info.items() if v.height > 0})
        for i in missing:
            st["info"][i] = info.get(i)
            st["units"][i] = units.get(i, []) if i in info else []

    def _pair(self, ln: place.Link) -> PairMatch | None:
        st = self._store()
        key = (ln.a_item, ln.b_item)
        if key not in st["pairs"]:
            ia, ib = st["info"].get(ln.a_item), st["info"].get(ln.b_item)
            st["pairs"][key] = (match_pair(ln, st["units"][ln.a_item], st["units"][ln.b_item], ia, ib)
                                if ia is not None and ib is not None else None)
        return st["pairs"][key]

    def _step(self, item_id: int, index: int) -> list[tuple[place.Link, PairMatch, int, int, int]]:
        """(link, pair match, index in a, the other photo, index there) for each matched neighbour of a unit."""
        st = self._store()
        out = []
        for ln in st["links"].get(item_id, []):
            other = ln.b_item if ln.a_item == item_id else ln.a_item
            self._load([other])
            pm = self._pair(ln)
            if pm is None or pm.swapped:
                continue
            if ln.a_item == item_id:
                j = pm.pairs.get(index)
                if j is not None:
                    out.append((ln, pm, index, other, j))
            else:
                for i, j in pm.pairs.items():
                    if j == index:
                        out.append((ln, pm, i, other, i))
                        break
        return out

    def _chains(self, start: int) -> dict[int, tuple[float, dict[str, Any]]]:
        """{person: (strength, fragment)} for the unit at index `start` of this photo."""
        st = self._store()
        iid = int(self.item["id"])
        t0 = st["info"][iid].t if st["info"].get(iid) is not None else None
        best: dict[int, tuple[float, dict[str, Any]]] = {}
        seen = {(iid, start)}
        frontier: list[tuple[int, int, float, str, float | None]] = [(iid, start, math.inf, "", None)]
        for hop in range(1, HOPS + 1):
            self._links([f[0] for f in frontier])
            nxt: list[tuple[int, int, float, str, float | None]] = []
            for item_id, index, strength, kind0, offset0 in frontier:
                here = st["units"][item_id][index]
                for ln, pm, ai, other, oi in self._step(item_id, index):
                    if (other, oi) in seen:
                        continue
                    there = st["units"][other][oi]
                    a_unit, b_unit = (here, there) if ln.a_item == item_id else (there, here)
                    e = link_llr(ln, pm, ai, a_unit, b_unit, exclude=there.person_id)
                    if e is None:
                        continue
                    s = min(strength, e)
                    # the chain's weakest link names how the photos lined up; the first link's offset is the box's own
                    kind = ln.kind if e <= strength else kind0
                    offset = pm.residual.get(ai) if offset0 is None else offset0
                    if there.person_id is None:
                        seen.add((other, oi))
                        nxt.append((other, oi, s, kind, offset))
                        continue
                    pid = int(there.person_id)
                    total = s - HOP_COST * (hop - 1)
                    if total <= 0 or (pid in best and best[pid][0] >= total):
                        continue
                    info = st["info"].get(other)
                    seconds = before = None
                    if t0 is not None and info is not None and info.t is not None:
                        seconds = round(abs(info.t - t0), 1)
                        before = bool(info.t <= t0)
                    cos = pm.clothes.get(ai) if hop == 1 else None
                    best[pid] = (total, {
                        "ref_item_uid": info.uid if info is not None else None,
                        "ref_face_id": there.face_id,
                        "ref_body_id": there.body_id if there.kind == "body" else None,
                        "seconds_apart": seconds,
                        "ref_before": before,
                        "hops": hop,
                        "offset": round(float(offset), 2) if offset is not None else None,
                        "kind": kind,
                        "others": len(pm.kept_people - {pid}),
                        "clothes": round(cos, 4) if cos is not None else None,
                        "llr": round(total, 2),
                    })
            frontier = nxt
            if not frontier:
                break
        return best

    def score_all(self, face_scores: dict[int, dict[int, float]],
                  rejected: dict[tuple[str, int], set[int]]) -> dict[tuple[str, int], PlaceResult]:
        """A PlaceResult per open box that position says something about."""
        from app.analysis.guess import FACE_CONTRADICTION, FACE_WEAK

        iid = int(self.item["id"])
        self._load([iid])
        self._links([iid])
        st = self._store()
        units = st["units"].get(iid) or []
        index = {(u.kind, u.id): k for k, u in enumerate(units)}
        named: set[int] | None = None
        out: dict[tuple[str, int], PlaceResult] = {}
        for b in self.boxes:
            key = (b.kind, b.id)
            k = index.get(key)
            if k is None:
                continue
            found = self._chains(k)
            res = PlaceResult({p: round(v[0], 4) for p, v in found.items()}, {p: v[1] for p, v in found.items()})
            if b.kind == "face" and res.llr:
                # the trait path's face veto: another person the face matches drops P, an unnamed group only when
                # the face hardly looks like P
                fs = face_scores.get(b.id) or {}
                strong = [o for o, s in fs.items() if s >= FACE_CONTRADICTION]
                if strong:
                    if named is None:
                        named = _named(self.conn, {o for scores in face_scores.values() for o in scores})
                    res = res.without({p for p in res.llr if any(
                        o != p and (o in named or fs.get(p, 0.0) < FACE_WEAK) for o in strong)})
            res = res.without(self.present)
            if res.llr:
                out[key] = res
        return out


def _named(conn: sqlite3.Connection, person_ids: set[int]) -> set[int]:
    ids = sorted({int(p) for p in person_ids})
    out: set[int] = set()
    for start in range(0, len(ids), 400):
        chunk = ids[start:start + 400]
        out.update(int(r[0]) for r in conn.execute(
            f"SELECT id FROM persons WHERE id IN ({','.join('?' * len(chunk))}) AND name IS NOT NULL AND name != ''", chunk))
    return out


def context(conn: sqlite3.Connection, item: sqlite3.Row, boxes: list[Box],
            cache: dict[str, Any] | None) -> PlaceContext | None:
    if not settings.place_enabled or not boxes:
        return None
    if item["kind"] != "image" or not item["taken_at"]:
        return None
    iid = int(item["id"])
    known = cache.get("place", {}).get("links", {}).get(iid) if cache is not None else None
    if known is None:
        hit = conn.execute(
            "SELECT 1 FROM place_links WHERE (a_item = ? OR b_item = ?) AND rev = ? AND kind != 'none' LIMIT 1",
            (iid, iid, place.PLACE_REV)).fetchone()
        if hit is None:
            return None
    elif not known:
        return None
    from app.analysis.guess import _present_people

    return PlaceContext(conn, item, boxes, cache, present=_present_people(conn, iid))


# ------------------------------------------------------------------ fusion
def posterior(llr: float) -> float:
    """sigmoid(strength): the tiers are log odds of the measured share of right answers."""
    return float(1.0 / (1.0 + math.exp(-float(llr))))


def admitted(t: PlaceResult) -> int | None:
    """The person a position-only guess may name, or None."""
    if not (settings.place_enabled and settings.place_only_guesses):
        return None
    pid, best, runner = t.best()
    if pid is None or best < ADMIT_LLR or best - max(0.0, runner) < LEAD_LLR:
        return None
    return pid


def _with_place(c: Candidate, t: PlaceResult, pid: int) -> Candidate:
    from app.analysis.guess import Candidate as Cand

    ev = copy.deepcopy(c.evidence)
    ev.setdefault("score", round(float(c.score), 4))  # a tied candidate has none yet
    if not ev.get("basis"):
        ev["basis"] = "clothes+place"
    ev["place_llr"] = round(float(t.llr[pid]), 2)
    ev["place"] = copy.deepcopy(t.evidence[pid])
    return Cand(c.person_id, c.score, ev)


def _place_candidate(t: PlaceResult, pid: int) -> Candidate:
    from app.analysis.guess import Candidate as Cand

    llr = float(t.llr[pid])
    score = round(min(MAX_SCORE, posterior(llr)), 4)
    frag = copy.deepcopy(t.evidence[pid])
    ev = {"basis": "place", "score": score, "place_llr": round(llr, 2), "ref_item_uid": frag.get("ref_item_uid"),
          "ref_face_id": frag.get("ref_face_id"), "ref_body_id": frag.get("ref_body_id"),
          "seconds_apart": frag.get("seconds_apart"), "place": frag}
    return Cand(pid, score, ev)


def fuse(win: Candidate | None, c_ranked: list[Candidate], t: PlaceResult) -> Candidate | None:
    """The second fuse step (see the module contract)."""
    from app.analysis.guess import MARGIN

    if win is not None:
        pid = win.person_id
        other = admitted(t)
        if other is not None and other != pid:
            return None
        if t.llr.get(pid, 0.0) > 0:
            return _with_place(win, t, pid)
        return win
    if c_ranked:
        top = c_ranked[0].score
        tied = [c for c in c_ranked if top - c.score < MARGIN]
        backed = [c for c in tied if t.llr.get(c.person_id, 0.0) >= TIE_LLR]
        if len(backed) == 1 and all(t.llr.get(c.person_id, 0.0) <= 0 for c in tied if c is not backed[0]):
            return _with_place(backed[0], t, backed[0].person_id)
    pid = admitted(t)
    if pid is not None and (not c_ranked or pid in {c.person_id for c in c_ranked}):
        return _place_candidate(t, pid)
    return win


def outcome_terms(evidence: dict[str, Any]) -> dict[str, Any]:
    """place_llr, the gap, the hops and how the photos lined up, of a 'place' guess's evidence."""
    frag = evidence.get("place") if isinstance(evidence, dict) else None
    frag = frag if isinstance(frag, dict) else {}
    return {"place_llr": evidence.get("place_llr") if isinstance(evidence, dict) else None,
            "seconds_apart": frag.get("seconds_apart"), "hops": frag.get("hops"), "kind": frag.get("kind"),
            "others": frag.get("others")}
