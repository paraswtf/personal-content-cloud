"""The looks of a person, for the 3D heads (docs/ARCHITECTURE.md section 36). onnxruntime and numpy only.

A look is how someone looked for a while: an age band, how much facial hair and how much hair. plan() reads a
person's good face photos (at most MAX_SCORED, spread over the years), puts them into bins of (age band, facial
hair, hair), keeps the fullest bins (at most max_looks, every age band first) and picks the photo of each that makes
the best input for the generator, with three runners-up. write_picture() cuts that photo's head and shoulders from
the original at full quality.

What is read per face:
- from a square HEAD_CROP face sizes wide around it in the 2048 px preview (what the generator will see): the hair as
  a level in 0..1, zero-shot with SigLIP over the ladder below; a covered head and a hand or an object at the face
  the same way; the head's turn, nod and tilt from the face detector's eye, nose and mouth points on that crop;
- from the stored face crop (the tight view shows it better): the facial hair as a level, sunglasses and glasses,
  and the light on the face;
- from the pose of the person box around the face, when there is one (DWPose, section 34): a wrist at the face, or a
  wrist or an elbow up beside the head;
- the age: FairFace's expected age (app.analysis.ages), and for photos with a camera time that are no forwards the
  year minus the person's birth year, which is the median of (year minus expected age) over such adult faces.
A photo is counted against as an input (Reading.faults) when another face of the photo is inside the square, the
square leaves the picture, a hand, an arm or an object is at the face or up by the head, the head is turned or
tilted far, the eyes are behind sunglasses, the head is covered, or the light on the face is coloured or very dark.
Looks are made from photos without a fault (a turned head or a second person in the crop also bends the hair and
facial hair readings), and a day's photos share the day's median levels, as hair does not change within a day. Only
when no look comes out of that do the faulty photos count. Without SigLIP the hair and facial hair are unknown and
the bins are age bands only; without the face detector the turn is unknown.

Measured on the owner's library (2026-11-13, a person with 3,907 source faces, 397 read, and 320 clear faces of 185
other people; docs section 36): the facial hair ladder on the face crop puts 87 percent of the other people (mostly
clean faces) under that bearded person's 10th percentile, against 70 percent on the head crop and 53 to 75 percent
for two-way and three-way prompt sets; clean faces read about 0.28 to 0.44, stubble and short beards 0.44 to 0.53,
beards above. The hair ladder reads shaved heads and caps under 0.3, short hair around 0.35 to 0.55, long hair above
0.76. Short noun prompts ("a hat", "sunglasses", "a phone in front of the face") separate cleanly where sentences
did not: the first covered-head set flagged 52 of 397 bare-headed photos, the one below none of them and 17 of 320
other faces, all with a hat, cap, turban, helmet or hood. The zero-shot "at the face" flag finds phones, bottles and
hands on the face but not raised arms or a hand under the chin, which is what the pose check is for. With every check
on, 38 of that person's 237 sampled photos had no fault, which is plenty: a look needs three.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.analysis import ages
from app.analysis.faces import face_weight
from app.util import parse_iso

log = logging.getLogger(__name__)

MIN_WEIGHT = 0.6           # face weight (section 10) a source face needs
MIN_FACE_PX = 120          # and its shorter side in source pixels
MAX_SCORED = 240           # faces read per plan (two SigLIP passes, a detection and a pose: 0.8 s a face on a CPU)
YEAR_FLOOR = 12            # every year keeps at least this many of its faces in the sample
MIN_PHOTOS = 3             # usable photos a bin needs to be a look
CHILD_MIN_PHOTOS = 1       # ... in the child band, where photos are few
MIN_USABLE = 3             # good face photos a person needs before a model can be made
MIN_BIRTH_FACES = 10       # dated adult faces needed to tell the birth year
CHILD_AGE = 13.0           # a face read this young keeps its read age (a re-shot print carries today's date)
HEAD_CROP = 2.6            # the reading square, in face sizes
READ_SIDE = 384            # and its side in pixels for the models
PICTURE_SIDE = 1024        # the generator's input picture
FRAMING_MIN, FRAMING_MAX, FRAMING_DEFAULT = 1.8, 3.2, 2.6
FACE_ABOVE = 0.08          # the face centre sits this share of the square above its middle
CUT_OFF_SHARE = 0.15       # the square may leave the picture by this share of its area
OTHER_FACE_INSIDE = 0.30   # another face counts when this share of it is inside the square ...
OTHER_FACE_SIZE = 0.30     # ... and it is at least this large against the face itself
# Head pose bounds, set by eye on the owner's photos: about a three-quarter view at 0.28, a profile above 1;
# chin up under 0.36, looking down above 0.75 (0.57 is the median facing the camera).
YAW_MAX = 0.28             # nose offset from the eye to mouth line, in eye distances
PITCH_MIN, PITCH_MAX = 0.36, 0.75  # where the nose sits between the eye line (0) and the mouth line (1)
ROLL_MAX = 25.0            # degrees the eye line may lean
# zero-shot probability from which a flag counts
P_FLAG = {"covered_head": 0.50, "sunglasses": 0.60, "covered_face": 0.40}
# The light on the face, from the middle of its crop in CIE Lab: skin of any tone is on the red and yellow side
# (a and b above zero), so a face that is not has coloured light on it (or the photo is black and white); the
# generator would copy that colour onto the head. On the test person's 297 faces: a under 1 or b under 2 on 9 percent.
LIGHT_L_MIN, LIGHT_A_MIN, LIGHT_B_MIN = 28.0, 1.0, 2.0
LIGHT_L_GOOD = 40.0        # from here up the light costs a photo nothing in the ranking
POSE_SEEN = 0.4            # a pose point counts from this score (the usual cut for DWPose)
HAND_AT_FACE = 1.1         # a wrist this near the face centre, in face sizes
ARM_UP_REACH = 1.6         # a raised wrist or elbow counts within this many face sizes to either side of the face
ARM_UP_ABOVE = 0.15        # ... when it is this far above the shoulder line, in face sizes
LOOKS_MIN, LOOKS_MAX, LOOKS_DEFAULT = 1, 12, 6
RUNNERS_UP = 3

AGE_BANDS: tuple[tuple[float, float, str], ...] = (
    (0, 10, "under 10"), (10, 18, "10 to 17"), (18, 25, "18 to 24"), (25, 35, "25 to 34"), (35, 45, "35 to 44"),
    (45, 60, "45 to 59"), (60, 200, "60 and over"),
)
FACIAL_BINS: tuple[tuple[float, str], ...] = ((0.44, "little or no facial hair"), (0.53, "stubble or a short beard"),
                                              (9.0, "a beard"))
HAIR_BINS: tuple[tuple[float, str], ...] = ((0.30, "shaved or very short hair"), (0.52, "short hair"),
                                            (0.76, "medium hair"), (9.0, "long hair"))

# Zero-shot ladders: (level, prompts). The reading is the probability-weighted level (SiglipEngine.zero_shot).
FACIAL_LADDER: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.0, ("a photo of a clean-shaven face", "a photo of a person with no beard and no moustache")),
    (0.3, ("a photo of a face with light stubble", "a photo of a person with a few days of stubble")),
    (0.4, ("a photo of a person with a moustache and no beard",)),
    (0.65, ("a photo of a person with a short trimmed beard",)),
    (1.0, ("a photo of a person with a full thick beard", "a photo of a person with a long beard")),
)
HAIR_LADDER: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.0, ("a photo of a person with a bald head", "a photo of a person with a shaved head")),
    (0.2, ("a photo of a person with very short cropped hair", "a photo of a person with a buzz cut")),
    (0.4, ("a photo of a person with short hair",)),
    (0.65, ("a photo of a person with medium length hair", "a photo of a person with hair down to the chin")),
    (1.0, ("a photo of a person with long hair", "a photo of a person with hair below the shoulders")),
)
# Flags: the first group is the flag, the rest are its opposites. Short noun prompts, on purpose (see above).
COVERED_HEAD: tuple[tuple[str, ...], ...] = (
    ("a hat", "a cap", "a beanie", "a helmet", "a hood", "a turban", "a headscarf"),
    ("hair", "a hairstyle", "a haircut"),
)
GLASSES: tuple[tuple[str, ...], ...] = (
    ("sunglasses", "dark sunglasses", "tinted glasses"),
    ("eyeglasses", "clear glasses", "spectacles"),
    ("a face without glasses", "bare eyes"),
)
COVERED_FACE: tuple[tuple[str, ...], ...] = (
    ("a hand on the face", "a hand over the mouth", "hands on the cheeks", "a hand touching the chin",
     "arms raised above the head", "holding something above the head", "a bottle at the mouth", "a cup at the mouth",
     "a phone in front of the face", "a mask over the mouth"),
    ("a face", "a portrait", "a clear view of a face", "a head and shoulders portrait"),
)
# The flags the zero-shot readings may raise. One that stops separating on real photos is taken out here; its
# probability is still stored.
ZERO_SHOT_FAULTS = frozenset({"covered_head", "sunglasses", "covered_face"})
FAULTS = ("other_face", "cut_off", "turned", "covered_face", "raised_hand", "sunglasses", "covered_head", "odd_light")

FAULT_WORDS = {
    "other_face": "another face is in the picture",
    "cut_off": "the head or shoulders leave the picture",
    "covered_face": "a hand or an object is at the face",
    "raised_hand": "a hand or an arm is up by the head",
    "turned": "the head is turned or tilted far",
    "sunglasses": "sunglasses",
    "covered_head": "the head is covered",
    "odd_light": "the light on the face is coloured or very dark",
}


class NoPhotos(Exception):
    """The person has too few good face photos for a model."""


@dataclass
class Reading:
    face_id: int
    item_id: int
    item_uid: str
    taken_at: str | None
    year: float | None                       # the photo's year (with its fraction) when the time is the camera's own
    box: tuple[float, float, float, float]   # the face box, normalised to the upright picture
    width: int                               # the picture's upright size
    height: int
    weight: float
    size_px: int
    sharpness: float | None
    expected_age: float | None
    child: bool
    has_body: bool = False
    shoulders: bool | None = None            # the person box reaches well below the chin (None: no person box)
    age: float | None = None
    facial_hair: float | None = None
    hair: float | None = None
    covered_head: float | None = None
    sunglasses: float | None = None
    glasses: float | None = None
    covered_face: float | None = None
    light: tuple[float, float, float] | None = None  # CIE L (0 to 100), a and b of the middle of the face crop
    hand_up: bool | None = None              # the pose shows a hand at the face or an arm up (None: no pose)
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    outside: float = 0.0
    faults: list[str] = field(default_factory=list)

    def score(self) -> float:
        """How good an input the photo is, apart from its faults: clear, large, sharp, facing the camera."""
        size = min(1.0, self.size_px / 400.0)
        sharp = 1.0 if self.sharpness is None else min(1.0, max(0.0, self.sharpness) / 40.0)
        frontal = 1.0
        if self.yaw is not None:
            frontal *= max(0.2, 1.0 - abs(self.yaw) / YAW_MAX * 0.6)
        if self.pitch is not None:
            frontal *= max(0.2, 1.0 - abs(self.pitch - 0.55) / 0.35 * 0.4)
        body = 1.0 if self.shoulders else 0.85 if self.shoulders is None else 0.7
        light = 1.0
        if self.light is not None:
            light = min(1.0, max(0.4, (self.light[0] - LIGHT_L_MIN) / (LIGHT_L_GOOD - LIGHT_L_MIN) * 0.6 + 0.4))
        return float(self.weight * (0.4 + 0.6 * size) * (0.5 + 0.5 * sharp) * frontal * body * light)


@dataclass
class Look:
    key: str
    age: float | None
    age_band: str
    hair: float | None
    hair_label: str | None
    facial_hair: float | None
    facial_hair_label: str | None
    photos: int
    face_id: int
    item_id: int
    item_uid: str
    taken_at: str | None
    box: tuple[float, float, float, float]
    faults: list[str]
    runners_up: list[int]                    # face ids, best first

    @property
    def label(self) -> str:
        return look_label(self.age, self.age_band, self.hair_label, self.facial_hair_label)


@dataclass
class Plan:
    looks: list[Look]
    scanned: int                             # faces read
    usable: int                              # of them, in some bin
    birth_year: float | None
    readings: list[Reading] = field(default_factory=list)


# ------------------------------------------------------------------ geometry (pure)
def square_for(box: tuple[float, float, float, float], width: int, height: int,
               framing: float = HEAD_CROP) -> tuple[float, float, float]:
    """(left, top, side) in pixels of a width x height picture: a square `framing` face sizes wide around the face,
    the face centre FACE_ABOVE of the side above the square's middle. It may leave the picture."""
    x, y, w, h = box
    fw, fh = w * width, h * height
    side = float(framing) * max(fw, fh)
    cx = (x + w / 2.0) * width
    cy = (y + h / 2.0) * height + FACE_ABOVE * side
    return cx - side / 2.0, cy - side / 2.0, side


def outside_share(square: tuple[float, float, float], width: int, height: int) -> float:
    """The share of the square's area that lies outside the picture."""
    left, top, side = square
    if side <= 0:
        return 1.0
    iw = max(0.0, min(left + side, float(width)) - max(left, 0.0))
    ih = max(0.0, min(top + side, float(height)) - max(top, 0.0))
    return float(1.0 - (iw * ih) / (side * side))


def other_face_inside(square: tuple[float, float, float], face_px: tuple[float, float, float, float],
                      others_px: list[tuple[float, float, float, float]]) -> bool:
    """Another face of the photo (boxes as x, y, w, h in pixels) is in the square: at least OTHER_FACE_INSIDE of it,
    and it is at least OTHER_FACE_SIZE of the face's own size (a far-away face in the background is let be)."""
    left, top, side = square
    own = max(face_px[2], face_px[3])
    for ox, oy, ow, oh in others_px:
        if ow <= 0 or oh <= 0 or max(ow, oh) < OTHER_FACE_SIZE * own:
            continue
        iw = max(0.0, min(ox + ow, left + side) - max(ox, left))
        ih = max(0.0, min(oy + oh, top + side) - max(oy, top))
        if iw * ih >= OTHER_FACE_INSIDE * ow * oh:
            return True
    return False


def head_pose(kps: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) from the detector's five points (left eye, right eye, nose, mouth corners):
    yaw: the nose's offset from the line between the eyes' middle and the mouth's middle, in eye distances (0 facing
    the camera, sign = side); pitch: where the nose sits between the eye line (0) and the mouth line (1), about 0.55
    facing the camera, smaller with the head tilted back; roll: the lean of the eye line in degrees."""
    k = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    eyes, mouth, nose = (k[0] + k[1]) / 2.0, (k[3] + k[4]) / 2.0, k[2]
    eye_dist = float(np.linalg.norm(k[1] - k[0]))
    axis = mouth - eyes
    length = float(np.linalg.norm(axis))
    if eye_dist < 1e-6 or length < 1e-6:
        return 0.0, 0.55, 0.0
    unit = axis / length
    rel = nose - eyes
    along = float(rel @ unit)
    across = float(rel[0] * unit[1] - rel[1] * unit[0])
    roll = math.degrees(math.atan2(float(k[1][1] - k[0][1]), float(k[1][0] - k[0][0])))
    return across / eye_dist, along / length, roll


def cut(rgb: np.ndarray, square: tuple[float, float, float], out_side: int) -> np.ndarray:
    """The square cut from an RGB picture and resized to out_side; where it leaves the picture the edge is
    repeated."""
    import cv2

    h, w = rgb.shape[:2]
    left, top, side = square
    x0, y0 = int(math.floor(left)), int(math.floor(top))
    s = max(2, int(round(side)))
    x1, y1 = x0 + s, y0 + s
    pad = (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))
    part = rgb[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if part.size == 0:
        return np.zeros((out_side, out_side, 3), dtype=np.uint8)
    if any(pad):
        part = cv2.copyMakeBorder(part, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if part.shape[0] > out_side else cv2.INTER_CUBIC
    return cv2.resize(part, (out_side, out_side), interpolation=interp)


# ------------------------------------------------------------------ bins and the choice (pure)
def age_band(age: float | None) -> int | None:
    if age is None or not math.isfinite(age):
        return None
    for i, (lo, hi, _label) in enumerate(AGE_BANDS):
        if lo <= age < hi:
            return i
    return None


def _bin(level: float | None, bins: tuple[tuple[float, str], ...]) -> int | None:
    if level is None:
        return None
    for i, (below, _label) in enumerate(bins):
        if level < below:
            return i
    return len(bins) - 1


def bin_of(r: Reading) -> tuple[int, int | None, int | None] | None:
    """(age band, facial hair bin, hair bin) of a reading, or None when it cannot be placed: no age, or a covered
    head (the hair under a cap is anyone's guess). Unknown levels (no SigLIP) give None in their place."""
    band = age_band(r.age)
    if band is None or "covered_head" in r.faults:
        return None
    return band, _bin(r.facial_hair, FACIAL_BINS), _bin(r.hair, HAIR_BINS)


def look_label(age: float | None, band: str, hair: str | None, facial: str | None) -> str:
    first = f"About {int(round(age))}" if age is not None else band.capitalize()
    return ", ".join([first, *[p for p in (hair, facial) if p]])


def settle_ages(readings: list[Reading]) -> float | None:
    """Set every reading's age and return the birth year (None with too few dated adult faces)."""
    births = [r.year - r.expected_age for r in readings
              if r.year is not None and r.expected_age is not None and not r.child and r.expected_age > CHILD_AGE]
    birth = float(np.median(births)) if len(births) >= MIN_BIRTH_FACES else None
    for r in readings:
        young = r.child or (r.expected_age is not None and r.expected_age <= CHILD_AGE)
        if birth is not None and r.year is not None and not young and 0.0 <= r.year - birth <= 110.0:
            r.age = float(r.year - birth)
        else:
            r.age = r.expected_age
    return birth


def share_day_levels(readings: list[Reading]) -> None:
    """Hair does not change within a day: every dated photo takes the median hair and facial hair levels of its
    day's photos, from those without a fault when the day has any (a turned head or a second person in the crop
    bends a single reading)."""
    days: dict[str, list[Reading]] = defaultdict(list)
    for r in readings:
        if r.year is not None and r.taken_at:
            days[r.taken_at[:10]].append(r)
    for group in days.values():
        if len(group) < 2:
            continue
        for name in ("hair", "facial_hair"):
            clean = [getattr(r, name) for r in group if not r.faults and getattr(r, name) is not None]
            every = [getattr(r, name) for r in group if getattr(r, name) is not None]
            values = clean or every
            if not values:
                continue
            level = round(float(np.median(values)), 3)
            for r in group:
                if getattr(r, name) is not None:
                    setattr(r, name, level)


def best_first(readings: list[Reading]) -> list[Reading]:
    """Fewest faults first, then the best score, then the lower face id (stable)."""
    return sorted(readings, key=lambda r: (len(r.faults), -r.score(), r.face_id))


def usable(r: Reading) -> bool:
    """A photo a look may be made from: it can be placed in a bin and has no fault."""
    return not r.faults and bin_of(r) is not None


def choose(readings: list[Reading], max_looks: int = LOOKS_DEFAULT) -> list[Look]:
    """The looks: bins of photos without a fault that hold enough of them, first the fullest bin of every age band
    (the fullest bands first when max_looks is smaller than their number), then the fullest remaining; listed by
    age, hair, facial hair. When no bin has enough such photos, the one fullest bin of all the photos that can be
    placed, faults or not, so a person with few photos still gets a head from their best one."""
    cap = max(LOOKS_MIN, min(LOOKS_MAX, int(max_looks)))
    bins: dict[tuple[int, int | None, int | None], list[Reading]] = defaultdict(list)
    for r in readings:
        if usable(r):
            bins[bin_of(r)].append(r)  # type: ignore[index]
    ok = {b: rs for b, rs in bins.items() if len(rs) >= (CHILD_MIN_PHOTOS if b[0] == 0 else MIN_PHOTOS)}
    if not ok:
        loose: dict[tuple[int, int | None, int | None], list[Reading]] = defaultdict(list)
        for r in readings:
            b = bin_of(r)
            if b is not None:
                loose[b].append(r)
        if not loose:
            return []
        top = min(loose, key=lambda b: (-len(loose[b]), b[0], -1 if b[2] is None else b[2], -1 if b[1] is None else b[1]))
        ok, cap = {top: loose[top]}, 1
    order = lambda b: (-len(ok[b]), b[0], -1 if b[2] is None else b[2], -1 if b[1] is None else b[1])  # noqa: E731
    fullest: dict[int, tuple[int, int | None, int | None]] = {}
    for b in sorted(ok, key=order):
        fullest.setdefault(b[0], b)
    picked = sorted(fullest.values(), key=order)[:cap]
    rest = [b for b in sorted(ok, key=order) if b not in picked]
    picked += rest[:cap - len(picked)]
    picked.sort(key=lambda b: (b[0], -1 if b[2] is None else b[2], -1 if b[1] is None else b[1]))
    looks: list[Look] = []
    for n, b in enumerate(picked, start=1):
        ranked = best_first(ok[b])
        top = ranked[0]
        known = lambda vals: [v for v in vals if v is not None]  # noqa: E731
        ages_ = known([r.age for r in ok[b]])
        hairs, facials = known([r.hair for r in ok[b]]), known([r.facial_hair for r in ok[b]])
        looks.append(Look(
            key=f"l{n}",
            age=round(float(np.median(ages_)), 1) if ages_ else None,
            age_band=AGE_BANDS[b[0]][2],
            hair=round(float(np.median(hairs)), 3) if hairs else None,
            hair_label=None if b[2] is None else HAIR_BINS[b[2]][1],
            facial_hair=round(float(np.median(facials)), 3) if facials else None,
            facial_hair_label=None if b[1] is None else FACIAL_BINS[b[1]][1],
            photos=len(ok[b]), face_id=top.face_id, item_id=top.item_id, item_uid=top.item_uid, taken_at=top.taken_at,
            box=top.box, faults=list(top.faults), runners_up=[r.face_id for r in ranked[1:1 + RUNNERS_UP]],
        ))
    return looks


def spread(rows: list[Any], limit: int = MAX_SCORED) -> list[Any]:
    """At most `limit` rows, the best of every year: each year keeps YEAR_FLOOR of its rows (all of them when it has
    fewer), the rest of the room goes to the years by their share. Rows need "year" (None: undated), "child",
    "quality" and "id"; child-band rows are all kept."""
    kids = [r for r in rows if r["child"]]
    rest = [r for r in rows if not r["child"]]
    room = max(0, limit - len(kids))
    if len(rest) <= room:
        return kids + rest
    by_year: dict[Any, list[Any]] = defaultdict(list)
    for r in rest:
        by_year[None if r["year"] is None else int(r["year"])].append(r)
    for group in by_year.values():
        group.sort(key=lambda r: (-r["quality"], r["id"]))
    take = {y: min(len(g), YEAR_FLOOR) for y, g in by_year.items()}
    left = room - sum(take.values())
    if left < 0:  # more years than room: the floor shrinks evenly
        per = max(1, room // max(1, len(by_year)))
        take = {y: min(len(g), per) for y, g in by_year.items()}
        left = max(0, room - sum(take.values()))
    spare = {y: len(g) - take[y] for y, g in by_year.items()}
    total_spare = sum(spare.values())
    if left > 0 and total_spare > 0:
        for y in sorted(by_year, key=lambda v: (v is None, v)):
            take[y] += min(spare[y], int(left * spare[y] / total_spare))
    picked = [r for y, g in by_year.items() for r in g[:take[y]]]
    return kids + picked


# ------------------------------------------------------------------ reading the faces
@dataclass
class Engines:
    siglip: Any | None
    faces: Any | None
    pose: Any | None = None


def engines() -> Engines:
    """The SigLIP, face and pose engines when their models are present and allowed (PCC_NO_MODELS turns them
    off). Each is optional: what it would have read stays unknown."""
    from app.analysis.pipeline import _model

    out = Engines(siglip=None, faces=None, pose=None)
    for name in ("siglip", "faces", "pose"):
        mod = _model(name)
        if mod is None:
            continue
        try:
            setattr(out, name, mod.get_engine())
        except Exception as e:  # noqa: BLE001 - a model that does not load is a missing model
            log.warning("looks: the %s model did not load: %s", name, e)
    return out


def hand_at_head(kps: np.ndarray, face_px: tuple[float, float, float, float]) -> bool:
    """Whether a pose (133 COCO-WholeBody points with scores, in the picture's pixels) shows a hand at the face or
    an arm up beside the head: a seen wrist within HAND_AT_FACE face sizes of the face centre, or a seen wrist or
    elbow more than ARM_UP_ABOVE face sizes above the shoulder line and within ARM_UP_REACH to either side."""
    k = np.asarray(kps, dtype=np.float64)
    x, y, w, h = face_px
    size = max(w, h)
    if size <= 0 or k.shape[0] < 11:
        return False
    cx, cy = x + w / 2.0, y + h / 2.0
    seen = lambda i: k[i, 2] >= POSE_SEEN  # noqa: E731
    for i in (9, 10):  # wrists
        if seen(i) and math.hypot(k[i, 0] - cx, k[i, 1] - cy) <= HAND_AT_FACE * size:
            return True
    shoulders = [k[i, 1] for i in (5, 6) if seen(i)]
    line = min(shoulders) if shoulders else y + h + 0.5 * size
    for i in (7, 8, 9, 10):  # elbows and wrists
        if seen(i) and k[i, 1] < line - ARM_UP_ABOVE * size and abs(k[i, 0] - cx) <= ARM_UP_REACH * size:
            return True
    return False


SOURCE_SQL = (
    "SELECT f.id, f.item_id, f.x, f.y, f.w, f.h, f.det_score, f.size_px, f.clarity, f.sharpness, f.age_groups, "
    "i.uid, i.version, i.ext, i.kind, i.width, i.height, i.taken_at, i.exif_datetime, i.forward "
    "FROM faces f JOIN items i ON i.id = f.item_id "
    "WHERE f.person_id = ? AND i.kind = 'image' AND i.deleted_at IS NULL AND i.nsfw = 0 "
    "AND i.width > 0 AND i.height > 0 AND f.size_px >= ? ORDER BY f.id"
)


def _year(taken_at: str | None, exif_datetime: str | None, forward: Any) -> float | None:
    """The photo's year with its fraction when the time is the camera's own (a forward's or a re-saved file's date
    says nothing about when the picture was taken)."""
    if not taken_at or not exif_datetime or forward:
        return None
    t = parse_iso(taken_at)
    if t is None:
        return None
    return t.year + (t.timetuple().tm_yday - 1) / 365.25


def source_rows(conn: sqlite3.Connection, person_id: int) -> list[dict[str, Any]]:
    """The person's faces a look may come from: on photos, the item live and outside the vault, clear and large."""
    out: list[dict[str, Any]] = []
    banded = ages.ready(conn)  # the child band counts only once the library's ages are all read (section 10)
    for r in conn.execute(SOURCE_SQL, (int(person_id), MIN_FACE_PX)):
        w = face_weight(r["det_score"], r["size_px"], r["clarity"], r["sharpness"])
        if w < MIN_WEIGHT:
            continue
        d = dict(r)
        d["weight"] = float(w)
        d["child"] = bool(banded and ages.in_child_band(r["age_groups"], w))
        d["year"] = _year(r["taken_at"], r["exif_datetime"], r["forward"])
        d["quality"] = float(w) * min(1.0, r["size_px"] / 400.0)
        out.append(d)
    return out


def usable_photos(conn: sqlite3.Connection, person_id: int) -> int:
    """How many good face photos the person has (cheap: no pixels are read)."""
    return len(source_rows(conn, person_id))


def _flag(engine: Any, embs: np.ndarray, groups: tuple[tuple[str, ...], ...]) -> np.ndarray:
    """P(first group) for each embedding, a softmax over the groups."""
    return engine.zero_shot(embs, {str(i): list(g) for i, g in enumerate(groups)})[:, 0]


def _ladder(engine: Any, embs: np.ndarray, ladder: tuple[tuple[float, tuple[str, ...]], ...]) -> np.ndarray:
    probs = engine.zero_shot(embs, {str(i): list(p) for i, (_lv, p) in enumerate(ladder)})
    return probs @ np.asarray([lv for lv, _p in ladder], dtype=np.float32)


def _face_points(engine: Any, crop_rgb: np.ndarray, side: float, framing: float) -> np.ndarray | None:
    """The detector's five points for the face in the middle of a head crop, or None when it finds none there."""
    bgr = np.ascontiguousarray(crop_rgb[:, :, ::-1])
    found = engine.detect(bgr, det_size=READ_SIDE, threshold=0.3)
    if not found:
        return None
    # the face was cut to sit at the middle, FACE_ABOVE of the side above it
    cx, cy = side / 2.0, side / 2.0 - FACE_ABOVE * side
    want = side / framing
    best, best_d = None, None
    for f in found:
        fx, fy = float(f.bbox[0] + f.bbox[2]) / 2.0, float(f.bbox[1] + f.bbox[3]) / 2.0
        d = math.hypot(fx - cx, fy - cy)
        if d <= 0.5 * want and (best_d is None or d < best_d):
            best, best_d = f, d
    return None if best is None else np.asarray(best.kps, dtype=np.float32)


def _face_crop(face_id: int, rgb: np.ndarray, face_px: tuple[float, float, float, float]) -> Any:
    """The tight view of a face as a PIL RGB picture: its stored crop, else the face box cut from the picture."""
    from PIL import Image

    from app import storage

    try:
        with Image.open(storage.face_crop_path(face_id)) as im:
            return im.convert("RGB")
    except Exception:  # noqa: BLE001 - no stored crop: cut the box
        x, y, w, h = face_px
        side = 1.15 * max(w, h)
        return Image.fromarray(cut(rgb, (x + w / 2.0 - side / 2.0, y + h / 2.0 - side / 2.0, side), 192))


def _faults(r: Reading) -> list[str]:
    out: list[str] = []
    if r.outside > CUT_OFF_SHARE:
        out.append("cut_off")
    if r.yaw is not None and r.pitch is not None and r.roll is not None and (
            abs(r.yaw) > YAW_MAX or not PITCH_MIN <= r.pitch <= PITCH_MAX or abs(r.roll) > ROLL_MAX):
        out.append("turned")
    if r.covered_face is not None and "covered_face" in ZERO_SHOT_FAULTS and r.covered_face >= P_FLAG["covered_face"]:
        out.append("covered_face")
    if r.hand_up:
        out.append("raised_hand")
    for name, value in (("sunglasses", r.sunglasses), ("covered_head", r.covered_head)):
        if name in ZERO_SHOT_FAULTS and value is not None and value >= P_FLAG[name]:
            out.append(name)
    if r.light is not None and (r.light[0] < LIGHT_L_MIN or r.light[1] < LIGHT_A_MIN or r.light[2] < LIGHT_B_MIN):
        out.append("odd_light")
    return out


def face_light(face: Any) -> tuple[float, float, float]:
    """(L, a, b) of the middle half of a face crop (a PIL picture): L from 0 to 100, a and b around 0."""
    import cv2

    rgb = np.asarray(face.convert("RGB"))
    h, w = rgb.shape[:2]
    mid = np.ascontiguousarray(rgb[h // 4:max(h // 4 + 1, 3 * h // 4), w // 4:max(w // 4 + 1, 3 * w // 4)])
    lab = cv2.cvtColor(mid, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32).mean(axis=0)
    return round(float(lab[0]) * 100.0 / 255.0, 1), round(float(lab[1]) - 128.0, 1), round(float(lab[2]) - 128.0, 1)


def read_faces(conn: sqlite3.Connection, rows: list[dict[str, Any]], eng: Engines | None = None,
               progress: Callable[[int, int], None] | None = None) -> list[Reading]:
    """A Reading per source row. Pictures are read one photo at a time; SigLIP runs once per photo's faces."""
    from app.analysis.trait_regions import load_preview

    eng = eng if eng is not None else engines()
    by_item: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_item[int(r["item_id"])].append(r)
    out: list[Reading] = []
    done = 0
    for item_id, group in by_item.items():
        first = group[0]
        width, height = int(first["width"]), int(first["height"])
        others = [(o["id"], o["x"] * width, o["y"] * height, o["w"] * width, o["h"] * height)
                  for o in conn.execute("SELECT id, x, y, w, h FROM faces WHERE item_id = ?", (item_id,))]
        bodies = {b["face_id"]: b for b in conn.execute(
            "SELECT face_id, x, y, w, h FROM bodies WHERE item_id = ? AND face_id IS NOT NULL", (item_id,))}
        rgb = load_preview(first) if (eng.siglip is not None or eng.faces is not None or eng.pose is not None) else None
        crops: list[np.ndarray] = []
        tight: list[Any] = []
        made: list[Reading] = []
        for r in group:
            box = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            sq = square_for(box, width, height)
            face_px = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
            body = bodies.get(r["id"])
            rd = Reading(
                face_id=int(r["id"]), item_id=item_id, item_uid=r["uid"], taken_at=r["taken_at"], year=r["year"],
                box=box, width=width, height=height, weight=float(r["weight"]), size_px=int(r["size_px"]),
                sharpness=None if r["sharpness"] is None else float(r["sharpness"]),
                expected_age=ages.expected_age(r["age_groups"]), child=bool(r["child"]), has_body=body is not None,
                shoulders=None if body is None else bool((body["y"] + body["h"]) >= box[1] + box[3] + 0.6 * box[3]),
                outside=round(outside_share(sq, width, height), 4),
            )
            if other_face_inside(sq, face_px, [o[1:] for o in others if o[0] != r["id"]]):
                rd.faults.append("other_face")
            if rgb is not None:
                # the preview has the picture's shape at another size: the same square, scaled
                scale = rgb.shape[1] / float(width)
                crop = cut(rgb, (sq[0] * scale, sq[1] * scale, sq[2] * scale), READ_SIDE)
                crops.append(crop)
                tight.append(_face_crop(int(r["id"]), rgb, (face_px[0] * scale, face_px[1] * scale,
                                                             face_px[2] * scale, face_px[3] * scale)))
                try:
                    rd.light = face_light(tight[-1])
                except Exception as e:  # noqa: BLE001 - the light stays unknown
                    log.warning("looks: could not read the light of face %s: %s", r["id"], e)
                if eng.faces is not None:
                    try:
                        pts = _face_points(eng.faces, crop, float(READ_SIDE), HEAD_CROP)
                    except Exception as e:  # noqa: BLE001 - the turn stays unknown
                        log.warning("looks: the face detector failed on a crop: %s", e)
                        pts = None
                    if pts is not None:
                        yaw, pitch, roll = head_pose(pts)
                        rd.yaw, rd.pitch, rd.roll = round(yaw, 3), round(pitch, 3), round(roll, 1)
            made.append(rd)
        if eng.pose is not None and rgb is not None:
            scale = rgb.shape[1] / float(width)
            posed = [(rd, bodies[rd.face_id]) for rd in made if rd.face_id in bodies]
            try:
                boxes = [(b["x"] * rgb.shape[1], b["y"] * rgb.shape[0], (b["x"] + b["w"]) * rgb.shape[1],
                          (b["y"] + b["h"]) * rgb.shape[0]) for _rd, b in posed]
                found = eng.pose.estimate(np.ascontiguousarray(rgb[:, :, ::-1]), boxes) if boxes else []
                for (rd, _b), kps in zip(posed, found):
                    fx, fy, fw, fh = rd.box
                    rd.hand_up = hand_at_head(kps, (fx * width * scale, fy * height * scale, fw * width * scale,
                                                    fh * height * scale))
            except Exception as e:  # noqa: BLE001 - the hands stay unknown for this photo
                log.warning("looks: the pose model failed on item %s: %s", item_id, e)
        if eng.siglip is not None and crops:
            try:
                from PIL import Image

                embs = eng.siglip.embed_images([Image.fromarray(c) for c in crops])
                close = eng.siglip.embed_images(tight)
                facial, hair = _ladder(eng.siglip, close, FACIAL_LADDER), _ladder(eng.siglip, embs, HAIR_LADDER)
                covered = _flag(eng.siglip, embs, COVERED_HEAD)
                glasses = eng.siglip.zero_shot(close, {str(i): list(g) for i, g in enumerate(GLASSES)})
                lower = _flag(eng.siglip, embs, COVERED_FACE)
                for k, rd in enumerate(made):
                    rd.facial_hair, rd.hair = round(float(facial[k]), 3), round(float(hair[k]), 3)
                    rd.covered_head = round(float(covered[k]), 3)
                    rd.sunglasses, rd.glasses = round(float(glasses[k, 0]), 3), round(float(glasses[k, 1]), 3)
                    rd.covered_face = round(float(lower[k]), 3)
            except Exception as e:  # noqa: BLE001 - the levels stay unknown for this photo
                log.warning("looks: SigLIP failed on item %s: %s", item_id, e)
        for rd in made:
            rd.faults.extend(f for f in _faults(rd) if f not in rd.faults)
        out.extend(made)
        done += len(group)
        if progress is not None:
            progress(done, len(rows))
    return out


def plan(conn: sqlite3.Connection, person_id: int, max_looks: int = LOOKS_DEFAULT, *, eng: Engines | None = None,
         progress: Callable[[int, int], None] | None = None) -> Plan:
    """Read the person's good face photos and choose the looks. NoPhotos when there are fewer than MIN_USABLE."""
    rows = source_rows(conn, person_id)
    if len(rows) < MIN_USABLE:
        raise NoPhotos(f"{len(rows)} good face photos")
    sample = spread(rows, MAX_SCORED)
    readings = read_faces(conn, sample, eng, progress)
    birth = settle_ages(readings)
    share_day_levels(readings)
    looks = choose(readings, max_looks)
    return Plan(looks=looks, scanned=len(readings), usable=sum(1 for r in readings if usable(r)),
                birth_year=None if birth is None else round(birth, 1), readings=readings)


# ------------------------------------------------------------------ the generator's picture
def picture(conn: sqlite3.Connection, face_id: int, framing: float = FRAMING_DEFAULT,
            side: int = PICTURE_SIDE) -> np.ndarray | None:
    """The head and shoulders of a face as the generator's input: a square `framing` face sizes wide cut from the
    original at full quality (the analysis picture when the original does not read), RGB, side x side. None when the
    face or its pixels are gone."""
    from app.analysis.trait_regions import load_original, load_preview

    row = conn.execute(
        "SELECT f.x, f.y, f.w, f.h, i.uid, i.version, i.ext, i.kind, i.width, i.height FROM faces f "
        "JOIN items i ON i.id = f.item_id WHERE f.id = ? AND i.deleted_at IS NULL AND i.nsfw = 0", (int(face_id),)
    ).fetchone()
    if row is None:
        return None
    rgb = load_original(row)
    if rgb is None:
        rgb = load_preview(row)
    if rgb is None:
        return None
    framing = min(FRAMING_MAX, max(FRAMING_MIN, float(framing)))
    h, w = rgb.shape[:2]
    return cut(rgb, square_for((row["x"], row["y"], row["w"], row["h"]), w, h, framing), side)


def write_picture(conn: sqlite3.Connection, face_id: int, path: Path, framing: float = FRAMING_DEFAULT) -> bool:
    """Write picture() as a PNG. False when there is nothing to write."""
    from PIL import Image

    rgb = picture(conn, face_id, framing)
    if rgb is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "PNG", compress_level=3)
    return True


def reading_dict(r: Reading) -> dict[str, Any]:
    return asdict(r)
