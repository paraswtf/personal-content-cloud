"""Recognition trained on one person (docs/ARCHITECTURE.md sections 10 and 18).

The owner says a person's faces are right ("Train recognition on NAME alone"). train() then fits, for that person
only, a small model on the frozen ArcFace vectors: a logistic regression on the 512 values (the person's faces against
faces known not to be theirs) stacked with the usual top-3 similarity and the face weight into one probability. It is
not in train.LEARNERS: nothing retrains it but the owner. One file per person, learned/person_<id>.json, so a learning
reset (store.clear) forgets it like every learned file, and with learning switched off load() answers None.

Why not the top-3 score alone: on the owner's library two other people score as high against the person as the
person's own photos do, so one cut on that score either lets them in or keeps a quarter of the person's own faces
out. The regression is trained with those people's faces as negatives (the faces the owner took off the person) and
separates them; for blurred, small or turned faces the two together find more at the same number of mistakes.

Examples:
- positives: the person's faces of weight > 0 outside the child band, each counted by its face weight (times
  index.USER_WEIGHT when the owner assigned it);
- trusted negatives: (a) faces the owner took off the person (guess_rejections), (b) faces the owner assigned to
  someone else, (c) the other faces of photos the person has a face in, except ambiguous ones (they look like the
  person at AMBIGUOUS or more: a mirror, a print or a collage shows one person twice) unless such a face sits on
  another NAMED person it matches better by AMBIGUOUS_LEAD (an unnamed group may hold the person's own faces, so
  it never makes a face a negative). Never the faces the app put on other people elsewhere, nor loose faces:
  those are what a rescan looks at (app.analysis.person_scan);
- conflicts: a positive the owner did not lock that looks like an (a) or (b) negative (cosine CONFLICT_SIM or
  more: a suspect) and that the person's trusted faces do not vouch for. Trusted are the owner's own faces, the
  faces that are no suspects, and each suspect whose top-3 to the trusted faces reaches its cosine to that
  negative; this repeats until no more suspect is cleared, so suspects never vouch for one another (a group of
  one look-alike's faces on the person would clear itself otherwise). The plain rule (nearer to the negative
  than its top-3 to ALL of the person's other faces) always counts, and it alone counts when the rule above
  would set aside more face weight than it keeps: the trusted faces do not stand for the person then. Conflicts
  are left out of the training and of the reference, and listed for the owner to settle.

The check is honest about new photos: five folds by the local calendar day of the photo, so no fold is tested on a
day it was trained on, and the stacker is checked nested (fitted on four folds of out-of-fold rows, scored on the
fifth). A photo without a date folds by itself, and when the days do not spread the examples (a person seen on
one day) every photo does: the check records which folding it ran (check.folds, 'day' or 'photo'), and the per-photo
one is the weaker test, since photos of one day share clothes, light and angle. Everything is deterministic: two
trainings on the same rows give the same file but for the time.

State outside the file: whether the model is paused (its guesses were taken back) is persons.recognition_paused_at,
a journaled column, so the undo of taking the guesses back un-pauses it. forget() removes the file through
app.undo.discard, so the undo of an owner action that deleted or merged the person brings the model back with them.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import zlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.config import settings
from app.learning import store
from app.util import local_date, now_iso

log = logging.getLogger(__name__)

MODEL_REV = 1
DIM = 512
MIN_POSITIVE_WEIGHT = 20.0  # total face weight of the positives a training needs
MIN_NEGATIVES = 20
FOLDS = 5
CONFLICT_SIM = 0.55   # cluster.FOLLOW_SIM: this close to a face the owner took off looks like the same wrong person
CONFLICT_ROUNDS = 50  # how often the conflict rule looks again at the suspects the trusted faces have not cleared
AMBIGUOUS = 0.5       # another face of a photo the person is in that scores this against them is not a sure negative
AMBIGUOUS_LEAD = 0.05
GUESS_P = 0.5         # a face at this probability or more becomes a guess
SURE_PRECISION = 0.995
GATE_P = 0.2          # under this the automatic matching does not put a face on the person (cluster._score)
UNCLEAR_WEIGHT = 0.5  # faces.CLEAR_WEIGHT: the check reports blurred, small or turned faces apart
CONFLICTS_KEPT = 500
_CHUNK = 2048
_LR = dict(l2=1e-3, steps=400, lr=20.0)
_STACK = dict(l2=1e-4, steps=3000, lr=5.0)

Progress = Callable[[int, int, str], None]

_lock = threading.Lock()
_models: dict[int, "Model"] = {}
_refs: dict[int, tuple[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
_listing: tuple[tuple[str, int], list[int]] | None = None
_clear_weights: dict[int, tuple[tuple[Any, ...], tuple[float, int]]] = {}
_NAME = re.compile(r"person_(\d+)\.json")


class NotEnough(Exception):
    """Training cannot run; the message says why in plain words."""


def name_of(person_id: int) -> str:
    return f"person_{int(person_id)}"


# ------------------------------------------------------------------ numbers
def _topk(q: np.ndarray, ref: np.ndarray, ref_w: np.ndarray, k: int, q_item: np.ndarray | None = None,
          ref_item: np.ndarray | None = None) -> np.ndarray:
    """For each row of q: the weighted mean of its k best cosines to the reference rows (index._topk_means for one
    person). With the item ids, a reference in the query's own item does not count."""
    out = np.zeros(q.shape[0], dtype=np.float32)
    n = int(ref.shape[0])
    if not n or not q.shape[0]:
        return out
    kk = max(1, min(int(k), n))
    for s in range(0, q.shape[0], _CHUNK):
        sims = q[s:s + _CHUNK] @ ref.T
        if q_item is not None and ref_item is not None:
            sims = np.where(q_item[s:s + _CHUNK, None] == ref_item[None, :], np.float32(-2.0), sims)
        if kk < n:
            best = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        else:
            best = np.broadcast_to(np.arange(n), (sims.shape[0], n))
        ts = np.take_along_axis(sims, best, axis=1).astype(np.float64)
        tw = np.where(ts > -1.5, ref_w[best].astype(np.float64), 0.0)
        den = tw.sum(axis=1)
        out[s:s + _CHUNK] = np.where(den > 0, (ts * tw).sum(axis=1) / np.maximum(den, 1e-12), 0.0)
    return out


def _fit(x: np.ndarray, y: np.ndarray, sw: np.ndarray, *, l2: float, steps: int, lr: float) -> tuple[np.ndarray, float]:
    """Weighted logistic regression by gradient descent with momentum, from zero: the same steps on the same
    rows give the same weights."""
    xd = np.ascontiguousarray(x, dtype=np.float64)
    yd = np.asarray(y, dtype=np.float64)
    wt = np.asarray(sw, dtype=np.float64)
    wt = wt / max(float(wt.sum()), 1e-12)
    w = np.zeros(xd.shape[1], dtype=np.float64)
    b = 0.0
    vw = np.zeros_like(w)
    vb = 0.0
    for _ in range(int(steps)):
        z = xd @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        g = wt * (p - yd)
        vw = 0.9 * vw - lr * (xd.T @ g + l2 * w)
        vb = 0.9 * vb - lr * float(g.sum())
        w = w + vw
        b = b + vb
    return w, float(b)


def _balanced(pos_w: np.ndarray, n_neg: int) -> np.ndarray:
    """Sample weights: the positives (by their own weights) and the negatives (equal) each sum to a half."""
    return np.concatenate([0.5 * pos_w / max(float(pos_w.sum()), 1e-12), np.full(n_neg, 0.5 / max(n_neg, 1))])


def _features(top3: np.ndarray, logit: np.ndarray, weight: np.ndarray) -> np.ndarray:
    t = np.asarray(top3, dtype=np.float64)
    lg = np.asarray(logit, dtype=np.float64) / 10.0
    w = np.asarray(weight, dtype=np.float64)
    return np.stack([t, lg, w, t * (1.0 - w), lg * (1.0 - w)], axis=1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ------------------------------------------------------------------ the rows a training reads
@dataclass
class _Rows:
    ids: np.ndarray
    person: np.ndarray   # -1 = nobody
    user: np.ndarray
    item: np.ndarray
    weight: np.ndarray
    child: np.ndarray
    video: np.ndarray
    fold_day: np.ndarray
    fold_item: np.ndarray
    emb: np.ndarray


def _read(conn: sqlite3.Connection) -> _Rows:
    from app.analysis.index import _ages_ready, _child_of, _weight_of

    rows = conn.execute(
        "SELECT f.id, f.person_id, f.assigned_by, f.item_id, f.det_score, f.size_px, f.clarity, f.sharpness, "
        "f.age_groups, f.embedding, i.kind, i.taken_at FROM faces f JOIN items i ON i.id = f.item_id ORDER BY f.id"
    ).fetchall()
    rows = [r for r in rows if r[9] is not None and len(r[9]) == DIM * 4]
    n = len(rows)
    emb = np.zeros((n, DIM), dtype=np.float32)
    for k, r in enumerate(rows):
        emb[k] = np.frombuffer(r[9], dtype=np.float32, count=DIM)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    weight = _weight_of(rows)
    child = _child_of(rows, weight, _ages_ready(conn), col=8)
    days: dict[str | None, str | None] = {}
    fold_day = np.zeros(n, dtype=np.int64)
    fold_item = np.zeros(n, dtype=np.int64)
    for k, r in enumerate(rows):
        taken = r[11]
        if taken not in days:
            days[taken] = local_date(taken, settings.local_timezone)
        key = days[taken] or f"item{int(r[3])}"
        fold_day[k] = zlib.crc32(key.encode("utf-8")) % FOLDS
        fold_item[k] = zlib.crc32(f"item{int(r[3])}".encode("utf-8")) % FOLDS
    return _Rows(
        ids=np.fromiter((int(r[0]) for r in rows), dtype=np.int64, count=n),
        person=np.fromiter((-1 if r[1] is None else int(r[1]) for r in rows), dtype=np.int64, count=n),
        user=np.fromiter((r[2] == "user" for r in rows), dtype=bool, count=n),
        item=np.fromiter((int(r[3]) for r in rows), dtype=np.int64, count=n),
        weight=weight.astype(np.float32), child=np.asarray(child, dtype=bool),
        video=np.fromiter((r[10] == "video" for r in rows), dtype=bool, count=n),
        fold_day=fold_day, fold_item=fold_item, emb=emb,
    )


def _folds(f: _Rows, pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, str]:
    """(the fold of each row, which folding it is): by day ('day'), or by photo ('photo') when the days do not
    spread the examples (every training part needs both kinds). The check stores which one ran."""
    for scheme, fold in (("day", f.fold_day), ("photo", f.fold_item)):
        if all((pos & (fold != k)).any() and (neg & (fold != k)).any() for k in range(FOLDS)):
            return fold, scheme
    raise NotEnough("These faces come from too few photos to check the training on photos it has not seen.")


def _conflicting(f: _Rows, pos: np.ndarray, free: np.ndarray, near: np.ndarray, ref_w: np.ndarray, k: int) -> np.ndarray:
    """Which of the positives the owner did not lock (rows `free`; `near` is each one's best cosine to a face the
    owner said is not the person) are conflicts: see the module text. Returns a mask over `free`."""
    suspect = near >= CONFLICT_SIM
    rows = np.nonzero(suspect)[0]
    if not rows.size:
        return suspect
    pi = np.nonzero(pos)[0]
    plain = np.zeros_like(suspect)
    plain[rows] = near[rows] > _topk(f.emb[free[rows]], f.emb[pi], ref_w[pi], k, f.item[free[rows]], f.item[pi])
    flag = suspect.copy()
    for _ in range(CONFLICT_ROUNDS):
        rows = np.nonzero(flag)[0]
        if not rows.size:
            break
        trusted = pos.copy()
        trusted[free[rows]] = False
        ti = np.nonzero(trusted)[0]
        own = _topk(f.emb[free[rows]], f.emb[ti], ref_w[ti], k, f.item[free[rows]], f.item[ti])
        cleared = near[rows] <= own
        if not cleared.any():
            break
        flag[rows[cleared]] = False
    hit = flag | plain
    kept = pos.copy()
    kept[free[hit]] = False
    if float(f.weight[free[hit]].sum()) > float(f.weight[kept].sum()):
        return plain  # it would set aside more than it keeps: the trusted faces do not stand for the person
    return hit


def clear_weight(conn: sqlite3.Connection, person_id: int) -> tuple[float, int]:
    """(the face weight a training of this person starts from, how many faces they have): the weight of their
    faces of weight > 0 outside the child band, as train() counts its positives before it sets conflicts aside.
    train() needs MIN_POSITIVE_WEIGHT of it. Remembered until the faces change."""
    from app.analysis.index import _ages_ready, _child_of, _weight_of
    from app.db import faces_version

    pid = int(person_id)
    key = (str(settings.db_path), int(faces_version(conn)))
    with _lock:
        hit = _clear_weights.get(pid)
    if hit is not None and hit[0] == key:
        return hit[1]
    rows = conn.execute(
        "SELECT id, person_id, assigned_by, item_id, det_score, size_px, clarity, sharpness, age_groups FROM faces "
        "WHERE person_id = ? AND embedding IS NOT NULL AND length(embedding) = ?", (pid, DIM * 4)).fetchall()
    weight = _weight_of(rows)
    child = np.asarray(_child_of(rows, weight, _ages_ready(conn), col=8), dtype=bool)
    faces = int(conn.execute("SELECT COUNT(*) FROM faces WHERE person_id = ?", (pid,)).fetchone()[0])
    out = (float(weight[(weight > 0) & ~child].sum()), faces)
    with _lock:
        if len(_clear_weights) > 256:
            _clear_weights.clear()
        _clear_weights[pid] = (key, out)
    return out


def owner_answers(conn: sqlite3.Connection, person_id: int) -> tuple[int, int]:
    """(faces the owner put on the person, faces taken off them): what "answers since the training" counts."""
    a = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id = ? AND assigned_by = 'user'", (int(person_id),)).fetchone()[0]
    b = conn.execute("SELECT COUNT(*) FROM guess_rejections WHERE kind = 'face' AND person_id = ?",
                     (int(person_id),)).fetchone()[0]
    return int(a), int(b)


# ------------------------------------------------------------------ training
def train(conn: sqlite3.Connection, person_id: int, progress: Progress | None = None) -> dict[str, Any]:
    """Fit and store the person's model. The slow part: call with no write transaction open. Raises NotEnough."""
    from app.analysis.index import USER_WEIGHT
    from app.db import faces_version

    pid = int(person_id)
    if not store.enabled(conn):
        raise NotEnough("Learning is switched off in Settings, so nothing can be trained.")
    total = FOLDS + 4
    done = 0

    def tick(step: str) -> None:
        nonlocal done
        if progress is not None:
            progress(done, total, step)
        done += 1

    tick("read")
    k = max(1, int(settings.cluster_topk))
    f = _read(conn)
    ok = (f.weight > 0) & ~f.child
    on = f.person == pid
    pos = on & ok
    ref_w = (f.weight * np.where(f.user, USER_WEIGHT, 1.0)).astype(np.float32)

    # ---- trusted negatives
    removed = {int(r[0]) for r in conn.execute(
        "SELECT ref_id FROM guess_rejections WHERE kind = 'face' AND person_id = ?", (pid,))}
    a = np.isin(f.ids, np.fromiter(removed, dtype=np.int64, count=len(removed))) & ~on & ok
    b = f.user & (f.person >= 0) & ~on & ok & ~a
    photos = np.unique(f.item[on & ~f.video])
    c_all = ~on & np.isin(f.item, photos) & ~f.video & ok & ~a & ~b
    pi = np.nonzero(pos)[0]
    ci = np.nonzero(c_all)[0]
    alike = np.zeros(f.ids.shape[0], dtype=np.float32)
    alike[ci] = _topk(f.emb[ci], f.emb[pi], ref_w[pi], k, f.item[ci], f.item[pi])
    amb = c_all & (alike >= AMBIGUOUS)
    elsewhere = np.zeros_like(amb)
    named = {int(r[0]) for r in conn.execute(
        "SELECT id FROM persons WHERE name IS NOT NULL AND name != '' AND kind = 'person'")}
    for i in np.nonzero(amb)[0].tolist():
        q = int(f.person[i])
        if q not in named:
            continue  # nobody, or an unnamed group: it may hold the person's own faces (a scan takes them out)
        mates = np.nonzero((f.person == q) & (f.weight > 0))[0]
        mates = mates[mates != i]
        if mates.size >= 3 and float(_topk(f.emb[i:i + 1], f.emb[mates], ref_w[mates], k)[0]) >= float(alike[i]) + AMBIGUOUS_LEAD:
            elsewhere[i] = True  # it sits on someone it matches better: a look-alike, not the person twice
    c = (c_all & ~amb) | elsewhere
    neg = a | b | c

    # ---- conflicts: a face on the person that looks like a face the owner took off or gave to someone else
    conflicts: list[dict[str, Any]] = []
    conflict = np.zeros_like(pos)
    said = np.nonzero(a | b)[0]
    free = np.nonzero(pos & ~f.user)[0]
    if said.size and free.size:
        near = np.zeros(free.size, dtype=np.float32)
        which = np.zeros(free.size, dtype=np.int64)
        for s in range(0, free.size, _CHUNK):
            sims = f.emb[free[s:s + _CHUNK]] @ f.emb[said].T
            which[s:s + _CHUNK] = sims.argmax(axis=1)
            near[s:s + _CHUNK] = sims.max(axis=1)
        hit = _conflicting(f, pos, free, near, ref_w, k)
        conflict[free[hit]] = True
        # strongest first, then by face id: the order the list is paged in (a keyset on the rounded value)
        for j in sorted(np.nonzero(hit)[0].tolist(), key=lambda j: (-round(float(near[j]), 4), int(f.ids[free[j]]))):
            other = int(said[which[j]])
            conflicts.append({"face_id": int(f.ids[free[j]]), "like_face_id": int(f.ids[other]),
                              "like_person_id": int(f.person[other]) if f.person[other] >= 0 else None,
                              "sim": round(float(near[j]), 4)})
    pos = pos & ~conflict
    pi = np.nonzero(pos)[0]
    ni = np.nonzero(neg)[0]
    if float(f.weight[pi].sum()) < MIN_POSITIVE_WEIGHT:
        if conflict.any():
            raise NotEnough(
                f"{int(conflict.sum())} of this person's faces look the same as faces you took off them or tagged as "
                f"someone else, and without those too few clear faces are left to train on: about "
                f"{int(MIN_POSITIVE_WEIGHT)} are needed. Tag the faces that are right as this person yourself, then try again.")
        raise NotEnough(f"There are too few clear faces on this person to train on: about {int(MIN_POSITIVE_WEIGHT)} are needed.")
    if ni.size < MIN_NEGATIVES:
        raise NotEnough(
            f"There are too few faces known not to be this person ({MIN_NEGATIVES} are needed): other faces in their "
            "photos, faces you tagged as someone else, or faces you took off them.")
    fold, scheme = _folds(f, pos, neg)

    # ---- out-of-fold scores
    n = f.ids.shape[0]
    oof_logit = np.full(n, np.nan, dtype=np.float64)
    oof_top3 = np.full(n, np.nan, dtype=np.float64)
    for fo in range(FOLDS):
        tick("fit")
        tp = np.nonzero(pos & (fold != fo))[0]
        tn = np.nonzero(neg & (fold != fo))[0]
        te = np.nonzero((pos | neg) & (fold == fo))[0]
        if not te.size:
            continue
        w, bias = _fit(np.concatenate([f.emb[tp], f.emb[tn]]), np.concatenate([np.ones(tp.size), np.zeros(tn.size)]),
                       _balanced(ref_w[tp], tn.size), **_LR)
        oof_logit[te] = f.emb[te].astype(np.float64) @ w + bias
        oof_top3[te] = _topk(f.emb[te], f.emb[tp], ref_w[tp], k, f.item[te], f.item[tp])
    tick("check")
    rows = np.nonzero(pos | neg)[0]
    y = pos[rows].astype(np.float64)
    feats = _features(oof_top3[rows], oof_logit[rows], f.weight[rows])
    cls = np.where(y == 1, 0.5 / max(float(y.sum()), 1.0), 0.5 / max(float((1 - y).sum()), 1.0))
    nested = np.zeros(rows.size, dtype=np.float64)
    rf = fold[rows]
    for fo in range(FOLDS):
        tr = rf != fo
        if not (~tr).any():
            continue
        sw_, sb_ = _fit(feats[tr], y[tr], cls[tr], **_STACK)
        nested[~tr] = _sigmoid(feats[~tr] @ sw_ + sb_)
    unclear = f.weight[rows] < UNCLEAR_WEIGHT

    def cut(p: float) -> dict[str, Any]:
        sel = nested >= p - 1e-12
        return {"p": round(float(p), 2), "found": int((sel & (y == 1)).sum()), "mistaken": int((sel & (y == 0)).sum()),
                "unclear_found": int((sel & (y == 1) & unclear).sum()),
                "unclear_mistaken": int((sel & (y == 0) & unclear).sum())}

    sure_p = 1.0
    for hundredths in range(int(round(GUESS_P * 100)), 100):
        got = cut(hundredths / 100.0)
        if got["found"] > 0 and got["found"] / (got["found"] + got["mistaken"]) >= SURE_PRECISION:
            sure_p = hundredths / 100.0
            break
    check = {
        "folds": scheme, "positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
        "unclear_positives": int(((y == 1) & unclear).sum()), "unclear_negatives": int(((y == 0) & unclear).sum()),
        "cuts": [cut(GUESS_P), cut(0.9), cut(sure_p)],
    }

    # ---- the model itself: every cleaned row, and the stacker on every out-of-fold row
    tick("fit")
    w, bias = _fit(np.concatenate([f.emb[pi], f.emb[ni]]), np.concatenate([np.ones(pi.size), np.zeros(ni.size)]),
                   _balanced(ref_w[pi], ni.size), **_LR)
    sw_, sb_ = _fit(feats, y, cls, **_STACK)
    tick("save")
    owner_faces, rejections = owner_answers(conn, pid)
    model = {
        "rev": MODEL_REV, "person_id": pid, "trained_at": now_iso(), "faces_version": int(faces_version(conn)),
        "w": [round(float(v), 6) for v in w], "b": round(float(bias), 6),
        "stack": {"w": [round(float(v), 6) for v in sw_], "b": round(float(sb_), 6)},
        "guess_p": GUESS_P, "sure_p": round(float(sure_p), 2),
        "ref_ids": [int(i) for i in f.ids[pi].tolist()],
        "counts": {
            "positives": int(pi.size), "positive_weight": round(float(f.weight[pi].sum()), 2),
            "negatives": int(ni.size), "rejected": int(a.sum()), "owner_others": int(b.sum()),
            "same_photo": int(c.sum()), "ambiguous": int((amb & ~elsewhere).sum()), "conflicts": int(conflict.sum()),
            "owner_faces": owner_faces, "rejections": rejections,
        },
        "check": check,
        "conflicts": conflicts[:CONFLICTS_KEPT],
    }
    store.save(name_of(pid), model)
    _forget_cached(pid)
    resume(conn, pid)  # a new training makes guesses again
    if progress is not None:
        progress(total, total, "save")
    log.info("person model: person %d trained on %d faces against %d (%d conflicts set aside)",
             pid, pi.size, ni.size, int(conflict.sum()))
    return model


# ------------------------------------------------------------------ the stored model
@dataclass
class Model:
    person_id: int
    trained_at: str
    w: np.ndarray
    b: float
    stack_w: np.ndarray
    stack_b: float
    guess_p: float
    sure_p: float
    ref_ids: list[int]
    raw: dict[str, Any]

    @staticmethod
    def from_raw(raw: dict[str, Any]) -> "Model":
        w = np.asarray(raw["w"], dtype=np.float64)
        sw = np.asarray(raw["stack"]["w"], dtype=np.float64)
        if w.shape != (DIM,) or sw.shape != (5,):
            raise ValueError("bad shapes")
        return Model(
            person_id=int(raw["person_id"]), trained_at=str(raw["trained_at"]), w=w, b=float(raw["b"]),
            stack_w=sw, stack_b=float(raw["stack"]["b"]), guess_p=float(raw.get("guess_p", GUESS_P)),
            sure_p=float(raw.get("sure_p", 1.0)), ref_ids=[int(i) for i in raw.get("ref_ids") or []], raw=raw,
        )

    def reference(self, conn: sqlite3.Connection | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(vectors, weights, item ids) of the faces the model was trained on that are still the person's: what a
        face's top-3 is measured against. With a connection the face index is synced first; without, the caller
        has synced it."""
        from app.analysis.index import USER_WEIGHT, get_index

        idx = get_index()
        with idx.lock:
            if conn is not None:
                idx.sync(conn)
            key = (id(idx), idx.generation, self.trained_at)
            with _lock:
                hit = _refs.get(self.person_id)
            if hit is not None and hit[0] == key:
                return hit[1]
            _, emb, weight, user, item = idx.examples_of(self.ref_ids, self.person_id)
            keep = weight > 0
            ref = (emb[keep], (weight[keep] * np.where(user[keep], USER_WEIGHT, 1.0)).astype(np.float32), item[keep])
            with _lock:
                _refs[self.person_id] = (key, ref)
            return ref

    def top3(self, embs: np.ndarray, *, exclude_items: np.ndarray | None = None,
             conn: sqlite3.Connection | None = None) -> np.ndarray:
        emb, weight, item = self.reference(conn)
        q = np.asarray(embs, dtype=np.float32).reshape(-1, DIM)
        return _topk(q, emb, weight, max(1, int(settings.cluster_topk)),
                     None if exclude_items is None else np.asarray(exclude_items, dtype=np.int64), item)

    def logit(self, embs: np.ndarray) -> np.ndarray:
        return np.asarray(embs, dtype=np.float64).reshape(-1, DIM) @ self.w + self.b

    def probability(self, embs: np.ndarray, weights: Any, *, exclude_items: np.ndarray | None = None,
                    top3: Any = None, conn: sqlite3.Connection | None = None) -> np.ndarray:
        """P(the face is the person) for each row, from its top-3 against the reference (computed here unless
        given), the regression and its face weight."""
        q = np.asarray(embs, dtype=np.float32).reshape(-1, DIM)
        t = self.top3(q, exclude_items=exclude_items, conn=conn) if top3 is None else np.asarray(top3, dtype=np.float64).reshape(-1)
        feats = _features(t, self.logit(q), np.asarray(weights, dtype=np.float64).reshape(-1))
        return _sigmoid(feats @ self.stack_w + self.stack_b)

    def gate(self, emb: np.ndarray, weight: float, top3: float) -> bool:
        """Whether the automatic matching may put this face on the person (its score there as the top-3)."""
        return bool(self.probability(np.asarray(emb, dtype=np.float32)[None, :], [weight], top3=[top3])[0] >= GATE_P)


def _forget_cached(person_id: int) -> None:
    global _listing
    with _lock:
        _models.pop(int(person_id), None)
        _refs.pop(int(person_id), None)
        _listing = None


def file_changed(person_id: int) -> None:
    """The person's file was put back or taken away outside this module (an undo): drop what is remembered of it."""
    _forget_cached(int(person_id))


def load(person_id: int) -> Model | None:
    """The person's model, or None: never trained, learning switched off, or written by another revision."""
    if not store.enabled():
        return None
    raw = store.load(name_of(person_id))
    if not raw or raw.get("rev") != MODEL_REV:
        return None
    pid = int(person_id)
    with _lock:
        hit = _models.get(pid)
        if hit is not None and hit.raw is raw:
            return hit
    try:
        model = Model.from_raw(raw)
    except (KeyError, TypeError, ValueError):
        log.warning("person model: the file of person %d cannot be read", pid)
        return None
    with _lock:
        _models[pid] = model
    return model


def stored(person_id: int) -> dict[str, Any] | None:
    """The stored file whatever the switch says (the status route shows a model that is switched off)."""
    raw = store.load(name_of(person_id))
    return raw if raw and raw.get("rev") == MODEL_REV else None


def trained_ids() -> list[int]:
    """The people who have a model (none while learning is off). One stat per call, a listing per change."""
    global _listing
    if not store.enabled():
        return []
    d = store.learned_dir()
    try:
        stamp = (str(d), d.stat().st_mtime_ns)
    except OSError:
        return []
    with _lock:
        if _listing is not None and _listing[0] == stamp:
            return list(_listing[1])
    ids = sorted(int(m.group(1)) for p in d.glob("person_*.json") if (m := _NAME.fullmatch(p.name)))
    with _lock:
        _listing = (stamp, ids)
    return list(ids)


def update(person_id: int, **fields: Any) -> dict[str, Any] | None:
    """Rewrite the stored file with these top-level fields (the last scan); None without a file."""
    raw = store.load(name_of(person_id))
    if not raw:
        return None
    model = {**raw, **fields}
    store.save(name_of(person_id), model)
    _forget_cached(int(person_id))
    return model


def forget(person_id: int) -> None:
    """Remove the person's model. Inside an owner action the file goes to the action's undo stash (app.undo), so
    an undo that brings the person back brings their model back too; outside one it is deleted."""
    from app import undo

    undo.discard(store.path(name_of(person_id)))
    store.remove(name_of(person_id))
    _forget_cached(int(person_id))


# ------------------------------------------------------------------ paused (the guesses were taken back)
def paused(conn: sqlite3.Connection, person_ids: Any) -> set[int]:
    """Those of the people whose model is paused: it makes no guesses until the next look or training. The flag is
    persons.recognition_paused_at, a journaled column, so undoing "Take back its guesses" un-pauses the model."""
    ids = sorted({int(p) for p in person_ids})
    out: set[int] = set()
    for s in range(0, len(ids), 500):
        part = ids[s:s + 500]
        out.update(int(r[0]) for r in conn.execute(
            f"SELECT id FROM persons WHERE id IN ({','.join('?' * len(part))}) AND recognition_paused_at IS NOT NULL", part))
    return out


def pause(conn: sqlite3.Connection, person_id: int) -> None:
    from app.db import tx

    with tx(conn):
        conn.execute("UPDATE persons SET recognition_paused_at = ? WHERE id = ?", (now_iso(), int(person_id)))


def resume(conn: sqlite3.Connection, person_id: int) -> None:
    from app.db import tx

    if conn.execute("SELECT 1 FROM persons WHERE id = ? AND recognition_paused_at IS NOT NULL",
                    (int(person_id),)).fetchone() is None:
        return
    with tx(conn):
        conn.execute("UPDATE persons SET recognition_paused_at = NULL WHERE id = ?", (int(person_id),))
