#!/usr/bin/env python3
"""Check position tracking (docs/ARCHITECTURE.md section 35) on a SNAPSHOT of the library, never on the live one.

Setup (by hand, before running):
  sqlite3 data/cloud.db ".backup <snapshot>/cloud.db"   and copy data/learned to <snapshot>/learned

Usage:
  place_eval.py --snapshot DIR --previews DATA_DIR/previews [--limit-items N] [--loo 400] [--out report.json]
                [--around N] [--modes position,clothes,both]
                [--skip-scan] [--threads N] [--live-data DIR]

What it does:
- refuses a snapshot that is (or lies inside) a live data folder, wherever the script runs from: the data folder
  of this checkout and of the main checkout when this one is a git worktree, DATA_DIR, the folder that holds the
  previews given, and any folder that holds the app's media folders. The report is never written into one of
  them. The previews are read from the given folder, read-only;
- lines the photos up (app.analysis.place.run_pending) unless --skip-scan, newest first, and reports the time per
  photo and per pair, the peak memory and the projected time for the whole library;
- reports how the pairs lined up, by the gap between the two photos (still, moved, burst, none);
- leave-one-out: for boxes whose person is known, in photos that are lined up with a neighbour, the person is taken
  off the box inside a transaction that is rolled back, the box is guessed again, and the guess is compared with
  the person it had. Two kinds of box: the face box with its face evidence switched off (so only position and
  clothes speak), and the person box around it with the face deleted (a faceless box). Three modes: position alone
  (no clothes path), clothes alone (PLACE_ENABLED off), and both. Body traits are off throughout. Per mode, kind of
  box, basis, gap and photos walked: decisions, right, precision with its Wilson interval, coverage, wrong names,
  and the same for the decisions where the box's person and the guessed person both have a name (a group without
  a name is the app's own clustering: one person split over two groups counts as a wrong name here), and how many
  of the wrong names are such a split by the faces ("wrong_but_the_same_face": the box's own face and the face it
  was followed from match at the face cut).
- with --around N the person is taken off every photo within N links of the box as well. A chain of more than N
  photos can still name them (how good are chains), and every guess from N photos away or nearer is wrong
  ("guessed_from_nearer_than_the_person_is"): that is how often someone not known nearby is named after whoever
  stood in their spot.
- what it cannot measure: real open boxes. Their faces are the unclear ones (a clear face that matches nobody
  starts a person of its own), and nobody has said who they are.

The pass bar for position-only guesses (section 35): precision 0.95 or more on 20 or more decisions of a tier, else
that tier never names alone. This report gives the end-to-end numbers of section 35 (the leave-one-out tables, the
chains and the count of people not known nearby). The tier strengths themselves, the head point, the clothes
floors and the movement finding come from one-off measurements that are not in the repository; section 35 says
which is which.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
for p in (str(BACKEND), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

GAPS = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 60.0), (60.0, 300.0), (300.0, math.inf))
MEDIA_DIRS = ("previews", "thumbs", "originals", "staging", "faces", "bodies")   # a live data folder holds these


def main_checkout(repo: Path = REPO) -> Path | None:
    """The main checkout of the repository when `repo` is one of its git worktrees, else None."""
    try:
        got = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    line = got.stdout.strip()
    if got.returncode != 0 or not line:
        return None
    common = Path(line)
    root = (common if common.is_absolute() else repo / common).resolve().parent
    return root if root != repo.resolve() else None


def live_dirs(te: Any, extra: list[str], repo: Path = REPO) -> list[Path]:
    """The live data folders: trait_eval's list (this checkout's data folder, DATA_DIR and its .env), and the same
    of the main checkout when this checkout is a git worktree of it (a worktree has no data folder of its own)."""
    out = list(te._live_dirs(extra))
    main = main_checkout(repo)
    if main is not None:
        out.append((main / "data").resolve())
        env = main / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith("DATA_DIR="):
                    out.append(Path(line.split("=", 1)[1].strip().strip("'\"")).expanduser().resolve())
    return out


def refusal(te: Any, snapshot: Path, previews: Path, out: Path | None, live: list[Path]) -> str | None:
    """Why this run may not start, or None. Wherever the script runs from: the folder that holds the previews it
    reads is a live data folder, so is any folder that holds the app's media folders, and the report is never
    written into one of them."""
    snap = snapshot.expanduser().resolve()
    prev = previews.expanduser().resolve()
    live = [*live, prev.parent]
    why = te.refuse_live(snap, live)
    if why:
        return why
    held = [d for d in MEDIA_DIRS if (snap / d).is_dir()]
    if held:
        return f"{snap} holds {', '.join(held)}: that is a live data folder, a snapshot holds cloud.db and learned/"
    if out is not None:
        o = out.expanduser().resolve()
        for d in live:
            if o == d or d in o.parents:
                return f"the report {o} would be written into the live data folder {d}"
    return None


def gap_of(seconds: float | None) -> str:
    """The bucket of a gap, closed on the right as the tiers of place_match are (exactly 10 s is in 3-10s)."""
    if seconds is None:
        return "?"
    for lo, hi in GAPS:
        if (seconds > lo or lo == 0.0) and seconds <= hi:
            return f"{lo:g}-{hi:g}s" if math.isfinite(hi) else f"{lo:g}s+"
    return "?"


def scan(conn, limit_items: int | None) -> dict[str, Any]:
    from app.analysis import place, place_align

    total = place.progress(conn)["eligible"]
    todo = min(total, limit_items) if limit_items else total
    t0 = time.perf_counter()
    done = pairs = aligned = 0
    while True:
        left = None if not limit_items else limit_items - done
        if left is not None and left <= 0:
            break
        r = place.run_pending(conn, budget_s=60.0, limit=left)
        done += r["items"]
        pairs += r["pairs"]
        aligned += r["aligned"]
        print(f"progress: {done}/{todo}", flush=True)
        if r["complete"] or not r["items"]:
            break
    took = time.perf_counter() - t0
    st = place_align.stats()
    all_pairs = conn.execute(place._SEQUENCE_SQL + "SELECT SUM(length(sig) - length(replace(sig, ',', ''))) FROM s",
                             {"gap": place.MAX_GAP_S + 0.5}).fetchone()[0] or 0
    per_item = took / max(1, done)
    from app.config import settings

    return {"photos": done, "pairs": pairs, "lined_up": aligned, "seconds": round(took, 1),
            "seconds_per_photo": round(per_item, 3), "seconds_per_pair": round(took / max(1, pairs), 3),
            "library_photos": int(total), "library_pairs": int(all_pairs), "threads": int(settings.place_threads),
            "projected_library_minutes": round(per_item * total / 60.0, 1), "pictures_described": st["described"],
            "fits_on_people": st["on_people"], "fits_refused_without_background": st["refused"]}


def link_table(conn) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in conn.execute("SELECT kind, dt_s FROM place_links"):
        out[gap_of(float(r["dt_s"]))][str(r["kind"])] += 1
    return {g: dict(v) for g, v in out.items()}


def leave_one_out(conn, n: int, wilson, around: int = 0,
                  modes: tuple[str, ...] = ("position", "clothes", "both")) -> dict[str, Any]:
    from app.analysis import guess, place
    from app.config import settings

    rng = random.Random(5)
    linked = {int(r[0]) for r in conn.execute(
        "SELECT a_item FROM place_links WHERE rev = ? AND kind != 'none' UNION "
        "SELECT b_item FROM place_links WHERE rev = ? AND kind != 'none'", (place.PLACE_REV, place.PLACE_REV))}
    pool = [r for r in conn.execute(
        "SELECT f.id, f.item_id, f.person_id FROM faces f JOIN items i ON i.id = f.item_id "
        "WHERE f.person_id IS NOT NULL AND f.assigned_by IN ('user', 'auto') "
        "AND i.kind = 'image' AND i.deleted_at IS NULL AND i.nsfw = 0 ORDER BY f.id") if int(r["item_id"]) in linked]
    rng.shuffle(pool)
    pool = pool[:n]
    holder: dict[int, int] = {}  # the person box around each face (the first by id)
    for r in conn.execute("SELECT id, face_id FROM bodies WHERE face_id IS NOT NULL ORDER BY id"):
        holder.setdefault(int(r["face_id"]), int(r["id"]))
    named = {int(r[0]) for r in conn.execute("SELECT id FROM persons WHERE name IS NOT NULL AND name != ''")}
    real_inputs, real_scores = guess._clothes_inputs, guess._face_scores
    saved = (settings.trait_matching, settings.place_enabled)
    # n_named, right_named: the decisions where the box's person and the guessed person both have a name. A group
    # without a name is the app's own clustering, and one person is often split over several such groups.
    # same_face: wrong names where the box's own face and the face it was followed from match at the face cut
    # (guess.FACE_CONTRADICTION): the same person under two person ids, most often two groups without a name.
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "right": 0, "wrong": 0, "n_named": 0, "right_named": 0, "same_face": 0})

    def face_cos(a: int, b: int | None) -> float | None:
        rows = [conn.execute("SELECT embedding FROM faces WHERE id = ?", (i,)).fetchone() for i in (a, b) if i is not None]
        if len(rows) != 2 or any(r is None or not r[0] for r in rows):
            return None
        va, vb = (np.frombuffer(r[0], dtype=np.float32) for r in rows)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        return float(va @ vb) / (na * nb) if va.shape == vb.shape and na > 0 and nb > 0 else None
    trials: dict[str, int] = defaultdict(int)
    too_near: dict[str, int] = defaultdict(int)
    settings.trait_matching = False
    try:
        for done, r in enumerate(pool):
            if done % 50 == 0:
                print(f"progress: {done}/{len(pool)}", flush=True)
            fid, iid, pid = int(r["id"]), int(r["item_id"]), int(r["person_id"])
            body_id = holder.get(fid)
            near = sorted(place.linked_around(conn, [iid], around) - {iid}) if around > 0 else []
            for box in ("face", "box"):
                if box == "box" and body_id is None:
                    continue
                for mode in modes:
                    trials[f"{mode}:{box}"] += 1
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        # the person leaves this photo: the box, the box around it, their other boxes and their tag;
                        # with --around, they leave the photos that many links around it too
                        for j in (iid, *near):
                            conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'auto' WHERE item_id = ? AND person_id = ?", (j, pid))
                            conn.execute("UPDATE bodies SET person_id = NULL, assigned_by = 'auto' WHERE item_id = ? AND person_id = ?", (j, pid))
                            conn.execute("DELETE FROM item_tags WHERE item_id = ? AND person_id = ? AND relation = 'in'", (j, pid))
                        conn.execute("DELETE FROM guess_rejections WHERE person_id = ? AND ((kind = 'face' AND ref_id = ?) "
                                     "OR (kind = 'body' AND ref_id = ?))", (pid, fid, body_id if body_id is not None else -1))
                        if box == "box":
                            conn.execute("DELETE FROM faces WHERE id = ?", (fid,))  # the box around it is faceless now
                        guess._face_scores = lambda c, b: {}  # type: ignore[assignment]
                        guess._clothes_inputs = (lambda c, it, ca: None) if mode == "position" else real_inputs  # type: ignore[assignment]
                        settings.place_enabled = mode != "clothes"
                        guess.guess_item(conn, iid)
                        table, key = ("faces", fid) if box == "face" else ("bodies", int(body_id))
                        g = conn.execute(f"SELECT guess_person_id, guess_evidence FROM {table} WHERE id = ?", (key,)).fetchone()
                    finally:
                        conn.execute("ROLLBACK")
                        guess._clothes_inputs, guess._face_scores = real_inputs, real_scores  # type: ignore[assignment]
                        settings.place_enabled = saved[1]
                    if g is None or g["guess_person_id"] is None:
                        continue
                    ev = json.loads(g["guess_evidence"] or "{}")
                    basis = ev.get("basis") or "clothes"
                    frag = ev.get("place") or {}
                    # a guess with a position part is counted by the gap to the photo it was followed from and the
                    # photos walked; a plain clothes guess has no such gap (its reference is minutes away)
                    k = f"{mode}:{box}:{basis}"
                    if frag:
                        k += f":{gap_of(frag.get('seconds_apart'))}:hops{frag.get('hops')}"
                        if around > 0 and int(frag.get("hops") or 0) <= around:
                            too_near[f"{mode}:{box}"] += 1  # the person is not that near: this guess cannot be right
                    s = stats[k]
                    right = int(g["guess_person_id"]) == pid
                    s["n"] += 1
                    s["right" if right else "wrong"] += 1
                    if pid in named and int(g["guess_person_id"]) in named:
                        s["n_named"] += 1
                        s["right_named"] += int(right)
                    if not right and frag:
                        cos = face_cos(fid, frag.get("ref_face_id"))   # after the rollback: the box's face is back
                        s["same_face"] += int(cos is not None and cos >= guess.FACE_CONTRADICTION)
    finally:
        settings.trait_matching, settings.place_enabled = saved
        guess._clothes_inputs, guess._face_scores = real_inputs, real_scores  # type: ignore[assignment]
    out: dict[str, Any] = {}
    for k in sorted(stats):
        s = stats[k]
        mode, box = k.split(":")[:2]
        out[k] = {"n": s["n"], "right": s["right"], "wrong_names": s["wrong"], "precision": round(s["right"] / s["n"], 4),
                  "wilson95": wilson(s["right"], s["n"]), "coverage": round(s["n"] / max(1, trials[f"{mode}:{box}"]), 4),
                  "pass_bar": bool(s["n"] >= 20 and s["right"] / s["n"] >= 0.95),
                  "both_named": {"n": s["n_named"], "right": s["right_named"],
                                 "wilson95": wilson(s["right_named"], s["n_named"])},
                  "wrong_but_the_same_face": s["same_face"]}
    totals: dict[str, Any] = {}
    for mb, t in sorted(trials.items()):
        mine = [v for k, v in stats.items() if k.startswith(mb + ":")]
        n_, right = sum(v["n"] for v in mine), sum(v["right"] for v in mine)
        nn, rn = sum(v["n_named"] for v in mine), sum(v["right_named"] for v in mine)
        same = sum(v["same_face"] for v in mine)
        totals[mb] = {"trials": t, "decisions": n_, "right": right, "precision": round(right / n_, 4) if n_ else None,
                      "wilson95": wilson(right, n_), "coverage": round(n_ / max(1, t), 4),
                      "both_named": {"n": nn, "right": rn, "wilson95": wilson(rn, nn)},
                      "wrong_but_the_same_face": same}
        if around > 0:
            totals[mb]["guessed_from_nearer_than_the_person_is"] = too_near[mb]
    return {"targets": len(pool), "around": around, "by_tier": out, "totals": totals}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check position tracking on a library snapshot (never the live library).")
    ap.add_argument("--snapshot", required=True, type=Path, help="folder with cloud.db (sqlite3 .backup) and learned/")
    ap.add_argument("--previews", required=True, type=Path, help="the live DATA_DIR/previews, read only")
    ap.add_argument("--limit-items", type=int, help="line up at most this many photos (newest first)")
    ap.add_argument("--loo", type=int, default=400, help="leave-one-out targets (0 skips)")
    ap.add_argument("--around", type=int, default=0,
                    help="take the person off the photos this many links around the box as well (chains, strangers)")
    ap.add_argument("--modes", default="position,clothes,both", help="the leave-one-out modes to run")
    ap.add_argument("--threads", type=int, help="OpenCV threads for the lining up (default: PLACE_THREADS)")
    ap.add_argument("--out", type=Path, help="report JSON (default <snapshot>/place_eval.json)")
    ap.add_argument("--skip-scan", action="store_true", help="use the links already in the snapshot")
    ap.add_argument("--live-data", action="append", default=[], help="another live data folder to refuse")
    args = ap.parse_args(argv)

    import trait_eval as te

    why = refusal(te, args.snapshot, args.previews, args.out, live_dirs(te, args.live_data))
    if why:
        print(f"refusing: {why}", file=sys.stderr)
        return 2
    snapshot = args.snapshot.expanduser().resolve()
    os.environ["DATA_DIR"] = str(snapshot)
    os.environ["WORKERS_ENABLED"] = "0"
    os.environ.pop("ORIGINALS_DIR", None)
    out = (args.out or snapshot / "place_eval.json").expanduser().resolve()

    from app.config import reload_settings, settings

    reload_settings()
    if args.threads:
        settings.place_threads = max(1, int(args.threads))
    previews = args.previews.expanduser().resolve()
    te._redirect_storage(previews, None)
    from app import db, storage

    storage.preview_small_path = lambda uid, version: previews / uid[:2] / f"{uid}.v{version}.sm.webp"  # type: ignore[assignment]
    db.init_db()
    conn = db.connect()
    report: dict[str, Any] = {"snapshot": str(snapshot)}
    if not args.skip_scan:
        report["scan"] = scan(conn, args.limit_items)
    report["links_by_gap"] = link_table(conn)
    if args.loo:
        t0 = time.perf_counter()
        modes = tuple(m for m in args.modes.split(",") if m in ("position", "clothes", "both"))
        report["leave_one_out"] = leave_one_out(conn, args.loo, te.wilson, max(0, args.around), modes)
        report["leave_one_out"]["seconds"] = round(time.perf_counter() - t0, 1)
    report["peak_rss_mb"] = te.peak_rss_mb()
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
