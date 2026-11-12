#!/usr/bin/env python3
"""Check position tracking (docs/ARCHITECTURE.md section 35) on a SNAPSHOT of the library, never on the live one.

Setup (by hand, before running):
  sqlite3 data/cloud.db ".backup <snapshot>/cloud.db"   and copy data/learned to <snapshot>/learned

Usage:
  place_eval.py --snapshot DIR --previews DATA_DIR/previews [--limit-items N] [--loo 400] [--out report.json]
                [--skip-scan] [--threads 2] [--live-data DIR]

What it does:
- refuses a snapshot that is (or lies inside) a live data folder, as scripts/trait_eval.py does, and reads the
  previews from the given folder, read-only;
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
  a name is the app's own clustering: one person split over two groups counts as a wrong name here).
- what it cannot measure: a box of someone who is not known in the photos around it. Every guess for such a box
  is wrong, and no leave-one-out contains one (section 35 has a count made by hand).

The pass bar for position-only guesses (section 35): precision 0.95 or more on 20 or more decisions of a tier, else
PLACE_ONLY_GUESSES stays off for that tier. Constants in app/analysis/place_match.py are the ones this report
supports.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
for p in (str(BACKEND), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

GAPS = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 60.0), (60.0, 300.0), (300.0, math.inf))


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
    return {"photos": done, "pairs": pairs, "lined_up": aligned, "seconds": round(took, 1),
            "seconds_per_photo": round(per_item, 3), "seconds_per_pair": round(took / max(1, pairs), 3),
            "library_photos": int(total), "library_pairs": int(all_pairs),
            "projected_library_minutes": round(per_item * total / 60.0, 1), "pictures_described": st["described"]}


def link_table(conn) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in conn.execute("SELECT kind, dt_s FROM place_links"):
        out[gap_of(float(r["dt_s"]))][str(r["kind"])] += 1
    return {g: dict(v) for g, v in out.items()}


def leave_one_out(conn, n: int, wilson) -> dict[str, Any]:
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
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "right": 0, "wrong": 0, "n_named": 0, "right_named": 0})
    trials: dict[str, int] = defaultdict(int)
    settings.trait_matching = False
    try:
        for done, r in enumerate(pool):
            if done % 50 == 0:
                print(f"progress: {done}/{len(pool)}", flush=True)
            fid, iid, pid = int(r["id"]), int(r["item_id"]), int(r["person_id"])
            body_id = holder.get(fid)
            for box in ("face", "box"):
                if box == "box" and body_id is None:
                    continue
                for mode in ("position", "clothes", "both"):
                    trials[f"{mode}:{box}"] += 1
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        # the person leaves this photo: the box, the box around it, their other boxes and their tag
                        conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'auto' WHERE item_id = ? AND person_id = ?", (iid, pid))
                        conn.execute("UPDATE bodies SET person_id = NULL, assigned_by = 'auto' WHERE item_id = ? AND person_id = ?", (iid, pid))
                        conn.execute("DELETE FROM item_tags WHERE item_id = ? AND person_id = ? AND relation = 'in'", (iid, pid))
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
                    s = stats[k]
                    right = int(g["guess_person_id"]) == pid
                    s["n"] += 1
                    s["right" if right else "wrong"] += 1
                    if pid in named and int(g["guess_person_id"]) in named:
                        s["n_named"] += 1
                        s["right_named"] += int(right)
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
                                 "wilson95": wilson(s["right_named"], s["n_named"])}}
    totals: dict[str, Any] = {}
    for mb, t in sorted(trials.items()):
        mine = [v for k, v in stats.items() if k.startswith(mb + ":")]
        n_, right = sum(v["n"] for v in mine), sum(v["right"] for v in mine)
        nn, rn = sum(v["n_named"] for v in mine), sum(v["right_named"] for v in mine)
        totals[mb] = {"trials": t, "decisions": n_, "right": right, "precision": round(right / n_, 4) if n_ else None,
                      "wilson95": wilson(right, n_), "coverage": round(n_ / max(1, t), 4),
                      "both_named": {"n": nn, "right": rn, "wilson95": wilson(rn, nn)}}
    return {"targets": len(pool), "by_tier": out, "totals": totals}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check position tracking on a library snapshot (never the live library).")
    ap.add_argument("--snapshot", required=True, type=Path, help="folder with cloud.db (sqlite3 .backup) and learned/")
    ap.add_argument("--previews", required=True, type=Path, help="the live DATA_DIR/previews, read only")
    ap.add_argument("--limit-items", type=int, help="line up at most this many photos (newest first)")
    ap.add_argument("--loo", type=int, default=400, help="leave-one-out targets (0 skips)")
    ap.add_argument("--threads", type=int, default=2, help="OpenCV threads for this run")
    ap.add_argument("--out", type=Path, help="report JSON (default <snapshot>/place_eval.json)")
    ap.add_argument("--skip-scan", action="store_true", help="use the links already in the snapshot")
    ap.add_argument("--live-data", action="append", default=[], help="another live data folder to refuse")
    args = ap.parse_args(argv)

    import trait_eval as te

    why = te.refuse_live(args.snapshot, te._live_dirs(args.live_data))
    if why:
        print(f"refusing: {why}", file=sys.stderr)
        return 2
    snapshot = args.snapshot.expanduser().resolve()
    os.environ["DATA_DIR"] = str(snapshot)
    os.environ["WORKERS_ENABLED"] = "0"
    os.environ.pop("ORIGINALS_DIR", None)
    out = (args.out or snapshot / "place_eval.json").expanduser().resolve()

    import cv2

    cv2.setNumThreads(max(1, args.threads))
    from app.config import reload_settings

    reload_settings()
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
        report["leave_one_out"] = leave_one_out(conn, args.loo, te.wilson)
        report["leave_one_out"]["seconds"] = round(time.perf_counter() - t0, 1)
    report["peak_rss_mb"] = te.peak_rss_mb()
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
