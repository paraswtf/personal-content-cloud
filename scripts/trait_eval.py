#!/usr/bin/env python3
"""Check body traits (docs/ARCHITECTURE.md section 34) on a SNAPSHOT of the library, never on the live one.

Setup (by hand, before running):
  sqlite3 data/cloud.db ".backup <snapshot>/cloud.db"   and copy data/learned to <snapshot>/learned

Usage:
  trait_eval.py --snapshot DIR --previews DATA_DIR/previews [--originals ORIGINALS_ROOT/originals]
                [--limit-items N] [--threads 2] [--out report.json] [--sheets DIR] [--loo 300]
                [--bodies 14836,14913,...] [--watch-person NAME ...] [--skip-scan] [--live-data DIR]

What it does:
- refuses a snapshot that is (or lies inside) the live DATA_DIR (the repo's data/, $DATA_DIR of the calling shell,
  --live-data, or a DATA_DIR line in the repo's .env), or whose cloud.db is the live file;
- sets DATA_DIR=<snapshot>, WORKERS_ENABLED=0 and TRAITS_THREADS, and reads the previews and originals from the given
  folders, read-only (writes into them raise);
- measures (app.analysis.traits.run_pending) unless --skip-scan, then takes a calibration sample and trains the
  traits learner;
- reports: stage timings, peak RSS and the projected backfill time, DB growth, rows per kind, part and status, the
  gate sheet (top 40 tattoo crops by p_tattoo), hand spans when stored, open faces without a body, the calibration
  sample and the per-person reference counts, the per-query verification table (same person / other person / unnamed,
  at 8, 12, 20 and 30 inliers, and how many verified pairs the ratio check removes), the top 20 different-person
  verified pairs after every gate, the watched bodies (amendment 9) with their rows, statuses, verified pairs and the
  guess the code makes, and leave-one-out on owner-assigned bodies with the trained params, the defaults and the
  tier ceilings (precision and coverage per basis and kind, Wilson intervals, the false names);
- writes contact sheets (PNG) next to the report.

The pass bar (amendment 7): tattoo trait-only precision 0.95 or more on 20 or more leave-one-out decisions, no
clothes winner changed by agreement, and no fabric, watch or bangle among the tattoo crops of the gate sheet (by
eye). The probe's figures cover one heavily tattooed person over about 8 days; this report says nothing more general.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

WATCH_BODIES = (14836, 14913, 14837, 14838, 14839, 13792, 13794, 14494, 12917, 13988, 15017)


# ------------------------------------------------------------------ safety
def _live_dirs(extra: list[str]) -> list[Path]:
    out = [REPO / "data"]
    if os.environ.get("DATA_DIR"):
        out.append(Path(os.environ["DATA_DIR"]))
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("DATA_DIR="):
                out.append(Path(line.split("=", 1)[1].strip().strip("'\"")))
    out.extend(Path(x) for x in extra)
    return [p.expanduser().resolve() for p in out]


def refuse_live(snapshot: Path, live: list[Path]) -> str | None:
    """Why the snapshot may not be used, or None."""
    snap = snapshot.expanduser().resolve()
    if not (snap / "cloud.db").is_file():
        return f"{snap}/cloud.db does not exist (make it with sqlite3 .backup)"
    for d in live:
        if snap == d or d in snap.parents:
            return f"{snap} is the live data folder {d} (or inside it)"
        db = d / "cloud.db"
        try:
            if db.exists() and os.path.samefile(db, snap / "cloud.db"):
                return f"{snap}/cloud.db is the live database {db}"
        except OSError:
            pass
    return None


def _redirect_storage(previews: Path, originals: Path | None) -> None:
    """Previews and originals come from the given folders; nothing may be written there."""
    from app import storage

    def preview_path(uid: str, version: int) -> Path:
        return previews / uid[:2] / f"{uid}.v{version}.webp"

    def original_path(uid: str, version: int, ext: str) -> Path:
        base = originals if originals is not None else storage.root() / "originals"
        return base / uid[:2] / f"{uid}.v{version}{ext}"

    storage.preview_path = preview_path  # type: ignore[assignment]
    storage.original_path = original_path  # type: ignore[assignment]
    guarded = [previews.resolve(), *([originals.resolve()] if originals is not None else [])]
    write = storage.write_bytes_atomic

    def safe_write(path: Path, data: bytes, *a: Any, **k: Any) -> Any:
        p = Path(path).resolve()
        if any(p == g or g in p.parents for g in guarded):
            raise PermissionError(f"trait_eval reads {p.parent} only")
        return write(path, data, *a, **k)

    storage.write_bytes_atomic = safe_write  # type: ignore[assignment]


# ------------------------------------------------------------------ small helpers
def peak_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return 0.0


def db_bytes(snapshot: Path) -> int:
    return sum((snapshot / n).stat().st_size for n in ("cloud.db", "cloud.db-wal") if (snapshot / n).exists())


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def contact_sheet(conn, rows: list[tuple[int, str]], path: Path, cols: int = 8, side: int = 160) -> str | None:
    """A grid of the UI crops of these trait ids, each with a caption."""
    if not rows:
        return None
    import io

    from PIL import Image, ImageDraw

    n = len(rows)
    sheet = Image.new("RGB", (cols * side, math.ceil(n / cols) * (side + 28)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (tid, caption) in enumerate(rows):
        r = conn.execute("SELECT crop FROM body_traits WHERE id = ?", (tid,)).fetchone()
        x, y = (i % cols) * side, (i // cols) * (side + 28)
        if r is not None and r["crop"]:
            img = Image.open(io.BytesIO(bytes(r["crop"]))).convert("RGB")
            img.thumbnail((side, side))
            sheet.paste(img, (x, y))
        draw.text((x + 2, y + side + 2), caption[:26], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)


# ------------------------------------------------------------------ stages
def scan(conn, snapshot: Path, limit_items: int | None) -> dict[str, Any]:
    from app.analysis import traits as atraits

    eligible = atraits.progress(conn)["eligible"]
    t0 = time.monotonic()
    items = bodies = 0
    aborted = False
    complete = False
    paused = None
    while True:
        left = None if limit_items is None else limit_items - items
        if left is not None and left <= 0:
            break
        res = atraits.run_pending(conn, budget_s=60, limit=None if left is None else min(left, 50))
        items += int(res.get("items") or 0)
        bodies += int(res.get("bodies") or 0)
        print(f"progress: {items}/{limit_items or eligible}", flush=True)
        if res.get("paused"):  # a model is missing: nothing can be measured
            paused = res["paused"]
            break
        if res.get("aborted"):
            aborted = True
            break
        if res.get("complete") or not res.get("items"):
            complete = bool(res.get("complete"))
            break
    took = time.monotonic() - t0
    per_body = took / bodies if bodies else None
    return {"items": items, "bodies": bodies, "seconds": round(took, 1), "complete": complete, "aborted": aborted, "paused": paused,
            "eligible_bodies": eligible, "s_per_body": round(per_body, 3) if per_body else None,
            "projected_backfill_min": round(per_body * eligible / 60.0, 1) if per_body else None,
            "progress": atraits.progress(conn), "peak_rss_mb": peak_rss_mb()}


def counts(conn) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": {}, "scans": {}}
    for r in conn.execute("SELECT kind, part, COUNT(*) AS n FROM body_traits GROUP BY kind, part ORDER BY kind, part"):
        out["rows"].setdefault(r["kind"], {})[r["part"]] = int(r["n"])
    for r in conn.execute("SELECT status, COUNT(*) AS n FROM trait_scans GROUP BY status ORDER BY status"):
        out["scans"][r["status"]] = int(r["n"])
    out["bare_scans"] = int(conn.execute("SELECT COUNT(*) FROM trait_scans WHERE bare NOT IN ('', '{}')").fetchone()[0])
    out["blob_mb"] = round(float(conn.execute(
        "SELECT COALESCE(SUM(LENGTH(detail)), 0) + COALESCE(SUM(LENGTH(crop)), 0) + COALESCE(SUM(LENGTH(vec)), 0) "
        "FROM body_traits").fetchone()[0]) / 1e6, 1)
    return out


def gate_sheet(conn, sheets: Path) -> dict[str, Any]:
    rows = []
    for r in conn.execute("SELECT id, body_id, attrs FROM body_traits WHERE kind = 'tattoo'"):
        a = json.loads(r["attrs"] or "{}")
        p = a.get("p") if isinstance(a.get("p"), dict) else {}
        rows.append((float(p.get("tattoo") or 0.0), int(r["id"]), int(r["body_id"]), p))
    rows.sort(key=lambda x: (-x[0], x[1]))
    top = rows[:40]
    return {"top": [{"trait_id": t, "body_id": b, "p": p} for _, t, b, p in top],
            "jewel_ge_0.10": sum(1 for _, _, _, p in top if float(p.get("jewel") or 0) >= 0.10),
            "sheet": contact_sheet(conn, [(t, f"{t} p{v:.2f}") for v, t, _, _ in top], sheets / "gate_top40.png"),
            "unclear_band": "not stored: the scan keeps no row for an unclear crop (attrs.p of the rows only)"}


def hand_spans(conn) -> dict[str, Any]:
    spans = []
    for r in conn.execute("SELECT attrs FROM body_traits WHERE kind = 'nails'"):
        a = json.loads(r["attrs"] or "{}")
        if isinstance(a.get("span"), (int, float)):
            spans.append(float(a["span"]))
    if not spans:
        return {"note": "no hand span stored in nails attrs"}
    hist: dict[str, int] = {}
    for s in spans:
        k = f"{int(s // 32) * 32}-{int(s // 32) * 32 + 31}"
        hist[k] = hist.get(k, 0) + 1
    return {"n": len(spans), "histogram_px": dict(sorted(hist.items(), key=lambda kv: int(kv[0].split('-')[0])))}


def open_faces_without_body(conn) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) FROM faces f JOIN items i ON i.id = f.item_id WHERE f.person_id IS NULL "
        "AND f.assigned_by IN ('auto', 'guess') AND i.deleted_at IS NULL AND i.kind = 'image' "
        "AND NOT EXISTS (SELECT 1 FROM bodies b WHERE b.face_id = f.id)").fetchone()[0])


def verification_table(conn, max_verify: int) -> dict[str, Any]:
    """Amendment 7: for each usable tattoo crop, the top 20 crops of the same family from OTHER days (any person,
    never its own photo) by DINOv2 cosine, verified; tallied by same person, other person and unnamed."""
    from app.analysis import trait_match as tm
    from app.analysis import trait_sift

    ld = tm._Loader(conn, {})
    rows = [r for r in ld.rows("t.kind = 'tattoo'") if r.vec is not None and r.has_detail]
    ver = tm._Verifier(conn, max_verify)
    levels = (8, 12, 20, 30)
    tally = {k: {f"ge{n}": 0 for n in levels} for k in ("named_same", "named_other", "unnamed_named")}
    queries = {"named": 0, "unnamed": 0}
    per_query_hits = {"named_same_only": {n: 0 for n in levels}, "named_any_other": {n: 0 for n in levels},
                      "unnamed_hit": {n: 0 for n in levels}}
    ratio_removed = 0
    diff_pairs: list[dict[str, Any]] = []
    unnamed_hits: list[dict[str, Any]] = []
    pairs_by_trait: dict[int, list[dict[str, Any]]] = {}
    for q in rows:
        qi = ld.items[q.item_id]
        cands = []
        for r in rows:
            if r.id == q.id or r.item_id == q.item_id or r.fam != q.fam:
                continue
            ri = ld.items[r.item_id]
            if qi.date is not None and qi.date == ri.date:
                continue
            cands.append((tm._cos(q.vec, r.vec), r))
        cands.sort(key=lambda x: (-x[0], x[1].id))
        queries["named" if q.person_id else "unnamed"] += 1
        best_same = {n: False for n in levels}
        best_other = {n: False for n in levels}
        best_unnamed = {n: False for n in levels}
        for c, r in cands[:20]:
            v = ver.rows(q, r)
            if v is None:
                continue
            if v.inliers >= 12 and v.ratio < trait_sift.MIN_RATIO:
                ratio_removed += 1
            entry = {"a": q.id, "b": r.id, "a_body": q.body_id, "b_body": r.body_id, "a_person": q.person_id,
                     "b_person": r.person_id, "inliers": v.inliers, "ratio": v.ratio, "cos": round(c, 3),
                     "a_uid": qi.uid, "b_uid": ld.items[r.item_id].uid}
            if v.inliers >= 8:
                pairs_by_trait.setdefault(q.id, []).append(entry)
            for n in levels:
                if not trait_sift.passes(v, n):   # the shipped rule: inliers and the ratio check
                    continue
                if q.person_id and r.person_id == q.person_id:
                    tally["named_same"][f"ge{n}"] += 1
                    best_same[n] = True
                elif q.person_id and r.person_id:
                    tally["named_other"][f"ge{n}"] += 1
                    best_other[n] = True
                elif not q.person_id and r.person_id:
                    tally["unnamed_named"][f"ge{n}"] += 1
                    best_unnamed[n] = True
            if q.person_id and r.person_id and r.person_id != q.person_id and trait_sift.passes(v, 12) and q.id < r.id:
                diff_pairs.append(entry)
            if not q.person_id and r.person_id and v.inliers >= 8:
                unnamed_hits.append(entry)
        for n in levels:
            if q.person_id:
                per_query_hits["named_same_only"][n] += int(best_same[n] and not best_other[n])
                per_query_hits["named_any_other"][n] += int(best_other[n])
            else:
                per_query_hits["unnamed_hit"][n] += int(best_unnamed[n])
    diff_pairs.sort(key=lambda e: (-e["inliers"], e["a"], e["b"]))
    return {"queries": queries, "pairs": tally, "per_query": per_query_hits, "ratio_check_removed": ratio_removed,
            "verified": ver.spent, "budget_hit": ver.left == 0, "top_different_person": diff_pairs[:20],
            "unnamed_hits": sorted(unnamed_hits, key=lambda e: -e["inliers"])[:100], "_by_trait": pairs_by_trait}


def watch(conn, body_ids: list[int], names: list[str], by_trait: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    from app.analysis import guess
    from app.analysis.index import reset_index

    ids = list(body_ids)
    for name in names:
        ids.extend(int(r[0]) for r in conn.execute(
            "SELECT DISTINCT b.id FROM bodies b JOIN persons p ON p.id = b.person_id JOIN body_traits t ON t.body_id = b.id "
            "WHERE p.name LIKE ? AND t.kind = 'tattoo' ORDER BY b.id LIMIT 40", (f"%{name}%",)))
    out = []
    for bid in dict.fromkeys(ids):
        b = conn.execute("SELECT b.*, i.uid, i.taken_at FROM bodies b JOIN items i ON i.id = b.item_id WHERE b.id = ?",
                         (bid,)).fetchone()
        if b is None:
            out.append({"body_id": bid, "missing": True})
            continue
        scan = conn.execute("SELECT status, rev, bare FROM trait_scans WHERE body_id = ?", (bid,)).fetchone()
        rows = [dict(r) for r in conn.execute("SELECT id, kind, part, attrs FROM body_traits WHERE body_id = ?", (bid,))]
        conn.execute("BEGIN IMMEDIATE")
        try:
            guess.guess_item(conn, int(b["item_id"]))
            g = conn.execute("SELECT guess_person_id, guess_evidence FROM bodies WHERE id = ?", (bid,)).fetchone()
            if b["face_id"] is not None:
                g = conn.execute("SELECT guess_person_id, guess_evidence FROM faces WHERE id = ?", (b["face_id"],)).fetchone()
        finally:
            conn.execute("ROLLBACK")
            reset_index()
        ev = json.loads(g["guess_evidence"]) if g and g["guess_evidence"] else None
        out.append({"body_id": bid, "item_uid": b["uid"], "taken_at": b["taken_at"], "person_id": b["person_id"],
                    "assigned_by": b["assigned_by"], "scan": dict(scan) if scan else None,
                    "rows": [{"id": r["id"], "kind": r["kind"], "part": r["part"]} for r in rows],
                    "verified_pairs": [p for r in rows for p in by_trait.get(int(r["id"]), [])][:20],
                    "guess": {"person_id": g["guess_person_id"] if g else None,
                              "basis": (ev or {}).get("basis"), "trait_llr": (ev or {}).get("trait_llr")}})
    return out


def leave_one_out(conn, n: int, variants: dict[str, Any]) -> dict[str, Any]:
    """Each trial inside a rolled-back transaction: the person is removed from the photo (faces and bodies back to
    auto, the 'in' tag deleted, and the same for its near-duplicates), the box's own face is deleted (a faceless box,
    so no face check decides for the traits), then guess_item runs with each param set."""
    from app.analysis import guess
    from app.analysis import trait_match as tm
    from app.analysis.index import reset_index
    from app.config import settings
    from app.learning import traits as ltraits

    from app.analysis import faces as faces_mod

    owner = conn.execute(
        "SELECT DISTINCT b.id, b.item_id, b.person_id, b.face_id FROM bodies b JOIN items i ON i.id = b.item_id "
        "LEFT JOIN faces f ON f.id = b.face_id JOIN body_traits t ON t.body_id = b.id "
        "WHERE b.person_id IS NOT NULL AND (b.assigned_by = 'user' OR f.assigned_by = 'user') AND i.deleted_at IS NULL "
        "AND i.nsfw = 0 AND i.kind = 'image' AND t.kind = 'tattoo' ORDER BY i.taken_at DESC LIMIT ?", (int(n),)).fetchall()
    # Second pool (the review's point: few tattooed boxes were confirmed by hand): bodies whose own face put them
    # in a named person's cluster, with a clear face (weight >= 0.5). Truth is the face cluster, reported apart.
    seen = {int(t["id"]) for t in owner}
    face_pool = []
    for t in conn.execute(
            "SELECT DISTINCT b.id, b.item_id, b.person_id, b.face_id, f.det_score, f.size_px, f.clarity, f.sharpness "
            "FROM bodies b JOIN items i ON i.id = b.item_id JOIN faces f ON f.id = b.face_id "
            "JOIN persons p ON p.id = b.person_id JOIN body_traits t ON t.body_id = b.id "
            "WHERE f.assigned_by = 'auto' AND p.name IS NOT NULL AND i.deleted_at IS NULL AND i.nsfw = 0 "
            "AND i.kind = 'image' AND t.kind = 'tattoo' ORDER BY i.taken_at DESC"):
        if int(t["id"]) in seen:
            continue
        if faces_mod.face_weight(t["det_score"], t["size_px"], t["clarity"], t["sharpness"]) >= 0.5:
            face_pool.append(t)
            seen.add(int(t["id"]))
        if len(face_pool) >= n:
            break
    trials = [(t, "owner") for t in owner] + [(t, "face") for t in face_pool]
    ld = tm._Loader(conn, {})
    results: dict[str, list[dict[str, Any]]] = {k: [] for k in variants}
    real_params = ltraits.params
    for k, (t, pool) in enumerate(trials):
        pid, iid = int(t["person_id"]), int(t["item_id"])
        it = ld.item_info(iid)
        near = [iid]
        if it is not None:
            for r in conn.execute(
                    "SELECT DISTINCT item_id FROM bodies WHERE person_id = ? UNION SELECT item_id FROM faces WHERE person_id = ?",
                    (pid, pid)):
                o = ld.item_info(int(r[0]))
                if o is not None and o.id != iid and (o.t is None or it.t is None or
                                                      abs((o.t - it.t).total_seconds()) <= 600) and ld.near_dup(it, o):
                    near.append(o.id)
        for name, params in variants.items():
            conn.execute("BEGIN IMMEDIATE")
            try:
                for j in near:
                    conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'auto' WHERE item_id = ? AND person_id = ?",
                                 (j, pid))
                    conn.execute("UPDATE bodies SET person_id = NULL, assigned_by = 'auto' WHERE item_id = ? AND person_id = ?",
                                 (j, pid))
                    conn.execute("DELETE FROM item_tags WHERE item_id = ? AND person_id = ? AND relation = 'in'", (j, pid))
                if t["face_id"] is not None:
                    # A faceless box, the case traits are for: with the face kept, the face veto alone would leave
                    # only the true person (the box's own face scores them at 0.40 or more), so "all right" would be
                    # built in. The rollback brings the face back.
                    conn.execute("DELETE FROM faces WHERE id = ?", (int(t["face_id"]),))
                reset_index()
                saved_window = settings.guess_window_h
                if params == "clothes":
                    saved = settings.trait_matching
                    settings.trait_matching = False
                elif params == "no_clothes":
                    # the trait path alone: no photo in the clothes window, so only trait-only guesses can appear
                    settings.guess_window_h = 0.0
                else:
                    ltraits.params = (lambda p=params: p)  # type: ignore[assignment]
                try:
                    guess.guess_item(conn, iid)
                finally:
                    if params == "clothes":
                        settings.trait_matching = saved
                    settings.guess_window_h = saved_window
                    ltraits.params = real_params  # type: ignore[assignment]
                g = conn.execute("SELECT guess_person_id, guess_evidence FROM bodies WHERE id = ?", (int(t["id"]),)).fetchone()
            finally:
                conn.execute("ROLLBACK")
                reset_index()
            ev = json.loads(g["guess_evidence"]) if g and g["guess_evidence"] else {}
            kinds = sorted({e.get("kind") for e in ev.get("traits", [])}) if ev else []
            lead = (ev.get("traits") or [{}])[0] if ev else {}
            days = lead.get("days_apart") if isinstance(lead, dict) else None
            results[name].append({"body_id": int(t["id"]), "truth": pid, "guess": g["guess_person_id"] if g else None,
                                  "basis": ev.get("basis") or ("clothes" if ev else None), "kinds": kinds,
                                  "pool": pool, "days": days})
        print(f"progress: {k + 1}/{len(trials)}", flush=True)
    report: dict[str, Any] = {"trials": len(trials), "pools": {"owner": len(owner), "face": len(face_pool)}}
    for name, lst in results.items():
        by: dict[str, dict[str, Any]] = {}
        for r in lst:
            if r["guess"] is None:
                continue
            span = "" if r["days"] is None else ("|same day" if r["days"] < 1 else "|later days")
            key = f"{r['pool']}:" + r["basis"] + ("" if not r["kinds"] else ":" + "+".join(r["kinds"])) + span
            d = by.setdefault(key, {"n": 0, "right": 0, "false_names": []})
            d["n"] += 1
            if r["guess"] == r["truth"]:
                d["right"] += 1
            else:
                d["false_names"].append({"body_id": r["body_id"], "truth": r["truth"], "guess": r["guess"]})
        for d in by.values():
            d["precision"] = round(d["right"] / d["n"], 3)
            d["wilson95"] = wilson(d["right"], d["n"])
            d["coverage"] = round(d["n"] / max(1, len(lst)), 3)
        report[name] = by
    if "clothes" in results and "trained" in results:
        base = {r["body_id"]: r["guess"] for r in results["clothes"]}
        report["agreement_changed_clothes_winner"] = [
            r["body_id"] for r in results["trained"]
            if r["basis"] == "clothes+trait" and base.get(r["body_id"]) is not None and base[r["body_id"]] != r["guess"]]
    # judged on the trait path alone (no photo in the clothes window), each box counted once
    trait_only = [r for r in results.get("no_clothes", []) if r["basis"] == "trait" and "tattoo" in r["kinds"]]
    right = sum(1 for r in trait_only if r["guess"] == r["truth"])
    later = [r for r in trait_only if r["days"] is not None and r["days"] >= 1]
    report["pass_bar"] = {
        "tattoo_trait_only_decisions": len(trait_only),
        "tattoo_trait_only_precision": round(right / len(trait_only), 3) if trait_only else None,
        "tattoo_trait_only_wilson95": wilson(right, len(trait_only)) if trait_only else None,
        "later_days_decisions": len(later),
        "later_days_right": sum(1 for r in later if r["guess"] == r["truth"]),
        "precision_ok": bool(trait_only) and len(trait_only) >= 20 and right / len(trait_only) >= 0.95,
        "agreement_ok": not report.get("agreement_changed_clothes_winner"),
        "gate_sheet": "check by eye: no fabric, watch or bangle among the tattoo crops"}
    return report


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check body traits on a library snapshot (never the live library).")
    ap.add_argument("--snapshot", required=True, type=Path, help="folder with cloud.db (sqlite3 .backup) and learned/")
    ap.add_argument("--previews", required=True, type=Path, help="the live DATA_DIR/previews, read only")
    ap.add_argument("--originals", type=Path, help="the live originals folder (ORIGINALS_DIR/originals), read only")
    ap.add_argument("--limit-items", type=int, help="measure at most this many items")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, help="report JSON (default <snapshot>/trait_eval.json)")
    ap.add_argument("--sheets", type=Path, help="contact sheets folder (default next to the report)")
    ap.add_argument("--loo", type=int, default=300, help="leave-one-out trials (0 skips)")
    ap.add_argument("--bodies", default=",".join(map(str, WATCH_BODIES)), help="body ids to report one by one")
    ap.add_argument("--watch-person", action="append", default=[], help="also report the tattoo bodies of this name")
    ap.add_argument("--skip-scan", action="store_true", help="use the rows already in the snapshot")
    ap.add_argument("--max-verify", type=int, default=30000, help="verifications for the per-query table")
    ap.add_argument("--live-data", action="append", default=[], help="another live data folder to refuse")
    args = ap.parse_args(argv)

    why = refuse_live(args.snapshot, _live_dirs(args.live_data))
    if why:
        print(f"refusing: {why}", file=sys.stderr)
        return 2
    snapshot = args.snapshot.expanduser().resolve()
    os.environ["DATA_DIR"] = str(snapshot)
    os.environ["WORKERS_ENABLED"] = "0"
    os.environ["TRAITS_THREADS"] = str(max(1, args.threads))
    os.environ.pop("ORIGINALS_DIR", None)
    out = (args.out or snapshot / "trait_eval.json").expanduser().resolve()
    sheets = (args.sheets or out.parent / "trait_eval_sheets").expanduser().resolve()

    from app.config import reload_settings

    reload_settings()
    _redirect_storage(args.previews.expanduser().resolve(),
                      args.originals.expanduser().resolve() if args.originals else None)
    from app import db
    from app.analysis import trait_match as tm
    from app.analysis import traits as atraits
    from app.db import kv_set, tx
    from app.learning import traits as ltraits
    from app.analysis import trait_sift

    trait_sift.FEATURE_CACHE = 1024  # the eval visits references all over the library (about 400 MB)

    db.init_db()
    conn = db.connect()
    report: dict[str, Any] = {"snapshot": str(snapshot), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    size0 = db_bytes(snapshot)
    if not args.skip_scan:
        report["scan"] = scan(conn, snapshot, args.limit_items)
    if not atraits.ready(conn):
        with tx(conn):
            kv_set(conn, atraits.READY_KEY, atraits.wanted_rev())
        report["forced_ready"] = "the snapshot was marked ready for this report (partial scan)"
    report["db_growth_mb"] = round((db_bytes(snapshot) - size0) / 1e6, 1)
    report["counts"] = counts(conn)
    t = time.monotonic()
    report["calibration"] = tm.calibration_sample(conn, force=True)
    report["calibration_s"] = round(time.monotonic() - t, 1)
    report["learner"] = ltraits.train(conn)
    trained = ltraits.params()
    report["params"] = trained.to_dict()
    report["gate"] = gate_sheet(conn, sheets)
    report["hands"] = hand_spans(conn)
    report["open_faces_without_body"] = open_faces_without_body(conn)
    t = time.monotonic()
    table = verification_table(conn, args.max_verify)
    by_trait = table.pop("_by_trait")
    report["verification"] = table
    report["verification_s"] = round(time.monotonic() - t, 1)
    report["top_different_sheet"] = contact_sheet(
        conn, [(x, f"{e['a_body']}/{e['b_body']} {e['inliers']}") for e in table["top_different_person"]
               for x in (e["a"], e["b"])], sheets / "top_different_pairs.png", cols=8)
    report["watch"] = watch(conn, [int(x) for x in args.bodies.split(",") if x.strip()], args.watch_person, by_trait)
    if args.loo > 0:
        import dataclasses

        ceil = dataclasses.replace(trained, tattoo_tiers=tuple(
            (lo, c) for (lo, _), c in zip(ltraits.TATTOO_TIERS, ltraits.TATTOO_CEILINGS)))
        variants = {"trained": trained, "defaults": ltraits.default_params(), "ceilings": ceil, "clothes": "clothes",
                    "no_clothes": "no_clothes"}
        t = time.monotonic()
        report["leave_one_out"] = leave_one_out(conn, args.loo, variants)
        report["leave_one_out_s"] = round(time.monotonic() - t, 1)
    report["peak_rss_mb"] = peak_rss_mb()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    conn.close()
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
