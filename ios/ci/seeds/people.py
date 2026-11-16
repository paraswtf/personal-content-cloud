"""The people area's seed (ios/UITests/PeopleScreens.swift), run by ci/seed_server.py.

Its own people, all named after the area, so the tests that rename, merge, tag or move change nothing another
area looks at: "People Alpha" (faces, a cover, two guesses, a tattoo), "People Beta" and "People Gamma" (faces
that look alike, so they are suggested as the same person), "People Delta" and "People Epsilon" (merged by a
test), "People Zeta" (renamed by a test), the pet "People Pet" (a cat with two crops), and two faces nobody
named, drawn 333 pixels wide so a test can tell them from the core's. Everything is dated March 2026 and before.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np

SKIN, HAIR = (214, 170, 140), (40, 30, 26)
UNNAMED_PX = 333


def library(kit) -> dict:
    from tests.helpers import make_trait, set_traits_ready

    conn = kit.conn
    rng = np.random.default_rng(5150)
    base = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)

    def unit(v: np.ndarray) -> np.ndarray:
        return (v / np.linalg.norm(v)).astype(np.float32)

    def random_vec() -> np.ndarray:
        return unit(rng.standard_normal(512).astype(np.float32))

    def near(center: np.ndarray) -> np.ndarray:
        return unit(center + 0.15 * rng.standard_normal(512).astype(np.float32) / np.sqrt(512))

    def person(name: str, kind: str = "person", species: str | None = None) -> int:
        pid = kit.make_person(conn, name)
        if kind != "person":
            with kit.tx(conn):
                conn.execute("UPDATE persons SET kind = ?, species = ? WHERE id = ?", (kind, species, pid))
        return pid

    ids = {name: person(name) for name in
           ("People Alpha", "People Beta", "People Gamma", "People Delta", "People Epsilon", "People Zeta")}
    pet = person("People Pet", kind="pet", species="cat")
    looks = {name: random_vec() for name in ids}
    looks["People Gamma"] = near(looks["People Beta"])  # Beta and Gamma look alike: a merge suggestion

    def photo(k: int, faces: list[tuple[tuple, int | None, dict]], when: datetime | None = None,
              gps: tuple[float, float] | None = None) -> tuple[dict, list[int]]:
        """A drawn photo with faces [(box, person id or None, extra face columns)]."""
        img = kit.scene(5100 + k, [(box, SKIN, HAIR) for box, _, _ in faces])
        taken = (when or base - timedelta(days=3 * k, hours=k)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        extra = {"gps_lat": gps[0], "gps_lon": gps[1]} if gps else {}
        item = kit.make_item(conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename=f"PEOPLE_{k:02d}.JPG",
                             taken_at=taken, camera="CI Camera", category="personal", **extra)
        kit.previews(item, img)
        face_ids = []
        for box, pid, extra in faces:
            extra = dict(extra)
            size_px = extra.pop("size_px", int(box[2] * kit.W))
            fid = kit.make_face(conn, int(item["id"]), pid, box=box, size_px=size_px, **extra)
            kit.crop(img, box).save(kit.storage.face_crop_path(fid), "WEBP", quality=85)
            face_ids.append(fid)
        return item, face_ids

    left, right = (0.20, 0.30, 0.13, 0.20), (0.62, 0.32, 0.12, 0.18)
    alpha = ids["People Alpha"]
    faces: dict[str, list[int]] = {name: [] for name in ids}

    def mine(name: str) -> dict:
        return {"embedding": near(looks[name]), "assigned_by": "user"}

    # People Alpha: three photos, the first with a tattoo on the forearm
    alpha_items = []
    for k in range(3):
        item, (fid,) = photo(k, [(left, alpha, mine("People Alpha"))])
        alpha_items.append(item)
        faces["People Alpha"].append(fid)
    with kit.tx(conn):
        conn.execute("UPDATE persons SET cover_face_id = ? WHERE id = ?", (faces["People Alpha"][0], alpha))

    # two guesses for People Alpha, made by the server's own guessing (section 10): a photo of Alpha, and two more
    # taken minutes later at the same place with someone in the same clothes and a face that matches nobody
    torso = unit(rng.standard_normal(768).astype(np.float32))

    def same_clothes() -> np.ndarray:
        return unit(torso + 0.05 * rng.standard_normal(768).astype(np.float32) / np.sqrt(768))

    party, place = datetime(2026, 2, 14, 18, 0, tzinfo=timezone.utc), (12.9716, 77.5946)
    # the first face is named People Alpha in step 2, through the API, so the server guesses the other two
    _, (party_face,) = photo(3, [(left, None, {"embedding": random_vec(), "torso": same_clothes()})],
                             when=party, gps=place)
    guess_faces = []
    for k, minutes in ((4, 10), (12, 20)):
        _, (fid,) = photo(k, [(right, None, {"embedding": random_vec(), "torso": same_clothes()})],
                          when=party + timedelta(minutes=minutes), gps=place)
        guess_faces.append(fid)

    # Beta with Delta, Gamma with Epsilon, Beta with Zeta, Gamma with Zeta: Beta and Gamma never share a photo
    for k, (a, b) in enumerate((("People Beta", "People Delta"), ("People Gamma", "People Epsilon"),
                                ("People Beta", "People Zeta"), ("People Gamma", "People Zeta")), start=5):
        _, (fa, fb) = photo(k, [(left, ids[a], mine(a)), (right, ids[b], mine(b))])
        faces[a].append(fa)
        faces[b].append(fb)

    # two faces nobody named, told apart from the core's by their size
    _, unnamed = photo(9, [(left, None, {"embedding": random_vec(), "size_px": UNNAMED_PX}),
                           (right, None, {"embedding": random_vec(), "size_px": UNNAMED_PX})])

    # People Pet: a cat in two photos (the crops are cut from the landscape; the detector is not run here)
    crops = []
    for k in (10, 11):
        img = kit.scene(5100 + k, [])
        item = kit.make_item(conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename=f"PEOPLE_{k:02d}.JPG",
                             taken_at=(base - timedelta(days=3 * k)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                             camera="CI Camera", category="personal")
        kit.previews(item, img)
        box = (0.40, 0.55, 0.20, 0.25)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        with kit.tx(conn):
            cur = conn.execute(
                "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, created_at) "
                "VALUES (?, ?, 'user', ?, ?, ?, ?, 0.9, 'cat', ?)", (int(item["id"]), pet, *box, now))
            aid = int(cur.lastrowid)
            conn.execute("UPDATE items SET animal_count = 1 WHERE id = ?", (int(item["id"]),))
        path = kit.storage.animal_crop_path(aid)
        path.parent.mkdir(parents=True, exist_ok=True)
        kit.crop(img, box).save(path, "WEBP", quality=85)
        crops.append(aid)

    # a tattoo on People Alpha's forearm (section 34), so the person page has its "Tattoos and nails" tab
    body = kit.make_body(conn, int(alpha_items[0]["id"]), alpha, box=(0.12, 0.28, 0.30, 0.65), assigned_by="user")
    trait = make_trait(conn, body, "tattoo", "forearm_l", vec=rng.standard_normal(384).astype(np.float32))
    set_traits_ready(conn)

    out = {"people": ids, "pet": pet, "faces": faces, "party_face": party_face, "guess_faces": guess_faces,
           "unnamed_faces": unnamed, "crops": crops, "trait": trait}
    (kit.uploads_dir / "people-seed.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def api(kit) -> dict:
    """Names the party photo's face People Alpha as the owner would, so the server's guessing (section 10) finds
    the two faces in the same clothes minutes later and asks about them; waits until it has."""
    seed = json.loads((kit.uploads_dir / "people-seed.json").read_text(encoding="utf-8"))
    alpha = seed["people"]["People Alpha"]
    kit.client.call("POST", "/api/faces/assign", {"face_ids": [seed["party_face"]], "person_id": alpha})
    guesses = 0
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        guesses = kit.client.call("GET", f"/api/review/guesses?person={alpha}&limit=60")["total"]
        if guesses >= len(seed["guess_faces"]):
            break
        time.sleep(1)
    if guesses < len(seed["guess_faces"]):
        print(f"people seed: the server made {guesses} of {len(seed['guess_faces'])} guesses for People Alpha")
    return {"alpha": alpha, "guesses": guesses}
