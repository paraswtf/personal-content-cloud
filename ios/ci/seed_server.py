"""A made-up library for the iOS app's UI tests on the CI machine (ios/README.md, "Tests on GitHub").

Two steps, both run with the backend's Python from the backend folder, with the same DATA_DIR,
ORIGINALS_DIR, BACKUP_DIR and MODELS_DIR as the server:

    python ../ios/ci/seed_server.py library            # before the server starts
    python ../ios/ci/seed_server.py api --server URL --out seed.json   # once it answers

"library" writes rows and files straight into DATA_DIR with the backend's own test helpers: photos spread
over six months (two thirds with a place), two named people with face boxes, three faces nobody named, a
vault photo, a favourite or two and a photo in the trash. Faces need rows the models would have written,
which is why this step uses the backend's code rather than the API. It also draws the photos the second
step uploads.

"api" talks to the running server over HTTP only, the way the website and the app do: it sets up the
owner account, opens the vault, creates a device token (Settings > Sessions > Devices), uploads three
photos with the app's resumable protocol and a text note, makes an album, shares it, and writes what the
tests need (server, token, counts) to --out and to stdout.

Everything is generated: no real people, no real photos, nothing from any owner's library.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent / "backend"

USERNAME = "owner"
PASSWORD = "ci-only-password"  # a throwaway account on a server that lives for one CI run
DEVICE_NAME = "iPhone simulator"

# Places the photos were "taken" at: (latitude, longitude).
PLACES = [(15.4909, 73.8278), (19.0760, 72.8777), (12.9716, 77.5946), (28.6139, 77.2090)]
PEOPLE = [
    # name, skin, hair: the drawn faces of one person look alike
    ("Test Person", (224, 182, 150), (60, 40, 30)),
    ("Test Visitor", (176, 128, 96), (20, 20, 24)),
]
UPLOADS = 3


def uploads_dir() -> Path:
    """Where "library" leaves the photos "api" uploads: next to DATA_DIR, outside what the server reads."""
    data = Path(os.environ["DATA_DIR"]).resolve()
    return data.parent / "seed-uploads"


# ------------------------------------------------------------------ step 1: rows and files
def library() -> dict:
    sys.path.insert(0, str(BACKEND))
    import io
    import random
    from datetime import datetime, timedelta, timezone

    import numpy as np
    from PIL import Image, ImageDraw

    from app import db, storage
    from app.db import tx
    from tests.helpers import make_face, make_item, make_person

    storage.ensure_dirs()
    db.init_db()
    conn = db.connect()

    W, H = 1024, 768

    def scene(seed: int, faces: list[tuple[tuple[float, float, float, float], tuple, tuple]]) -> Image.Image:
        """A landscape: sky, sun, hills, water, and a cartoon head for every face box."""
        rnd = random.Random(seed)
        skies = [((86, 156, 222), (198, 228, 250)), ((240, 132, 92), (252, 214, 150)),
                 ((32, 44, 96), (110, 120, 180)), ((120, 180, 210), (230, 240, 240))]
        top, bottom = skies[seed % len(skies)]
        t = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
        sky = (np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t)
        img = np.broadcast_to(sky, (H, W, 3)).copy()
        im = Image.fromarray(img.astype(np.uint8))
        d = ImageDraw.Draw(im)
        sx, sy, sr = rnd.randint(100, W - 100), rnd.randint(60, 220), rnd.randint(30, 60)
        d.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=(255, 236, 170))
        for layer in range(3):
            base = int(H * (0.5 + 0.12 * layer))
            amp, freq, phase = rnd.randint(20, 70), rnd.uniform(1.0, 3.0), rnd.uniform(0, 6.28)
            green = (40 + 30 * layer + rnd.randint(0, 30), 90 + 30 * layer + rnd.randint(0, 40), 50 + 10 * layer)
            pts = [(x, base - amp * math.sin(freq * x / W * 6.28 + phase)) for x in range(0, W + 16, 16)]
            d.polygon([(0, H), *pts, (W, H)], fill=green)
        if seed % 3 == 0:
            d.rectangle((0, int(H * 0.82), W, H), fill=(70, 120, 170))
        for (x, y, w, h), skin, hair in faces:
            x0, y0, x1, y1 = x * W, y * H, (x + w) * W, (y + h) * H
            d.ellipse((x0 - 0.35 * (x1 - x0), y1 - 0.1 * (y1 - y0), x1 + 0.35 * (x1 - x0), y1 + 1.4 * (y1 - y0)),
                      fill=(rnd.randint(40, 200), rnd.randint(40, 200), rnd.randint(40, 200)))  # shoulders
            d.ellipse((x0, y0, x1, y1), fill=skin)
            d.chord((x0, y0 - 0.08 * (y1 - y0), x1, y0 + 0.55 * (y1 - y0)), 180, 360, fill=hair)
            ew = 0.12 * (x1 - x0)
            for ex in (x0 + 0.32 * (x1 - x0), x0 + 0.68 * (x1 - x0)):
                ey = y0 + 0.5 * (y1 - y0)
                d.ellipse((ex - ew / 2, ey - ew / 2, ex + ew / 2, ey + ew / 2), fill=(30, 30, 30))
            d.arc((x0 + 0.3 * (x1 - x0), y0 + 0.55 * (y1 - y0), x0 + 0.7 * (x1 - x0), y0 + 0.82 * (y1 - y0)),
                  20, 160, fill=(120, 40, 40), width=max(2, int(0.03 * (x1 - x0))))
        # a little grain, so the quality check does not take a smooth drawing for a blurred mis-click
        arr = np.asarray(im, dtype=np.int16)
        arr += np.random.default_rng(seed).integers(-10, 11, size=arr.shape, dtype=np.int16)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def jpeg(img: Image.Image, exif: Image.Exif | None = None) -> bytes:
        buf = io.BytesIO()
        kw = {"quality": 88}
        if exif is not None:
            kw["exif"] = exif
        img.save(buf, "JPEG", **kw)
        return buf.getvalue()

    def crop(img: Image.Image, box) -> Image.Image:
        x, y, w, h = box
        pad = 0.25
        x0, y0 = max(0, (x - pad * w) * W), max(0, (y - pad * h) * H)
        x1, y1 = min(W, (x + w + pad * w) * W), min(H, (y + h + pad * h) * H)
        return img.crop((int(x0), int(y0), int(x1), int(y1))).resize((192, 192))

    end = datetime(2026, 10, 30, 17, 0, tzinfo=timezone.utc)
    person_ids = {name: make_person(conn, name) for name, _, _ in PEOPLE}
    counts = {"photos": 0, "with_place": 0, "faces": 0, "unnamed_faces": 0, "vault": 0, "trash": 0, "favorites": 0}
    for i in range(36):
        taken = end - timedelta(days=5 * i, hours=i % 7)
        faces = []
        if i % 3 == 0:  # the first person
            faces.append(((0.18, 0.30, 0.13, 0.20), PEOPLE[0][0]))
        if i % 4 == 1:  # the second person
            faces.append(((0.62, 0.34, 0.12, 0.18), PEOPLE[1][0]))
        if i in (5, 17, 29):  # someone nobody named
            faces.append(((0.44, 0.40, 0.08, 0.12), None))
        drawn = [(box, *(next((s, h) for n, s, h in PEOPLE if n == who) if who else ((205, 170, 140), (90, 70, 50))))
                 for box, who in faces]
        img = scene(1000 + i, drawn)
        place = PLACES[i % len(PLACES)] if i % 3 != 2 else None
        extra = {"gps_lat": place[0], "gps_lon": place[1]} if place else {}
        item = make_item(conn, data=jpeg(img), width=W, height=H, filename=f"IMG_{4000 + i}.JPG",
                         taken_at=taken.strftime("%Y-%m-%dT%H:%M:%S.000Z"), camera="CI Camera",
                         category="places" if place else "misc", favorite=1 if i in (2, 9) else 0,
                         nsfw=1 if i == 11 else 0, **extra)
        # the helper keeps a downscaled copy as thumb and previews; write them from the picture itself
        for path, side in ((storage.thumb_path(item["uid"], 1), 480), (storage.preview_path(item["uid"], 1), 2048),
                           (storage.preview_small_path(item["uid"], 1), 1280)):
            copy = img.copy()
            copy.thumbnail((side, side))
            copy.save(path, "WEBP", quality=80)
        counts["photos"] += 1
        counts["with_place"] += 1 if place else 0
        counts["vault"] += 1 if i == 11 else 0
        counts["favorites"] += 1 if i in (2, 9) else 0
        for box, who in faces:
            face_id = make_face(conn, int(item["id"]), person_ids[who] if who else None, box=box,
                                size_px=int(box[2] * W), assigned_by="user" if who and i < 6 else "auto")
            crop(img, box).save(storage.face_crop_path(face_id), "WEBP", quality=85)
            counts["faces" if who else "unnamed_faces"] += 1
        if i == 7:
            with tx(conn):
                conn.execute("UPDATE items SET deleted_at = ? WHERE id = ?",
                             ((end - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"), item["id"]))
            counts["trash"] += 1

    # the photos step 2 uploads: drawn here, where Pillow is at hand, with a date and a place in their EXIF
    out = uploads_dir()
    out.mkdir(parents=True, exist_ok=True)
    for k in range(UPLOADS):
        exif = Image.Exif()
        exif[0x0110] = "CI Camera"
        when = end + timedelta(days=1, hours=k)
        exif.get_ifd(0x8769)[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
        lat, lon = PLACES[k % len(PLACES)]
        gps = exif.get_ifd(0x8825)

        def dms(v: float) -> tuple[float, float, float]:
            deg = int(v)
            minutes = int((v - deg) * 60)
            return (float(deg), float(minutes), round(((v - deg) * 60 - minutes) * 60, 2))

        gps[1], gps[2] = ("N" if lat >= 0 else "S"), dms(abs(lat))
        gps[3], gps[4] = ("E" if lon >= 0 else "W"), dms(abs(lon))
        (out / f"IMG_{5000 + k}.JPG").write_bytes(jpeg(scene(2000 + k, []), exif))
    (out / "notes.txt").write_text(
        "Packing list for the weekend\n\nCamera, charger, two lenses, a rain jacket, the train tickets.\n"
        "Leave on Friday at 7, back on Sunday evening.\n", encoding="utf-8")
    conn.close()
    counts["people"] = len(person_ids)
    counts["upload_files"] = UPLOADS + 1
    print(json.dumps({"library": counts}))
    return counts


# ------------------------------------------------------------------ step 2: the API, as a client
class Client:
    def __init__(self, server: str):
        self.server = server.rstrip("/")
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def call(self, method: str, path: str, body=None, *, raw: bytes | None = None, ok=(200, 201),
             headers: dict | None = None):
        data, hdrs = None, {"X-Requested-With": "pcc", **(headers or {})}
        if raw is not None:
            data = raw
            hdrs.setdefault("Content-Type", "application/octet-stream")
        elif body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(self.server + path, data=data, method=method, headers=hdrs)
        try:
            with self.opener.open(req, timeout=60) as r:
                status, payload = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, payload = e.code, e.read()
        if status not in ok:
            raise SystemExit(f"{method} {path} answered {status}: {payload[:300]!r}")
        return json.loads(payload) if payload else None

    def upload(self, path: Path) -> dict:
        """The app's resumable protocol (section 28): create, chunks, complete, then wait until stored."""
        data = path.read_bytes()
        up = self.call("POST", "/api/uploads", {"filename": path.name, "size": len(data), "client_id": "ios-ci-seed",
                                                "source": "api"})
        if up["status"] == "uploading":
            offset = int(up["received_bytes"])
            while offset < len(data):
                part = data[offset: offset + int(up["chunk_size"])]
                offset = int(self.call("PUT", f"/api/uploads/{up['id']}/chunk?offset={offset}", raw=part)["received_bytes"])
            up = self.call("POST", f"/api/uploads/{up['id']}/complete")
        deadline = time.monotonic() + 180
        while up["status"] in ("queued", "processing", "uploading") and time.monotonic() < deadline:
            time.sleep(1)
            up = self.call("GET", f"/api/uploads/{up['id']}")
        if up["status"] not in ("stored", "duplicate"):
            raise SystemExit(f"upload of {path.name} ended as {up['status']}: {up.get('message')}")
        return up


def api(server: str, out: Path | None) -> dict:
    c = Client(server)
    deadline = time.monotonic() + 120
    while True:
        try:
            state = c.call("GET", "/api/auth/state")
            break
        except (SystemExit, OSError):
            if time.monotonic() > deadline:
                raise SystemExit(f"the server at {server} did not answer")
            time.sleep(1)
    if state.get("setup_required"):
        c.call("POST", "/api/auth/setup", {"username": USERNAME, "password": PASSWORD})
    else:
        c.call("POST", "/api/auth/login", {"username": USERNAME, "password": PASSWORD})
    c.call("POST", "/api/auth/vault/unlock", {"password": PASSWORD})
    device = c.call("POST", "/api/auth/devices", {"name": DEVICE_NAME})

    uploaded = [c.upload(p) for p in sorted(uploads_dir().glob("IMG_*.JPG"))]
    note = c.upload(uploads_dir() / "notes.txt")

    items = c.call("GET", "/api/items?limit=60")["items"]
    uids = [it["uid"] for it in items]
    album = c.call("POST", "/api/albums", {"name": "Weekend trip", "description": "Made by the CI seed",
                                           "item_uids": uids[:8]})
    c.call("POST", "/api/albums", {"name": "Hills", "item_uids": uids[8:14]})
    share = c.call("POST", "/api/shares", {"kind": "album", "album_id": album["id"], "title": "Weekend trip"})

    # what the tests check against, read back as the app will see it (the device token, not the cookie)
    bearer = {"Authorization": f"Bearer {device['token']}"}
    tokened = Client(server)
    people = tokened.call("GET", "/api/people?limit=50", headers=bearer)
    library_items = tokened.call("GET", "/api/items?limit=100", headers=bearer)
    result = {
        "server": c.server,
        "token": device["token"],
        "username": USERNAME,
        "uploaded": [u.get("item_uid") for u in uploaded],
        "note": note.get("item_uid"),
        "album_id": album["id"],
        "share_path": share.get("path"),
        "people": [p.get("name") for p in people.get("items", [])],
        "items": library_items.get("total", len(library_items.get("items", []))),
    }
    text = json.dumps(result, indent=2)
    if out:
        out.write_text(text, encoding="utf-8")
    print(text)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("library", help="rows and files, before the server starts")
    a = sub.add_parser("api", help="account, token, uploads, album and share, once the server answers")
    a.add_argument("--server", required=True)
    a.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.step == "library":
        library()
    else:
        api(args.server, args.out)


if __name__ == "__main__":
    main()
