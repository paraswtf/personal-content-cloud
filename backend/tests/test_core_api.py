"""Core HTTP API: uploads, items, trash, review, duplicates, stats, files, system and the app shell."""
from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image

from tests.helpers import (
    H,
    jpeg_bytes,
    make_face,
    make_item,
    make_person,
    make_video,
    noise_image,
    pdf_bytes,
    run_jobs,
    upload_bytes,
    upload_status,
)


def noop(conn, *args) -> None:
    """Stand-in handler for jobs of other areas (analysis, documents)."""


@pytest.fixture(autouse=True)
def quiet_jobs(monkeypatch):
    """Only core's own handlers run here: analysis and documents jobs complete without doing anything
    (no model is loaded by these tests)."""
    from app import worker

    for kind in ("analyze", "text", "guess"):
        monkeypatch.setitem(worker.HANDLERS, kind, (__name__, "noop", worker.HANDLERS[kind][2]))
    monkeypatch.setitem(worker.HANDLERS, "recluster", (__name__, "noop", lambda p: ()))


def add(client, data: bytes, name: str, **kw) -> dict:
    """Upload through the chunked protocol, run the jobs, return the item detail."""
    up = upload_bytes(client, data, name, **kw)
    run_jobs()
    st = upload_status(client, up["id"])
    assert st["status"] in ("stored", "duplicate"), st
    r = client.get(f"/api/items/{st['item_uid']}")
    assert r.status_code == 200, r.text
    return r.json()


def conn_():
    from app.db import connect

    return connect()


# ------------------------------------------------------------------ auth gates
def test_everything_requires_login(anon_client):
    for path in ("/api/items", "/api/config", "/api/stats", "/api/trash", "/api/duplicates", "/api/system",
                 "/api/review/misclicks", "/api/uploads/mine?client_id=x", "/files/thumb/abc", "/files/original/abc",
                 "/files/face/1"):
        r = anon_client.get(path)
        assert r.status_code == 401, (path, r.status_code)
        assert r.json() == {"detail": "login_required"}
    assert anon_client.post("/api/uploads", json={"filename": "a.jpg", "size": 1}).status_code == 403  # csrf
    assert anon_client.post("/api/uploads", json={"filename": "a.jpg", "size": 1}, headers=H).status_code == 401


def test_app_shell(anon_client):
    r = anon_client.get("/api/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "private, no-store"
    r = anon_client.post("/share-target", data={"title": "x"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/upload?shared=1"
    r = anon_client.get("/")
    assert r.status_code == 200 and "not built" in r.text


def test_spa_fallback(anon_client):
    from app.config import settings

    dist = settings.frontend_dist
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>PCC</title>")
    (dist / "assets" / "app.js").write_text("console.log('x');" * 200)
    (dist / "sw.js").write_text("self.addEventListener('fetch', () => {});")
    (dist / "manifest.webmanifest").write_text('{"name": "PCC"}')
    r = anon_client.get("/sw.js")
    assert r.headers["content-type"].startswith("text/javascript") and r.headers["cache-control"] == "no-cache"
    r = anon_client.get("/manifest.webmanifest")
    assert r.headers["content-type"] == "application/manifest+json" and r.headers["cache-control"] == "no-cache"
    r = anon_client.get("/people/3")
    assert r.status_code == 200 and "PCC" in r.text and r.headers["cache-control"] == "no-cache"
    r = anon_client.get("/assets/app.js", headers={"accept-encoding": "gzip"})
    assert r.status_code == 200 and "immutable" in r.headers["cache-control"]
    assert anon_client.get("/assets/missing.js").status_code == 404


# ------------------------------------------------------------------ uploads
def test_upload_round_trip_and_byte_identical_download(client):
    data = jpeg_bytes(noise_image(1200, 900, seed=1), quality=95)
    assert len(data) > 3 * 256 * 1024  # several chunks
    up = upload_bytes(client, data, "Holiday Photo.jpeg", source="camera")
    assert up["status"] == "queued"
    run_jobs()
    st = upload_status(client, up["id"])
    assert st["status"] == "stored" and st["item_uid"] == up["id"] and st["item_thumb_url"]
    uid = st["item_uid"]
    r = client.get(f"/files/original/{uid}", params={"download": 1})
    assert r.status_code == 200
    assert hashlib.sha256(r.content).hexdigest() == hashlib.sha256(data).hexdigest()
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["content-disposition"].startswith('attachment; filename="Holiday Photo.jpeg"')
    detail = client.get(f"/api/items/{uid}").json()
    assert detail["kind"] == "image" and detail["filename"] == "Holiday Photo.jpeg"
    assert detail["size_bytes"] == len(data) and (detail["width"], detail["height"]) == (1200, 900)
    assert detail["urls"]["download"].endswith("download=1")
    r = client.get(detail["urls"]["thumb"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert r.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert client.get(f"/files/thumb/{uid}").headers["cache-control"] == "private, no-cache"
    assert client.get(detail["urls"]["preview_small"]).status_code == 200
    r = client.get(f"/files/original/{uid}", headers={"range": "bytes=0-9"})
    assert r.status_code == 206 and r.content == data[:10]
    mine = client.get("/api/uploads/mine", params={"client_id": "test-client"}).json()["uploads"]
    assert [u["id"] for u in mine] == [up["id"]]
    many = client.get("/api/uploads", params={"ids": f"{up['id']},nope"}).json()["uploads"]
    assert len(many) == 1
    with conn_() as c:
        assert c.execute("SELECT source FROM uploads").fetchone()[0] == "camera"


def test_upload_protocol_errors(client):
    data = b"x" * 1000
    r = client.post("/api/uploads", json={"filename": "a.bin", "size": len(data)})
    up = r.json()
    r = client.put(f"/api/uploads/{up['id']}/chunk", params={"offset": 10}, content=b"abc",
                   headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 409 and r.json() == {"detail": "offset_mismatch", "received_bytes": 0}
    r = client.post(f"/api/uploads/{up['id']}/complete")
    assert r.status_code == 409
    r = client.delete(f"/api/uploads/{up['id']}")
    assert r.json()["status"] == "cancelled"
    from app.config import settings

    r = client.post("/api/uploads", json={"filename": "huge.mkv", "size": settings.max_upload_bytes + 1})
    assert r.status_code == 413
    assert client.get("/api/uploads/doesnotexist").status_code == 404


def test_exact_duplicates(client):
    data = jpeg_bytes(noise_image(300, 300, seed=2))
    first = add(client, data, "one.jpg")
    up = upload_bytes(client, data, "again.jpg", send_hash=True)
    assert up["status"] == "duplicate" and up["item_uid"] == first["uid"] and up["received_bytes"] == 0
    up = upload_bytes(client, data, "third.jpg")
    assert up["status"] == "duplicate" and up["item_uid"] == first["uid"]
    assert client.get("/api/items").json()["total"] == 1


def test_simple_upload_and_documents(client):
    from app import storage

    pdf = pdf_bytes([["Aadhaar card"], ["Second page"], ["Third"]])
    r = client.post("/api/uploads/simple", files={"file": ("id card.pdf", pdf, "application/pdf")},
                    data={"source": "share"})
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "queued"
    run_jobs()
    uid = upload_status(client, r.json()["id"])["item_uid"]
    d = client.get(f"/api/items/{uid}").json()
    assert (d["kind"], d["page_count"], d["category"]) == ("pdf", 3, "documents")
    assert len(d["urls"]["pages"]) == 3
    pages = client.get(f"/api/items/{uid}/pages").json()
    assert [p["page_no"] for p in pages] == [1, 2, 3]
    # Every page is rendered at ingest; a missing render is made again on first request.
    assert storage.page_path(uid, 1, 3).is_file()
    storage.page_path(uid, 1, 2).unlink()
    r = client.get(pages[1]["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert storage.page_path(uid, 1, 2).is_file()
    assert client.get(f"/files/page/{uid}/4").status_code == 404
    r = client.get(f"/files/original/{uid}")
    assert r.headers["content-security-policy"] == "default-src 'none'; sandbox allow-same-origin"
    assert r.headers["x-frame-options"] == "SAMEORIGIN"
    assert r.headers["content-disposition"].startswith("inline")
    assert client.get(f"/api/items/{uid}/text").status_code in (200, 404)
    big = b"x" * 10
    r = client.post("/api/uploads/simple", files={"file": ("e.txt", b"", "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/uploads/simple", files={"file": ("a.txt", big, "text/plain")}, data={"source": "nope"})
    assert r.status_code == 422


def test_unknown_file_kept_and_served_as_download(client):
    data = bytes(range(256)) * 20
    d = add(client, data, "firmware.bin")
    assert d["kind"] == "other" and d["mime"] == "application/octet-stream"
    r = client.get(d["urls"]["original"])
    assert r.content == data and r.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert client.get(d["urls"]["thumb"]).status_code == 404


def test_video_files_and_frames(client, tmp_path):
    path = make_video(tmp_path / "clip.mp4", seconds=2)
    d = add(client, path.read_bytes(), "clip.mp4")
    assert d["kind"] == "video" and d["urls"]["video"]
    r = client.get(d["urls"]["video"])
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    r = client.get(f"/files/frame/{d['uid']}", params={"t": 1.0, "download": 1})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "clip-0m01.0s.jpg" in r.headers["content-disposition"]
    assert Image.open(io.BytesIO(r.content)).size == (640, 360)
    r = client.get(f"/files/frame/{d['uid']}", params={"t": 0.5, "size": "preview"})
    assert r.status_code == 200
    assert client.get(f"/files/frame/{d['uid']}", params={"t": "nan"}).status_code == 422


# ------------------------------------------------------------------ near duplicates
def test_near_duplicates_collapse_and_best(client, fixture_file):
    src = fixture_file("armstrong.jpg")
    img = Image.open(src).convert("RGB")
    best = add(client, src.read_bytes(), "armstrong.jpg")
    copy = add(client, jpeg_bytes(img.resize((396, 495)), quality=60), "small.jpg")
    other = add(client, fixture_file("aldrin.jpg").read_bytes(), "aldrin.jpg")
    assert copy["dup_group"] == best["uid"]
    page = client.get("/api/items").json()
    assert page["total"] == 2
    cards = {c["uid"]: c for c in page["items"]}
    assert set(cards) == {best["uid"], other["uid"]}
    assert cards[best["uid"]]["dup_count"] == 1
    assert client.get("/api/items", params={"dups": "all"}).json()["total"] == 3
    groups = client.get("/api/duplicates").json()
    assert groups["total"] == 1
    assert groups["groups"][0]["best"]["uid"] == best["uid"]
    assert [c["uid"] for c in groups["groups"][0]["copies"]] == [copy["uid"]]

    assert client.post(f"/api/duplicates/{copy['uid']}/best").status_code == 200
    uids = {c["uid"] for c in client.get("/api/items").json()["items"]}
    assert uids == {copy["uid"], other["uid"]}
    assert client.post(f"/api/duplicates/{other['uid']}/best").status_code == 409

    # Trashing the best shows the remaining copy.
    assert client.delete(f"/api/items/{copy['uid']}").status_code == 200
    uids = {c["uid"] for c in client.get("/api/items").json()["items"]}
    assert uids == {best["uid"], other["uid"]}
    assert client.get("/api/duplicates").json()["total"] == 0


# ------------------------------------------------------------------ trash
def test_trash_restore_and_purge(client):
    from app import storage

    d = add(client, jpeg_bytes(noise_image(320, 240, seed=5)), "t.jpg")
    uid = d["uid"]
    assert client.delete(f"/api/trash/{uid}").status_code == 409  # not in the trash yet
    assert client.delete(f"/api/items/{uid}").status_code == 200
    assert client.get("/api/items").json()["total"] == 0
    trash = client.get("/api/trash").json()
    assert [c["uid"] for c in trash["items"]] == [uid]
    detail = client.get(f"/api/items/{uid}").json()
    assert detail["deleted_at"]
    assert client.get(f"/files/thumb/{uid}").status_code == 200  # the trash page shows it
    r = client.post(f"/api/items/{uid}/restore")
    assert r.status_code == 200 and r.json()["uid"] == uid and r.json()["deleted_at"] is None
    assert client.get("/api/items").json()["total"] == 1
    client.delete(f"/api/items/{uid}")
    assert client.delete(f"/api/trash/{uid}").status_code == 200
    assert client.get(f"/api/items/{uid}").status_code == 404
    assert not storage.original_path(uid, 1, ".jpg").exists()
    assert not storage.thumb_path(uid, 1).exists()

    a = add(client, jpeg_bytes(noise_image(100, 100, seed=6)), "a.jpg")
    b = add(client, jpeg_bytes(noise_image(100, 100, seed=7)), "b.jpg")
    client.post("/api/items/bulk", json={"uids": [a["uid"], b["uid"]], "action": "delete"})
    r = client.post("/api/trash/empty")
    assert r.json() == {"updated": 2}
    assert client.get("/api/trash").json()["total"] == 0


# ------------------------------------------------------------------ bulk and patch
def test_bulk_actions(client):
    c = conn_()
    try:
        items = [make_item(c, taken_at=f"2024-01-0{i}T10:00:00.000Z") for i in range(1, 5)]
        flagged = make_item(c, misclick=1, misclick_reasons='{"blurry": 12.5}')
    finally:
        c.close()
    uids = [i["uid"] for i in items]

    def bulk(action, which, value=None):
        r = client.post("/api/items/bulk", json={"uids": which, "action": action, "value": value})
        assert r.status_code == 200, r.text
        return r.json()["updated"]

    assert bulk("favorite", uids[:2]) == 2
    assert client.get("/api/items", params={"favorite": True}).json()["total"] == 2
    assert bulk("unfavorite", uids[:1]) == 1
    assert bulk("category", uids[:3], "food") == 3
    page = client.get("/api/items", params={"category": "food"}).json()
    assert page["total"] == 3
    assert client.get(f"/api/items/{uids[0]}").json()["category_source"] == "user"
    assert client.post("/api/items/bulk", json={"uids": uids, "action": "category", "value": "bogus"}).status_code == 422
    assert bulk("keep", [flagged["uid"]]) == 1
    d = client.get(f"/api/items/{flagged['uid']}").json()
    assert d["misclick"] is False and d["misclick_reasons"] is None
    assert bulk("delete", uids[:2]) == 2
    assert bulk("purge", uids[1:3]) == 1  # only the trashed one
    assert bulk("restore", uids) == 1
    assert bulk("nsfw", [uids[3]], "true") == 1
    assert client.get(f"/api/items/{uids[3]}").status_code == 403
    # NSFW items are skipped while the vault is locked.
    assert bulk("favorite", [uids[3]]) == 0
    assert bulk("purge", ["nonexistent"]) == 0


def test_patch_item(client):
    c = conn_()
    try:
        row = make_item(c, category="food", category_scores='{"food": 0.7, "places": 0.2, "nsfw": 0.9, "skipped": "x"}',
                        misclick=1, misclick_reasons='{"dark": true}')
    finally:
        c.close()
    uid = row["uid"]
    r = client.patch(f"/api/items/{uid}", json={"nsfw": True})
    assert r.status_code == 200 and r.json()["nsfw"] is True and r.json()["category"] == "nsfw"
    assert client.get(f"/api/items/{uid}").status_code == 403
    assert client.patch(f"/api/items/{uid}", json={"nsfw": False}).status_code == 403  # vault locked
    with conn_() as c:
        c.execute("UPDATE items SET nsfw = 0")
        c.commit()
    r = client.patch(f"/api/items/{uid}", json={"misclick": False, "favorite": True, "filename": "renamed.jpg",
                                                "category": "places"})
    d = r.json()
    assert d["misclick"] is False and d["misclick_reasons"] is None and d["favorite"] is True
    assert d["filename"] == "renamed.jpg" and d["category"] == "places" and d["category_source"] == "user"
    assert client.patch(f"/api/items/{uid}", json={"category": "nope"}).status_code == 422
    assert client.patch("/api/items/missing", json={"favorite": True}).status_code == 404


def test_patch_nsfw_round_trip_restores_category(vault_client):
    c = conn_()
    try:
        row = make_item(c, category="pets", category_scores='{"pets": 0.5, "food": 0.3, "nsfw": 0.1}')
    finally:
        c.close()
    uid = row["uid"]
    d = vault_client.patch(f"/api/items/{uid}", json={"nsfw": True}).json()
    assert d["category"] == "nsfw"
    assert vault_client.get("/api/items").json()["total"] == 0
    assert vault_client.get("/api/items", params={"vault": 1}).json()["total"] == 1
    d = vault_client.patch(f"/api/items/{uid}", json={"nsfw": False}).json()
    assert d["nsfw"] is False and d["category"] == "pets"


# ------------------------------------------------------------------ listing
def test_items_filters_and_cursors(client):
    c = conn_()
    try:
        made = []
        for day in range(1, 8):
            made.append(make_item(c, taken_at=f"2024-03-0{day}T06:00:00.000Z", category="places" if day % 2 else "food"))
        video = make_item(c, kind="video", ext=".mp4", mime="video/mp4", taken_at="2024-03-10T06:00:00.000Z")
        untaken = make_item(c)
        flagged = make_item(c, misclick=1, taken_at="2024-03-04T07:00:00.000Z")
        trashed = make_item(c, deleted_at="2024-03-05T00:00:00.000Z")
        hidden = make_item(c, nsfw=1)
        named = make_person(c, "Meera")
        make_face(c, made[0]["id"], named)
        make_face(c, made[2]["id"], named)
    finally:
        c.close()

    page = client.get("/api/items").json()
    uids = [i["uid"] for i in page["items"]]
    assert page["total"] == 9
    assert flagged["uid"] not in uids and trashed["uid"] not in uids and hidden["uid"] not in uids
    # sort=taken: newest first, items without a capture time by their upload time (now).
    assert uids[0] == untaken["uid"] and uids[1] == video["uid"] and uids[-1] == made[0]["uid"]

    seen, cursor = [], None
    while True:
        r = client.get("/api/items", params={"limit": 2, **({"cursor": cursor} if cursor else {})}).json()
        seen += [i["uid"] for i in r["items"]]
        cursor = r["next_cursor"]
        if not cursor:
            break
    assert seen == uids

    added = [i["uid"] for i in client.get("/api/items", params={"sort": "added"}).json()["items"]]
    assert added[0] == untaken["uid"] and set(added) == set(uids)
    assert client.get("/api/items", params={"kind": "video"}).json()["total"] == 1
    assert client.get("/api/items", params={"category": "food"}).json()["total"] == 3
    assert client.get("/api/items", params={"person": named}).json()["total"] == 2
    r = client.get("/api/items", params={"from": "2024-03-02", "to": "2024-03-04"}).json()
    assert {i["uid"] for i in r["items"]} == {m["uid"] for m in made[1:4]}
    # Local dates: 2024-03-04T20:00Z is already 5 March in Asia/Kolkata.
    assert client.get("/api/items", params={"misclicks": "only"}).json()["total"] == 1
    assert client.get("/api/items", params={"misclicks": "include"}).json()["total"] == 10
    assert client.get("/api/review/misclicks").json()["items"][0]["uid"] == flagged["uid"]
    assert client.get("/api/items", params={"cursor": "garbage!"}).status_code == 400
    assert client.get("/api/items", params={"from": "03/02/2024"}).status_code == 422
    assert client.get("/api/items", params={"vault": 1}).status_code == 403
    assert client.get("/api/items", params={"kind": "hologram"}).status_code == 422


def test_local_day_bounds(data_dir):
    from app.api.library import local_day_start_utc

    assert local_day_start_utc("2024-03-05") == "2024-03-04T18:30:00.000Z"
    assert local_day_start_utc("2024-03-05", next_day=True) == "2024-03-05T18:30:00.000Z"


# ------------------------------------------------------------------ vault and files
def test_nsfw_files_need_the_vault(client):
    c = conn_()
    try:
        row = make_item(c, nsfw=1)
        face = make_face(c, row["id"])
    finally:
        c.close()
    uid = row["uid"]
    for path in (f"/files/thumb/{uid}", f"/files/preview/{uid}", f"/files/preview-small/{uid}",
                 f"/files/original/{uid}", f"/files/page/{uid}/1", f"/files/doc-photo/{uid}", f"/files/face/{face}"):
        r = client.get(path)
        assert r.status_code == 403 and r.json() == {"detail": "vault_locked"}, path
    assert client.get(f"/api/items/{uid}").status_code == 403
    assert client.delete(f"/api/items/{uid}").status_code == 403


def test_nsfw_files_with_vault_open(vault_client):
    c = conn_()
    try:
        row = make_item(c, nsfw=1)
        face = make_face(c, row["id"])
    finally:
        c.close()
    r = vault_client.get(f"/files/thumb/{row['uid']}", params={"v": 1})
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    assert vault_client.get(f"/files/face/{face}").headers["cache-control"] == "private, no-store"
    assert vault_client.get(f"/api/items/{row['uid']}").json()["nsfw"] is True


def test_crops_and_image_page(client):
    c = conn_()
    try:
        row = make_item(c)
        face = make_face(c, row["id"])
    finally:
        c.close()
    r = client.get(f"/files/face/{face}")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, max-age=86400"
    assert client.get("/files/face/999").status_code == 404
    assert client.get("/files/body/1").status_code == 404
    assert client.get(f"/files/page/{row['uid']}/1").status_code == 200  # an image's page 1 is its preview
    assert client.get(f"/files/page/{row['uid']}/2").status_code == 404
    detail = client.get(f"/api/items/{row['uid']}").json()
    assert len(detail["faces"]) == 1 and detail["faces"][0]["url"].split("?")[0] == f"/files/face/{face}"


# ------------------------------------------------------------------ stats, config, system, reprocess
def test_stats_and_config(client):
    c = conn_()
    try:
        make_item(c, category="food")
        make_item(c, kind="pdf", ext=".pdf", mime="application/pdf", category="documents")
        make_item(c, misclick=1)
        make_item(c, nsfw=1)
        make_item(c, deleted_at="2024-01-01T00:00:00.000Z")
        make_person(c, "A")
        make_person(c)
    finally:
        c.close()
    s = client.get("/api/stats").json()
    assert s["items"] == 3 and s["trash"] == 1 and s["nsfw"] == 1 and s["misclicks"] == 1
    assert s["by_kind"]["image"] == 2 and s["by_kind"]["pdf"] == 1
    assert s["by_category"]["food"] == 1 and "nsfw" not in s["by_category"]
    assert s["people"] == 2 and s["named_people"] == 1
    assert len(s["recent"]) == 2  # mis-clicks, NSFW and trashed items are not shown
    assert s["disk"]["free_bytes"] > 0 and s["queue"]["failed"] == 0
    cfg = client.get("/api/config").json()
    assert "application/pdf" in cfg["accept"] and ".docx" in cfg["accept"] and ".cr3" in cfg["accept"]
    assert "nsfw" in cfg["categories"] and cfg["trash_days"] == 30


@pytest.fixture()
def no_gpu(monkeypatch):
    """Keep the system test off the GPU (the first gpu_info() call starts CUDA)."""
    from app.config import settings

    monkeypatch.setattr(settings, "gpu_enabled", "false")
    try:
        from app.ml import runtime
    except ImportError:
        yield
        return
    reset = getattr(runtime, "reset", None)
    if reset:
        reset()
    yield
    if reset:
        reset()


def test_system_and_housekeeping(client, no_gpu):
    r = client.get("/api/system")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] and "available" in body["gpu"] and isinstance(body["models"], list)
    assert body["ffmpeg"] is True and body["workers"] is False
    assert "configured" in body["wallet"]
    assert client.post("/api/system/housekeeping").json() == {"ok": True}


def test_reprocess_endpoint(client):
    d = add(client, jpeg_bytes(noise_image(200, 150, seed=11)), "r.jpg")
    r = client.post(f"/api/items/{d['uid']}/reprocess", json={"stages": ["analyze"]})
    assert r.status_code == 200
    with conn_() as c:
        jobs = [row[0] for row in c.execute("SELECT type FROM jobs WHERE status = 'queued'")]
    assert jobs == ["analyze"]
    assert client.post(f"/api/items/{d['uid']}/reprocess", json={"stages": ["bogus"]}).status_code == 422
