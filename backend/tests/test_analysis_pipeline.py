"""The analyze job without models (every model reported unavailable): quality, categories, bursts,
overrides and idempotency. The model-backed end-to-end run is in test_analysis_models.py."""
from __future__ import annotations

import json

import pytest
from PIL import Image, ImageFilter

from app import storage
from app.analysis import bodies, pipeline
from app.db import tx
from tests.helpers import flat_image, jpeg_bytes, make_item, noise_image, png_bytes, text_image


@pytest.fixture()
def no_models(monkeypatch):
    monkeypatch.setattr(pipeline, "_model", lambda name: None)
    monkeypatch.setattr(bodies, "available", lambda: False)


def _item(conn, img: Image.Image, **kw):
    data = kw.pop("data", None) or jpeg_bytes(img)
    return make_item(conn, data=data, width=img.width, height=img.height, analysis_status="pending", **kw)


def _row(conn, item_id: int):
    return conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()


def test_plain_photo_without_models(conn, no_models):
    item = _item(conn, noise_image(800, 600, seed=1))
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["analysis_status"] == "done"
    q = json.loads(row["quality_scores"])
    assert q["blur_var"] > 100 and {"faces", "siglip", "nsfw", "bodies"} <= set(q["skipped"])
    assert set(json.loads(row["category_scores"])["skipped"]) == set(q["skipped"])
    assert row["category"] == "misc" and row["misclick"] == 0 and row["nsfw"] == 0
    assert row["face_count"] == 0 and row["body_count"] == 0
    assert conn.execute("SELECT COUNT(*) FROM item_embeddings").fetchone()[0] == 0


def test_dark_and_blurry_photos_are_flagged(conn, no_models):
    dark = _item(conn, flat_image(640, 480, (8, 8, 10)))
    blurry = _item(conn, noise_image(640, 480, seed=2).filter(ImageFilter.GaussianBlur(8)))
    for it in (dark, blurry):
        pipeline.analyze_item(conn, it["id"], 1)
    d, b = _row(conn, dark["id"]), _row(conn, blurry["id"])
    assert d["misclick"] == 1 and "dark" in json.loads(d["misclick_reasons"])
    assert b["misclick"] == 1 and "blurry" in json.loads(b["misclick_reasons"])


def test_text_photo_is_a_document(conn, no_models):
    page = text_image([f"Invoice line {i}: item and amount 1{i}0.00" for i in range(14)], w=1000, h=1200)
    item = _item(conn, page)
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["doc_likeness"] >= 0.55
    assert row["category"] == "documents" and row["misclick"] == 0


def test_screenshot_by_size(conn, no_models):
    img = noise_image(1170, 2532, seed=3)
    item = _item(conn, img, data=png_bytes(img), ext=".png", mime="image/png")
    pipeline.analyze_item(conn, item["id"], 1)
    assert _row(conn, item["id"])["category"] == "screenshots"


def test_pdf_uses_page_one(conn, no_models):
    item = make_item(conn, kind="pdf", ext=".pdf", mime="application/pdf", width=850, height=1100,
                     analysis_status="pending", page_count=1)
    page = storage.page_path(item["uid"], 1, 1)
    page.parent.mkdir(parents=True, exist_ok=True)
    text_image(["Page one"] * 10, w=850, h=1100).save(page, "WEBP")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["analysis_status"] == "done" and row["category"] == "documents" and row["misclick"] == 0
    assert "image" not in json.loads(row["quality_scores"] or "{}").get("skipped", [])


def test_other_kind_completes(conn, no_models):
    item = make_item(conn, kind="other", ext=".bin", mime="application/octet-stream", width=None, height=None,
                     analysis_status="pending")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["analysis_status"] == "done" and row["category"] == "misc"


def test_skips_done_gone_and_outdated(conn, no_models):
    item = _item(conn, flat_image(320, 240, (5, 5, 5)))
    pipeline.analyze_item(conn, item["id"], 2)  # wrong version
    assert _row(conn, item["id"])["analysis_status"] == "pending"
    pipeline.analyze_item(conn, 99999, 1)  # gone
    pipeline.analyze_item(conn, item["id"], 1)
    assert _row(conn, item["id"])["misclick"] == 1
    with tx(conn):
        conn.execute("UPDATE items SET misclick = 0 WHERE id = ?", (item["id"],))
    pipeline.analyze_item(conn, item["id"], 1)  # done already: nothing changes
    assert _row(conn, item["id"])["misclick"] == 0


def test_reprocess_requested_during_a_run_is_not_lost(conn, no_models, monkeypatch):
    from app import jobs

    item = _item(conn, noise_image(800, 600, seed=3))
    real = pipeline._run_models

    def run_and_request_again(*args, **kwargs):
        # Someone clicks reprocess while the models run: another analyze job is queued.
        jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1}, unique=True)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_run_models", run_and_request_again)
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["category"] == "misc" and row["quality_scores"]  # this run's results are visible
    assert row["analysis_status"] == "pending"  # and the queued job will run in full
    monkeypatch.setattr(pipeline, "_run_models", real)
    with tx(conn):  # the worker claims the queued job and runs it
        conn.execute("UPDATE jobs SET status = 'running' WHERE type = 'analyze' AND status = 'queued'")
    pipeline.analyze_item(conn, item["id"], 1)
    assert _row(conn, item["id"])["analysis_status"] == "done"


def test_user_choices_win(conn, no_models):
    item = _item(conn, flat_image(320, 240, (5, 5, 5)), category="food", category_source="user",
                 misclick=0, misclick_source="user", nsfw=1, nsfw_source="user")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["category"] == "food" and row["misclick"] == 0 and row["nsfw"] == 1
    assert row["analysis_status"] == "done"


def test_burst_keeps_the_sharpest(conn, no_models):
    base = noise_image(640, 480, seed=9)
    soft = base.filter(ImageFilter.GaussianBlur(1.2))
    kw = dict(camera="Pixel 8", phash="a5a5a5a5a5a5a5a5")
    first = _item(conn, soft, taken_at="2026-05-01T10:00:00.000Z", **kw)
    pipeline.analyze_item(conn, first["id"], 1)
    r1 = _row(conn, first["id"])
    assert r1["misclick"] == 0
    second = _item(conn, base, taken_at="2026-05-01T10:00:01.000Z", **kw)
    pipeline.analyze_item(conn, second["id"], 1)
    r1, r2 = _row(conn, first["id"]), _row(conn, second["id"])
    assert r2["misclick"] == 0 and "burst_best" in json.loads(r2["misclick_reasons"])
    reasons = json.loads(r1["misclick_reasons"])
    assert r1["misclick"] == 1 and reasons["burst"]["best_uid"] == second["uid"]


def test_mark_failed(conn):
    item = make_item(conn, analysis_status="pending", write_files=False, data=None)
    pipeline.mark_failed(conn, item["id"], 1)
    assert _row(conn, item["id"])["analysis_status"] == "error"


def test_pcc_no_models_turns_every_model_off(monkeypatch):
    monkeypatch.setenv("PCC_NO_MODELS", "1")
    for name in ("siglip", "faces", "nsfw", "persons"):
        assert pipeline._model(name) is None
    assert bodies.available() is False
