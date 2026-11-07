"""End-to-end analysis with the real models (faces, SigLIP, NSFW, persons). Skipped when the models are
missing or PCC_NO_MODELS=1. Run alone: one model-backed pytest process at a time, ORT_THREADS=2."""
from __future__ import annotations

import json

from PIL import Image

from app.analysis import pipeline
from tests.conftest import skip_without_models
from tests.helpers import make_item


def _item(conn, fixture_file, name: str, taken_at: str):
    path = fixture_file(name)
    with Image.open(path) as im:
        w, h = im.size
    return make_item(conn, data=path.read_bytes(), width=w, height=h, taken_at=taken_at, filename=name,
                     analysis_status="pending")


def test_obama_photos_cluster_into_one_person(conn, fixture_file):
    skip_without_models("faces", "siglip", "nsfw")
    a = _item(conn, fixture_file, "obama_2009.jpg", "2009-01-13T10:00:00.000Z")
    b = _item(conn, fixture_file, "obama_2012.jpg", "2012-12-05T10:00:00.000Z")
    for it in (a, b):
        pipeline.analyze_item(conn, it["id"], 1)
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM items")}
    for it in (a, b):
        row = rows[it["id"]]
        assert row["analysis_status"] == "done"
        assert row["category"] == "personal", (row["category"], row["category_scores"])
        assert row["nsfw"] == 0 and row["misclick"] == 0
        assert row["face_count"] >= 1
        q = json.loads(row["quality_scores"])
        assert "faces" not in q.get("skipped", []) and "siglip" not in q.get("skipped", [])
    people = conn.execute(
        "SELECT item_id, person_id FROM faces WHERE det_score >= 0.7 ORDER BY item_id").fetchall()
    persons = {r["person_id"] for r in people}
    assert len(persons) == 1 and None not in persons
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 1
    # Embeddings, labels and torso vectors were stored.
    assert conn.execute("SELECT COUNT(*) FROM item_embeddings WHERE frame_index = 0").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM item_labels").fetchone()[0] >= 2
    assert conn.execute("SELECT COUNT(*) FROM faces WHERE torso_embedding IS NOT NULL").fetchone()[0] >= 2
