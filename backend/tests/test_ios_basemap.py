"""The iPhone app draws the same plain basemap as the website: ios/Sources/Features/Map/basemap.json is a copy of
frontend/public/map/basemap.json (Natural Earth land, lakes and borders), bundled so the plain map needs nothing
from the internet. This test fails when the two files differ, so a change to the website's basemap is copied to
the app too (cp frontend/public/map/basemap.json ios/Sources/Features/Map/basemap.json)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEBSITE = REPO / "frontend" / "public" / "map" / "basemap.json"
APP = REPO / "ios" / "Sources" / "Features" / "Map" / "basemap.json"


def test_app_basemap_is_the_websites():
    assert APP.is_file(), f"{APP.relative_to(REPO)} is missing: copy {WEBSITE.relative_to(REPO)} there"
    assert APP.read_bytes() == WEBSITE.read_bytes(), (
        f"{APP.relative_to(REPO)} differs from {WEBSITE.relative_to(REPO)}: copy the website's file over the app's"
    )


def test_basemap_has_the_kinds_the_app_draws():
    data = json.loads(APP.read_text(encoding="utf-8"))
    kinds = {f["properties"].get("kind") for f in data["features"]}
    assert {"land", "lake", "boundary"} <= kinds
