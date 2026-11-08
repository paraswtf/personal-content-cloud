"""The served site is a copy of the build taken at startup (app.main.snapshot_site): a later build
does not change what a running server serves, and a missing build serves the hint."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _build(dist: Path, marker: str) -> None:
    (dist / "assets").mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text(f"<!doctype html><title>{marker}</title>")
    (dist / "assets" / "app.js").write_text(f"console.log('{marker}')")


def test_snapshot_serves_the_build_as_it_was_at_startup(data_dir, tmp_path, monkeypatch):
    from app import main
    from app.config import settings

    dist = tmp_path / "dist"
    _build(dist, "first")
    monkeypatch.setattr(settings, "frontend_dist", dist)
    with TestClient(main.create_app()) as c:
        assert main.site_dir() == settings.data_dir / "site"
        assert "first" in c.get("/").text
        assert "first" in c.get("/assets/app.js").text
        _build(dist, "second")
        (dist / "assets" / "app.js").unlink()
        assert "first" in c.get("/").text, "a rebuilt dist must not swap the site under a running server"
        assert c.get("/assets/app.js").status_code == 200
    # a new start takes the new build
    with TestClient(main.create_app()) as c:
        assert "second" in c.get("/").text
        assert c.get("/assets/app.js").status_code == 404


def test_missing_build_serves_the_hint(data_dir, tmp_path, monkeypatch):
    from app import main
    from app.config import settings

    monkeypatch.setattr(settings, "frontend_dist", tmp_path / "nope")
    with TestClient(main.create_app()) as c:
        assert main.site_dir() == tmp_path / "nope"
        r = c.get("/")
        assert r.status_code == 200 and "build" in r.text.lower()
