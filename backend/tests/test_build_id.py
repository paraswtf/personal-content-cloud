"""API build id (app/build_id.py): stable for the same files, sensitive to them, reported by
GET /api/auth/state, and the frontend's Node implementation (frontend/scripts/api-build.mjs) agrees."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app import build_id

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "frontend" / "scripts" / "api-build.mjs"


def _fake_app(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    (app / "api").mkdir(parents=True)
    (app / "schemas.py").write_text("class A:\n    pass\n")
    (app / "urls.py").write_text("x = 1\n")
    (app / "api" / "b.py").write_text("b = 2\n")
    (app / "api" / "a.py").write_text("a = 1\n")
    (app / "api" / "notes.txt").write_text("ignored\n")
    (app / "other.py").write_text("ignored too\n")
    return app


def test_compute_is_stable_and_sensitive(tmp_path):
    app = _fake_app(tmp_path)
    first = build_id.compute(app)
    assert len(first) == 12 and first == build_id.compute(app)
    assert [p.name for p in build_id.api_files(app)] == ["a.py", "b.py", "schemas.py", "urls.py"]
    (app / "api" / "a.py").write_text("a = 2\n")
    assert build_id.compute(app) != first
    (app / "api" / "a.py").write_text("a = 1\n")
    assert build_id.compute(app) == first
    (app / "api" / "notes.txt").write_text("changed\n")
    (app / "other.py").write_text("changed\n")
    assert build_id.compute(app) == first
    (app / "api" / "c.py").write_text("")
    assert build_id.compute(app) != first


def test_module_value_describes_the_real_app():
    assert build_id.API_BUILD == build_id.compute()
    assert len(build_id.API_BUILD) == 12 and build_id.API_BUILD.isalnum()


def test_state_reports_the_build(anon_client):
    assert anon_client.get("/api/auth/state").json()["api_build"] == build_id.API_BUILD


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_frontend_script_agrees(tmp_path):
    app = _fake_app(tmp_path)
    out = subprocess.run(["node", str(SCRIPT), str(app)], capture_output=True, text=True, check=True, timeout=60)
    assert out.stdout.strip() == build_id.compute(app)
    real = subprocess.run(["node", str(SCRIPT)], capture_output=True, text=True, check=True, timeout=60)
    assert real.stdout.strip() == build_id.compute()
