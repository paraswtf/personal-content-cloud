"""Shared pytest fixtures. Every test gets a fresh data dir + database; background worker threads
are disabled (run jobs with app.worker.drain()).

Model-backed tests call `models_available(key)` and skip when the model files are missing or when
PCC_NO_MODELS=1 is set (parallel agents must not all load the models at once)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CSRF = {"X-Requested-With": "pcc"}
ACCOUNT = {"username": "owner", "password": "correct horse battery staple"}


def models_available(*keys: str) -> bool:
    """True when every named model (app.ml.models.REGISTRY keys) is on disk and models are allowed."""
    if os.environ.get("PCC_NO_MODELS") == "1":
        return False
    try:
        from app.ml import models
    except Exception:
        return False
    return all(models.present(k) for k in keys)


def skip_without_models(*keys: str) -> None:
    if not models_available(*keys):
        pytest.skip(f"model(s) {', '.join(keys)} not available")


@pytest.fixture()
def data_dir(tmp_path):
    from app.config import reload_settings

    d = tmp_path / "data"
    # frontend_dist points into tmp: the app writes .gz siblings there at startup, never into frontend/
    reload_settings(data_dir=d, frontend_dist=tmp_path / "dist", workers_enabled=False, min_free_disk_bytes=0,
                    chunk_size=256 * 1024, session_days=30)
    from app import db, storage

    storage.ensure_dirs()
    db.init_db()
    try:
        from app.auth.passwords import reset_all

        reset_all()
    except Exception:
        pass
    try:
        from app.analysis.index import reset_index

        reset_index()
    except Exception:
        pass
    try:
        from app.search import service

        service.reset_matrix()
    except Exception:
        pass
    yield d
    reload_settings()


@pytest.fixture()
def conn(data_dir):
    from app.db import connect

    c = connect()
    yield c
    c.close()


@pytest.fixture()
def app(data_dir):
    from app.main import create_app

    return create_app()


@pytest.fixture()
def anon_client(app):
    """No session: for auth tests and 401 checks."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client(app):
    """Logged in through the real setup + login flow; sends the CSRF header on every request."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        r = c.post("/api/auth/setup", json=ACCOUNT, headers=CSRF)
        assert r.status_code in (200, 201), r.text
        c.headers.update(CSRF)
        yield c


@pytest.fixture()
def vault_client(client):
    """Logged in with the NSFW vault unlocked."""
    r = client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]})
    assert r.status_code == 200, r.text
    return client


@pytest.fixture()
def fixture_file():
    def get(name: str) -> Path:
        p = FIXTURES / name
        assert p.exists(), p
        return p

    return get


@pytest.fixture(autouse=True)
def _reset_learning_flag_cache():
    """The learning switch is remembered for 30 s in the process (store._flag); a test that turned it
    off must not reach into the next test's fresh database."""
    from app.learning import store

    store._flag = None
    yield
    store._flag = None
