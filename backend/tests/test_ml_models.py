"""app.ml.models: registry pins, presence, statuses and the downloader (file:// sources, no network)."""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pytest

from tests.conftest import skip_without_models


@pytest.fixture()
def models_mod():
    from app.ml import models

    return models


@pytest.fixture(autouse=True)
def _restore_download_state():
    """Download tests leave errors in module state; restore it so other areas' tests see a clean state."""
    from app.ml import models

    with models._state_lock:
        state, errors = dict(models._state), dict(models._errors)
    yield
    with models._state_lock:
        models._state.clear()
        models._state.update(state)
        models._errors.clear()
        models._errors.update(errors)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def fake_registry(tmp_path, monkeypatch, models_mod):
    """A registry with two tiny models served from file:// URLs, models_dir in tmp."""
    from app.config import reload_settings

    src = tmp_path / "src"
    src.mkdir()
    a = b"alpha model bytes" * 10
    b = b"beta config"
    (src / "a.onnx").write_bytes(a)
    (src / "b.json").write_bytes(b)
    zbytes = b"zipped member" * 7
    with zipfile.ZipFile(src / "pack.zip", "w") as zf:
        zf.writestr("pack/m.onnx", zbytes)
        zf.writestr("pack/other.txt", b"ignored")
    M = models_mod
    reg = {
        "one": M.ModelSpec(key="one", label="One", subdir="one", source="test", license="test", files=(
            M.ModelFile("a.onnx", len(a), _sha(a), url=(src / "a.onnx").as_uri()),
            M.ModelFile("sub/b.json", len(b), _sha(b), url=(src / "b.json").as_uri()),
        )),
        "zip": M.ModelSpec(key="zip", label="Zip", subdir="zipped", source="test", license="test",
                           archive_url=(src / "pack.zip").as_uri(), archive_size_estimate=1000, files=(
                               M.ModelFile("m.onnx", len(zbytes), _sha(zbytes), member="m.onnx"),
                           )),
    }
    monkeypatch.setattr(M, "REGISTRY", reg)
    reload_settings(models_dir=tmp_path / "models", min_free_disk_bytes=0)
    yield reg
    reload_settings()


def test_registry_keys_and_pins(models_mod):
    assert set(models_mod.REGISTRY) == {"faces", "siglip", "nsfw", "persons", "ocr"}
    for key, spec in models_mod.REGISTRY.items():
        assert spec.key == key and spec.label and spec.files
        for f in spec.files:
            assert f.size > 0 and len(f.sha256) == 64
            if spec.package is None and spec.archive_url is None:
                assert f.url and f.url.startswith("https://"), (key, f.rel)
    assert models_mod.REGISTRY["faces"].archive_url.endswith("/v0.7/buffalo_l.zip")
    assert "c3b037a9ef99ed5f116440790f1e098fb388db49" in models_mod.REGISTRY["siglip"].files[0].url
    assert "1ceb3c7fe1e9f3f2507e6df577437f23a9149fd5" in models_mod.REGISTRY["nsfw"].files[0].url


def test_models_module_does_not_import_onnxruntime():
    import subprocess
    import sys

    code = "import sys; import app.ml.models; print('onnxruntime' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]), check=True)
    assert out.stdout.strip() == "False"


def test_statuses_shape(models_mod):
    rows = models_mod.statuses()
    assert [r["key"] for r in rows] == list(models_mod.REGISTRY)
    for r in rows:
        assert set(r) == {"key", "label", "present", "loaded", "provider", "size_bytes", "downloading", "error"}
        assert isinstance(r["present"], bool) and r["size_bytes"] > 0
    st = models_mod.download_state()
    assert set(st) == {"running", "progress", "error"}


@pytest.mark.parametrize("key", ["faces", "siglip", "nsfw", "persons", "ocr"])
def test_present_for_every_key(models_mod, key):
    skip_without_models(key)
    assert models_mod.present(key)
    assert models_mod.missing_files(key) == []
    for f in models_mod.REGISTRY[key].files:
        assert models_mod.path(key, f.rel).is_file()


def test_ocr_bundled_files_match_pins(models_mod):
    skip_without_models("ocr")
    assert models_mod.verify("ocr") == []


def test_download_is_noop_when_present(models_mod, monkeypatch):
    skip_without_models(*models_mod.REGISTRY)

    def no_network(*a, **kw):
        raise AssertionError("network used")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    seen: list[str] = []
    models_mod.download(progress=seen.append)
    assert seen == []
    assert models_mod.download_state()["running"] is False


def test_download_files_and_archive(models_mod, fake_registry):
    M = models_mod
    assert not M.present("one") and not M.present("zip")
    msgs: list[str] = []
    M.download(progress=msgs.append)
    assert M.present("one") and M.present("zip")
    assert M.verify("one") == [] and M.verify("zip") == []
    assert M.path("one", "sub/b.json").read_bytes() == b"beta config"
    d = M.model_dir("zip")
    assert sorted(p.name for p in d.iterdir()) == ["m.onnx"]  # zip and .part files removed
    assert not list(M.model_dir("one").rglob("*.part"))
    assert any("one: ready" in m for m in msgs)
    st = M.download_state()
    assert st["running"] is False and st["error"] is None
    # second run: nothing to do
    msgs.clear()
    M.download(progress=msgs.append)
    assert msgs == []


def test_download_checksum_mismatch_reports_model(models_mod, fake_registry, monkeypatch):
    M = models_mod
    one = fake_registry["one"]
    bad = M.ModelSpec(key="one", label="One", subdir="one", source="t", license="t", files=(
        M.ModelFile("a.onnx", one.files[0].size, "0" * 64, url=one.files[0].url),
    ))
    monkeypatch.setitem(M.REGISTRY, "one", bad)
    with pytest.raises(M.DownloadError) as ei:
        M.download()
    assert set(ei.value.failed) == {"one"}
    assert "sha256 mismatch" in ei.value.failed["one"]
    assert M.present("zip")  # the other model still downloaded
    assert not M.model_dir("one").joinpath("a.onnx").exists()
    assert not list(M.model_dir("one").rglob("*.part"))
    row = {r["key"]: r for r in M.statuses()}["one"]
    assert row["present"] is False and "sha256" in row["error"]
    assert "one" in M.download_state()["error"]


def test_download_refuses_without_free_disk(models_mod, fake_registry, monkeypatch):
    M = models_mod
    monkeypatch.setattr(M, "_free_bytes", lambda p: 10)
    with pytest.raises(M.DownloadError) as ei:
        M.download(["one"])
    assert "free disk" in ei.value.failed["one"]


def test_download_unknown_key(models_mod):
    with pytest.raises(KeyError):
        models_mod.download(["nope"])


def test_download_script_noop_and_failure(models_mod, fake_registry, tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "download_models.py"
    spec = importlib.util.spec_from_file_location("download_models_script", script)
    mod = importlib.util.module_from_spec(spec)
    import sys

    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep scripts/ free of __pycache__
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    assert mod.main(["--quiet"]) == 0
    assert models_mod.present("one") and models_mod.present("zip")
    assert mod.main(["--quiet", "--verify"]) == 0
    assert mod.main(["--only", "nope"]) == 2
    models_mod.path("one", "a.onnx").unlink()
    models_mod.REGISTRY["one"] = models_mod.ModelSpec(
        key="one", label="One", subdir="one", source="t", license="t",
        files=(models_mod.ModelFile("a.onnx", 5, "0" * 64, url=(tmp_path / "missing.bin").as_uri()),))
    assert mod.main(["--quiet"]) == 1
