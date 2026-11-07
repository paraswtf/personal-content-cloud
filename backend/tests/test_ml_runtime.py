"""app.ml.runtime: CUDA on this machine (skip when unavailable), CPU when GPU_ENABLED=false."""
from __future__ import annotations

import shutil

import pytest

from tests.conftest import skip_without_models


def _tiny_model():
    from app.ml import models

    return models.path("ocr", "ch_ppocr_mobile_v2.0_cls_mobile.onnx")


def test_cuda_provider_on_this_machine():
    skip_without_models("ocr")
    import onnxruntime as ort

    from app.ml import runtime

    if "CUDAExecutionProvider" not in ort.get_available_providers() or not shutil.which("nvidia-smi"):
        pytest.skip("no CUDA provider or GPU here")
    if not runtime.cuda_usable():
        pytest.skip(f"CUDA listed but not usable: {runtime.status()['error']}")
    assert runtime.providers()[0] == "CUDAExecutionProvider"
    sess = runtime.session(_tiny_model(), threads=1)
    assert runtime.session_provider(sess) == "CUDAExecutionProvider"
    info = runtime.gpu_info()
    assert info["available"] is True and info["provider"] == "CUDAExecutionProvider"
    assert info["name"]


def test_status_shape():
    skip_without_models("ocr")  # status() verifies CUDA once (a CUDA context): not under PCC_NO_MODELS
    from app.ml import runtime

    st = runtime.status()
    for k in ("available", "name", "provider", "setting", "verified", "error"):
        assert k in st
    assert st["provider"] in ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_gpu_disabled_gives_cpu_sessions():
    skip_without_models("ocr")
    from app.config import reload_settings
    from app.ml import runtime

    try:
        reload_settings(gpu_enabled="false")
        runtime.reset()
        assert runtime.providers() == ["CPUExecutionProvider"]
        assert runtime.gpu_info()["available"] is False
        sess = runtime.session(_tiny_model(), threads=1)
        assert sess.get_providers() == ["CPUExecutionProvider"]
    finally:
        reload_settings()
        runtime.reset()


def test_missing_model_file(tmp_path):
    from app.ml import runtime

    with pytest.raises(FileNotFoundError):
        runtime.session(tmp_path / "nope.onnx")


class _FakeSession:
    def __init__(self, providers):
        self.calls = []
        self._providers = providers

    def get_providers(self):
        return self._providers

    def get_inputs(self):
        return ["x"]

    def run(self, out, feed, run_options=None):
        self.calls.append(run_options)
        return [feed["x"]]


def test_cuda_sessions_free_their_arena_after_each_run(monkeypatch):
    import onnxruntime as ort

    from app.ml import runtime

    monkeypatch.setattr(runtime.settings, "gpu_arena_shrink", True)
    fake = _FakeSession([runtime.CUDA, runtime.CPU])
    wrapped = runtime.wrap_session(fake)
    assert wrapped is not fake and wrapped.shrinks_arena
    assert wrapped.run(None, {"x": 1}) == [1]
    ro = fake.calls[0]
    assert isinstance(ro, ort.RunOptions)  # the shrink option travels with every run
    assert ro.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:0"
    assert wrapped.get_inputs() == ["x"] and wrapped.get_providers()[0] == runtime.CUDA
    assert runtime.wrap_session(wrapped) is wrapped  # idempotent
    own = ort.RunOptions()
    wrapped.run(None, {"x": 2}, own)
    assert fake.calls[1] is own  # a caller's own options are kept

    cpu = _FakeSession([runtime.CPU])
    assert runtime.wrap_session(cpu) is cpu  # nothing to shrink on the CPU
    monkeypatch.setattr(runtime.settings, "gpu_arena_shrink", False)
    off = _FakeSession([runtime.CUDA])
    assert runtime.wrap_session(off) is off  # off: the session itself
