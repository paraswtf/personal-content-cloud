"""Model registry and downloader (docs/ARCHITECTURE.md section 8).

Contract:
  REGISTRY: dict[str, ModelSpec]      keys 'faces', 'siglip', 'nsfw', 'ages', 'persons', 'ocr', 'ocr_devanagari',
                                      'pose', 'hands', 'dino' (the body traits models)
  present(key) -> bool                every file of the model is on disk with its pinned size (no hashing)
  path(key, relpath) -> Path          a file of the model (relative to the model's folder)
  model_dir(key) -> Path              the model's folder
  verify(key) -> list[str]            files whose size or sha256 differ from the pins (hashes every file)
  statuses() -> list[dict]            one ModelStatus dict per key (key, label, present, loaded, provider,
                                      size_bytes, downloading, error) for /api/system
  download(keys=None, progress=None)  fetch missing files of the given models (all when None); a no-op when
                                      everything is present; raises DownloadError naming the failed models
  download_state() -> dict            {running, progress, error} of the current or last download
  start_download(keys=None) -> bool   download() in a daemon thread; False when one is already running
  devanagari_font() -> Path | None    Noto Sans Devanagari Regular (from 'ocr_devanagari') when it is on disk
                                      with its pinned size, for page renders that draw Hindi or Marathi

Every file is pinned by size and sha256 and downloaded from a pinned source: a release asset or a
Hugging Face 'resolve' URL at a fixed commit. A file is streamed to '<name>.part' next to its target,
checked and renamed into place, so a partial download never looks present. The InsightFace models come
inside a release zip: the zip is streamed to a temp file, the two members are extracted, checked and
the zip is deleted. The OCR models ship inside the rapidocr package; they cannot be downloaded here and
present() checks that the package and its bundled model files are installed. 'ocr_devanagari' is the
second OCR recogniser (Hindi, Marathi) plus the Devanagari font; its files sit in models/ocr and
models/fonts (subdir '' with the folder in each file's rel).

This module does not import onnxruntime (scripts/download_models.py runs it on every start).
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from app.config import settings

log = logging.getLogger(__name__)

USER_AGENT = "personal-content-cloud/0.1 (model download)"
SPARE_BYTES = 200 * 1024 * 1024      # margin on top of settings.min_free_disk_bytes
CHUNK = 1 << 20

Progress = Callable[[str], None]


@dataclass(frozen=True)
class ModelFile:
    rel: str                  # path inside the model's folder
    size: int
    sha256: str
    url: str | None = None    # direct download; None for archive members and package files
    member: str | None = None  # file name inside the model's archive


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    subdir: str               # folder under settings.models_dir ('' for package models)
    files: tuple[ModelFile, ...]
    source: str
    license: str
    archive_url: str | None = None
    archive_size_estimate: int = 0   # for the free-disk check only (the zip is not pinned, its members are)
    package: str | None = None       # python package that ships the files (models live in <package>/models)

    @property
    def size_bytes(self) -> int:
        return sum(f.size for f in self.files)


_HF_SIGLIP = "https://huggingface.co/immich-app/ViT-B-16-SigLIP__webli/resolve/c3b037a9ef99ed5f116440790f1e098fb388db49/"
_HF_NSFW = "https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/resolve/1ceb3c7fe1e9f3f2507e6df577437f23a9149fd5/"
_HF_RTMPOSE = "https://huggingface.co/Tau-J/RTMPose/resolve/cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd/"
_HF_HANDS = "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/4b2a0b446e5cf2f11fb6b2c7251091c035d2c1f7/"
_HF_DINO = "https://huggingface.co/onnx-community/dinov2-small/resolve/8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/"
_HF_AGES = "https://huggingface.co/onnx-community/fairface_age_image_detection-ONNX/resolve/fd04cddf04982f6acc1ac76fb1a31b44c9285596/"

REGISTRY: dict[str, ModelSpec] = {
    "faces": ModelSpec(
        key="faces",
        label="Faces (InsightFace buffalo_l: SCRFD-10GF + ArcFace R50)",
        subdir="buffalo_l",
        files=(
            ModelFile("det_10g.onnx", 16_923_827,
                      "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91", member="det_10g.onnx"),
            ModelFile("w600k_r50.onnx", 174_383_860,
                      "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43", member="w600k_r50.onnx"),
        ),
        source="InsightFace release v0.7 (buffalo_l.zip)",
        license="InsightFace pretrained models: non-commercial research use",
        archive_url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        archive_size_estimate=290 * 1024 * 1024,
    ),
    "siglip": ModelSpec(
        key="siglip",
        label="Search and categories (SigLIP ViT-B/16 WebLI)",
        subdir="search/ViT-B-16-SigLIP__webli",
        files=tuple(
            ModelFile(rel, size, sha, url=_HF_SIGLIP + rel)
            for rel, size, sha in (
                ("config.json", 506, "23294da09d33a548acddb4feea6aa85d446c00803ad36fbc603d34a92a75dd62"),
                ("visual/preprocess_cfg.json", 154,
                 "d78a80d49dca4e62d17f802a685d67d821f7102844cb89f65a0bb1a5d0601f43"),
                ("visual/model.onnx", 371_674_292,
                 "0981d03d194baa6153106d41b5a5ca73faa514e8d9a007a3332eecb63be2c74a"),
                ("textual/model.onnx", 441_295_409,
                 "3983f96348daba5024f5e3085d2ea8cc67c11ce553e4061f193e8ea37aabd792"),
                ("textual/tokenizer.json", 2_421_038,
                 "ec6fe74c37f2a65be3857ef5e8fbca7662ebb15fcfa569a845a59958db969f76"),
                ("textual/tokenizer_config.json", 20_670,
                 "0c58fcd8b4706112d9fa3c330ba4c29597d71bfc228612ba36de48e39820acda"),
                ("textual/special_tokens_map.json", 2_533,
                 "acc196266a358dbcf2445f7b18f610c3c8763a5f0fdbf753f6aa0607631e7ba9"),
            )
        ),
        source="Hugging Face immich-app/ViT-B-16-SigLIP__webli at commit c3b037a",
        license="Apache 2.0",
    ),
    "nsfw": ModelSpec(
        key="nsfw",
        label="NSFW classifier (Falconsai ViT)",
        subdir="nsfw",
        files=(
            ModelFile("model.onnx", 343_401_688,
                      "a4316a4fb750169ac4fcabaabee1fcbd982b0ee8c0cc63fe3e944954bb9a7d9c",
                      url=_HF_NSFW + "onnx/model.onnx"),
            ModelFile("config.json", 764, "0e01783cf842a0acaa6b3b6594b941bee4d39a1918756b27d23f503878889696",
                      url=_HF_NSFW + "config.json"),
            ModelFile("preprocessor_config.json", 353,
                      "ae9bb157b9629887cc74913a4e7c12c9308f374f0930e8072320e8f2e1583c5e",
                      url=_HF_NSFW + "preprocessor_config.json"),
        ),
        source="Hugging Face onnx-community/nsfw_image_detection-ONNX at commit 1ceb3c7",
        license="Apache 2.0",
    ),
    "ages": ModelSpec(
        key="ages",
        label="Face ages (FairFace ViT, nine age groups)",
        subdir="ages",
        files=(
            ModelFile("onnx/model_int8.onnx", 87_339_068,
                      "c6a6c231140e2111267d9ade2017543c86ff4e685795be4006f1e94e0534c8c0",
                      url=_HF_AGES + "onnx/model_int8.onnx"),
            ModelFile("config.json", 1_010, "b7ebbfec7938cc4db21d972bd1aecdaf084204e24ddc1d6bde8ada64bb768652",
                      url=_HF_AGES + "config.json"),
            ModelFile("preprocessor_config.json", 353,
                      "ae9bb157b9629887cc74913a4e7c12c9308f374f0930e8072320e8f2e1583c5e",
                      url=_HF_AGES + "preprocessor_config.json"),
        ),
        source="Hugging Face onnx-community/fairface_age_image_detection-ONNX at commit fd04cdd (int8 export of "
               "dima806/fairface_age_image_detection, a ViT trained on FairFace)",
        license="Apache 2.0",
    ),
    "persons": ModelSpec(
        key="persons",
        label="Person detector (YOLOX-S)",
        subdir="yolox",
        files=(
            ModelFile("yolox_s.onnx", 35_858_002,
                      "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
                      url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"),
        ),
        source="Megvii YOLOX release 0.1.1rc0",
        license="Apache 2.0",
    ),
    "ocr": ModelSpec(
        key="ocr",
        label="Text recognition (RapidOCR, PP-OCRv6 small)",
        subdir="",
        files=(
            ModelFile("PP-OCRv6_det_small.onnx", 9_929_594,
                      "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
            ModelFile("PP-OCRv6_rec_small.onnx", 21_234_383,
                      "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
            ModelFile("ch_ppocr_mobile_v2.0_cls_mobile.onnx", 585_532,
                      "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
        ),
        source="bundled with the rapidocr 3.10 package",
        license="Apache 2.0",
        package="rapidocr",
    ),
    "ocr_devanagari": ModelSpec(
        key="ocr_devanagari",
        label="Devanagari text recognition (Hindi, Marathi: PP-OCRv5 mobile) and Noto Sans Devanagari font",
        subdir="",
        files=(
            ModelFile("ocr/devanagari_PP-OCRv5_rec_mobile.onnx", 7_940_361,
                      "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e",
                      url="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/"
                          "devanagari_PP-OCRv5_rec_mobile.onnx"),
            ModelFile("fonts/NotoSansDevanagari-Regular.ttf", 277_680,
                      "573e525d9f4923d34645fe752475419f45f8dec5a8e43f3e83c5c5f43d1f2ff9",
                      url="https://raw.githubusercontent.com/notofonts/notofonts.github.io/86ce02c0fe7890f5936c5dffb26fca0797d1be64/fonts/"
                          "NotoSansDevanagari/full/ttf/NotoSansDevanagari-Regular.ttf"),
        ),
        source="RapidOCR model zoo v3.10.0 on ModelScope (PP-OCRv5 Devanagari mobile recogniser); "
               "Noto fonts (notofonts.github.io, Noto Sans Devanagari Regular)",
        license="Recogniser: Apache 2.0; font: SIL Open Font License 1.1",
    ),
    "pose": ModelSpec(
        key="pose",
        label="Body pose (DWPose-m, RTMPose whole body 133 points)",
        subdir="pose",
        files=(
            ModelFile("end2end.onnx", 71_792_372,
                      "94ca58fa2d6c4530b6957ac9548084ebc2fa27ed71e4e01f0b73844306ed01a6", member="end2end.onnx"),
        ),
        source="Hugging Face Tau-J/RTMPose at commit cd4d709 (rtmposev1 onnx_sdk, rtmpose-m_simcc-ucoco_dw-ucoco_"
               "270e-256x192 zip, member end2end.onnx)",
        license="Code: Apache 2.0. Weights are trained on COCO-WholeBody: research and non-commercial use only "
                "(acceptable for this personal, non-commercial repository)",
        archive_url=_HF_RTMPOSE + "rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
        archive_size_estimate=67 * 1024 * 1024,
    ),
    "hands": ModelSpec(
        key="hands",
        label="Hand landmarks (MediaPipe handpose, OpenCV Zoo 2023feb)",
        subdir="hands",
        files=(
            ModelFile("handpose.onnx", 4_099_621,
                      "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
                      url=_HF_HANDS + "handpose_estimation_mediapipe_2023feb.onnx"),
        ),
        source="Hugging Face opencv/handpose_estimation_mediapipe at commit 4b2a0b4",
        license="Apache 2.0",
    ),
    "dino": ModelSpec(
        key="dino",
        label="Image descriptors (DINOv2 small, fp32)",
        subdir="dino",
        files=(
            ModelFile("dinov2_small.onnx", 88_532_934,
                      "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
                      url=_HF_DINO + "onnx/model.onnx"),
        ),
        source="Hugging Face onnx-community/dinov2-small at commit 8b1f705 (onnx/model.onnx, fp32)",
        license="Apache 2.0",
    ),
}

DEVANAGARI_FONT_REL = "fonts/NotoSansDevanagari-Regular.ttf"

# engine module per key: each exposes loaded() -> bool and provider() -> str | None
_ENGINES = {
    "faces": "app.ml.faces",
    "siglip": "app.ml.siglip",
    "nsfw": "app.ml.nsfw",
    "ages": "app.ml.ages",
    "persons": "app.ml.persons",
    "ocr": "app.ml.ocr",
    "ocr_devanagari": "app.ml.ocr_devanagari",
    "pose": "app.ml.pose",
    "hands": "app.ml.hands",
    "dino": "app.ml.dino",
}


class DownloadError(RuntimeError):
    """One or more models failed to download. `failed` maps model key -> reason."""

    def __init__(self, failed: dict[str, str]):
        self.failed = dict(failed)
        super().__init__("; ".join(f"{k}: {v}" for k, v in self.failed.items()))


class DownloadBusy(RuntimeError):
    """Another download is running in this process."""


# ------------------------------------------------------------------ files

def _spec(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {', '.join(REGISTRY)}") from None


def _package_dir(package: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def model_dir(key: str) -> Path:
    spec = _spec(key)
    if spec.package:
        base = _package_dir(spec.package)
        if base is None:
            return Path(settings.models_dir) / f"{spec.package}-not-installed"
        return base / "models"
    return Path(settings.models_dir) / spec.subdir


def path(key: str, relpath: str) -> Path:
    return model_dir(key) / relpath


def _file_ok(p: Path, size: int) -> bool:
    try:
        return p.is_file() and p.stat().st_size == size
    except OSError:
        return False


def missing_files(key: str) -> list[ModelFile]:
    spec = _spec(key)
    if spec.package and _package_dir(spec.package) is None:
        return list(spec.files)
    d = model_dir(key)
    return [f for f in spec.files if not _file_ok(d / f.rel, f.size)]


def present(key: str) -> bool:
    """Every file of the model exists with its pinned size. Cheap: no hashing."""
    return not missing_files(key)


def devanagari_font() -> Path | None:
    """The Noto Sans Devanagari font when it is on disk with its pinned size, else None."""
    spec = REGISTRY.get("ocr_devanagari")
    if spec is None:
        return None
    pinned = next((f for f in spec.files if f.rel == DEVANAGARI_FONT_REL), None)
    if pinned is None:
        return None
    p = path("ocr_devanagari", pinned.rel)
    return p if _file_ok(p, pinned.size) else None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(key: str) -> list[str]:
    """Relative paths of files that are missing or whose size or sha256 differ from the pins."""
    spec = _spec(key)
    d = model_dir(key)
    bad = []
    for f in spec.files:
        p = d / f.rel
        if not _file_ok(p, f.size) or sha256_file(p) != f.sha256:
            bad.append(f.rel)
    return bad


# ----------------------------------------------------------------- status

_state_lock = threading.Lock()
_run_lock = threading.Lock()
_state: dict[str, Any] = {"running": False, "progress": None, "error": None, "key": None}
_errors: dict[str, str] = {}


def download_state() -> dict[str, Any]:
    with _state_lock:
        return {"running": bool(_state["running"]), "progress": _state["progress"], "error": _state["error"]}


def _set_state(**kw: Any) -> None:
    with _state_lock:
        _state.update(kw)


def _engine_info(key: str) -> tuple[bool, str | None]:
    mod = sys.modules.get(_ENGINES.get(key, ""))
    if mod is None:
        return False, None
    try:
        loaded = bool(mod.loaded())
        return loaded, (mod.provider() if loaded else None)
    except Exception:  # noqa: BLE001
        return False, None


def _engine_error(key: str) -> str | None:
    """A load failure the engine module keeps (only app.ml.ocr_devanagari has one), for the status row."""
    mod = sys.modules.get(_ENGINES.get(key, ""))
    fn = getattr(mod, "load_error", None) if mod is not None else None
    if fn is None:
        return None
    try:
        err = fn()
    except Exception:  # noqa: BLE001
        return None
    return f"could not be loaded: {err}" if err else None


def statuses() -> list[dict[str, Any]]:
    with _state_lock:
        current = _state["key"] if _state["running"] else None
        errors = dict(_errors)
    out = []
    for key, spec in REGISTRY.items():
        loaded, provider = _engine_info(key)
        out.append({
            "key": key,
            "label": spec.label,
            "present": present(key),
            "loaded": loaded,
            "provider": provider,
            "size_bytes": spec.size_bytes,
            "downloading": current == key,
            "error": errors.get(key) or _engine_error(key),
        })
    return out


# --------------------------------------------------------------- download

def _free_bytes(target: Path) -> int:
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _fetch(url: str, dest: Path, *, size: int | None, sha256: str | None, label: str,
           progress: Progress | None) -> None:
    """Stream url into dest (a .part or temp path); check size and sha256 when given."""
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = 0.0
    done = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = size or int(resp.headers.get("Content-Length") or 0)
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= 2.0:
                last = now
                if total:
                    msg = f"{label}: {done * 100 // total}% ({done // 2**20} of {total // 2**20} MB)"
                else:
                    msg = f"{label}: {done // 2**20} MB"
                _set_state(progress=msg)
                if progress:
                    progress(msg)
        out.flush()
        os.fsync(out.fileno())
    if size is not None and done != size:
        raise RuntimeError(f"size mismatch for {url}: got {done} bytes, expected {size}")
    if sha256 and h.hexdigest() != sha256:
        raise RuntimeError(f"sha256 mismatch for {url}: got {h.hexdigest()}")


def _download_files(spec: ModelSpec, files: list[ModelFile], progress: Progress | None) -> None:
    d = model_dir(spec.key)
    for f in files:
        if not f.url:
            raise RuntimeError(f"{f.rel} has no download source")
        final = d / f.rel
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.name + ".part")
        label = f"{spec.key}: {f.rel} ({max(1, f.size // 2**20)} MB)"
        if progress:
            progress(f"downloading {label}")
        _set_state(progress=f"downloading {label}")
        try:
            _fetch(f.url, part, size=f.size, sha256=f.sha256, label=f"{spec.key}/{f.rel}", progress=progress)
            part.replace(final)
        except BaseException:
            part.unlink(missing_ok=True)
            raise


def _download_archive(spec: ModelSpec, files: list[ModelFile], progress: Progress | None) -> None:
    d = model_dir(spec.key)
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{spec.key}.", suffix=".zip.part", dir=d)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if progress:
            progress(f"downloading {spec.archive_url}")
        _set_state(progress=f"downloading {spec.key} archive")
        _fetch(spec.archive_url or "", tmp, size=None, sha256=None, label=f"{spec.key} archive", progress=progress)
        with zipfile.ZipFile(tmp) as zf:
            names = {Path(n).name: n for n in zf.namelist()}
            for f in files:
                member = f.member or Path(f.rel).name
                if member not in names:
                    raise RuntimeError(f"{member} not found in {spec.archive_url}")
                final = d / f.rel
                final.parent.mkdir(parents=True, exist_ok=True)
                part = final.with_name(final.name + ".part")
                try:
                    with zf.open(names[member]) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst, CHUNK)
                    got = part.stat().st_size
                    if got != f.size:
                        raise RuntimeError(f"size mismatch for {member}: got {got}, expected {f.size}")
                    digest = sha256_file(part)
                    if digest != f.sha256:
                        raise RuntimeError(f"sha256 mismatch for {member}: got {digest}")
                    part.replace(final)
                except BaseException:
                    part.unlink(missing_ok=True)
                    raise
    finally:
        tmp.unlink(missing_ok=True)


def _download_one(key: str, progress: Progress | None) -> None:
    spec = _spec(key)
    files = missing_files(key)
    if not files:
        return
    if spec.package:
        raise RuntimeError(f"the {spec.package} package or its bundled models are missing; "
                           f"reinstall it with the backend environment (uv sync)")
    need = sum(f.size for f in files) + (spec.archive_size_estimate if spec.archive_url else 0)
    want = need + max(0, int(settings.min_free_disk_bytes)) + SPARE_BYTES
    free = _free_bytes(model_dir(key))
    if free < want:
        raise RuntimeError(f"not enough free disk in {model_dir(key)}: {free // 2**20} MB free, "
                           f"need {want // 2**20} MB including the free-space floor")
    if spec.archive_url:
        _download_archive(spec, files, progress)
    else:
        _download_files(spec, files, progress)
    if not present(key):
        raise RuntimeError("files still missing after download")


def download(keys: Iterable[str] | None = None, progress: Progress | None = None) -> None:
    """Download every missing file of the named models (all models when keys is None).

    A no-op when everything is present. Models are tried one after another; a failure does not stop
    the others. Raises DownloadError (with .failed) at the end when any model failed, and DownloadBusy
    when another download is running in this process.
    """
    wanted = list(REGISTRY) if keys is None else [k for k in keys]
    for k in wanted:
        _spec(k)
    todo = [k for k in wanted if not present(k)]
    if not todo:
        return
    if not _run_lock.acquire(blocking=False):
        raise DownloadBusy("a model download is already running")
    failed: dict[str, str] = {}
    try:
        _set_state(running=True, progress=None, error=None, key=None)
        for k in todo:
            _set_state(key=k)
            with _state_lock:
                _errors.pop(k, None)
            try:
                _download_one(k, progress)
                log.info("model %s ready in %s", k, model_dir(k))
                if progress:
                    progress(f"{k}: ready")
            except Exception as e:  # noqa: BLE001 - reported per model
                reason = str(e) or e.__class__.__name__
                failed[k] = reason
                with _state_lock:
                    _errors[k] = reason
                log.warning("model %s download failed: %s", k, reason)
                if progress:
                    progress(f"{k}: failed: {reason}")
        error = "; ".join(f"{k}: {v}" for k, v in failed.items()) or None
        _set_state(running=False, key=None, error=error, progress="failed" if failed else "done")
    except BaseException as e:
        _set_state(running=False, key=None, error=str(e) or e.__class__.__name__)
        raise
    finally:
        _run_lock.release()
    if failed:
        raise DownloadError(failed)


def start_download(keys: Iterable[str] | None = None) -> bool:
    """Run download() in a daemon thread for the system API. False when a download is already running."""
    if _run_lock.locked():
        return False
    wanted = None if keys is None else list(keys)

    def run() -> None:
        try:
            download(wanted)
        except DownloadBusy:
            pass
        except Exception as e:  # noqa: BLE001 - state holds the error
            log.warning("model download failed: %s", e)

    threading.Thread(target=run, name="pcc-model-download", daemon=True).start()
    return True
