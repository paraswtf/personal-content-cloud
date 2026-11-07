"""Semantic search embeddings with Google SigLIP ViT-B/16 (WebLI), onnxruntime on the CPU.

The ONNX export by Immich (immich-app/ViT-B-16-SigLIP__webli, Apache 2.0) has two models
that map photos and text into one 768-d space:
  visual/model.onnx   image (1, 3, 224, 224) float32 -> embedding (1, 768)
  textual/model.onnx  text  (1, 64) int32 token ids  -> embedding (1, 768)
Both exports have a fixed batch size of 1, so a "batch" is run one item at a time.

Preprocessing follows the export config (visual/preprocess_cfg.json, config.json) and
open_clip's SigLIP pipeline exactly:
  image: RGB, "squash" resize to 224x224 with PIL bicubic, scale to [0, 1], then
         (x - 0.5) / 0.5 per channel, NCHW.
  text:  open_clip "canonicalize" cleaning (HTML entities decoded, "_" to space, ASCII
         punctuation removed, lower-cased, whitespace collapsed), SentencePiece tokenizer
         from textual/tokenizer.json (appends </s>), truncated and padded to the context
         length (64) with the pad token </s> (id 1).

Relevance: SigLIP is trained with a sigmoid loss, so sigmoid(logit_scale * cos + logit_bias)
is the model's own probability that an image and a text match. The learned values of this
checkpoint (timm/ViT-B-16-SigLIP open_clip weights) are logit_scale = exp(4.764997) = 117.33
and logit_bias = -12.932437. True matches typically land between 0.01 and 0.5 on this
scale and unrelated pairs below 0.0001; DEFAULT_MIN_RELEVANCE was picked on real photos
(see tests/test_search_engine.py and tests/fixtures/search/README.md).

This module is the only place that knows about the model. Swapping models means a new
MODEL_ID (stored with every embedding, so old rows are simply re-indexed) and re-checking
DEFAULT_MIN_RELEVANCE.
"""
from __future__ import annotations

import html
import json
import logging
import os
import string
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.config import REPO_ROOT, settings

log = logging.getLogger(__name__)

MODEL_ID = "ViT-B-16-SigLIP__webli"

# Learned SigLIP temperature and bias of this checkpoint.
LOGIT_SCALE = 117.33076
LOGIT_BIAS = -12.932437

# Lowest relevance a search hit needs (settings.search_min_relevance defaults to this).
DEFAULT_MIN_RELEVANCE = 0.001

# Search queries are embedded as "a photo of <query>": on real photos this raised the
# relevance of true matches 2 to 5 times for short queries ("cat", "dog", "woman") while
# unrelated pairs stayed below DEFAULT_MIN_RELEVANCE. Queries that already say so are kept.
SEARCH_PROMPT = "a photo of {}"
_PROMPT_PREFIXES = ("a photo of ", "photo of ", "a picture of ", "picture of ", "an image of ", "image of ")

REQUIRED_FILES = ("visual/model.onnx", "textual/model.onnx", "textual/tokenizer.json")

_PUNCT = str.maketrans("", "", string.punctuation)
_TEXT_CACHE_SIZE = 256


def models_root() -> Path:
    """Folder that holds search/<MODEL_ID>/ (settings.search_models_dir)."""
    value = getattr(settings, "search_models_dir", None)
    if value is None:
        value = os.environ.get("SEARCH_MODELS_DIR") or (REPO_ROOT / "models")
    return Path(value)


def model_dir(models_dir: Path | str | None = None) -> Path:
    return Path(models_dir if models_dir is not None else models_root()) / "search" / MODEL_ID


def search_available(models_dir: Path | str | None = None) -> bool:
    """True when every model file is present (search can run)."""
    d = model_dir(models_dir)
    return all((d / f).is_file() for f in REQUIRED_FILES)


def normalize_text(text: str) -> str:
    """open_clip "canonicalize" cleaning used for SigLIP text."""
    text = html.unescape(html.unescape(str(text))).strip()
    text = text.replace("_", " ").translate(_PUNCT).lower()
    return " ".join(text.split())


def search_prompt(query: str) -> str:
    """The text actually embedded for a search query (cleaned, with SEARCH_PROMPT)."""
    q = normalize_text(query)
    if not q or (q + " ").startswith(_PROMPT_PREFIXES):
        return q
    return SEARCH_PROMPT.format(q)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(n, 1e-12)).astype(np.float32)


class SearchEngine:
    """Holds the image and text sessions and the tokenizer. Thread-safe."""

    def __init__(self, models_dir: Path | str | None = None, threads: int = 2):
        from tokenizers import Tokenizer  # imported here so the app starts without it

        d = model_dir(models_dir)
        for f in REQUIRED_FILES:
            if not (d / f).is_file():
                raise FileNotFoundError(f"search model missing: {d / f} (run scripts/download_search_model.py)")
        self.model_id = MODEL_ID
        self.dir = d

        pre = {}
        if (d / "visual" / "preprocess_cfg.json").is_file():
            pre = json.loads((d / "visual" / "preprocess_cfg.json").read_text())
        cfg = {}
        if (d / "config.json").is_file():
            cfg = json.loads((d / "config.json").read_text())
        size = pre.get("size", [224, 224])
        if isinstance(size, int):
            size = [size, size]
        self.image_size = (int(size[1]), int(size[0]))  # PIL (width, height)
        self.mean = np.array(pre.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(pre.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        if pre.get("resize_mode", "squash") != "squash" or pre.get("interpolation", "bicubic") != "bicubic":
            raise RuntimeError(f"unsupported preprocessing in {d}: {pre}")
        self.context_length = int(cfg.get("text_cfg", {}).get("context_length", 64))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self._vis = ort.InferenceSession(str(d / "visual" / "model.onnx"), opts, providers=providers)
        self._txt = ort.InferenceSession(str(d / "textual" / "model.onnx"), opts, providers=providers)
        vin = self._vis.get_inputs()[0]
        tin = self._txt.get_inputs()[0]
        self._vis_in = vin.name
        self._txt_in = tin.name
        self._vis_batch = vin.shape[0] if isinstance(vin.shape[0], int) else 16
        self._txt_dtype = np.int64 if tin.type == "tensor(int64)" else np.int32
        out_dim = self._vis.get_outputs()[0].shape[-1]
        self.dim = int(out_dim) if isinstance(out_dim, int) else int(cfg.get("embed_dim", 768))

        tok = Tokenizer.from_file(str(d / "textual" / "tokenizer.json"))
        self.pad_id = 1
        tcfg_path = d / "textual" / "tokenizer_config.json"
        if tcfg_path.is_file():
            pad_token = json.loads(tcfg_path.read_text()).get("pad_token")
            if isinstance(pad_token, dict):
                pad_token = pad_token.get("content")
            if pad_token and tok.token_to_id(pad_token) is not None:
                self.pad_id = int(tok.token_to_id(pad_token))
        tok.no_padding()
        tok.enable_truncation(max_length=self.context_length)
        self._tok = tok

        self._vis_lock = threading.Lock()
        self._txt_lock = threading.Lock()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ images
    def preprocess(self, img: Image.Image) -> np.ndarray:
        """One PIL image -> (3, H, W) float32, normalised as the model expects."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(self.image_size, Image.Resampling.BICUBIC)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """PIL RGB images -> (n, dim) float32, L2-normalised."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(images), self.dim), dtype=np.float32)
        step = max(1, int(self._vis_batch))
        for i in range(0, len(images), step):
            blob = np.stack([self.preprocess(im) for im in images[i:i + step]])
            with self._vis_lock:
                emb = self._vis.run(None, {self._vis_in: blob})[0]
            out[i:i + len(blob)] = emb.reshape(len(blob), -1)
        return _l2norm(out)

    # -------------------------------------------------------------------- text
    def tokenize(self, text: str) -> np.ndarray:
        """Text -> (1, context_length) token ids, cleaned, truncated and padded with </s>."""
        ids = self._tok.encode(normalize_text(text)).ids[: self.context_length]
        arr = np.full((1, self.context_length), self.pad_id, dtype=self._txt_dtype)
        arr[0, : len(ids)] = ids
        return arr

    def embed_text(self, text: str, prompt: bool = True) -> np.ndarray:
        """Search query -> (dim,) float32, L2-normalised. Recent queries are cached.

        With prompt=True (the default, and what DEFAULT_MIN_RELEVANCE is calibrated for)
        the query is embedded as search_prompt(text); prompt=False embeds the text as is.
        """
        key = search_prompt(text) if prompt else normalize_text(text)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit.copy()
        tokens = self.tokenize(key)
        with self._txt_lock:
            emb = self._txt.run(None, {self._txt_in: tokens})[0]
        vec = _l2norm(emb.reshape(-1))
        with self._cache_lock:
            self._cache[key] = vec
            self._cache.move_to_end(key)
            while len(self._cache) > _TEXT_CACHE_SIZE:
                self._cache.popitem(last=False)
        return vec.copy()

    # --------------------------------------------------------------- relevance
    @staticmethod
    def relevance(cos: np.ndarray | float) -> np.ndarray:
        """Cosine similarity -> SigLIP match probability in [0, 1]."""
        x = np.asarray(cos, dtype=np.float64) * LOGIT_SCALE + LOGIT_BIAS
        return _sigmoid(np.atleast_1d(x)).reshape(np.shape(x))


_engine: SearchEngine | None = None
_engine_lock = threading.Lock()


def get_search_engine() -> SearchEngine:
    """Shared engine, loaded on first use (about 3 s and 1 GB of memory)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SearchEngine(models_root(), int(getattr(settings, "search_threads", 2)))
                log.info("search model %s loaded from %s", MODEL_ID, _engine.dir)
    return _engine
