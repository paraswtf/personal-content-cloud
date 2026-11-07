"""SigLIP ViT-B/16 (WebLI) image and text embeddings: semantic search, zero-shot categories, torso crops.

Contract (docs/ARCHITECTURE.md sections 8, 9 and 12):
  available() -> bool                  the model files are present (app.ml.models 'siglip')
  get_engine() -> SiglipEngine         process-wide engine, loaded on first use; reset() drops it
  loaded() / provider()                for app.ml.models.statuses()
  MODEL_ID, LOGIT_SCALE, LOGIT_BIAS, DEFAULT_MIN_RELEVANCE, SEARCH_PROMPT
  engine.embed_images(images) -> (n, 768) float32, L2-normalised. images are BGR uint8 arrays of any
      size (HxWx3; HxW grey and HxWx4 BGRA are accepted) or PIL images; each is squashed to 224x224 bicubic.
  engine.embed_text(text, prompt=True) -> (768,) float32, L2-normalised; recent texts are cached
  engine.relevance(cos) -> sigmoid(cos * LOGIT_SCALE + LOGIT_BIAS), the model's match probability
  engine.zero_shot(image_embs, groups) -> (n, len(groups)) probabilities, rows sum to 1, columns in the
      dict order of groups. A 1-D embedding (768,) is treated as one row and gives shape (1, len(groups)).

The ONNX export by Immich (immich-app/ViT-B-16-SigLIP__webli, Apache 2.0) has two models that map
photos and text into one 768-d space:
  visual/model.onnx   image (1, 3, 224, 224) float32 -> embedding (1, 768)
  textual/model.onnx  text  (1, 64) int32 token ids  -> embedding (1, 768)
Both exports have a fixed batch size of 1, so a "batch" is run one item at a time. Sessions come from
app.ml.runtime (CUDA when usable, else CPU).

Preprocessing follows the export config (visual/preprocess_cfg.json, config.json) and open_clip's
SigLIP pipeline exactly:
  image: RGB, "squash" resize to 224x224 with PIL bicubic, scale to [0, 1], then (x - 0.5) / 0.5 per
         channel, NCHW.
  text:  open_clip "canonicalize" cleaning (HTML entities decoded, "_" to space, ASCII punctuation
         removed, lower-cased, whitespace collapsed), SentencePiece tokenizer from textual/tokenizer.json
         (appends </s>), truncated and padded to the context length (64) with the pad token </s> (id 1).

Relevance: SigLIP is trained with a sigmoid loss, so sigmoid(logit_scale * cos + logit_bias) is the
model's own probability that an image and a text match. The learned values of this checkpoint are
logit_scale = exp(4.764997) = 117.33 and logit_bias = -12.932437. True matches typically land between
0.01 and 0.5 and unrelated pairs below 0.0001 (tests/fixtures/search/README.md).

Zero-shot: each group is a list of prompts. The prompts are embedded as written (no search prefix),
their normalised embeddings averaged and normalised again; the image-group cosines are multiplied by
LOGIT_SCALE and a softmax over the groups gives the probabilities (the bias cancels in a softmax).
"""
from __future__ import annotations

import html
import json
import logging
import string
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings

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
_GROUP_CACHE_SIZE = 256


def models_root() -> Path:
    """Folder that holds search/<MODEL_ID>/ (settings.models_dir)."""
    return Path(settings.models_dir)


def model_dir(models_dir: Path | str | None = None) -> Path:
    return Path(models_dir if models_dir is not None else models_root()) / "search" / MODEL_ID


def available() -> bool:
    """True when every model file is present with its pinned size (the engine can load)."""
    from app.ml import models

    return models.present("siglip")


def search_available(models_dir: Path | str | None = None) -> bool:
    """True when the required files exist in models_dir (any size; kept from the hub)."""
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


class SiglipEngine:
    """Holds the image and text sessions and the tokenizer. Thread-safe."""

    def __init__(self, models_dir: Path | str | None = None, threads: int = 2):
        from tokenizers import Tokenizer  # imported here so the app starts without it

        d = model_dir(models_dir)
        for f in REQUIRED_FILES:
            if not (d / f).is_file():
                raise FileNotFoundError(f"search model missing: {d / f} (run scripts/download_models.py)")
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

        from app.ml import runtime

        self._vis = runtime.session(d / "visual" / "model.onnx", threads=threads)
        self._txt = runtime.session(d / "textual" / "model.onnx", threads=threads)
        self.provider = runtime.session_provider(self._vis)
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
        self._groups: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------ images
    def preprocess(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """One image (PIL, or a BGR / grey / BGRA array) -> (3, H, W) float32, normalised."""
        img = _to_pil_rgb(img)
        img = img.resize(self.image_size, Image.Resampling.BICUBIC)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)

    def embed_images(self, images: list[np.ndarray] | list[Image.Image]) -> np.ndarray:
        """BGR arrays (or PIL images) -> (n, dim) float32, L2-normalised."""
        if images is None or len(images) == 0:
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
        return relevance(cos)

    # -------------------------------------------------------------- zero-shot
    def group_embedding(self, prompts: list[str] | tuple[str, ...]) -> np.ndarray:
        """Mean of the normalised prompt embeddings, normalised again. Cached per prompt list."""
        key = tuple(str(p) for p in prompts)
        if not key:
            raise ValueError("a zero-shot group needs at least one prompt")
        with self._cache_lock:
            hit = self._groups.get(key)
            if hit is not None:
                self._groups.move_to_end(key)
                return hit
        vecs = np.stack([self.embed_text(p, prompt=False) for p in key])
        vec = _l2norm(vecs.mean(axis=0))
        with self._cache_lock:
            self._groups[key] = vec
            while len(self._groups) > _GROUP_CACHE_SIZE:
                self._groups.popitem(last=False)
        return vec

    def zero_shot(self, image_embs: np.ndarray, groups: dict[str, list[str]]) -> np.ndarray:
        """Softmax over prompt groups for each image embedding: (n, len(groups)), rows sum to 1."""
        embs = np.atleast_2d(np.asarray(image_embs, dtype=np.float32))
        if not groups:
            raise ValueError("zero_shot needs at least one group")
        if embs.shape[0] == 0:
            return np.zeros((0, len(groups)), dtype=np.float32)
        g = np.stack([self.group_embedding(p) for p in groups.values()])  # (k, dim)
        logits = (embs @ g.T).astype(np.float64) * LOGIT_SCALE
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def relevance(cos: np.ndarray | float) -> np.ndarray:
    """Cosine similarity -> SigLIP match probability in [0, 1] (module-level form of engine.relevance)."""
    x = np.asarray(cos, dtype=np.float64) * LOGIT_SCALE + LOGIT_BIAS
    return _sigmoid(np.atleast_1d(x)).reshape(np.shape(x))


def _to_pil_rgb(img: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img if img.mode == "RGB" else img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {arr.shape}")
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]), "RGB")


SearchEngine = SiglipEngine  # hub name

_engine: SiglipEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> SiglipEngine:
    """Shared engine, loaded on first use (about 3 s and 1 GB of memory on the CPU)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SiglipEngine(models_root(), int(getattr(settings, "search_threads", 2)))
                log.info("SigLIP %s loaded from %s on %s", MODEL_ID, _engine.dir, _engine.provider)
    return _engine


get_search_engine = get_engine  # hub name


def reset() -> None:
    """Drop the shared engine (tests, or after the model files changed)."""
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None
