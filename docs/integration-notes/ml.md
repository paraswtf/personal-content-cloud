# Integration notes: ml area

## Contract or doc changes for the integrator

1. ARCHITECTURE.md section 8 says rapidocr ships PP-OCRv4. The installed rapidocr 3.10.0 ships
   `PP-OCRv6_det_small.onnx`, `PP-OCRv6_rec_small.onnx` and `ch_ppocr_mobile_v2.0_cls_mobile.onnx` in the
   package. The registry pins those three files. Update the table row for `ocr`. Do not change rapidocr's
   language, model type or OCR version params: a model that is not bundled is downloaded on first use.
2. onnxruntime-gpu 1.31.1 is built for CUDA 13. Its CUDA 13 libraries come from the
   `onnxruntime-gpu[cuda,cudnn]` extras (nvidia-cublas 13, nvidia-cuda-runtime 13, nvidia-cudnn-cu13), and they
   load only after `ort.preload_dlls()`. `app.ml.runtime.prepare()` does that once per process. Without the
   preload, sessions come up on the CPU without raising, and the provider bridge prints errors about
   `libcublasLt.so.13`. The `nvidia-*-cu12` pins in `backend/pyproject.toml` are not used by onnxruntime.
   They can probably be dropped to save disk, but I did not check whether anything else needs them.
3. `tests/helpers.py` `text_image()` asks for `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`, which does
   not exist on this machine. It then falls back to `ImageFont.load_default()`, a 10 px font. A lone line in
   that font on a blank 1000x640 page is not detected by the OCR (multi-line pages are). Suggested fallback
   chain: DejaVu, then `/usr/share/fonts/truetype/freefont/FreeSans.ttf`, then `ImageFont.load_default(size)`.
   The ml OCR test monkeypatches `ImageFont.truetype` to that chain. Documents-area tests that OCR
   `text_image` output may need the same fix.
4. `siglip.zero_shot(image_embs, groups)` always returns 2-D `(n, len(groups))`. A 1-D embedding `(768,)`
   is treated as one row and gives `(1, len(groups))`. Callers that pass one embedding take `[0]`.
5. `settings.search_models_dir` is gone (the config has no such field). SigLIP reads `settings.models_dir`.

## What the area exposes beyond the listed signatures

- Every engine module (`faces`, `siglip`, `nsfw`, `persons`, `ocr`) has `available()`, `get_engine()`,
  `reset()`, `loaded()` and `provider()`. `models.statuses()` uses `loaded()` and `provider()` only for
  modules that are already imported, so a status call never loads a model.
- `models.start_download(keys=None) -> bool` runs `download()` in a daemon thread for
  `POST /api/system/models/download`. It returns False when a download is already running. `download()`
  raises `models.DownloadBusy` in that case and `models.DownloadError` (`.failed: dict[key, reason]`) when
  a model fails. Per-model errors show up in `statuses()[i]["error"]`.
- `models.verify(key)` hashes the files. `present()` checks sizes only, because conftest and run.sh call it often.
- `runtime.status()` returns `gpu_info()` plus `setting`, `cuda_listed`, `verified`, `error` and
  `onnxruntime` (version). `runtime.cuda_provider_options()` returns the CUDA EP options.
- `ocr.text_of(boxes) -> str` joins OCR boxes into lines. `ocr.available()` does not look at
  `settings.ocr_enabled`, so callers check that setting.
- `siglip.relevance(cos)` also exists at module level. `SiglipEngine` keeps the hub alias `SearchEngine`,
  and `get_engine` keeps the alias `get_search_engine`.
- `persons.detect` returns boxes best score first, clipped to the image, after NMS at 0.45.

## Measurements on this machine (RTX 2080, CUDA verified)

- With all five engines loaded in one process, GPU memory is about 2.6 GB.
- CUDA verification and library preload take about 1.3 s, once per process.
- OCR on a dense 1131x1600 page takes 0.6 to 0.9 s on the GPU. Every word was read at font sizes 11, 14 and 19 px.
- The SigLIP "dog" relevance on `tests/fixtures/search/dog.jpg` is 0.090 on CUDA, the same as the CPU
  calibration in the fixtures README.

## For the system API and its tests

- The first call to `runtime.gpu_info()`, `runtime.status()`, `runtime.providers()` or `runtime.session()`
  does the one-time CUDA check: library preload, a CUDA context and a tiny test session, about 1.3 s.
  System API tests that run under `PCC_NO_MODELS=1` should call `reload_settings(gpu_enabled="false")`
  and `app.ml.runtime.reset()` (then restore both) so they do not touch the GPU.
- Runtime fallback to the CPU covers session creation only. A CUDA error while a session runs (for
  example out of memory) raises to the caller, and the analysis job's retry handles it.
