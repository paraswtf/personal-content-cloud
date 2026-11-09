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

## Hindi and Marathi (Devanagari) OCR

### What was added

- Registry key `ocr_devanagari` (label "Devanagari text recognition (Hindi, Marathi: PP-OCRv5 mobile) and
  Noto Sans Devanagari font", subdir `""`). It pins two files:
  - `models/ocr/devanagari_PP-OCRv5_rec_mobile.onnx`: 7,940,361 bytes, from the RapidOCR model zoo
    v3.10.0 on ModelScope, Apache 2.0;
  - `models/fonts/NotoSansDevanagari-Regular.ttf`: 277,680 bytes, from notofonts, SIL OFL 1.1.

  Both carry URLs, so `scripts/download_models.py` and `POST /api/system/models/download` fetch them
  like any other model. Engine state lives in `app.ml.ocr_devanagari`, which `_ENGINES` maps the key to,
  so Settings > System shows loaded and provider for it.
- `models.devanagari_font() -> Path | None` returns the font when it is on disk with its pinned size.
  It is for page renders (office and plain-text files) that need to draw Devanagari.
- `app.ml.ocr_devanagari` has `available()`, `get_engine()`, `recognise(crops)`, `choose(latin, deva)`,
  `is_devanagari(text)`, `load_error()`, `reset()`, `loaded()` and `provider()`.
  - The second engine is a RapidOCR instance built with the same CUDA and thread params, plus
    `Rec.model_path` and the enums `LangRec.DEVANAGARI`, `OCRVersion.PPOCRV5` and `ModelType.MOBILE`.
  - Only its recogniser is loaded. Detection and classification stay with the bundled engine.
  - Its character list (568 entries: Devanagari plus ASCII letters, digits and punctuation) sits in the
    ONNX metadata, so nothing is downloaded.
- `ocr.devanagari_active() -> bool`. `ocr.reset()` also resets the Devanagari engine.

### How `ocr.ocr()` works with two recognisers

When the model is present, `ocr.ocr()` runs rapidocr's steps itself:

1. preprocess;
2. detect and crop;
3. classify direction and rotate;
4. recognise every crop with the bundled recogniser and with the Devanagari recogniser;
5. choose one reading per crop;
6. map the boxes back to the input image;
7. drop empty readings and readings under rapidocr's `text_score` (0.5).

The steps run here because rapidocr's `build_final_output` drops every crop whose bundled reading is
empty, and that is how a Marathi line disappeared.

The `OcrBox` contract, `reading_order`, `text_of`, the run lock and the warm-up are unchanged. When the
model is absent, `ocr()` calls `eng(img)` as before. When the model fails to load, the failure is logged
once (until `reset()`), `load_error()` holds the reason, and `ocr()` also calls `eng(img)` as before. A
model downloaded while the app runs is loaded on the next `ocr()` call.

### Choice rule (`ocr_devanagari.choose`)

- The Devanagari reading wins when it has a Devanagari letter and its score is at least
  max(0.6, bundled score - 0.35).
- Otherwise the higher score wins, and ties go to the bundled reading.
- An empty reading counts as 0.
- A "Devanagari letter" is a letter (category L) in U+0900 to U+097F or U+A8E0 to U+A8FF. Devanagari
  digits, the danda and vowel signs alone do not count.

**Deviation from the rule as first specified (two guards, easy to revert in `choose()`):**

1. A Devanagari reading with no Devanagari letter never replaces a non-empty bundled reading.
   - Measured: on Latin lines the two recognisers often tie within 0.01. The Devanagari model then won
     with "Flight Al 865" (for "Flight AI"), "GsI" (for "GST") and "Price  499" (it dropped the ₹ sign).
2. The 0.35 margin applies only when the Devanagari reading has at least two Devanagari letters or no
   other letters.
   - Measured: the model reads ₹ as the letter र. "Price र 499" (0.932) would otherwise beat
     "Price ₹ 499" (0.958).
   - A reading with one Devanagari letter among Latin letters still wins on a higher score. That keeps
     "रु. 500 only" (0.962 against the bundled "5.500 only" at 0.947).

**Direction classifier.** `ch_ppocr_mobile_v2.0_cls` is not trained on Devanagari. It often turns an
upright Hindi or Marathi line upside down: the headline stroke looks like a Latin baseline, and one
Marathi line was flipped at 0.987. The Devanagari recogniser therefore also reads every crop the
classifier turned, in its upright form, and keeps the better of its two readings. The bundled
recogniser still reads the crops as rapidocr does. An upside-down Devanagari page that the classifier
does not turn is not read.

### English regression check

I rendered 24 images: the 3 document fixtures, 7 photo fixtures, and 14 synthetic lines in FreeSans,
FreeSerif and FreeMono at 14, 19 and 28 px, including ₹, €, ©, accents and punctuation. I compared the
output with and without the second recogniser.

- Before the guards (GPU run): 13 lines differed out of 139.
- With the guards and the upright re-read (CPU run, with two extra "रु." and "₹" lines on each synthetic
  page): 3 lines differed out of 157.
  - "रु. 500 only" is now read correctly. It is a Devanagari line.
  - "र. 500 only" in FreeMono, which has no Devanagari glyphs, was read as "." before.
  - "Total ₹ 1,200 paid" in FreeSans is now "Total र 1,200 paid" (0.982). The bundled reading
    "Total  1,200 paid" (0.974) had already lost the ₹, so the line was wrong either way.
- With the second recogniser on, small serif and mono lines that the classifier flips no longer come out
  as mirrored garbage. Without it they still do. This is a classifier issue, not one introduced here.

### Cost per page

- GPU (RTX 2080), measured before the upright re-read was added: 24 images took 7.93 s with the
  second recogniser and 7.75 s without, about 2 percent more. The 8-line Hindi, Marathi and English test
  page took 0.47 s once warm.
- CPU (4 threads): 23.3 s against 14.8 s for the same 24 images, about 57 percent more.
- The second recogniser takes about one more recognition pass over the crops, plus a pass over the crops
  the classifier turned.
- Loading both engines took 2.9 s with the warm-up.
- I could not measure the extra GPU memory: during this work an app server on port 41006 held 6.8 to
  7.4 GB of the 8 GB, and GPU runs of the tests failed with CUDA allocation errors. The tests were run
  on the CPU (`GPU_ENABLED=false`). The model file is 7.9 MB.

### Text form of Devanagari readings

- `choose()` returns a winning Devanagari reading in NFC. The model's character list has the precomposed
  nukta letters U+0958 to U+095F (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़). They are excluded from composition, so NFC turns
  each one into the letter plus U+093C (nukta), the form typed text and queries use. Bundled readings
  are returned unchanged.
- The list has no format characters: no ZWJ (U+200D) and no ZWNJ (U+200C).
- The marks OCR can emit (category Mn or Mc) are U+0900 to U+0903, U+093A to U+093C, U+093E to U+094F,
  U+0951 to U+0957 and U+0962 to U+0963. These cover candrabindu, anusvara, visarga, nukta, the vowel
  signs, virama, the Vedic accents and the vocalic vowel signs.
- It can also emit the danda (U+0964), the double danda (U+0965), the Devanagari digits U+0966 to U+096F,
  the abbreviation sign (U+0970) and the high spacing dot (U+0971, Lm).
- An FTS tokenizer that keeps Devanagari words whole needs these marks as token characters.

### Limits seen in testing

- If the Devanagari engine fails to load, the failure is latched until `ocr.reset()` or a restart, and
  Hindi and Marathi stay unread without a further error. One cause seen during this work was a
  transient CUDA allocation failure while another process held the GPU. `ocr_devanagari.load_error()`
  holds the reason. The `error` field of the `ocr_devanagari` row in `models.statuses()` shows it as
  "could not be loaded: ...".

- At 30 and 32 px the recogniser drops the subscript RA of "प्रिया" (it reads "प्िया"). The 26, 28 and
  34 to 40 px renders read exactly. The model-backed tests render at 36 px.
- The detector scales pages whose shorter side is under 736 px up before detecting. On a 247 px tall
  strip, it split a Marathi line into three overlapping word boxes, which repeated a letter. Pages of
  300 px and taller read correctly in a sweep of sizes. The tests use pages at least 800 px tall.

### For the integrator

- `tests/test_ml_models.py:73` asserts the exact key set `{"faces", "siglip", "nsfw", "persons", "ocr"}`.
  Add `"ocr_devanagari"`. The parametrize list at line 105 can add it too.
- `scripts/download_models.py`: its docstring lists the keys for `--only`. Add `ocr_devanagari`. No code
  change is needed.
- `ARCHITECTURE.md`:
  - section 8 model table: add the `ocr_devanagari` row;
  - line 371 ("Devanagari is not read"): Devanagari is now read when the model is present.

## Body traits models (2026-11-09, ARCHITECTURE sections 8 and 34)

- Three registry keys: `pose` (`pose/end2end.onnx`, DWPose-m from the Tau-J/RTMPose zip; the archive member is found
  by its basename `end2end.onnx`), `hands` (`hands/handpose.onnx`, MediaPipe handpose from OpenCV Zoo) and `dino`
  (`dino/dinov2_small.onnx`, DINOv2-small fp32). All three are pinned by size and sha256. The pose weights are
  trained on COCO-WholeBody, which is research and non-commercial use only; the registry's licence text says so.
- The engines follow `ml/ages.py` (lazy singleton, one lock, `runtime.session(path, threads=TRAITS_THREADS)`).
  `hands` tells its four outputs apart by shape at load time and refuses a graph with other shapes; the handedness
  polarity is not checked on labelled hands.
- `tests/test_ml_models.py` lists the new keys; `tests/test_ml_traits.py` needs the real files and runs alone with
  `ORT_THREADS=2`.
- SigLIP is used by the traits runner through the shared `siglip.get_engine()` (at `SEARCH_THREADS`). Crops go in as
  PIL RGB images: `_to_pil_rgb` reads a numpy array as BGR.
