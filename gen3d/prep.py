"""Getting a look's picture ready for the image-to-3D model: the background cut out (U2-Net through onnxruntime, on
the CPU), stray bits dropped, and the subject centred on a square with a margin. Returns RGBA, alpha 0 outside the
person. The model file is weights/u2net/u2net.onnx (the folder comes in U2NET_HOME)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

FILL = 0.86      # the subject's longer side as a share of the square
SIDE = 1024
NET_SIDE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_SUBJECT = 0.02   # of the picture: less than this is no subject
_session = None


def model_path() -> Path:
    return Path(os.environ.get("U2NET_HOME", "")) / "u2net.onnx"


def available() -> bool:
    return model_path().is_file()


def _net():
    global _session
    if _session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("GEN3D_THREADS", "4"))
        _session = ort.InferenceSession(str(model_path()), opts, providers=["CPUExecutionProvider"])
    return _session


def matte(image: Image.Image) -> np.ndarray:
    """The person's mask as uint8 (255 inside), at the picture's size."""
    rgb = image.convert("RGB")
    x = np.asarray(rgb.resize((NET_SIDE, NET_SIDE), Image.LANCZOS)).astype(np.float32)
    x = (x / max(float(x.max()), 1e-6) - MEAN) / STD
    net = _net()
    pred = net.run(None, {net.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    lo, hi = float(pred.min()), float(pred.max())
    pred = (pred - lo) / max(hi - lo, 1e-8)
    return np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize(rgb.size, Image.LANCZOS))


def _largest_part(alpha: np.ndarray) -> np.ndarray:
    import cv2

    mask = (alpha > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return alpha
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == keep, alpha, 0).astype(np.uint8)


def prepare(image: Image.Image, options: dict) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    alpha = _largest_part(matte(image))
    ys, xs = np.nonzero(alpha > 127)
    if ys.size < MIN_SUBJECT * alpha.size:
        raise ValueError("no person found in the picture")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    cut = Image.fromarray(np.dstack([rgb, alpha])[y0:y1, x0:x1], "RGBA")
    scale = FILL * SIDE / max(cut.size)
    cut = cut.resize((max(1, round(cut.size[0] * scale)), max(1, round(cut.size[1] * scale))), Image.LANCZOS)
    out = Image.new("RGBA", (SIDE, SIDE), (0, 0, 0, 0))
    out.paste(cut, ((SIDE - cut.size[0]) // 2, (SIDE - cut.size[1]) // 2))
    if options.get("debug") and options.get("out_dir") and options.get("key"):
        out.save(Path(options["out_dir"]) / f"{options['key']}-cut.png")
    return out
