"""Getting a look's picture ready for the image-to-3D model: the background cut out (U2-Net through onnxruntime, on
the CPU), stray bits dropped, and the subject centred on a square with a margin. The model file is
weights/u2net/u2net.onnx (the folder comes in U2NET_HOME).

`prepare` returns a Prepared: the square the model is given (RGBA, alpha 0 outside the person), and with it the
picture as it came (at its own size, up to 3072 pixels: the finishing lays it over the head as the texture), the
cut-out at that size, and how a point of the square maps back to a point of the picture. `whole` is the same record
for a picture that is used as it is (the stand-in backend): everything counts as the person.

A look's picture may be wider than the photo it was cut from: the app then fills the rest by repeating the photo's
edge. The job says which part is the photo itself (`inside`); it is kept with the record, and the finishing shows
no photo outside it. The model is still given the whole picture, as before."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FILL = 0.86      # the subject's longer side as a share of the square
SIDE = 1024
NET_SIDE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_SUBJECT = 0.02   # of the picture: less than this is no subject
CENTRE = (0.35, 0.25, 0.65, 0.6)   # left, top, right, bottom of the window the face is cut into, as shares
_session = None


@dataclass
class Prepared:
    """The picture in two frames. The square: SIDE x SIDE pixels, what the model saw (`rgba`, None when the model
    is given the picture itself). The source: the picture as it came, with the person's cut-out at its size.
    A square point (x, y) is the source point ((x - ox) / sx + x0, (y - oy) / sy + y0); pixel centres at +0.5 in
    both, so the same formula holds for them."""
    rgba: Image.Image | None
    source: np.ndarray | None   # (h, w, 3) uint8; None while a job keeps the record between its two passes
    alpha: np.ndarray           # (h, w) uint8, 255 inside the person
    side: int                   # the square's side in pixels
    x0: float                   # the corner, in the source, of the part that was cut out ...
    y0: float
    sx: float                   # ... how many square pixels one of its pixels became, across and down ...
    sy: float
    ox: float                   # ... and where that part was pasted in the square
    oy: float
    inside: tuple[float, float, float, float] | None = None   # left, top, right, bottom of the photo itself, source pixels

    def to_source(self, xy: np.ndarray) -> np.ndarray:
        """Square points (n, 2) as source points."""
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack([(xy[:, 0] - self.ox) / self.sx + self.x0, (xy[:, 1] - self.oy) / self.sy + self.y0], axis=1)

    def to_square(self, xy: np.ndarray) -> np.ndarray:
        """Source points (n, 2) as square points: the way back."""
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack([(xy[:, 0] - self.x0) * self.sx + self.ox, (xy[:, 1] - self.y0) * self.sy + self.oy], axis=1)

    def part(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) in whole source pixels of the part that is the photo itself: `inside` rounded
        inwards and kept within the picture, or the whole picture. Right and bottom are the first pixels past it;
        an `inside` that leaves nothing gives right <= left or bottom <= top."""
        h, w = self.alpha.shape[:2]
        if self.inside is None:
            return 0, 0, int(w), int(h)
        left, top, right, bottom = (float(t) for t in self.inside)
        return (max(0, int(np.ceil(left))), max(0, int(np.ceil(top))),
                min(int(w), int(np.floor(right))), min(int(h), int(np.floor(bottom))))


def _inside(options: dict | None) -> tuple[float, float, float, float] | None:
    given = (options or {}).get("inside")
    return None if given is None else tuple(float(t) for t in given)  # type: ignore[return-value]


def source_of(image: Image.Image) -> np.ndarray:
    """A picture as a record keeps it: (h, w, 3) uint8, RGB."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def whole(image: Image.Image, options: dict | None = None) -> Prepared:
    """The record for a picture the model is given as it is: the square is the picture, everything is the person.
    options: `inside`, as for prepare."""
    rgb = source_of(image)
    h, w = rgb.shape[:2]
    return Prepared(rgba=None, source=rgb, alpha=np.full((h, w), 255, dtype=np.uint8), side=SIDE,
                    x0=0.0, y0=0.0, sx=SIDE / w, sy=SIDE / h, ox=0.0, oy=0.0, inside=_inside(options))


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


def release() -> None:
    """Let the cut-out model go (it is loaded again when it is next needed)."""
    global _session
    _session = None


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


def _subject_part(alpha: np.ndarray) -> np.ndarray:
    """Of the separate blobs of the mask, the one the look's face sits in. The picture is cut with the face in the
    middle, a little above centre (app.analysis.looks), so the subject is the blob holding the most of a window
    there; a larger blob elsewhere (another person at the edge, a lamp) is dropped. Without any blob in the window,
    the largest one."""
    import cv2

    mask = (alpha > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return alpha
    h, w = mask.shape
    window = labels[int(h * CENTRE[1]):int(h * CENTRE[3]), int(w * CENTRE[0]):int(w * CENTRE[2])]
    inside = np.bincount(window.ravel(), minlength=n)[1:]
    keep = 1 + int(np.argmax(inside)) if int(inside.max()) > 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == keep, alpha, 0).astype(np.uint8)


def prepare(image: Image.Image, options: dict) -> Prepared:
    """The picture cut out and laid on the model's square. options: `inside` ([left, top, right, bottom] in pixels
    of the picture, the part that is the photo itself; left out when all of it is), and `debug`, `out_dir`, `key`
    to keep the square as a file."""
    rgb = source_of(image)
    alpha = _subject_part(matte(image))
    ys, xs = np.nonzero(alpha > 127)
    if ys.size < MIN_SUBJECT * alpha.size:
        raise ValueError("no person found in the picture")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    cut = Image.fromarray(np.dstack([rgb, alpha])[y0:y1, x0:x1], "RGBA")
    scale = FILL * SIDE / max(cut.size)
    wide, tall = max(1, round(cut.size[0] * scale)), max(1, round(cut.size[1] * scale))
    cut = cut.resize((wide, tall), Image.LANCZOS)
    ox, oy = (SIDE - wide) // 2, (SIDE - tall) // 2
    out = Image.new("RGBA", (SIDE, SIDE), (0, 0, 0, 0))
    out.paste(cut, (ox, oy))
    if options.get("debug") and options.get("out_dir") and options.get("key"):
        out.save(Path(options["out_dir"]) / f"{options['key']}-cut.png")
    return Prepared(rgba=out, source=rgb, alpha=np.ascontiguousarray(alpha), side=SIDE, x0=float(x0), y0=float(y0),
                    sx=wide / (x1 - x0), sy=tall / (y1 - y0), ox=float(ox), oy=float(oy), inside=_inside(options))
