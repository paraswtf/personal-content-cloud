"""TripoSR (Stability AI and Tripo AI, MIT licence): one picture in, a mesh with a colour per vertex out.

Layout (scripts/setup_3d.sh makes it): WEIGHTS/TripoSR/{config.yaml, model.ckpt}, WEIGHTS/u2net/u2net.onnx,
WEIGHTS/hf (the cached config of the image encoder) and WEIGHTS/../src/TripoSR (the pinned checkout with the `tsr`
package). Nothing is downloaded at run time. Measured on an RTX 2080: about 6 s a head at detail 2, 2.4 GB of GPU
memory, 3.2 GB resident."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NAME = "triposr"
RESOLUTION = {1: 192, 2: 256, 3: 320}   # marching cubes grid per detail level
INPUT_SIDE = 512
GREY = 0.5                              # the background the model was trained on
CHUNK = 8192


class TripoSR:
    name = NAME
    version = "1.0"

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"

    @staticmethod
    def _weights(weights_dir: Path) -> Path:
        return Path(weights_dir) / "TripoSR"

    @staticmethod
    def _source(weights_dir: Path) -> Path:
        return Path(weights_dir).parent / "src" / "TripoSR"

    def available(self, weights_dir: Path) -> bool:
        w = self._weights(weights_dir)
        return ((w / "model.ckpt").is_file() and (w / "config.yaml").is_file()
                and (self._source(weights_dir) / "tsr" / "system.py").is_file()
                and (Path(weights_dir) / "u2net" / "u2net.onnx").is_file())

    def load(self, weights_dir: Path, device: str, options: dict) -> None:
        import torch

        shims = str(Path(__file__).resolve().parent.parent / "_shims")
        if shims not in sys.path:
            sys.path.append(shims)       # last: a real package of the same name wins
        source = str(self._source(weights_dir))
        if source not in sys.path:
            sys.path.insert(0, source)
        from tsr.system import TSR

        from omegaconf import OmegaConf

        self.device = device if device not in ("", "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
        # As TSR.from_pretrained, but the model goes to the device first and the checkpoint is memory-mapped, so the
        # 1.7 GB of weights are never held twice in memory (the review server's memory limit is shared with the app).
        w = self._weights(weights_dir)
        cfg = OmegaConf.load(str(w / "config.yaml"))
        OmegaConf.resolve(cfg)
        model = TSR(cfg)
        model.to(self.device)
        state = torch.load(str(w / "model.ckpt"), map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state)
        del state
        model.renderer.set_chunk_size(CHUNK)
        model.eval()
        self.model = model

    def generate(self, image, options: dict):
        import torch
        from PIL import Image

        from gen3d.backends import Mesh

        rgba = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4]) * GREY
        picture = Image.fromarray((rgb * 255.0).astype(np.uint8)).resize((INPUT_SIDE, INPUT_SIDE), Image.LANCZOS)
        resolution = RESOLUTION.get(int(options.get("detail", 2)), RESOLUTION[2])
        with torch.no_grad():
            codes = self.model([picture], device=self.device)
            mesh = self.model.extract_mesh(codes, True, resolution=resolution)[0]
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.uint32)
        c = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
        # TripoSR's object looks along +x with z up: turn it to +y up, +z towards the viewer
        v = np.ascontiguousarray(np.stack([v[:, 1], v[:, 2], v[:, 0]], axis=1))
        return Mesh(vertices=v, faces=f, colors=c)

    def close(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - closing never fails a job
            pass


def create() -> TripoSR:
    return TripoSR()
