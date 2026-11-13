"""The generator backends and the interface they implement.

A backend's module is imported only when it is asked for, so the fake one never imports PyTorch and a real one that
is not installed costs nothing. Importing a module must stay cheap (no PyTorch at import time: the check that runs
for status reads imports it). Each module exposes `BACKEND`, or a `create()` that returns it (called once): an
object with:

    name: str
    version: str
    available(weights_dir: Path) -> bool           its weights (and its libraries) are there
    load(weights_dir: Path, device: str, options: dict) -> None
    generate(image: PIL.Image.Image, options: dict) -> Mesh
        image: a square picture of the head and shoulders, the face centred. RGB, or RGBA with the background cut out
        (alpha 0 there) when gen3d.prep ran before. options: the job's options plus "out_dir" and the look's "key"
        (a backend may write a debug file there when options["debug"] is true).
        The mesh: +Y up, +Z towards the viewer, any scale (gen3d.mesh.finish centres and scales it). Its colours,
        when it has any: one sRGB byte triple per vertex, as in the picture.
    close() -> None
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol

from gen3d.mesh import Mesh

__all__ = ["Backend", "Mesh", "ORDER", "REGISTRY", "get", "is_available", "pick"]

REGISTRY: dict[str, str] = {
    "fake": "gen3d.backends.fake",
    "triposr": "gen3d.backends.triposr",
}
ORDER: tuple[str, ...] = ("triposr",)  # what "auto" tries; the first one available wins


class Backend(Protocol):
    name: str
    version: str

    def available(self, weights_dir: Path) -> bool: ...
    def load(self, weights_dir: Path, device: str, options: dict[str, Any]) -> None: ...
    def generate(self, image: Any, options: dict[str, Any]) -> Mesh: ...
    def close(self) -> None: ...


class UnknownBackend(KeyError):
    pass


_made: dict[str, Backend] = {}


def get(name: str) -> Backend:
    """The backend of that name: its module's `BACKEND`, or what its `create()` returns (made once).
    UnknownBackend for a name outside REGISTRY; ImportError when its module (or a library it needs) is not
    installed, or when the module offers neither."""
    module_name = REGISTRY.get(name)
    if module_name is None:
        raise UnknownBackend(name)
    module = importlib.import_module(module_name)
    backend = getattr(module, "BACKEND", None)
    if backend is not None:
        return backend
    if name not in _made:
        create = getattr(module, "create", None)
        if create is None:
            raise ImportError(f"{module_name} has neither BACKEND nor create()")
        _made[name] = create()
    return _made[name]


def is_available(name: str, weights_dir: Path) -> bool:
    """True when the backend imports and says its weights are there. Never raises."""
    try:
        return bool(get(name).available(Path(weights_dir)))
    except Exception:  # noqa: BLE001 - not installed, half installed, or broken: not available
        return False


def pick(weights_dir: Path) -> str | None:
    """The backend "auto" means: the first of ORDER that is available, or None."""
    for name in ORDER:
        if is_available(name, weights_dir):
            return name
    return None
