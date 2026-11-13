"""Stand-in for the rembg package: TripoSR's utils import it at the top, but the generator cuts backgrounds out
itself (gen3d.prep), so nothing here is ever called."""


def new_session(*args, **kwargs):
    raise RuntimeError("rembg is not installed; gen3d.prep cuts the background out")


def remove(*args, **kwargs):
    raise RuntimeError("rembg is not installed; gen3d.prep cuts the background out")
