"""Stand-in for the imageio package: TripoSR's utils import it for saving videos, which the generator never does."""


def mimsave(*args, **kwargs):
    raise RuntimeError("imageio is not installed")
