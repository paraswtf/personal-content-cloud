"""The 3D head generator (docs/ARCHITECTURE.md section 36): a separate program with its own Python environment.

The app never imports this package. It starts `python -m gen3d --job JOB.json` as a child process while a generation
runs, with the generator's own Python (settings GEN3D_PYTHON), so PyTorch stays out of the server and the GPU and the
memory are free the rest of the time. See gen3d/__main__.py for the job contract and gen3d/backends/__init__.py for
the interface a backend implements. The `fake` backend needs only numpy and Pillow.
"""

__version__ = "1"
