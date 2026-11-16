"""How overlapping boxes stack on a photo (docs/ARCHITECTURE.md section 14): frontend/scripts/check-boxes.mjs runs the
viewer's own stacking rule in node, when the frontend's packages are installed. No browser."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_a_smaller_box_is_drawn_above_a_larger_one(tmp_path):
    front = REPO / "frontend"
    node = shutil.which("node")
    if node is None or not (front / "node_modules" / "esbuild").is_dir():
        pytest.skip("node or the frontend's packages are not installed")
    done = subprocess.run([node, str(front / "scripts" / "check-boxes.mjs"), str(tmp_path)],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-3000:]
    assert "9 groups passed" in done.stdout
