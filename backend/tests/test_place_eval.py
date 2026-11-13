"""scripts/place_eval.py (section 35): it never takes a live data folder for a snapshot, wherever it runs from."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import place_eval  # noqa: E402
import trait_eval  # noqa: E402


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_main_checkout_of_a_worktree_is_found(tmp_path):
    main, wt = tmp_path / "main", tmp_path / "wt"
    subprocess.run(["git", "init", "-q", str(main)], check=True, capture_output=True)
    admin = main / ".git" / "worktrees" / "wt"                       # a worktree, as git lays one out
    admin.mkdir(parents=True)
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {admin}\n")
    (admin / "commondir").write_text("../..\n")
    (admin / "gitdir").write_text(f"{wt / '.git'}\n")
    (admin / "HEAD").write_text("ref: refs/heads/wt\n")
    assert place_eval.main_checkout(wt) == main.resolve()
    assert place_eval.main_checkout(main) is None
    (main / "data").mkdir()
    (main / ".env").write_text("DATA_DIR=/srv/library\n")
    live = place_eval.live_dirs(trait_eval, [], repo=wt)
    assert (main / "data").resolve() in live and Path("/srv/library") in live


def test_a_live_folder_is_refused_wherever_the_script_runs_from(tmp_path, capsys):
    live = tmp_path / "library"
    prev = live / "previews"
    prev.mkdir(parents=True)
    (live / "cloud.db").write_bytes(b"")
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "cloud.db").write_bytes(b"")
    te = trait_eval
    assert place_eval.refusal(te, snap, prev, None, []) is None
    assert "live" in place_eval.refusal(te, live, prev, None, [])               # the folder of the previews it reads
    inside = live / "inside"
    inside.mkdir()
    (inside / "cloud.db").write_bytes(b"")
    assert "live" in place_eval.refusal(te, inside, prev, None, [])
    other = tmp_path / "elsewhere"                                              # a live folder that nobody listed
    (other / "thumbs").mkdir(parents=True)
    (other / "cloud.db").write_bytes(b"")
    assert "thumbs" in place_eval.refusal(te, other, prev, None, [])
    assert place_eval.refusal(te, snap, prev, prev / "report.json", [])         # the report never goes in there
    assert place_eval.refusal(te, snap, prev, live / "report.json", [])
    assert place_eval.refusal(te, snap, prev, snap / "report.json", []) is None
    assert place_eval.main(["--snapshot", str(live), "--previews", str(prev)]) == 2
    assert "refusing" in capsys.readouterr().err
