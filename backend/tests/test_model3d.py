"""3D heads of a person (app.analysis.model3d, app.api.model3d, docs/ARCHITECTURE.md section 36): the job end to
end with the stand-in generator (a real child process, no PyTorch), its limits, the stored model, and the routes."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.analysis import ages, looks, model3d
from app.config import REPO_ROOT, settings
from app.db import tx
from tests.helpers import make_face, make_item, make_person

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def groups(index: int) -> bytes:
    g = np.zeros(9, dtype=np.float32)
    g[index] = 1.0
    return ages.encode(g)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """The job's state and the remembered generator check are module globals: no test inherits another's."""
    monkeypatch.setattr(model3d, "_state", None)
    monkeypatch.setattr(model3d, "_proc", None)
    monkeypatch.setattr(model3d, "_check", None)
    stamps = iter(f"20261113T{n:06d}" for n in range(100, 100000))
    monkeypatch.setattr(model3d, "_stamp", lambda: next(stamps))
    yield


@pytest.fixture()
def fake(monkeypatch):
    monkeypatch.setattr(settings, "gen3d_fake", True)
    monkeypatch.setattr(settings, "gen3d_sandbox", "auto")


@pytest.fixture()
def no_generator(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gen3d_fake", False)
    monkeypatch.setattr(settings, "gen3d_generator", "auto")
    monkeypatch.setattr(settings, "gen3d_dir", tmp_path / "nothing-here")
    monkeypatch.setattr(settings, "gen3d_python", None)


def person_with_looks(conn, name: str = "Asha", young: int = 4, older: int = 3) -> tuple[int, list[int]]:
    """A person with two looks when no model reads hair: `young` photos at about 25 and `older` at about 45."""
    person = make_person(conn, name)
    items = []
    for k in range(young):
        item = make_item(conn, taken_at=f"2024-03-{k + 1:02d}T10:00:00.000Z")
        make_face(conn, item["id"], person, box=(0.35, 0.3, 0.25, 0.3), size_px=200 + k, age_groups=groups(3))
        items.append(item["id"])
    for k in range(older):
        item = make_item(conn, taken_at=f"2025-06-{k + 1:02d}T10:00:00.000Z")
        make_face(conn, item["id"], person, box=(0.35, 0.3, 0.25, 0.3), size_px=300 + k, age_groups=groups(5))
        items.append(item["id"])
    return person, items


def sleeper(*_args, **_kw) -> list[str]:
    return [sys.executable, "-c", "import time; print('progress: 0/2', flush=True); time.sleep(120)"]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie is not running
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


# ------------------------------------------------------------------ the job
def test_job_end_to_end_with_the_stand_in_generator(conn, fake):
    person, items = person_with_looks(conn)
    model3d.start(conn, person, {"max_looks": 6, "detail": 3, "framing": 2.2})
    assert model3d.running() is None
    doc = model3d.manifest(person)
    rev = doc["rev"]
    base = model3d.person_dir(person)
    assert (base / "current").read_text() == rev and {p.name for p in base.iterdir()} == {"current", rev}
    assert doc["person_id"] == person and doc["generator"] == {"name": "fake", "version": "1"}
    assert doc["options"] == {"max_looks": 6, "detail": 3, "framing": 2.2}
    assert doc["scanned"] == 7 and doc["usable"] == 7 and doc["seconds"] >= 0
    assert [(lk["key"], lk["age_band"], lk["photos"], lk["label"]) for lk in doc["looks"]] == [
        ("l1", "25 to 34", 4, "About 25"), ("l2", "45 to 59", 3, "About 45")]
    for lk in doc["looks"]:
        mesh, picture = base / rev / lk["mesh"], base / rev / lk["picture"]
        assert mesh.read_bytes()[:4] == b"glTF" and lk["bytes"] == mesh.stat().st_size and lk["vertices"] > 500
        with Image.open(picture) as im:
            assert im.size == (model3d.PICTURE_SIDE, model3d.PICTURE_SIDE) and im.format == "JPEG"
        assert lk["item_id"] in items and lk["faults"] == []
    assert doc["looks"][0]["face_id"] != doc["looks"][1]["face_id"]

    status = model3d.status(conn, person)
    assert status["available"] and status["state"] == "done" and status["generator"] == "fake"
    assert status["usable_photos"] == 7 and status["progress"] is None and status["error"] is None
    assert [lk["mesh_url"] for lk in status["looks"]] == [
        f"/files/model3d/{person}/{rev}/l1.glb", f"/files/model3d/{person}/{rev}/l2.glb"]
    assert status["looks"][0]["picture_url"] == f"/files/model3d/{person}/{rev}/l1.jpg"
    assert model3d.file_path(person, rev, "l1.glb") == base / rev / "l1.glb"
    assert model3d.file_path(person, rev, "model.json") is None and model3d.file_path(person, "nope", "l1.glb") is None
    assert model3d.file_path(person, rev, "../current") is None

    # again with one look: the new model replaces the old one
    model3d.start(conn, person, {"max_looks": 1})
    again = model3d.manifest(person)
    assert again["rev"] != rev and len(again["looks"]) == 1 and again["looks"][0]["photos"] == 4
    assert {p.name for p in base.iterdir()} == {"current", again["rev"]}
    assert model3d.file_path(person, rev, "l1.glb") is None  # the old model's files are gone and not served

    model3d.remove(conn, person)
    assert not base.exists() and model3d.status(conn, person)["state"] == "none"


def test_options_are_kept_inside_their_ranges():
    assert model3d.clean_options(None) == {"max_looks": 6, "detail": 2, "framing": 2.6}
    assert model3d.clean_options({"max_looks": 99, "detail": 0, "framing": 9}) == {"max_looks": 12, "detail": 1, "framing": 3.2}
    assert model3d.clean_options({"generator": "fake", "junk": 1})["generator"] == "fake"
    assert "generator" not in model3d.clean_options({"generator": "triposr"})


def test_status_says_why_nothing_can_be_made(conn, no_generator, monkeypatch):
    person, _items = person_with_looks(conn)
    status = model3d.status(conn, person)
    assert not status["available"] and "scripts/setup_3d.sh" in status["reason"] and status["state"] == "none"
    assert status["generator"] is None and status["usable_photos"] == 7 and status["looks"] == []
    with pytest.raises(model3d.GeneratorMissing):
        model3d.start(conn, person, {})
    assert model3d.running() is None

    monkeypatch.setattr(settings, "gen3d_fake", True)
    few = make_person(conn, "Ravi")
    make_face(conn, make_item(conn)["id"], few, size_px=200, age_groups=groups(3))
    status = model3d.status(conn, few)
    assert not status["available"] and "1 found, 3 needed" in status["reason"]
    with pytest.raises(model3d.NotEnoughPhotos):
        model3d.start(conn, few, {})


def test_the_real_generator_is_asked_once_and_remembered(conn, monkeypatch, tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setattr(settings, "gen3d_fake", False)
    monkeypatch.setattr(settings, "gen3d_python", python)
    monkeypatch.setattr(settings, "gen3d_dir", tmp_path)
    monkeypatch.setattr(settings, "gen3d_sandbox", "off")
    asked = []

    def answer(installed: bool):
        def command(args, work, fake):
            asked.append(list(args))
            info = {"backends": {"fake": True, "triposr": installed}, "auto": "triposr" if installed else None,
                    "device": "cuda", "torch": "2.15.0"}
            return [sys.executable, "-c", f"print('noise'); print({json.dumps(json.dumps(info))})"]
        return command

    monkeypatch.setattr(model3d, "_command", answer(True))
    assert model3d.generator() == {"installed": True, "name": "triposr", "device": "cuda", "reason": None}
    assert model3d.generator()["installed"] and len(asked) == 1               # remembered
    assert asked[0] == ["--check", "--weights-dir", str(tmp_path / "weights")]
    assert model3d.generator(refresh=True)["installed"] and len(asked) == 2
    monkeypatch.setattr(model3d, "_command", answer(False))
    assert model3d.generator(refresh=True) == {"installed": False, "name": None, "device": None, "reason": model3d.MISSING}
    monkeypatch.setattr(settings, "gen3d_generator", "triposr")               # a named backend that is not there
    assert not model3d.generator()["installed"]
    env = model3d._child_env()
    assert env["HF_HOME"] == str(tmp_path / "weights" / "hf") and env["U2NET_HOME"] == str(tmp_path / "weights" / "u2net")
    assert env["PYTHONUNBUFFERED"] == "1"


def test_only_one_job_at_a_time(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    monkeypatch.setattr(model3d, "_state", {"person_id": 999, "step": "Building heads", "done": 1, "total": 3,
                                            "started_at": "", "cancel": False, "remove": False})
    with pytest.raises(model3d.Busy):
        model3d.start(conn, person, {})
    status = model3d.status(conn, person)
    assert status["busy"] and status["state"] == "none" and status["progress"] is None
    assert model3d.cancel(person) is False            # not this person's job
    assert model3d.manifest(person) is None


def _start_in_thread(conn_factory, person: int) -> threading.Thread:
    def run() -> None:
        c = conn_factory()
        try:
            model3d.start(c, person, {})
        finally:
            c.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _wait_for_child(timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = model3d._proc
        job = model3d.running()
        if proc is not None and job is not None and job["step"] == model3d.STEP_BUILD:
            return proc
        time.sleep(0.05)
    raise AssertionError("the child did not start")


def test_cancel_stops_the_child_and_leaves_nothing(conn, fake, monkeypatch):
    from app.db import connect

    person, _items = person_with_looks(conn)
    monkeypatch.setattr(model3d, "_command", sleeper)
    t = _start_in_thread(connect, person)
    proc = _wait_for_child()
    status = model3d.status(conn, person)
    assert status["state"] == "running" and status["progress"] == {"step": "Building heads", "done": 0, "total": 2}
    with pytest.raises(model3d.Busy):
        model3d.start(conn, person, {})
    assert model3d.cancel(person) is True
    t.join(30)
    assert not t.is_alive() and not alive(proc.pid)
    assert model3d.running() is None and model3d.manifest(person) is None
    base = model3d.person_dir(person)
    assert not (base / "work").exists() and not (base / "error.json").exists()
    assert model3d.status(conn, person)["state"] == "none"


def test_the_stop_route_keeps_the_model_in_use(client, conn, fake, monkeypatch):
    from app.db import connect

    person, _items = person_with_looks(conn)
    url = f"/api/people/{person}/model3d"
    r = client.post(f"{url}/stop")                     # nothing is running
    assert r.status_code == 200 and r.json() == {"stopped": False} and "x-action-id" not in r.headers
    model3d.start(conn, person, {})                    # a model exists
    before = model3d.current_rev(person)
    assert before is not None
    monkeypatch.setattr(model3d, "_command", sleeper)
    t = _start_in_thread(connect, person)
    proc = _wait_for_child()
    body = client.get(url).json()
    assert body["state"] == "running" and len(body["looks"]) == 2      # the older model shows while the new one is made
    assert client.post(f"{url}/stop").json() == {"stopped": True}
    t.join(30)
    assert not t.is_alive() and not alive(proc.pid)
    assert model3d.current_rev(person) == before and client.get(url).json()["state"] == "done"
    assert client.get(body["looks"][0]["mesh_url"]).status_code == 200
    assert client.post("/api/people/99999/model3d/stop").status_code == 404


def test_removing_a_person_s_model_stops_their_job(conn, fake, monkeypatch):
    from app.db import connect

    person, _items = person_with_looks(conn)
    model3d.start(conn, person, {})                    # a model exists
    assert model3d.manifest(person) is not None
    monkeypatch.setattr(model3d, "_command", sleeper)
    t = _start_in_thread(connect, person)
    proc = _wait_for_child()
    model3d.remove(conn, person)
    t.join(30)
    assert not t.is_alive() and not alive(proc.pid)
    assert not model3d.person_dir(person).exists()


def test_a_child_that_fails_leaves_its_reason(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    real = model3d._command
    monkeypatch.setattr(model3d, "_command", lambda *a, **k: [
        sys.executable, "-c", "import sys; print('progress: 0/2', flush=True); sys.stderr.write('noise\\nthe weights are missing\\n'); sys.exit(4)"])
    model3d.start(conn, person, {})
    status = model3d.status(conn, person)
    assert status["state"] == "error" and "the weights are missing" in status["error"] and status["looks"] == []
    assert model3d.manifest(person) is None and not (model3d.person_dir(person) / "work").exists()
    monkeypatch.setattr(model3d, "_command", real)     # the stand-in generator again: the error goes
    model3d.start(conn, person, {})
    status = model3d.status(conn, person)
    assert status["state"] == "done" and status["error"] is None


def test_a_child_that_cannot_start_is_an_error(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    monkeypatch.setattr(model3d, "_command", lambda *a, **k: ["/nonexistent/python", "-m", "gen3d"])
    model3d.start(conn, person, {})
    assert "could not be started" in model3d.status(conn, person)["error"]


def test_the_memory_limit_stops_the_child(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    monkeypatch.setattr(model3d, "_command", sleeper)
    monkeypatch.setattr(settings, "gen3d_max_rss_mb", 1)
    started = time.monotonic()
    model3d.start(conn, person, {})
    assert time.monotonic() - started < 60
    status = model3d.status(conn, person)
    assert status["state"] == "error" and "more memory" in status["error"] and "1 MB" in status["error"]
    assert model3d._proc is None


def test_the_time_limit_stops_the_child(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    monkeypatch.setattr(model3d, "_command", sleeper)
    monkeypatch.setattr(settings, "gen3d_look_timeout_s", 1.0)
    model3d.start(conn, person, {})
    assert "took longer" in model3d.status(conn, person)["error"]


def test_tree_memory_counts_the_descendants():
    import subprocess

    child = subprocess.Popen([sys.executable, "-c", (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; x = list(range(2_500_000)); print(1, flush=True); time.sleep(60)'], stdout=subprocess.PIPE)\n"
        "p.stdout.readline(); print('ready', flush=True); time.sleep(60)\n")], stdout=subprocess.PIPE, process_group=0)
    try:
        assert child.stdout.readline().strip() == b"ready"
        tree = model3d._tree(child.pid)
        assert child.pid in tree and len(tree) == 2
        assert model3d.tree_rss_mb(child.pid) > 60          # the grandchild's list (about 90 MB) is counted
        model3d._stop_child(child)
        assert child.poll() is not None and not any(alive(p) for p in tree)
    finally:
        if child.poll() is None:
            child.kill()


def test_a_look_that_fails_does_not_stop_the_others(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    real = looks.write_picture

    def spoil(conn_, face_id, path, framing=2.6):
        ok = real(conn_, face_id, path, framing)
        if path.name == "l1.png":
            path.write_bytes(b"not a picture")
        return ok

    monkeypatch.setattr(looks, "write_picture", spoil)
    model3d.start(conn, person, {})
    doc = model3d.manifest(person)
    assert [lk["key"] for lk in doc["looks"]] == ["l2"] and model3d.status(conn, person)["state"] == "done"

    monkeypatch.setattr(looks, "write_picture", lambda conn_, face_id, path, framing=2.6: (path.write_bytes(b"x"), True)[1])
    model3d.start(conn, person, {})
    status = model3d.status(conn, person)
    assert status["state"] == "done"                                   # the earlier model stays in use
    assert "No head could be built" in (model3d._error(person) or "")
    assert [lk["key"] for lk in status["looks"]] == ["l2"]


def test_no_readable_photo_is_an_error(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    monkeypatch.setattr(looks, "write_picture", lambda *a, **k: False)
    model3d.start(conn, person, {})
    assert "could not be read" in model3d.status(conn, person)["error"]
    nobody = make_person(conn, "Meera")
    for _ in range(3):
        make_face(conn, make_item(conn)["id"], nobody, size_px=200)    # no age read: no look can be told
    model3d.start(conn, nobody, {})
    assert "No look could be told" in model3d.status(conn, nobody)["error"]


def test_looks_whose_photo_left_the_library_are_not_listed(conn, fake):
    person, _items = person_with_looks(conn)
    model3d.start(conn, person, {})
    doc = model3d.manifest(person)
    first, second = doc["looks"]
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-11-13T00:00:00.000Z' WHERE id = ?", (first["item_id"],))
    status = model3d.status(conn, person)
    assert [lk["key"] for lk in status["looks"]] == ["l2"] and status["state"] == "done"
    with tx(conn):
        conn.execute("UPDATE items SET nsfw = 1 WHERE id = ?", (second["item_id"],))
    status = model3d.status(conn, person)
    assert status["looks"] == [] and status["state"] == "none" and status["generated_at"] is None
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = NULL WHERE id = ?", (first["item_id"],))
    assert [lk["key"] for lk in model3d.status(conn, person)["looks"]] == ["l1"]   # restored from the trash: back


def test_sweep_removes_the_models_of_people_who_are_gone(conn, fake):
    asha, _ = person_with_looks(conn, "Asha")
    ravi, _ = person_with_looks(conn, "Ravi")
    model3d.start(conn, asha, {})
    model3d.start(conn, ravi, {})
    (model3d.root() / "notes.txt").write_text("not a person's folder")
    assert model3d.sweep(conn) == 0
    with tx(conn):
        conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (asha,))
        conn.execute("DELETE FROM persons WHERE id = ?", (asha,))
    assert model3d.sweep(conn) == 1
    assert not model3d.person_dir(asha).exists() and model3d.manifest(ravi) is not None
    assert (model3d.root() / "notes.txt").exists()
    from app import worker

    with tx(conn):
        conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (ravi,))
        conn.execute("DELETE FROM persons WHERE id = ?", (ravi,))
    assert worker.housekeeping(conn).get("model3d") == 1               # housekeeping runs the sweep
    assert not model3d.person_dir(ravi).exists()


# ------------------------------------------------------------------ the sandbox
def test_the_sandbox_is_for_the_real_generator_unless_it_is_demanded(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gen3d_python", tmp_path / "python")
    monkeypatch.setattr(model3d, "sandbox_bin", lambda: "/usr/bin/bwrap")
    for mode, fake_run, wanted in (("auto", False, True), ("auto", True, False), ("on", True, True), ("off", False, False)):
        monkeypatch.setattr(settings, "gen3d_sandbox", mode)
        monkeypatch.setattr(settings, "gen3d_fake", fake_run)
        assert model3d.sandbox_wanted(fake_run) is wanted
        cmd = model3d._command(["--job", "/w/job.json"], Path("/w"), fake_run)
        assert cmd[:2] == ["/bin/sh", "-c"] and "oom_score_adj" in cmd[2] and cmd[-3:] == ["gen3d", "--job", "/w/job.json"]
        assert ("/usr/bin/bwrap" in cmd) is wanted
        if wanted:
            inner = cmd[cmd.index("/usr/bin/bwrap"):]
            for flag in ("--unshare-net", "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--new-session"):
                assert flag in inner
            assert inner[inner.index("--bind"):inner.index("--bind") + 3] == ["--bind", "/w", "/w"]
            assert inner[1:4] == ["--ro-bind", "/", "/"]
            env = {inner[i + 1]: inner[i + 2] for i, word in enumerate(inner) if word == "--setenv"}
            assert env == {"HOME": "/tmp", "XDG_CACHE_HOME": "/tmp/cache", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "MPLCONFIGDIR": "/tmp/mpl", "NUMBA_CACHE_DIR": "/tmp/numba"}
    monkeypatch.setattr(model3d, "sandbox_bin", lambda: None)
    monkeypatch.setattr(settings, "gen3d_sandbox", "auto")
    assert model3d.sandbox_wanted(False) is False                      # no bwrap: auto runs without
    monkeypatch.setattr(settings, "gen3d_sandbox", "on")
    with pytest.raises(model3d.GeneratorMissing):
        model3d.sandbox_wanted(True)
    monkeypatch.setattr(settings, "gen3d_fake", True)
    assert "bwrap" in model3d.generator()["reason"]


@pytest.mark.skipif(model3d.sandbox_bin() is None, reason="bwrap is not installed")
def test_the_stand_in_job_runs_through_the_sandbox(conn, fake, monkeypatch):
    import subprocess

    person, _items = person_with_looks(conn)
    monkeypatch.setattr(settings, "gen3d_sandbox", "on")
    seen = []
    real = model3d._command
    monkeypatch.setattr(model3d, "_command", lambda args, work, fake_run: seen.append(real(args, work, fake_run)) or seen[-1])
    model3d.start(conn, person, {})
    assert model3d.status(conn, person)["state"] == "done", model3d._error(person)
    assert any(model3d.sandbox_bin() in cmd and "--unshare-net" in cmd for cmd in seen)

    # what the sandbox means: only the work folder can be written, and there is no network
    work = model3d.person_dir(person) / "probe"
    work.mkdir(parents=True)
    probe = (
        "import json, socket, sys\n"
        "out = {}\n"
        "for name, path in (('work', sys.argv[1] + '/x'), ('repo', sys.argv[2] + '/x'), ('home', sys.argv[3] + '/x')):\n"
        "    try:\n        open(path, 'w').close(); out[name] = True\n    except OSError:\n        out[name] = False\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), 2).close(); out['net'] = True\n"
        "except OSError:\n    out['net'] = False\n"
        "print(json.dumps(out))\n")
    done = subprocess.run([*model3d.sandbox_prefix(work), sys.executable, "-c", probe, str(work), str(REPO_ROOT),
                           str(Path.home())], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"work": True, "repo": False, "home": False, "net": False}
    assert (work / "x").exists() and not (REPO_ROOT / "x").exists()


# ------------------------------------------------------------------ the routes
def test_routes(client, conn, fake):
    person, _items = person_with_looks(conn)
    url = f"/api/people/{person}/model3d"
    r = client.get(url)
    assert r.status_code == 200 and r.json()["state"] == "none" and r.json()["available"] is True
    assert r.json()["usable_photos"] == 7 and r.json()["generator"] == "fake"

    assert client.post(url, json={"max_looks": 99}).status_code == 422
    assert client.post(url, json={"detail": 0}).status_code == 422
    assert client.post(url, json={"framing": 9.0}).status_code == 422
    r = client.post(url, json={"max_looks": 2, "detail": 1, "framing": 2.0})
    assert r.status_code == 202 and r.json() == {"started": True}
    assert "x-action-id" not in r.headers                                  # not an undoable owner action
    body = client.get(url).json()
    assert body["state"] == "done" and body["options"] == {"max_looks": 2, "detail": 1, "framing": 2.0}
    assert [lk["key"] for lk in body["looks"]] == ["l1", "l2"] and body["looks"][0]["label"] == "About 25"
    assert body["generated_at"] and body["error"] is None and body["progress"] is None

    mesh = client.get(body["looks"][0]["mesh_url"])
    assert mesh.status_code == 200 and mesh.headers["content-type"] == "model/gltf-binary"
    assert mesh.headers["cache-control"] == "private, max-age=31536000, immutable" and mesh.content[:4] == b"glTF"
    assert mesh.headers["x-content-type-options"] == "nosniff"
    picture = client.get(body["looks"][0]["picture_url"])
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
    assert client.head(body["looks"][1]["mesh_url"]).status_code == 200
    rev = body["looks"][0]["mesh_url"].split("/")[4]
    for bad in (f"/files/model3d/{person}/{rev}/model.json", f"/files/model3d/{person}/{rev}/l9.glb",
                f"/files/model3d/{person}/20200101T000000/l1.glb", f"/files/model3d/{person}/{rev}/L1.GLB",
                f"/files/model3d/999/{rev}/l1.glb"):
        assert client.get(bad).status_code == 404, bad

    # a look whose photo went to the trash is neither listed nor served
    item_uid = body["looks"][0]["item_uid"]
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = '2026-11-13T00:00:00.000Z' WHERE uid = ?", (item_uid,))
    assert [lk["key"] for lk in client.get(url).json()["looks"]] == ["l2"]
    assert client.get(body["looks"][0]["mesh_url"]).status_code == 404
    assert client.get(body["looks"][0]["picture_url"]).status_code == 404
    assert client.get(body["looks"][1]["mesh_url"]).status_code == 200

    r = client.delete(url)
    assert r.status_code == 200 and "x-action-id" not in r.headers
    assert client.get(url).json()["state"] == "none"
    assert client.get(body["looks"][1]["mesh_url"]).status_code == 404


def test_routes_refuse(client, conn, no_generator, monkeypatch):
    person, _items = person_with_looks(conn)
    url = f"/api/people/{person}/model3d"
    body = client.get(url).json()
    assert body["available"] is False and "setup_3d.sh" in body["reason"]
    r = client.post(url, json={})
    assert r.status_code == 409 and r.json()["detail"] == "generator_missing"

    monkeypatch.setattr(settings, "gen3d_fake", True)
    few = make_person(conn, "Ravi")
    make_face(conn, make_item(conn)["id"], few, size_px=200, age_groups=groups(3))
    r = client.post(f"/api/people/{few}/model3d")
    assert r.status_code == 422 and r.json()["detail"] == "not_enough_photos"

    monkeypatch.setattr(model3d, "_state", {"person_id": 999, "step": "Building heads", "done": 0, "total": 1,
                                            "started_at": "", "cancel": False, "remove": False})
    r = client.post(url, json={})
    assert r.status_code == 409 and r.json()["detail"] == "busy"
    assert client.get(url).json()["busy"] is True
    monkeypatch.setattr(model3d, "_state", None)

    with tx(conn):
        pet = int(conn.execute("INSERT INTO persons(name, name_key, kind, species, created_at, updated_at) "
                               "VALUES ('Biscuit', 'biscuit', 'pet', 'dog', '', '')").lastrowid)
    for method in (client.get, client.post, client.delete):
        assert method(f"/api/people/{pet}/model3d").status_code == 404      # pets have no 3D heads
        assert method("/api/people/99999/model3d").status_code == 404

    from fastapi.testclient import TestClient

    anon = TestClient(client.app)                                            # the same running app, no session
    assert anon.get(url).status_code == 401
    assert anon.post(url, json={}, headers={"X-Requested-With": "pcc"}).status_code == 401
    assert anon.get(f"/files/model3d/{person}/20261113T000100/l1.glb").status_code == 401


def test_a_deleted_person_s_model_is_not_served(client, conn, fake):
    person, _items = person_with_looks(conn)
    url = f"/api/people/{person}/model3d"
    assert client.post(url, json={}).status_code == 202
    mesh_url = client.get(url).json()["looks"][0]["mesh_url"]
    assert client.delete(f"/api/people/{person}").status_code == 200
    assert client.get(url).status_code == 404 and client.get(mesh_url).status_code == 404
    assert model3d.person_dir(person).exists()                              # until housekeeping sweeps it
    assert model3d.sweep(conn) == 1 and not model3d.person_dir(person).exists()
