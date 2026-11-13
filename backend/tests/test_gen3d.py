"""The generator shell (the gen3d package at the repo root, docs/ARCHITECTURE.md section 36): the mesh clean-up, the
GLB writer read back, the fake backend, and `python -m gen3d` end to end with it. No PyTorch anywhere. The last test
hands the written files to the viewer's own code and three.js, in node, when the frontend's packages are installed."""
from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gen3d import backends, glb  # noqa: E402
from gen3d.backends.fake import BACKEND as FAKE  # noqa: E402
from gen3d.mesh import BadMesh, Mesh, components, finish, signed_volume  # noqa: E402


def _tetra(offset=(0.0, 0.0, 0.0), scale=1.0) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) * scale + np.asarray(offset, np.float32)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)  # outward
    return v, f


def _picture(path: Path, skin=(200, 150, 120), hair=(30, 20, 10)) -> Path:
    im = Image.new("RGB", (128, 128), skin)
    im.paste(Image.new("RGB", (128, 30), hair), (0, 0))
    im.save(path)
    return path


def _run(args: list[str], cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "gen3d", *args], cwd=str(cwd), capture_output=True, text=True,
                          timeout=120, check=False)


def _job(tmp_path: Path, looks: list[dict], **extra) -> Path:
    job = {"out_dir": str(tmp_path / "out"), "weights_dir": str(tmp_path / "weights"), "generator": "fake",
           "device": "auto", "options": {"detail": 2, "seed": 0}, "looks": looks, **extra}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


# ------------------------------------------------------------------ mesh
def test_finish_centres_scales_and_gives_unit_normals():
    v, f = _tetra(offset=(5, 5, 5), scale=3.0)
    m = finish(Mesh(vertices=v, faces=f, colors=np.full((4, 3), 200, np.uint8)))
    assert np.allclose(m.vertices.min(axis=0) + m.vertices.max(axis=0), 0, atol=1e-6)  # centred
    assert np.isclose(float((m.vertices.max(axis=0) - m.vertices.min(axis=0)).max()), 1.0)  # a unit box
    assert m.faces.dtype == np.uint32 and m.vertices.dtype == np.float32
    assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-5)
    assert m.colors.shape == (4, 3)


def test_finish_turns_an_inside_out_mesh_and_drops_loose_bits():
    v, f = _tetra()
    inside_out = f[:, [0, 2, 1]]
    assert signed_volume(v, inside_out) < 0
    assert signed_volume(finish(Mesh(v, inside_out)).vertices, finish(Mesh(v, inside_out)).faces.astype(np.int64)) > 0
    # a big part (a fan of 400 triangles) and a speck of 4, which is 1 percent of it
    ring = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 401)[:-1]], dtype=np.float32)
    fan_v = np.vstack([[0.0, 0.0, 0.5], ring])
    fan_f = np.array([[0, 1 + k, 1 + (k + 1) % 400] for k in range(400)], dtype=np.uint32)
    speck_v, speck_f = _tetra(offset=(9, 9, 9), scale=0.01)
    both = Mesh(np.vstack([fan_v, speck_v]).astype(np.float32), np.vstack([fan_f, speck_f + len(fan_v)]))
    labels = components(both.faces.astype(np.int64), both.vertices.shape[0])
    assert len(set(labels.tolist())) == 2
    cleaned = finish(both)
    assert cleaned.vertices.shape[0] == len(fan_v) and cleaned.faces.shape[0] == 400  # the speck is gone
    # a part of 2 percent or more stays (the stand-in head's neck is one)
    part_v, part_f = _tetra(offset=(3, 3, 3))
    kept = finish(Mesh(np.vstack([fan_v[:101], part_v]).astype(np.float32),
                       np.vstack([np.array([[0, 1 + k, 1 + (k + 1) % 100] for k in range(100)], np.uint32), part_f + 101])))
    assert kept.faces.shape[0] == 104


def test_finish_refuses_what_is_not_a_mesh():
    v, f = _tetra()
    with pytest.raises(BadMesh):
        finish(Mesh(v, np.array([[0, 1, 9]], dtype=np.uint32)))
    with pytest.raises(BadMesh):
        finish(Mesh(np.zeros((0, 3), np.float32), f))
    with pytest.raises(BadMesh):
        finish(Mesh(v * np.nan, f))
    with pytest.raises(BadMesh):
        finish(Mesh(v, f, colors=np.zeros((2, 3), np.uint8)))


# ------------------------------------------------------------------ the GLB
def test_glb_round_trip(tmp_path):
    v, f = _tetra()
    m = finish(Mesh(v, f, colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 8, 7]], np.uint8)))
    path = tmp_path / "t.glb"
    size = glb.write(path, m)
    data = path.read_bytes()
    assert size == len(data) and len(data) % 4 == 0
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    back = glb.read(path)
    assert np.allclose(back["positions"], m.vertices) and np.allclose(back["normals"], m.normals)
    assert np.array_equal(back["indices"].reshape(-1, 3), m.faces)
    assert np.array_equal(back["colors"][:, :3], m.colors) and (back["colors"][:, 3] == 255).all()
    doc = back["json"]
    assert doc["asset"]["version"] == "2.0"
    pos = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert pos["count"] == 4 and np.allclose(pos["min"], m.vertices.min(axis=0)) and np.allclose(pos["max"], m.vertices.max(axis=0))
    colour = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]]
    assert colour["normalized"] is True and colour["type"] == "VEC4" and colour["componentType"] == 5123
    # glTF vertex colours are linear: the file holds 16-bit linear values, not the picture's sRGB bytes
    view = doc["bufferViews"][colour["bufferView"]]
    stored = np.frombuffer(data, dtype="<u2", count=16, offset=len(data) - doc["buffers"][0]["byteLength"]
                           + view["byteOffset"]).reshape(4, 4)
    assert np.array_equal(stored[:, :3], glb.to_linear16(m.colors)) and (stored[:, 3] == 65535).all()
    assert glb.to_linear16(np.array([255, 0, 9], np.uint8)).tolist() == [65535, 0, round(9 / 255 / 12.92 * 65535)]
    for view in doc["bufferViews"]:
        assert view["byteOffset"] % 4 == 0  # vertex data sits on 4-byte boundaries
    assert doc["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] == 0.9
    assert "extensions" not in doc["materials"][0]  # lit, not KHR_materials_unlit


def test_glb_colours_round_trip_every_byte():
    every = np.arange(256, dtype=np.uint8)
    linear = glb.to_linear16(every)
    assert linear[0] == 0 and linear[255] == 65535 and (np.diff(linear.astype(int)) > 0).all()
    assert linear[128] == round(((128 / 255 + 0.055) / 1.055) ** 2.4 * 65535)  # mid grey is 21.6% linear
    assert np.array_equal(glb.to_srgb8(linear), every)


def test_glb_without_colours(tmp_path):
    v, f = _tetra()
    glb.write(tmp_path / "plain.glb", finish(Mesh(v, f)))
    back = glb.read(tmp_path / "plain.glb")
    assert back["colors"] is None and back["positions"].shape == (4, 3)


# ------------------------------------------------------------------ the backends
def test_registry_and_the_fake_backend(tmp_path):
    assert set(backends.REGISTRY) == {"fake", "triposr"} and backends.ORDER == ("triposr",)
    assert backends.get("fake") is FAKE and backends.is_available("fake", tmp_path)
    assert not backends.is_available("triposr", tmp_path)  # not installed here: False, never an exception
    assert backends.pick(tmp_path) is None
    with pytest.raises(backends.UnknownBackend):
        backends.get("nope")
    rgb = Image.open(_picture(tmp_path / "a.png"))
    one, two = FAKE.generate(rgb, {}), FAKE.generate(rgb.convert("RGBA"), {})  # RGB or RGBA, the same head
    assert np.array_equal(one.vertices, two.vertices) and np.array_equal(one.colors, two.colors)
    assert 1500 <= one.faces.shape[0] <= 2500
    m = finish(one)
    assert m.faces.shape[0] == one.faces.shape[0]  # the neck is not a loose bit
    # +Y up, +Z towards the viewer: the face (skin) looks along +Z, the hair is on top and at the back
    face = (m.vertices[:, 2] > 0.25) & (np.abs(m.vertices[:, 1] - 0.1) < 0.1)  # the middle of the front
    top = m.vertices[:, 1] > 0.42
    back = (m.vertices[:, 2] < -0.25) & (m.vertices[:, 1] > 0.1)
    assert face.sum() > 20 and top.sum() > 20 and back.sum() > 20
    assert np.abs(m.colors[face].astype(int) - [200, 150, 120]).max() < 12
    assert np.abs(m.colors[top].astype(int) - [30, 20, 10]).max() < 12
    assert np.abs(m.colors[back].astype(int) - [30, 20, 10]).max() < 12
    outward = np.einsum("ij,ij->i", m.vertices - m.vertices.mean(axis=0), m.normals) > 0
    assert outward.mean() > 0.95
    other = FAKE.generate(Image.open(_picture(tmp_path / "b.png", skin=(90, 120, 200))), {})
    assert not np.array_equal(one.colors, other.colors)  # the colours come from the picture


def test_a_backend_module_may_offer_create_instead(tmp_path, monkeypatch):
    (tmp_path / "pcc_test_backend_made.py").write_text(
        "made = 0\n"
        "class One:\n"
        "    name, version = 'made', '2'\n"
        "    def available(self, weights_dir):\n        return True\n"
        "def create():\n"
        "    global made\n    made += 1\n    return One()\n", encoding="utf-8")
    (tmp_path / "pcc_test_backend_bare.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(backends.REGISTRY, "made", "pcc_test_backend_made")
    monkeypatch.setitem(backends.REGISTRY, "bare", "pcc_test_backend_bare")
    monkeypatch.setattr(backends, "_made", {})
    one = backends.get("made")
    assert one.name == "made" and backends.get("made") is one          # made once
    assert sys.modules["pcc_test_backend_made"].made == 1 and backends.is_available("made", tmp_path)
    with pytest.raises(ImportError):
        backends.get("bare")
    assert not backends.is_available("bare", tmp_path)
    for name in ("pcc_test_backend_made", "pcc_test_backend_bare"):
        sys.modules.pop(name, None)


# ------------------------------------------------------------------ python -m gen3d
def test_job_builds_every_look_it_can(tmp_path):
    job = _job(tmp_path, [{"key": "l1", "image": str(_picture(tmp_path / "l1.png"))},
                          {"key": "l2", "image": str(tmp_path / "missing.png")},
                          {"key": "l3", "image": str(_picture(tmp_path / "l3.png", skin=(90, 120, 200)))}])
    done = _run(["--job", str(job)])
    assert done.returncode == 0, done.stderr
    lines = done.stdout.split("\n")
    assert lines[0] == "progress: 0/3" and "progress: 3/3" in lines
    assert any(ln.startswith("look: l1 ok ") for ln in lines) and any(ln.startswith("look: l2 error ") for ln in lines)
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["generator"] == {"name": "fake", "version": "1"}
    by_key = {r["key"]: r for r in result["looks"]}
    assert by_key["l1"]["ok"] and by_key["l1"]["mesh"] == "l1.glb" and by_key["l1"]["faces"] > 1000
    assert not by_key["l2"]["ok"] and by_key["l2"]["error"] and by_key["l2"]["mesh"] is None
    assert glb.read(tmp_path / "out" / "l3.glb")["positions"].shape[0] == by_key["l3"]["vertices"]
    assert not (tmp_path / "out" / "l2.glb").exists()
    # the picture is laid over the front of the stand-in head: a second part with the picture as its texture
    row, back = by_key["l1"], glb.read(tmp_path / "out" / "l1.glb")
    assert row["textured"] is True and row["note"] is None and row["overlap"] is None and row["overlap_head"] is None
    assert back["photo"] is not None and back["photo"]["indices"].size > 300 and back["texture"][:3] == b"\xff\xd8\xff"
    with Image.open(io.BytesIO(back["texture"])) as texture:
        assert list(texture.size) == row["texture"] and max(texture.size) <= 128   # the picture's own pixels, never more
    assert back["photo"]["alpha"].max() == 1.0 and back["photo"]["uv"].min() >= 0.0 and back["photo"]["uv"].max() <= 1.0
    assert by_key["l2"]["textured"] is False and by_key["l2"]["texture"] is None
    # deterministic: the same picture, the same bytes
    first = (tmp_path / "out" / "l1.glb").read_bytes()
    assert _run(["--job", str(job)]).returncode == 0
    assert (tmp_path / "out" / "l1.glb").read_bytes() == first


def test_job_options_for_the_finishing(tmp_path):
    picture = str(_picture(tmp_path / "l1.png"))
    # texture 0: a colour per vertex and nothing else, the file a head always was
    job = _job(tmp_path, [{"key": "l1", "image": picture}], options={"detail": 2, "smoothing": 0, "texture": 0})
    assert _run(["--job", str(job)]).returncode == 0
    row = json.loads((tmp_path / "out" / "result.json").read_text())["looks"][0]
    plain = glb.read(tmp_path / "out" / "l1.glb")
    assert row["ok"] and row["textured"] is False and row["texture"] is None and row["note"] is None
    assert plain["photo"] is None and len(plain["json"]["meshes"][0]["primitives"]) == 1
    # the part of the picture that is the photo itself: nothing below row 80 of 128 reaches the texture
    job = _job(tmp_path, [{"key": "l1", "image": picture, "inside": [0, 0, 128, 80]}])
    assert _run(["--job", str(job)]).returncode == 0
    row = json.loads((tmp_path / "out" / "result.json").read_text())["looks"][0]
    assert row["textured"] is True and row["texture"][1] <= 80
    whole = _job(tmp_path, [{"key": "l1", "image": picture, "inside": None}])        # not known: all of the picture
    assert _run(["--job", str(whole)]).returncode == 0
    assert json.loads((tmp_path / "out" / "result.json").read_text())["looks"][0]["texture"][1] > 80
    # a picture that shows nothing of the photo still gives its head, without the layer, and says why
    job = _job(tmp_path, [{"key": "l1", "image": picture, "inside": [0, 0, 0, 0]}])
    assert _run(["--job", str(job)]).returncode == 0
    row = json.loads((tmp_path / "out" / "result.json").read_text())["looks"][0]
    assert row["ok"] and row["textured"] is False and "nothing of the photo" in row["note"]
    # the relief's options: the stand-in never runs it, whatever they say, and a memory limit is taken
    job = _job(tmp_path, [{"key": "l1", "image": picture}], options={"detail": 2, "relief": 1, "framing": 3.2},
               max_rss_mb=5000)
    assert _run(["--job", str(job)]).returncode == 0
    row = json.loads((tmp_path / "out" / "result.json").read_text())["looks"][0]
    assert row["ok"] and row["textured"] is True and row["relief"] is False and row["note"] is None
    for bad in ({"options": {"smoothing": 4}}, {"options": {"texture": -1}}, {"options": {"texture": "fine"}},
                {"options": {"smoothing": True}}, {"options": {"relief": 2}}, {"options": {"relief": True}},
                {"options": {"relief": "on"}}, {"options": {"framing": 0.5}}, {"options": {"framing": 7}},
                {"options": {"framing": "wide"}}, {"options": {"framing": True}}, {"max_rss_mb": 0},
                {"max_rss_mb": "all of it"}, {"max_rss_mb": True},
                # numbers no number can be, and things that are not what they should be: a bad job, never a crash
                {"options": {"detail": float("inf")}}, {"options": {"texture": float("inf")}},
                {"options": {"smoothing": float("nan")}}, {"options": {"framing": 10 ** 400}},
                {"options": {"relief": float("inf")}}, {"max_rss_mb": 10 ** 400}, {"max_rss_mb": float("inf")},
                {"weights_dir": 5}, {"generator": {"a": 1}}, {"generator": ["fake"]},
                {"looks": [{"key": "l1", "image": picture, "inside": [0, 0, 10 ** 400, 80]}]},
                {"looks": [{"key": "l1", "image": picture, "inside": [0, 0, 128]}]},
                {"looks": [{"key": "l1", "image": picture, "inside": "all"}]},
                {"looks": [{"key": "l1", "image": picture, "inside": [0, 0, True, 80]}]}):
        job = _job(tmp_path, bad.pop("looks", [{"key": "l1", "image": picture}]), **bad)
        done = _run(["--job", str(job)])
        assert done.returncode == 2, (bad, done.stderr)


def test_job_exit_codes(tmp_path):
    none = _job(tmp_path, [{"key": "l1", "image": str(tmp_path / "missing.png")}])
    done = _run(["--job", str(none)])
    assert done.returncode == 3 and "no look could be built" in done.stderr.strip().splitlines()[-1]
    assert json.loads((tmp_path / "out" / "result.json").read_text())["looks"][0]["ok"] is False

    (tmp_path / "broken.json").write_text("{not json")
    assert _run(["--job", str(tmp_path / "broken.json")]).returncode == 2
    for bad in ({"looks": []}, {"looks": [{"key": "Bad Key", "image": "/x.png"}]},
                {"looks": [{"key": "a", "image": "relative.png"}]}, {"out_dir": "relative"},
                {"generator": "nope"}, {"options": {"detail": 7}},
                {"looks": [{"key": "a", "image": "/x.png"}, {"key": "a", "image": "/y.png"}]}):
        job = _job(tmp_path, bad.pop("looks", [{"key": "l1", "image": str(tmp_path / "x.png")}]), **bad)
        done = _run(["--job", str(job)])
        assert done.returncode == 2, (bad, done.stderr)
    assert _run([]).returncode == 2

    real = _job(tmp_path, [{"key": "l1", "image": str(_picture(tmp_path / "l1.png"))}], generator="triposr")
    done = _run(["--job", str(real)])
    assert done.returncode == 4 and done.stderr.strip()  # not installed here
    auto = _job(tmp_path, [{"key": "l1", "image": str(_picture(tmp_path / "l1.png"))}], generator="auto")
    done = _run(["--job", str(auto)])
    assert done.returncode == 4 and "setup_3d.sh" in done.stderr


def test_check_says_what_is_installed(tmp_path):
    # a PyTorch that leaves a note when it is imported: the plain check, which the app asks for status reads, must
    # not touch it
    (tmp_path / "torch.py").write_text(
        "import pathlib\n"
        "pathlib.Path(__file__).with_name('imported').write_text('x')\n"
        "__version__ = '9.9'\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return False\n", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()

    def check(*more: str) -> dict:
        done = subprocess.run([sys.executable, "-m", "gen3d", "--check", "--weights-dir", str(weights), *more],
                              cwd=str(REPO), capture_output=True, text=True, timeout=120, check=False,
                              env={**os.environ, "PYTHONPATH": str(tmp_path)})
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    info = check()
    assert info["backends"] == {"fake": True, "triposr": False} and info["auto"] is None
    assert info["device"] is None and info["torch"] is None and not (tmp_path / "imported").exists()
    info = check("--device")
    assert info["device"] == "cpu" and info["torch"] == "9.9" and (tmp_path / "imported").exists()


def test_sigterm_stops_the_job_cleanly(tmp_path):
    import signal
    import time

    # a backend that never finishes: a tiny module on the path that the registry is pointed at
    slow = tmp_path / "slowpkg"
    slow.mkdir()
    (slow / "slow_backend.py").write_text(
        "import time\n"
        "class B:\n"
        "    name = 'fake'\n    version = '0'\n"
        "    def available(self, w): return True\n"
        "    def load(self, w, d, o): pass\n"
        "    def generate(self, image, options):\n        time.sleep(60)\n"
        "    def close(self): open(__file__ + '.closed', 'w').close()\n"
        "BACKEND = B()\n")
    runner = tmp_path / "run.py"
    runner.write_text(
        "import sys\n"
        f"sys.path[:0] = [{str(slow)!r}, {str(REPO)!r}]\n"
        "from gen3d import backends\n"
        "backends.REGISTRY['fake'] = 'slow_backend'\n"
        "from gen3d.__main__ import main\n"
        "sys.exit(main(sys.argv[1:]))\n")
    job = _job(tmp_path, [{"key": "l1", "image": str(_picture(tmp_path / "l1.png"))}])
    proc = subprocess.Popen([sys.executable, str(runner), "--job", str(job)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "progress: 0/1"
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=20)
    assert proc.returncode == 5 and err.strip().splitlines()[-1] == "stopped"
    assert (slow / "slow_backend.py.closed").exists()  # the backend was closed
    assert json.loads((tmp_path / "out" / "result.json").read_text())["looks"] == []


LOGGING_BACKEND = """
import pathlib

import numpy as np

from gen3d.backends.fake import BACKEND as FAKE
from gen3d.mesh import Mesh

LOG = pathlib.Path(__file__).with_name("events.log")


def note(line):
    with LOG.open("a") as f:
        f.write(line + "\\n")


class B:
    name, version = "fake", "t"

    def available(self, weights_dir):
        return True

    def load(self, weights_dir, device, options):
        note("load")

    def generate(self, image, options):
        key = options["key"]
        note("generate " + key)
        if key == "l2":
            raise RuntimeError("the model failed")
        if key == "l4":
            return Mesh(np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint32))   # no mesh at all
        return FAKE.generate(image, options)

    def close(self):
        note("close")


BACKEND = B()
"""


def _never_left_behind(proc: subprocess.Popen) -> None:
    """A child this test started is gone when the test ends, also when one of its own assertions failed first."""
    if proc.poll() is None:
        proc.kill()
        proc.wait(10)


def _logging_job(tmp_path: Path, keys: list[str], patch: str) -> tuple[list[str], Path, Path]:
    """A job on a backend that writes down what it is asked, started through a small script that first applies
    `patch` (Python source). Returns (the command, the job's out folder, the backend's log)."""
    pkg = tmp_path / "logpkg"
    pkg.mkdir()
    (pkg / "logging_backend.py").write_text(LOGGING_BACKEND, encoding="utf-8")
    runner = tmp_path / "run.py"
    runner.write_text(
        "import sys, time\n"
        f"sys.path[:0] = [{str(pkg)!r}, {str(REPO)!r}]\n"
        "from gen3d import backends, glb\n"
        "import gen3d.__main__ as job\n"
        "backends.REGISTRY['fake'] = 'logging_backend'\n"
        f"{patch}\n"
        "sys.exit(job.main(sys.argv[1:]))\n", encoding="utf-8")
    looks = [{"key": key, "image": str(tmp_path / "missing.png" if key == "l3" else _picture(tmp_path / f"{key}.png"))}
             for key in keys]
    return [sys.executable, str(runner), "--job", str(_job(tmp_path, looks))], tmp_path / "out", pkg / "events.log"


def test_a_job_runs_in_two_passes_and_reports_a_look_when_its_file_is_there(tmp_path):
    """First every look goes through the model, then the model is closed, then the heads are finished and reported
    one by one. The writer is slowed down here: a line printed before its file was written would be caught."""
    command, out, log = _logging_job(tmp_path, ["l1", "l2", "l3", "l4", "l5"], patch=(
        "real = glb.write\n"
        "def slow(path, head):\n"
        "    time.sleep(0.3)\n"
        "    return real(path, head)\n"
        "glb.write = slow"))
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines, seen = [], {}
    try:
        for line in proc.stdout:
            line = line.strip()
            lines.append(line)
            if line.startswith("look: "):
                key, state = line.split()[1:3]
                seen[key] = {"state": state, "file": (out / f"{key}.glb").exists(),
                             "events": log.read_text().split("\n")[:-1], "result": (out / "result.json").exists()}
        _, err = proc.communicate(timeout=60)
    finally:
        _never_left_behind(proc)
    assert proc.returncode == 0, err
    # the first pass says a line per look when the model is done with it (l4's mesh is only found bad later); then
    # every look is finished in the job's order, each with its count
    assert lines[0] == "progress: 0/5"
    assert [ln.split()[:3] for ln in lines[1:6]] == [["made:", "l1", "ok"], ["made:", "l2", "error"],
                                                     ["made:", "l3", "error"], ["made:", "l4", "ok"], ["made:", "l5", "ok"]]
    assert all(float(ln.split()[3]) >= 0 for ln in lines[1:6])
    assert [ln.split()[0] for ln in lines[6:]] == ["look:", "progress:"] * 5
    assert [ln for ln in lines if ln.startswith("progress")] == [f"progress: {n}/5" for n in range(6)]
    assert [(key, got["state"]) for key, got in seen.items()] == [
        ("l1", "ok"), ("l2", "error"), ("l3", "error"), ("l4", "error"), ("l5", "ok")]
    # a look that worked has its file in place when its line is read; one that did not never gets a file
    assert seen["l1"]["file"] and seen["l5"]["file"]
    assert not any((out / f"{key}.glb").exists() for key in ("l2", "l3", "l4"))
    # before the first look was finished, the model had been asked for every look it could be asked for, and closed
    first_pass = ["load", "generate l1", "generate l2", "generate l4", "generate l5", "close"]   # l3 has no picture
    assert all(got["events"] == first_pass for got in seen.values()) and log.read_text().split("\n")[:-1] == first_pass
    assert not seen["l1"]["result"]                                      # the result file is written at the end
    result = json.loads((out / "result.json").read_text())
    rows = result["looks"]
    assert [r["key"] for r in rows] == ["l1", "l2", "l3", "l4", "l5"] and [r["ok"] for r in rows] == [True, False, False, False, True]
    assert rows[1]["error"] == "RuntimeError: the model failed"           # failed in the model: the others still ran
    assert rows[2]["error"].startswith("FileNotFoundError")              # failed before it
    assert rows[3]["error"].startswith("BadMesh")                        # failed in the finishing
    assert rows[0]["textured"] and rows[4]["textured"] and all(r["seconds"] >= 0 for r in rows)
    assert rows[0]["seconds"] >= 0.3                                      # both passes count for a look's time
    memory = result["memory_mb"]
    assert set(memory) == {"peak", "with_model", "released"}
    if sys.platform.startswith("linux"):
        assert all(isinstance(memory[k], int) and memory[k] > 10 for k in memory)
        assert memory["peak"] >= memory["released"] - 1 and memory["peak"] >= memory["with_model"] - 1


def test_sigterm_in_the_second_pass_keeps_what_was_reported(tmp_path):
    import signal

    command, out, log = _logging_job(tmp_path, ["l1", "l5", "l6"], patch=(
        "real, calls = job.finish_head, []\n"
        "def slow(mesh, prepared, options, reshape=None):\n"
        "    calls.append(1)\n"
        "    if len(calls) == 2:\n"
        "        time.sleep(60)\n"
        "    return real(mesh, prepared, options, reshape=reshape)\n"
        "job.finish_head = slow"))
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while True:
            line = proc.stdout.readline()
            assert line, "the job ended before its first look"
            if line.startswith("look: l1 ok"):
                break
        assert proc.stdout.readline().strip() == "progress: 1/3"
        proc.send_signal(signal.SIGTERM)                                  # while the second head is being finished
        rest, err = proc.communicate(timeout=20)
    finally:
        _never_left_behind(proc)
    assert proc.returncode == 5 and err.strip().splitlines()[-1] == "stopped" and "look:" not in rest
    result = json.loads((out / "result.json").read_text())
    assert [(r["key"], r["ok"]) for r in result["looks"]] == [("l1", True)]   # what was reported stays
    assert (out / "l1.glb").exists() and not (out / "l5.glb").exists() and not (out / "l6.glb").exists()
    assert glb.read(out / "l1.glb")["photo"] is not None
    assert log.read_text().split("\n")[:-1].count("close") == 1          # closed once, between the passes


# ------------------------------------------------------------------ the viewer reads what the writer wrote
def test_the_viewer_reads_the_heads(tmp_path):
    """frontend/scripts/check-head3d.mjs: the viewer's stage and three.js on files written here. No browser."""
    front = REPO / "frontend"
    node = shutil.which("node")
    if node is None or not all((front / "node_modules" / name).is_dir() for name in ("three", "esbuild")):
        pytest.skip("node or the frontend's packages are not installed")
    out = tmp_path / "glb"
    FAKE.load(tmp_path, "cpu", {})
    pictures = [((200, 150, 120), (30, 20, 10)), ((180, 130, 100), (90, 60, 30)), ((220, 180, 150), (200, 200, 200))]
    for n, (skin, hair) in enumerate(pictures, start=1):
        head = finish(FAKE.generate(Image.open(_picture(tmp_path / f"l{n}.png", skin=skin, hair=hair)), {}))
        glb.write(out / f"l{n}.glb", head)
    v, f = _tetra(offset=(10.0, 10.0, 10.0), scale=8.0)  # far from the origin and large: the viewer frames it itself
    glb.write(out / "plain.glb", Mesh(v, f))
    (out / "broken.glb").write_bytes(b"not a glb at all")
    # a head with its photo, as the job writes one: the finishing's own two-part file. With it the script also runs
    # its group for the photo layer (without it that group is left out, and nothing here would read such a file)
    from gen3d import prep
    from gen3d.finishing import finish_head

    with Image.open(_picture(tmp_path / "photo.png", skin=(190, 140, 110), hair=(40, 30, 20))) as image:
        image.load()
    with_photo = finish_head(FAKE.generate(image, {}), prep.whole(image), {"texture": 2})
    assert with_photo.photo is not None and with_photo.texture is not None
    glb.write(out / "photo.glb", with_photo)
    assert len(glb.read(out / "photo.glb")["json"]["meshes"][0]["primitives"]) == 2
    # the same head with its picture named by an address in place of the bytes in the file, which the writer never
    # does: the viewer must not ask for that address
    data = (out / "photo.glb").read_bytes()
    n_json = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + n_json])
    doc["images"] = [{"uri": "https://example.invalid/carried-out.jpg"}]
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    rest = data[20 + n_json:]
    (out / "outside.glb").write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + len(rest))
                                      + struct.pack("<II", len(text), 0x4E4F534A) + text + rest)
    done = subprocess.run([node, str(front / "scripts" / "check-head3d.mjs"), str(out), str(tmp_path / "work")],
                          capture_output=True, text=True, timeout=180, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-3000:]
    assert "Q a head with its photo" in done.stdout and "R a head that names an address" in done.stdout
    assert "19 groups passed" in done.stdout

