"""The generator shell (the gen3d package at the repo root, docs/ARCHITECTURE.md section 36): the mesh clean-up, the
GLB writer read back, the fake backend, and `python -m gen3d` end to end with it. No PyTorch anywhere. The last test
hands the written files to the viewer's own code and three.js, in node, when the frontend's packages are installed."""
from __future__ import annotations

import json
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
    # deterministic: the same picture, the same bytes
    first = (tmp_path / "out" / "l1.glb").read_bytes()
    assert _run(["--job", str(job)]).returncode == 0
    assert (tmp_path / "out" / "l1.glb").read_bytes() == first


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
    done = _run(["--check", "--weights-dir", str(tmp_path)])
    assert done.returncode == 0, done.stderr
    info = json.loads(done.stdout.strip().splitlines()[-1])
    assert info["backends"] == {"fake": True, "triposr": False} and info["auto"] is None
    assert info["device"] in ("cpu", "cuda") and "torch" in info


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
    done = subprocess.run([node, str(front / "scripts" / "check-head3d.mjs"), str(out), str(tmp_path / "work")],
                          capture_output=True, text=True, timeout=180, check=False)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-3000:]
    assert "16 groups passed" in done.stdout

