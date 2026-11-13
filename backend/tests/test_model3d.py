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

from app import storage
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
    monkeypatch.setattr(model3d, "_checking", False)
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


ENV_NAMES = {"PATH", "HOME", "LANG", "PYTHONUNBUFFERED", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "HF_HOME",
             "U2NET_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
             "LD_LIBRARY_PATH", "GEN3D_THREADS", "USER", "LOGNAME"}


def one_look_then(end: str):
    """A child that builds the first look for real, says so, and then does `end` (a line of Python)."""
    code = (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "from PIL import Image\n"
        "from gen3d import glb\n"
        "from gen3d.backends import get\n"
        "from gen3d.mesh import finish\n"
        "job = json.loads(Path(sys.argv[1]).read_text())\n"
        "look = job['looks'][0]\n"
        "print('progress: 0/%d' % len(job['looks']), flush=True)\n"
        "head = finish(get('fake').generate(Image.open(look['image']), {}))\n"
        "glb.write(Path(job['out_dir']) / (look['key'] + '.glb'), head)\n"
        "print('look: %s ok 0.1' % look['key'], flush=True)\n"
        "print('progress: 1/%d' % len(job['looks']), flush=True)\n"
        f"{end}\n")
    return lambda args, work, fake_run: [sys.executable, "-c", code, args[1]]


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
    assert doc["options"] == {"max_looks": 6, "detail": 3, "framing": 2.2, "smoothing": 2, "texture": 2}
    assert doc["scanned"] == 7 and doc["usable"] == 7 and doc["seconds"] >= 0
    assert all(isinstance(lk["textured"], bool) for lk in doc["looks"]) and "note" in doc["looks"][0]
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
    assert status["options"] == doc["options"]
    assert [lk["textured"] for lk in status["looks"]] == [lk["textured"] for lk in doc["looks"]]
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


DEFAULTS = {"max_looks": 6, "detail": 2, "framing": 2.6, "smoothing": 2, "texture": 2}


def test_options_are_kept_inside_their_ranges():
    assert model3d.clean_options(None) == DEFAULTS and model3d.clean_options({}) == DEFAULTS
    assert set(DEFAULTS) == set(model3d.OPTION_KEYS)
    # outside the range: the nearest end
    assert model3d.clean_options({"max_looks": 99, "detail": 0, "framing": 9, "smoothing": 9, "texture": -4}) == {
        "max_looks": 12, "detail": 1, "framing": 3.2, "smoothing": 3, "texture": 0}
    assert model3d.clean_options({"smoothing": 0, "texture": 3}) == {**DEFAULTS, "smoothing": 0, "texture": 3}
    # no number at all: the default, for every option
    for bad in (None, "strong", True, [2], {"a": 1}, float("nan"), float("inf")):
        assert model3d.clean_options({k: bad for k in DEFAULTS}) == DEFAULTS, bad
    assert model3d.clean_options("not a dict") == DEFAULTS  # type: ignore[arg-type]
    # a number written as text or with a fraction is still taken, as before
    assert model3d.clean_options({"smoothing": "1", "texture": 2.9, "detail": "3"}) == {
        **DEFAULTS, "smoothing": 1, "texture": 2, "detail": 3}
    assert model3d.clean_options({"generator": "fake", "junk": 1})["generator"] == "fake"
    assert "generator" not in model3d.clean_options({"generator": "triposr"})
    assert "junk" not in model3d.clean_options({"junk": 1})


def _spy_on_the_job(monkeypatch) -> dict:
    """The job file the child is started with, read just before the real child runs."""
    seen: dict = {}
    real = model3d._run_child

    def run(work, job_path, total, fake_run, stopped=None):
        seen.update(json.loads(Path(job_path).read_text(encoding="utf-8")))
        seen["pictures"] = {}
        for look in seen["looks"]:
            with Image.open(look["image"]) as im:
                seen["pictures"][look["key"]] = (im.size, im.mode, im.format)
        return real(work, job_path, total, fake_run, stopped)

    monkeypatch.setattr(model3d, "_run_child", run)
    return seen


def test_the_finishing_options_reach_the_job_and_the_manifest(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    job = _spy_on_the_job(monkeypatch)
    model3d.start(conn, person, {"smoothing": 3, "texture": 0, "detail": 1})
    assert job["options"] == {"detail": 1, "smoothing": 3, "texture": 0, "seed": 0}
    assert [look["key"] for look in job["looks"]] == ["l1", "l2"]
    doc = model3d.manifest(person)
    assert doc["options"] == {**DEFAULTS, "detail": 1, "smoothing": 3, "texture": 0}
    assert model3d.status(conn, person)["options"] == doc["options"]
    # nothing said: the defaults go to the generator
    model3d.start(conn, person, None)
    assert job["options"] == {"detail": 2, "smoothing": 2, "texture": 2, "seed": 0}
    assert model3d.manifest(person)["options"] == DEFAULTS


def test_a_look_s_picture_is_written_at_the_photo_s_own_size(conn, fake, monkeypatch):
    """The child gets each look's picture as large as the photo has it, inside 1024 to 3072 px; the picture kept
    with the model stays small."""
    from tests.helpers import flat_image, jpeg_bytes

    person = make_person(conn, "Asha")
    # faces 1,000 px wide in 4,000 px photos: a square of 2,200 px at framing 2.2
    for k in range(3):
        photo = jpeg_bytes(flat_image(4000, 3000, (200 - 9 * k, 160, 120)))
        item = make_item(conn, width=4000, height=3000, data=photo, taken_at=f"2024-03-{k + 1:02d}T10:00:00.000Z")
        make_face(conn, item["id"], person, box=(0.35, 0.3, 0.25, 0.3), size_px=900, age_groups=groups(3))
    job = _spy_on_the_job(monkeypatch)
    model3d.start(conn, person, {"framing": 2.2})
    assert job["pictures"] == {"l1": ((2200, 2200), "RGB", "PNG")}
    doc = model3d.manifest(person)
    with Image.open(model3d.person_dir(person) / doc["rev"] / doc["looks"][0]["picture"]) as im:
        assert im.size == (model3d.PICTURE_SIDE, model3d.PICTURE_SIDE) == (512, 512)
    assert not (model3d.person_dir(person) / "work").exists()


def _with_result_rows(monkeypatch, said: dict[str, dict], meshes: dict[str, bytes] | None = None) -> None:
    """Let the real stand-in child run, then make its result rows say exactly `said` about the finishing (and
    nothing where a look is not named), and swap in `meshes`, before the job collects them."""
    real = model3d._run_child

    def run(work, job_path, total, fake_run, stopped=None):
        out = real(work, job_path, total, fake_run, stopped)
        path = work / "out" / "result.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc["looks"]:
            for key in ("textured", "texture", "overlap", "note"):
                row.pop(key, None)
            row.update(said.get(row["key"], {}))
        path.write_text(json.dumps(doc), encoding="utf-8")
        for key, data in (meshes or {}).items():
            (work / "out" / f"{key}.glb").write_bytes(data)
        return out

    monkeypatch.setattr(model3d, "_run_child", run)


def _glb(textured: bool, vertices: int = 600, triangles: int = 1100) -> bytes:
    """The smallest file that reads as a .glb here: the header and a description, no data."""
    material: dict = {"pbrMetallicRoughness": {"metallicFactor": 0.0}}
    doc: dict = {
        "asset": {"version": "2.0"},
        "accessors": [{"count": vertices, "type": "VEC3"}, {"count": triangles * 3, "type": "SCALAR"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [material],
    }
    if textured:
        doc["images"] = [{"mimeType": "image/jpeg", "bufferView": 0}]
        doc["textures"] = [{"source": 0}]
        doc["materials"].append({"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}, "alphaMode": "BLEND"})
        doc["meshes"][0]["primitives"].append(
            {"attributes": {"POSITION": 0, "TEXCOORD_0": 0}, "indices": 1, "material": 1})
    text = json.dumps(doc).encode("utf-8")
    text += b" " * ((4 - len(text) % 4) % 4)
    return (0x46546C67).to_bytes(4, "little") + (2).to_bytes(4, "little") + (20 + len(text)).to_bytes(4, "little") \
        + len(text).to_bytes(4, "little") + (0x4E4F534A).to_bytes(4, "little") + text


def test_what_the_generator_says_about_the_finishing_is_kept(client, conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    _with_result_rows(monkeypatch, {
        "l1": {"textured": True, "texture": [1536, 2048], "overlap": 0.9712345, "note": None},
        "l2": {"textured": False, "texture": None, "overlap": 0.71,
               "note": "The photo did not line up with the head,\n so the head keeps its own colours.\x07"},
    })
    model3d.start(conn, person, {})
    l1, l2 = model3d.manifest(person)["looks"]
    assert (l1["textured"], l1["texture"], l1["overlap"], l1["note"]) == (True, [1536, 2048], 0.9712, None)
    assert (l2["textured"], l2["texture"], l2["overlap"]) == (False, None, 0.71)
    assert l2["note"] == "The photo did not line up with the head, so the head keeps its own colours."
    s1, s2 = model3d.status(conn, person)["looks"]
    assert s1["textured"] is True and "note" not in s1                 # a note is given only when there is one
    assert s2["textured"] is False and s2["note"] == l2["note"]
    assert "texture" not in s1 and "overlap" not in s1                 # those two stay in the manifest
    body = client.get(f"/api/people/{person}/model3d").json()
    assert [(lk["textured"], lk["note"]) for lk in body["looks"]] == [(True, None), (False, l2["note"])]


def test_a_result_without_the_finishing_keys_reads_as_not_textured(conn, fake, monkeypatch):
    """The generator before the finishing: its rows say nothing, and its files carry no picture."""
    person, _items = person_with_looks(conn)
    _with_result_rows(monkeypatch, {}, {"l1": _glb(False), "l2": _glb(False)})
    model3d.start(conn, person, {})
    for lk in model3d.manifest(person)["looks"]:
        assert (lk["textured"], lk["texture"], lk["overlap"], lk["note"]) == (False, None, None, None)
    assert all(lk["textured"] is False and "note" not in lk for lk in model3d.status(conn, person)["looks"])


def test_the_file_says_whether_it_is_textured_when_the_row_does_not(conn, fake, monkeypatch):
    """A head finished before the generator was stopped has no result row: its file is asked. And what a row
    says is only taken when it is of the right kind."""
    assert model3d._glb_textured(Path(__file__)) is False              # not a .glb at all
    person, _items = person_with_looks(conn)
    _with_result_rows(monkeypatch, {
        "l1": {},                                                                       # says nothing
        "l2": {"textured": "yes", "texture": [0, -5], "overlap": 7, "note": 12},        # says nonsense
    }, {"l1": _glb(True), "l2": _glb(False)})
    model3d.start(conn, person, {})
    l1, l2 = model3d.manifest(person)["looks"]
    assert (l1["textured"], l1["texture"], l1["overlap"], l1["note"]) == (True, None, None, None)
    assert (l2["textured"], l2["texture"], l2["overlap"], l2["note"]) == (False, None, None, None)
    assert (l1["vertices"], l1["faces"]) != (0, 0)

    # a size is kept only with a texture; a long note is cut; a note of blanks is none
    _with_result_rows(monkeypatch, {
        "l1": {"textured": False, "texture": [1024, 1024], "note": "x" * 500},
        "l2": {"textured": True, "texture": [4096, 4096.0], "overlap": True, "note": " \n\t "},
    })
    model3d.start(conn, person, {})
    l1, l2 = model3d.manifest(person)["looks"]
    assert (l1["textured"], l1["texture"], l1["note"]) == (False, None, "x" * model3d.NOTE_MAX)
    assert (l2["textured"], l2["texture"], l2["overlap"], l2["note"]) == (True, None, None, None)


def test_a_model_made_before_the_finishing_still_shows(client, conn, fake):
    """An older manifest has neither the two options nor anything about textures. The status gives the
    defaults for the options (the tab starts "Generate again" from them) and says the heads are not textured."""
    person, _items = person_with_looks(conn)
    model3d.start(conn, person, {"max_looks": 6, "detail": 3, "framing": 2.2})
    rev = model3d.current_rev(person)
    path = model3d.person_dir(person) / rev / "model.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["options"] = {"max_looks": 6, "detail": 3, "framing": 2.2}
    doc.pop("peak_rss_mb", None)
    for lk in doc["looks"]:
        for key in ("textured", "texture", "overlap", "note"):
            lk.pop(key, None)
    path.write_text(json.dumps(doc), encoding="utf-8")

    status = model3d.status(conn, person)
    assert status["state"] == "done" and len(status["looks"]) == 2
    assert status["options"] == {"max_looks": 6, "detail": 3, "framing": 2.2, "smoothing": 2, "texture": 2}
    assert all(lk["textured"] is False and "note" not in lk for lk in status["looks"])
    body = client.get(f"/api/people/{person}/model3d").json()
    assert body["options"]["smoothing"] == 2 and body["options"]["texture"] == 2
    assert [(lk["textured"], lk["note"]) for lk in body["looks"]] == [(False, None), (False, None)]
    assert client.get(body["looks"][0]["mesh_url"]).status_code == 200

    # a manifest whose options are missing or broken gives none, and the rest still shows
    for broken in (None, {}, "fast", ["a"]):
        doc["options"] = broken
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert model3d.status(conn, person)["options"] is None
        assert client.get(f"/api/people/{person}/model3d").json()["options"] is None
    assert model3d.shown_options({"detail": "x", "framing": None}) == DEFAULTS    # bad values read as the defaults


def test_the_manifest_says_how_much_memory_the_child_held(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    monkeypatch.setattr(model3d, "tree_rss_mb", lambda pid: 321.4)
    model3d.start(conn, person, {})
    assert model3d.manifest(person)["peak_rss_mb"] == 321 and model3d.peak_rss_mb() == 321
    monkeypatch.setattr(model3d, "tree_rss_mb", lambda pid: 0.0)           # a child that was gone at every look
    model3d.start(conn, person, {})
    assert model3d.manifest(person)["peak_rss_mb"] is None


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
    works = []

    def answer(installed: bool):
        def command(args, work, fake):
            asked.append(list(args))
            works.append((Path(work), Path(work).is_dir()))
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
    # the check wrote, if at all, in an empty folder of its own that is gone again: never among the models
    assert works and all(there and w.parent == storage.root() / "tmp" and not w.exists() for w, there in works)
    assert not any(model3d.root().iterdir())
    monkeypatch.setenv("GEN3D_THREADS", "3")
    monkeypatch.setenv("PCC_TEST_SECRET", "not for the generator")
    env = model3d._child_env()
    assert env["HF_HOME"] == str(tmp_path / "weights" / "hf") and env["U2NET_HOME"] == str(tmp_path / "weights" / "u2net")
    assert env["PYTHONUNBUFFERED"] == "1" and env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert set(env) <= ENV_NAMES and env["GEN3D_THREADS"] == "3" and env["PATH"] == "/usr/bin:/bin"
    assert model3d._child_env(tmp_path)["XDG_CACHE_HOME"] == f"{tmp_path}/cache"


def test_a_status_read_waits_for_the_check_only_once(monkeypatch, tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    for name, value in (("gen3d_fake", False), ("gen3d_generator", "auto"), ("gen3d_python", python),
                        ("gen3d_dir", tmp_path), ("gen3d_sandbox", "off"), ("workers_enabled", True)):
        monkeypatch.setattr(settings, name, value)
    asked = []
    release = threading.Event()

    def ask(_python):
        asked.append(1)
        if len(asked) > 1:
            release.wait(20)                                              # the generator's Python is slow today
        return {"installed": True, "name": f"triposr-{len(asked)}", "device": None, "reason": None}

    monkeypatch.setattr(model3d, "_ask_generator", ask)
    assert model3d.generator()["name"] == "triposr-1"                     # the first read asks, and waits
    at, key, out = model3d._check
    monkeypatch.setattr(model3d, "_check", (at - model3d.CHECK_TTL_S - 1, key, out))
    started = time.monotonic()
    assert model3d.generator()["name"] == "triposr-1"                     # an old answer: given at once ...
    assert model3d.generator()["name"] == "triposr-1" and time.monotonic() - started < 2
    deadline = time.monotonic() + 10
    while len(asked) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(asked) == 2                                                # ... and asked again once, behind the scenes
    release.set()
    while model3d.generator()["name"] != "triposr-2" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert model3d.generator()["name"] == "triposr-2" and len(asked) == 2


def test_the_child_gets_a_short_environment_of_its_own(conn, fake, monkeypatch, tmp_path):
    person, _items = person_with_looks(conn)
    monkeypatch.setenv("PCC_TEST_SECRET", "not for the generator")
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1/docker.sock")
    monkeypatch.setenv("GEN3D_THREADS", "2")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dump = tmp_path / "env.json"
    code = ("import json, os, sys; open(sys.argv[1], 'w').write(json.dumps(dict(os.environ))); "
            "print('progress: 0/2', flush=True)")
    monkeypatch.setattr(model3d, "_command", lambda *a, **k: [sys.executable, "-c", code, str(dump)])
    model3d.start(conn, person, {})
    env = json.loads(dump.read_text())
    work = model3d.person_dir(person) / "work"
    assert set(env) <= ENV_NAMES | {"LC_CTYPE"}
    assert "PCC_TEST_SECRET" not in env and "DOCKER_HOST" not in env and "CUDA_VISIBLE_DEVICES" not in env
    assert env["GEN3D_THREADS"] == "2" and env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HOME"] == str(work / "home") and not work.exists()        # no sandbox here: a home that goes with the job


def test_generator_fake_needs_no_generator_environment(conn, monkeypatch, tmp_path):
    for name, value in (("gen3d_fake", False), ("gen3d_generator", "fake"), ("gen3d_python", None),
                        ("gen3d_dir", tmp_path / "nothing-here"), ("gen3d_sandbox", "auto")):
        monkeypatch.setattr(settings, name, value)
    person, _items = person_with_looks(conn)
    assert model3d.generator()["name"] == "fake" and model3d._python() == Path(sys.executable)
    assert model3d._command(["--check"], tmp_path, True)[4:] == [sys.executable, "-m", "gen3d", "--check"]
    model3d.start(conn, person, {})
    assert model3d.status(conn, person)["state"] == "done", model3d._error(person)
    monkeypatch.setattr(settings, "gen3d_generator", "auto")              # asked for by the job alone: the same Python
    assert model3d._python(True) == Path(sys.executable)
    assert model3d._python(False) == tmp_path / "nothing-here" / "venv" / "bin" / "python"


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


def test_a_person_who_is_deleted_takes_their_running_job_along(client, conn, fake, monkeypatch):
    from app.db import connect

    person, _items = person_with_looks(conn)
    other, _ = person_with_looks(conn, "Ravi")
    real = model3d._command
    monkeypatch.setattr(model3d, "_command", sleeper)
    t = _start_in_thread(connect, person)
    proc = _wait_for_child()
    with pytest.raises(model3d.Busy):
        model3d.start(conn, other, {})
    assert client.delete(f"/api/people/{person}").status_code == 200
    assert client.post(f"/api/people/{person}/model3d/stop").status_code == 404    # no route reaches the job now ...
    t.join(30)
    assert not t.is_alive() and not alive(proc.pid)                                # ... and none has to: it saw it
    assert model3d.running() is None and model3d.manifest(person) is None
    monkeypatch.setattr(model3d, "_command", real)
    model3d.start(conn, other, {})                                                 # the one job slot is free again
    assert model3d.manifest(other) is not None


def test_stop_delete_and_a_deleted_person_end_the_job_while_looks_are_found(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    model3d.start(conn, person, {})                                    # a model exists
    first = model3d.current_rev(person)
    real = looks.read_faces
    calls: list[int] = []
    act = {"now": lambda: None}

    def reading(conn_, rows, eng=None, progress=None):
        def tick(done, total):
            calls.append(done)
            if len(calls) == 4:                                        # the second of seven photos has been read
                act["now"]()
            progress(done, total)
        return real(conn_, rows, eng, tick)

    monkeypatch.setattr(looks, "read_faces", reading)
    children = []
    command = model3d._command
    monkeypatch.setattr(model3d, "_command", lambda *a: children.append(a) or command(*a))

    def stop():
        assert model3d.cancel(person) is True
        assert model3d.status(conn, person)["stopping"] is True and model3d.status(conn, person)["state"] == "running"

    act["now"] = stop
    model3d.start(conn, person, {})
    assert calls == [0, 0, 1, 2] and children == [] and model3d.running() is None   # out at that photo, not after all
    assert model3d.current_rev(person) == first and model3d._error(person) is None  # Stop keeps the model in use
    status = model3d.status(conn, person)
    assert status["state"] == "done" and status["stopping"] is False

    def delete():
        model3d.remove(conn, person)
        assert model3d.manifest(person) is None and model3d.status(conn, person)["looks"] == []   # gone from now on

    calls.clear()
    act["now"] = delete
    model3d.start(conn, person, {})
    assert calls == [0, 0, 1, 2] and children == [] and not model3d.person_dir(person).exists()

    calls.clear()
    act["now"] = lambda: None
    model3d.start(conn, person, {})
    assert len(calls) == 9 and len(children) == 1 and model3d.manifest(person) is not None
    kept = model3d.current_rev(person)

    def gone():                                                        # deleted, or merged into someone else
        with tx(conn):
            conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (person,))
            conn.execute("DELETE FROM persons WHERE id = ?", (person,))

    calls.clear()
    children.clear()
    act["now"] = gone
    monkeypatch.setattr(model3d, "GONE_CHECK_S", 0.0)                  # look at every photo
    model3d.start(conn, person, {})
    assert calls == [0, 0, 1, 2] and children == [] and model3d.running() is None
    assert model3d.current_rev(person) == kept and model3d._error(person) is None   # the folder waits for the sweep
    assert model3d.sweep(conn) == 1


def test_a_gone_person_is_noticed_before_a_head_is_built_or_saved(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    real = looks.plan

    def plan(conn_, person_id, *a, **k):
        out = real(conn_, person_id, *a, **k)
        with tx(conn):                                                 # gone right after the looks were found
            conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (person,))
            conn.execute("DELETE FROM persons WHERE id = ?", (person,))
        return out

    monkeypatch.setattr(looks, "plan", plan)
    children = []
    monkeypatch.setattr(model3d, "_command", lambda *a: children.append(a) or sleeper())
    model3d.start(conn, person, {})                                    # the once-a-second look-up is not due yet
    assert children == [] and model3d.manifest(person) is None and model3d.running() is None


def test_heads_finished_before_the_generator_was_stopped_are_kept(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    monkeypatch.setattr(model3d, "_command", one_look_then("time.sleep(120)"))
    monkeypatch.setattr(settings, "gen3d_look_timeout_s", 4.0)         # the second head never comes
    model3d.start(conn, person, {})
    doc = model3d.manifest(person)
    assert [lk["key"] for lk in doc["looks"]] == ["l1"]
    assert doc["looks"][0]["vertices"] > 500 and doc["looks"][0]["faces"] > 1000   # read from the file itself
    status = model3d.status(conn, person)
    assert status["state"] == "done" and len(status["looks"]) == 1
    assert "took longer" in status["error"] and "1 of 2 heads was finished before that and is kept" in status["error"]
    assert not (model3d.person_dir(person) / "work").exists() and model3d._proc is None

    # the same when the generator breaks after a head: the head stays, the reason is told
    monkeypatch.setattr(model3d, "_command", one_look_then("sys.stderr.write('CUDA out of memory\\n'); sys.exit(1)"))
    model3d.start(conn, person, {})
    status = model3d.status(conn, person)
    assert [lk["key"] for lk in status["looks"]] == ["l1"] and model3d.current_rev(person) != doc["rev"]
    assert "CUDA out of memory" in status["error"] and "1 of 2 heads was finished" in status["error"]


def test_sweep_removes_what_an_interrupted_run_left_behind(conn, fake, monkeypatch):
    person, _items = person_with_looks(conn)
    model3d.start(conn, person, {})
    rev = model3d.current_rev(person)
    base = model3d.person_dir(person)
    (base / "work" / "out").mkdir(parents=True)                        # the server stopped in the middle of a job
    (base / "work" / "l1.png").write_bytes(b"a head and shoulders")
    (base / "20200101T000000").mkdir()                                 # ... or between saving and switching over
    (base / "20200101T000000" / "l1.glb").write_bytes(b"half a model")
    monkeypatch.setattr(model3d, "_state", {"person_id": 999, "step": "Building heads", "done": 0, "total": 1,
                                            "started_at": "", "cancel": False, "remove": False})
    assert model3d.sweep(conn) == 0 and (base / "work").exists()       # while any job runs, nothing is touched
    monkeypatch.setattr(model3d, "_state", None)
    assert model3d.sweep(conn) == 0
    assert {p.name for p in base.iterdir()} == {"current", rev}
    assert model3d.status(conn, person)["state"] == "done" and len(model3d.status(conn, person)["looks"]) == 2


def test_a_look_goes_when_its_face_is_no_longer_this_person_s(client, conn, fake):
    person, _items = person_with_looks(conn)
    other = make_person(conn, "Ravi")
    url = f"/api/people/{person}/model3d"
    assert client.post(url, json={}).status_code == 202
    body = client.get(url).json()
    l1, l2 = model3d.manifest(person)["looks"]
    assert len(l1["box"]) == 4
    keys = lambda: [lk["key"] for lk in client.get(url).json()["looks"]]  # noqa: E731
    with tx(conn):                                                     # look 1 was built from someone else's face
        conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (other, l1["face_id"]))
    assert keys() == ["l2"]
    assert client.get(body["looks"][0]["mesh_url"]).status_code == 404
    assert client.get(body["looks"][0]["picture_url"]).status_code == 404
    with tx(conn):
        conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person, l1["face_id"]))
    assert keys() == ["l1", "l2"]
    with tx(conn):                                                     # unassigned
        conn.execute("UPDATE faces SET person_id = NULL WHERE id = ?", (l1["face_id"],))
    assert keys() == ["l2"]
    # the photo of look 2 is processed again: its faces are new rows. The look stays when this person's face there
    # sits where the look's face was, and goes when it is someone else's place in the photo.
    row = conn.execute("SELECT x, y, w, h FROM faces WHERE id = ?", (l2["face_id"],)).fetchone()
    with tx(conn):
        conn.execute("DELETE FROM faces WHERE id = ?", (l2["face_id"],))
    assert keys() == []
    make_face(conn, l2["item_id"], person, box=(row["x"] + 0.01, row["y"], row["w"], row["h"]), size_px=300,
              age_groups=groups(5))
    assert keys() == ["l2"] and client.get(body["looks"][1]["mesh_url"]).status_code == 200
    with tx(conn):
        conn.execute("UPDATE faces SET x = 0.0, y = 0.0 WHERE item_id = ? AND person_id = ?", (l2["item_id"], person))
    assert keys() == []


# ------------------------------------------------------------------ the sandbox
def test_the_sandbox_is_for_the_real_generator_unless_it_is_demanded(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gen3d_python", tmp_path / "python")
    monkeypatch.setattr(model3d, "sandbox_bin", lambda: "/usr/bin/bwrap")
    home = str(Path.home())
    for mode, fake_run, wanted in (("auto", False, True), ("auto", True, False), ("on", True, True), ("off", False, False)):
        monkeypatch.setattr(settings, "gen3d_sandbox", mode)
        monkeypatch.setattr(settings, "gen3d_fake", fake_run)
        assert model3d.sandbox_wanted(fake_run) is wanted
        cmd = model3d._command(["--job", "/w/job.json"], Path("/w"), fake_run)
        assert cmd[:2] == ["/bin/sh", "-c"] and "oom_score_adj" in cmd[2] and cmd[-3:] == ["gen3d", "--job", "/w/job.json"]
        assert ("/usr/bin/bwrap" in cmd) is wanted
        if wanted:
            inner = cmd[cmd.index("/usr/bin/bwrap"):]
            for flag in ("--unshare-user", "--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                         "--unshare-cgroup-try", "--die-with-parent", "--new-session", "--clearenv"):
                assert flag in inner
            pairs = lambda flag: [(inner[n + 1], inner[n + 2]) for n, word in enumerate(inner) if word == flag]  # noqa: E731
            read_only = pairs("--ro-bind")
            assert pairs("--bind") == [("/w", "/w")]                   # the one place it can write
            assert ("/usr", "/usr") in read_only and ("/sys", "/sys") in read_only
            assert (str(REPO_ROOT / "gen3d"), str(REPO_ROOT / "gen3d")) in read_only
            assert inner[inner.index("--chdir") + 1] == str(REPO_ROOT)
            assert inner[inner.index("--remount-ro") + 1] == "/" and inner.index("--remount-ro") > inner.index("--bind")
            assert [inner[n + 1] for n, word in enumerate(inner) if word == "--tmpfs"] == ["/tmp", "/run"]
            assert inner[inner.index("--dev") + 1] == "/dev" and inner[inner.index("--proc") + 1] == "/proc"
            assert all(src.startswith("/dev/nvidia") for src, _ in pairs("--dev-bind"))
            # nothing wide: not the whole machine, not the home folder or a folder above it, not the data folder
            for src, dest in read_only:
                assert src == dest and src != "/" and not (home == src or home.startswith(src.rstrip("/") + "/")), src
                assert not str(storage.root()).startswith(src.rstrip("/") + "/") and src != str(storage.root()), src
            assert dict(pairs("--setenv")) == model3d._child_env("/tmp")
            assert inner.index("--tmpfs") < inner.index(str(REPO_ROOT / "gen3d"))  # binds come after the empty /tmp
    monkeypatch.setattr(model3d, "sandbox_bin", lambda: None)
    monkeypatch.setattr(settings, "gen3d_sandbox", "auto")
    assert model3d.sandbox_wanted(False) is False                      # no bwrap: auto runs without
    monkeypatch.setattr(settings, "gen3d_sandbox", "on")
    with pytest.raises(model3d.GeneratorMissing):
        model3d.sandbox_wanted(True)
    monkeypatch.setattr(settings, "gen3d_fake", True)
    assert "bwrap" in model3d.generator()["reason"]


def test_the_sandbox_shows_a_python_environment_but_not_a_folder_that_only_holds_one(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gen3d_dir", tmp_path / "gen3d")
    stray = tmp_path / "somewhere" / "bin" / "python"                  # GEN3D_PYTHON set to a loose file
    stray.parent.mkdir(parents=True)
    stray.write_text("")
    assert model3d.sandbox_reads(stray) == [str(REPO_ROOT / "gen3d")]
    venv = tmp_path / "gen3d" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "bin" / "python").symlink_to(sys.executable)
    reads = model3d.sandbox_reads(venv / "bin" / "python")
    assert reads[0] == str(venv) and str(tmp_path / "gen3d") in reads and str(REPO_ROOT / "gen3d") in reads
    assert "/" not in reads and "/usr" not in reads and len(reads) == len(set(reads))


@pytest.mark.skipif(model3d.sandbox_bin() is None, reason="bwrap is not installed")
def test_the_check_runs_in_the_sandbox_with_a_folder_of_its_own(monkeypatch, tmp_path, data_dir):
    import subprocess

    python = tmp_path / "gen3d" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    for name, value in (("gen3d_fake", False), ("gen3d_generator", "auto"), ("gen3d_python", None),
                        ("gen3d_dir", tmp_path / "gen3d"), ("gen3d_sandbox", "auto")):
        monkeypatch.setattr(settings, name, value)
    seen = {}

    def run(cmd, **kw):
        bind = Path(cmd[cmd.index("--bind") + 1])
        seen.update(cmd=cmd, bind=bind, there=bind.is_dir(), env=kw["env"])
        said = {"backends": {"fake": True, "triposr": True}, "auto": "triposr", "device": None, "torch": None}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(said).encode(), stderr=b"")

    monkeypatch.setattr(model3d.subprocess, "run", run)
    assert model3d.generator() == {"installed": True, "name": "triposr", "device": None, "reason": None}
    assert seen["there"] and seen["bind"].parent == storage.root() / "tmp" and not seen["bind"].exists()
    assert "people3d" not in " ".join(seen["cmd"]) and "--device" not in seen["cmd"]   # no PyTorch for a status read
    assert seen["cmd"][-6:] == [str(python), "-m", "gen3d", "--check", "--weights-dir", str(tmp_path / "gen3d" / "weights")]
    assert seen["env"]["HOME"] == "/tmp" and set(seen["env"]) <= ENV_NAMES


@pytest.mark.skipif(model3d.sandbox_bin() is None, reason="bwrap is not installed")
def test_the_stand_in_job_runs_through_the_sandbox(conn, fake, monkeypatch, tmp_path):
    import socket
    import subprocess

    person, _items = person_with_looks(conn)
    monkeypatch.setattr(settings, "gen3d_sandbox", "on")
    monkeypatch.setenv("PCC_TEST_SECRET", "not for the generator")
    seen = []
    real = model3d._command
    monkeypatch.setattr(model3d, "_command", lambda args, work, fake_run: seen.append(real(args, work, fake_run)) or seen[-1])
    model3d.start(conn, person, {})
    assert model3d.status(conn, person)["state"] == "done", model3d._error(person)
    assert any(model3d.sandbox_bin() in cmd and "--unshare-net" in cmd and "--clearenv" in cmd for cmd in seen)

    # what the sandbox means, asked from inside it with the app's own Python
    work = model3d.person_dir(person) / "probe"
    work.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not for the generator")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(outside / "host.sock"))
    server.listen(1)
    home = Path.home()
    probe = """
import json, os, socket, sys
work, repo, outside, home, data = sys.argv[1:6]
out = {"env": sorted(os.environ), "secret": os.environ.get("PCC_TEST_SECRET")}
def writes(path):
    try:
        open(path, "w").close()
        return True
    except OSError:
        return False
def connects(family, address):
    s = socket.socket(family, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect(address)
        return True
    except OSError:
        return False
    finally:
        s.close()
out["write_work"] = writes(work + "/x")
out["write_else"] = [p for p in (repo + "/gen3d/x", repo + "/x", "/usr/x", home + "/x", data + "/x") if writes(p)]
out["sees"] = [p for p in (outside + "/secret.txt", outside + "/host.sock", data + "/cloud.db") if os.path.exists(p)]
out["home"] = sorted(os.listdir(home)) if os.path.isdir(home) else None
out["repo"] = sorted(os.listdir(repo))
out["run"] = sorted(os.listdir("/run"))
out["unix"] = connects(socket.AF_UNIX, outside + "/host.sock")
out["tcp"] = connects(socket.AF_INET, ("1.1.1.1", 53))
print(json.dumps(out))
"""
    python = Path(sys.executable)
    try:
        done = subprocess.run([*model3d.sandbox_prefix(work, python), str(python), "-c", probe, str(work), str(REPO_ROOT),
                               str(outside), str(home), str(storage.root())],
                              capture_output=True, text=True, timeout=60, check=False, env=model3d._child_env())
    finally:
        server.close()
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["write_work"] and (work / "x").exists()                 # the work folder, and nowhere else
    assert got["write_else"] == [] and not (REPO_ROOT / "gen3d" / "x").exists() and not (storage.root() / "x").exists()
    assert got["sees"] == []                                           # no file outside, no database
    assert got["unix"] is False and got["run"] == []                   # no socket of the machine
    assert got["tcp"] is False                                         # no network
    assert got["secret"] is None and set(got["env"]) <= ENV_NAMES | {"LC_CTYPE", "PWD"}   # PWD is bwrap's own
    # of the home folder and the repository only the way to what was shown exists
    shown = [Path(r) for r in model3d.sandbox_reads(python)] + [work]
    ways = lambda top: {r.relative_to(top).parts[0] for r in shown if r != top and r.is_relative_to(top)}  # noqa: E731
    assert got["home"] is None or set(got["home"]) <= ways(home)
    assert "gen3d" in got["repo"] and set(got["repo"]) <= ways(REPO_ROOT)


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
    # the finishing options: whole numbers 0 to 3, and nothing that only looks like one
    for bad in ({"smoothing": 4}, {"smoothing": -1}, {"texture": 4}, {"texture": "2"}, {"smoothing": "2"},
                {"texture": 2.5}, {"smoothing": True}, {"texture": None}):
        assert client.post(url, json=bad).status_code == 422, bad
    assert client.get(url).json()["state"] == "none"                       # none of them started anything
    r = client.post(url, json={"max_looks": 2, "detail": 1, "framing": 2.0})
    assert r.status_code == 202 and r.json() == {"started": True}
    assert "x-action-id" not in r.headers                                  # not an undoable owner action
    body = client.get(url).json()
    assert body["state"] == "done"
    assert body["options"] == {"max_looks": 2, "detail": 1, "framing": 2.0, "smoothing": 2, "texture": 2}
    assert all(lk["textured"] in (True, False) and "note" in lk for lk in body["looks"])
    assert client.post(url, json={"max_looks": 2, "smoothing": 0, "texture": 3}).status_code == 202
    assert client.get(url).json()["options"] == {
        "max_looks": 2, "detail": 2, "framing": 2.6, "smoothing": 0, "texture": 3}
    chosen = {"max_looks": 2, "detail": 1, "framing": 2.0, "smoothing": 3, "texture": 1}
    assert client.post(url, json=chosen).status_code == 202
    body = client.get(url).json()
    assert body["options"] == chosen
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
