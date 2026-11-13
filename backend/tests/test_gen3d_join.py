"""The face's relief joined to the finishing and to the job (gen3d/__main__.py with gen3d.finishing and
gen3d.relief; docs/ARCHITECTURE.md section 36). The product's own stage and finishing run here on a made-up head
with stand-in depth models: no model file, no PyTorch, no photo.

The made-up head is the finishing tests' bust. The stand-in depth model answers with the depth of the same bust
with a nose on it, so the stage has something to add; the stand-in normal model says the surface is flat, so how
deep the relief is comes from the plain fit of the reading to the mesh."""
from __future__ import annotations

import gc
import json
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import gen3d.__main__ as job_runner  # noqa: E402
from gen3d import backends, raster, relief  # noqa: E402
from gen3d import finishing as F  # noqa: E402
from gen3d.finishing import Fit, View  # noqa: E402
from gen3d.mesh import Camera, Head, Mesh, clean  # noqa: E402
from gen3d.prep import Prepared  # noqa: E402
from tests.test_gen3d_finishing import SQUARE, SURE, _bust, _picture  # noqa: E402

FRAMING = 3.0                    # not the job's default, so a stage that ignored the option would read another crop
NOSE = 0.03                      # how far the nose stands out of the true head, in the mesh's units
NOSE_AT = (0.0, 0.24)            # where, across and up
SIZE = 512                       # the made-up picture's side


def _with_nose(vertices: np.ndarray) -> np.ndarray:
    out = np.array(vertices, dtype=np.float64)
    bump = NOSE * np.exp(-((out[:, 0] - NOSE_AT[0]) / 0.05) ** 2 - ((out[:, 1] - NOSE_AT[1]) / 0.06) ** 2)
    out[:, 2] += np.where(out[:, 2] > 0.0, bump, 0.0)
    return out


def _models(true_vertices: np.ndarray, faces: np.ndarray, view: View, picture: Prepared, calls: list | None = None):
    """Stand-in depth and normal models for the crop the stage cuts from `picture`."""
    h, w = picture.alpha.shape
    x0, y0, m = relief.face_crop(h, w, FRAMING, picture.inside)
    uv, depth = view.project(true_vertices)
    crop = (picture.to_source(uv) - np.array([x0, y0])) * (relief.MODEL_SIDE / m)
    zbuf = raster.depth_buffer(crop, depth, faces, relief.MODEL_SIDE, relief.MODEL_SIDE)
    seen = np.isfinite(zbuf)
    z = np.where(seen, zbuf, float(zbuf[seen].max()) + 0.3)                # a wall behind where the head is not
    reading = (0.48 - 0.03 * z).astype(np.float32)                         # larger means nearer
    flat = np.zeros((3, relief.MODEL_SIDE, relief.MODEL_SIDE), np.float32)
    flat[2] = 1.0

    def run_depth(rgb):
        if calls is not None:
            calls.append(("depth", rgb.shape, rgb.dtype))
        return reading

    def run_normals(rgb):
        if calls is not None:
            calls.append(("normals", rgb.shape, rgb.dtype))
        return flat

    return run_depth, run_normals


@pytest.fixture()
def scene(monkeypatch, tmp_path):
    """The bust, its picture, and the stand-in models installed where the stage looks for the real ones."""
    mesh = _bust(60, 64, camera=SURE)
    m = clean(mesh)
    view = View(SURE, Fit(), SQUARE)
    picture = _picture(m.vertices, m.faces, view, size=SIZE)
    calls: list = []
    monkeypatch.setattr(relief, "available", lambda weights: True)
    monkeypatch.setattr(relief, "runners", lambda weights: _models(_with_nose(m.vertices), m.faces, view, picture, calls))
    return {"mesh": mesh, "clean": m, "view": view, "picture": picture, "calls": calls, "weights": tmp_path / "weights"}


def _nose_height(head: Head, m: Mesh) -> float:
    """How far the nose's place stands in front of the cheeks beside it, in the written head's units."""
    v = m.vertices
    front = v[:, 2] > 0.0
    nose = front & (np.abs(v[:, 0] - NOSE_AT[0]) < 0.02) & (np.abs(v[:, 1] - NOSE_AT[1]) < 0.025)
    beside = front & (np.abs(np.abs(v[:, 0]) - 0.11) < 0.02) & (np.abs(v[:, 1] - NOSE_AT[1]) < 0.025)
    assert nose.sum() >= 3 and beside.sum() >= 3
    z = head.base.vertices[:, 2].astype(np.float64)
    return float(z[nose].mean() - z[beside].mean())


def test_the_stage_shapes_the_face_between_the_camera_and_the_photo(scene):
    m, picture = scene["clean"], scene["picture"]
    plain = F.finish_head(scene["mesh"], picture, {"texture": 2})
    said: dict = {}
    stage = job_runner._relief_stage({"framing": FRAMING}, scene["weights"], None, said)
    was: dict = {}

    def watched(vertices, faces, view, prepared):
        out = stage(vertices, faces, view, prepared)
        was.update(before=vertices.copy(), after=out, view=view)
        return out

    head = F.finish_head(scene["mesh"], picture, {"texture": 2}, reshape=watched)
    # the two models were each given the crop once, as a 512 x 512 RGB picture of bytes
    assert scene["calls"] == [("depth", (512, 512, 3), np.dtype(np.uint8)), ("normals", (512, 512, 3), np.dtype(np.uint8))]
    assert said["relief"] is True and said["moved"] > 200 and said["note"] == ""
    assert head.reshaped is True and head.reshape_note is None and head.note is None and head.photo is not None
    # the stage moved vertices along the camera's rays: each keeps its place in the picture, the others their place
    before, after, view = was["before"], was["after"], was["view"]
    changed = np.any(after != before, axis=1)
    assert int(changed.sum()) == said["moved"]
    assert float(np.abs(view.project(after[changed])[0] - view.project(before[changed])[0]).max()) < 1e-6
    assert float(np.abs(view.project(after[changed])[1] - view.project(before[changed])[1]).max()) > 0.01
    # how deep the relief is: the stand-in reading is 0.03 per unit of distance, and the fit finds that
    assert said["factor"] == pytest.approx(-1.0 / 0.03, rel=0.05)
    # the nose the depth model saw is on the head now: most of its height, in the head's unit box
    extent = float((m.vertices.max(axis=0) - m.vertices.min(axis=0)).max())
    grown = _nose_height(head, m) - _nose_height(plain, m)
    assert 0.5 * NOSE / extent < grown < 1.2 * NOSE / extent
    # and the photo still lies where the camera put it: the layer shows the same part of the picture
    assert head.texture_size == plain.texture_size
    assert abs(int(head.photo.faces.shape[0]) - int(plain.photo.faces.shape[0])) < 0.1 * plain.photo.faces.shape[0]
    # what the camera does not see keeps its place: in the written head it only moves as one, with the middle of
    # the box, which the nose pushed forward (the few smoothing rounds after the move aside)
    back = m.vertices[:, 2] < -0.05
    moved = head.base.vertices[back].astype(np.float64) - plain.base.vertices[back]
    assert float(np.abs(moved - moved.mean(axis=0)).max()) < 0.004
    # without the photo the same surface is written, with the model's own colours
    bare = F.finish_head(scene["mesh"], picture, {"texture": 0, "smoothing": 0},
                         reshape=job_runner._relief_stage({"framing": FRAMING}, scene["weights"], None, {}))
    under = F.finish_head(scene["mesh"], picture, {"texture": 2, "smoothing": 0},
                          reshape=job_runner._relief_stage({"framing": FRAMING}, scene["weights"], None, {}))
    assert bare.reshaped and bare.photo is None and np.allclose(bare.base.vertices, under.base.vertices, atol=1e-6)


def test_the_stage_is_given_the_finishings_camera_and_the_looks_picture(scene, monkeypatch):
    m, view = scene["clean"], scene["view"]
    picture = _picture(m.vertices, m.faces, view, size=SIZE, inside=(0, 0, SIZE, 400))
    got: dict = {}
    real = relief.refine

    def spy(vertices, faces, **named):
        got.update(named, vertices=vertices, faces=faces)
        return real(vertices, faces, **named)

    monkeypatch.setattr(relief, "refine", spy)
    said: dict = {}
    stage = job_runner._relief_stage({"framing": FRAMING}, scene["weights"], None, said)
    F.finish_head(scene["mesh"], picture, {"texture": 2}, reshape=stage)
    assert got["framing"] == FRAMING and got["inside"] == (0, 0, SIZE, 400) and got["side"] == SQUARE
    assert got["focal"] == pytest.approx(view.focal) and np.allclose(got["camera_position"], view.place)
    assert got["source_rgb"] is picture.source and got["alpha"] is picture.alpha and got["raster"] is raster.depth_buffer
    assert got["options"] == {"relief": 1, "weights_dir": str(scene["weights"])}
    assert "run_depth" not in got and "run_normals" not in got            # the installed files' runners are used
    points = np.array([[10.0, 20.0], [500.0, 900.0]])
    assert np.allclose(got["square_to_source"](points), picture.to_source(points))
    uv, depth = got["project"](m.vertices[:5])
    assert np.allclose(uv, view.project(m.vertices[:5])[0]) and np.allclose(depth, view.project(m.vertices[:5])[1])
    assert got["vertices"].shape == m.vertices.shape and got["faces"].shape == m.faces.shape
    # a job that does not say how its pictures are framed is read at the app's default
    job_runner._relief_stage({}, scene["weights"], None, {})(m.vertices.astype(np.float64), m.faces, view, picture)
    assert got["framing"] == job_runner.FRAMING == 2.6


def test_a_look_that_would_not_fit_in_memory_keeps_its_head_as_it_is(scene, monkeypatch):
    plain = F.finish_head(scene["mesh"], scene["picture"], {"texture": 2})
    monkeypatch.setattr(job_runner, "_resident_mb", lambda: 4000)
    said: dict = {}
    tight = job_runner._relief_stage({"framing": FRAMING}, scene["weights"], 4000 + job_runner.RELIEF_NEEDS_MB - 1, said)
    head = F.finish_head(scene["mesh"], scene["picture"], {"texture": 2}, reshape=tight)
    assert scene["calls"] == []                                           # no model was loaded
    assert said == {"relief": False, "note": f"too little memory was left for the depth models (4000 MB held, "
                                             f"{4000 + job_runner.RELIEF_NEEDS_MB - 1} MB allowed)"}
    assert head.reshaped is False and head.photo is not None and np.array_equal(head.base.vertices, plain.base.vertices)
    assert job_runner._note(head, True, said).startswith("the face was not shaped from the photo: too little memory")
    # with room, with no limit given, and where the memory cannot be read, the stage runs
    for limit, held in ((4000 + job_runner.RELIEF_NEEDS_MB, 4000), (None, 4000), (5000.0, None)):
        monkeypatch.setattr(job_runner, "_resident_mb", lambda held=held: held)
        said = {}
        stage = job_runner._relief_stage({"framing": FRAMING}, scene["weights"], limit, said)
        assert F.finish_head(scene["mesh"], scene["picture"], {"texture": 2}, reshape=stage).reshaped and said["relief"]


def test_a_stage_that_leaves_the_head_alone_says_why(scene, monkeypatch):
    flat = np.zeros((relief.MODEL_SIDE, relief.MODEL_SIDE), np.float32)       # a reading that says nothing
    normals = np.zeros((3, relief.MODEL_SIDE, relief.MODEL_SIDE), np.float32)
    normals[2] = 1.0
    monkeypatch.setattr(relief, "runners", lambda weights: ((lambda rgb: flat), (lambda rgb: normals)))
    said: dict = {}
    stage = job_runner._relief_stage({"framing": FRAMING}, scene["weights"], None, said)
    head = F.finish_head(scene["mesh"], scene["picture"], {"texture": 2}, reshape=stage)
    assert head.reshaped is False and head.reshape_note is None and head.photo is not None
    assert said["relief"] is False and "does not match the head" in said["note"]
    assert job_runner._note(head, True, said) == f"the face was not shaped from the photo: {said['note']}"


def test_what_a_look_says_about_its_photo_and_its_relief():
    base = Mesh(vertices=np.zeros((3, 3), np.float32), faces=np.zeros((1, 3), np.uint32))
    note = job_runner._note
    assert note(Head(base=base), False, {}) is None                       # nothing asked, nothing to say
    assert note(Head(base=base, reshaped=True), True, {"relief": True, "note": ""}) is None
    # the photo's reason comes first: it is also why the face was not shaped
    lined = "the photo did not line up with the head (outlines overlap 0.80, over the head 0.75)"
    assert note(Head(base=base, note=lined, reshape_note=lined), True, {}) == lined
    assert note(Head(base=base, note=lined), False, {}) == lined
    # no photo was asked for, so the reason is told as the relief's
    assert note(Head(base=base, reshape_note=lined), True, {}) == f"the face was not shaped from the photo: {lined}"
    assert note(Head(base=base, reshape_note="ValueError: no"), True, {"note": "never read"}) == (
        "the face was not shaped from the photo: ValueError: no")
    assert note(Head(base=base), True, {"relief": False, "note": "too little of the face is seen"}) == (
        "the face was not shaped from the photo: too little of the face is seen")
    assert note(Head(base=base), True, {}) == "the face was not shaped from the photo"
    assert note(Head(base=base, reshape_note="x"), False, {"note": "y"}) is None   # the stage was not part of the job


# ------------------------------------------------------------------ the job
class _CutOutBackend:
    """A backend that, unlike the stand-in, is given a cut-out: it returns the bust for every picture. It writes
    down what it is asked, fails the looks named in `failing`, and when it is closed it notes how many depth model
    calls there had been by then and runs `on_close`."""
    name, version = "bust", "1"

    def __init__(self, calls: list, failing: tuple[str, ...] = (), on_close=None) -> None:
        self.events: list[str] = []
        self.calls = calls
        self.failing = failing
        self.on_close = on_close
        self.calls_at_close: int | None = None

    def available(self, weights_dir) -> bool:
        return True

    def load(self, weights_dir, device, options) -> None:
        self.events.append("load")

    def generate(self, image, options) -> Mesh:
        self.events.append(f"generate {options['key']}")
        if options["key"] in self.failing:
            raise RuntimeError("the model failed")
        return _bust(60, 64, camera=SURE)

    def close(self) -> None:
        self.events.append("close")
        self.calls_at_close = len(self.calls)
        if self.on_close is not None:
            self.on_close()


def _run_job(tmp_path: Path, monkeypatch, capsys, scene, options: dict, failing: tuple[str, ...] = (), on_close=None,
             records: list | None = None, **extra) -> tuple[int, list[dict], list[str], _CutOutBackend]:
    """One job of two looks through gen3d.__main__.run, on the bust backend with a made-up cut-out. `records`
    collects a weak reference to every picture record the cut-out makes."""
    backend = _CutOutBackend(scene["calls"], failing, on_close)
    monkeypatch.setattr(backends, "get", lambda name: backend)
    picture = scene["picture"]
    square = Image.fromarray(np.dstack([picture.source, picture.alpha]), "RGBA").resize((SQUARE, SQUARE), Image.NEAREST)

    def prepare(image, opts):
        made = Prepared(rgba=square, source=np.asarray(image.convert("RGB")), alpha=picture.alpha, side=SQUARE, x0=0.0,
                        y0=0.0, sx=SQUARE / SIZE, sy=SQUARE / SIZE, ox=0.0, oy=0.0, inside=opts.get("inside"))
        if records is not None:
            records.append((opts["key"], weakref.ref(made)))
        return made

    monkeypatch.setattr(job_runner, "_prepare", lambda name: prepare)
    looks = []
    for key in ("l1", "l2"):
        path = tmp_path / f"{key}.png"
        Image.fromarray(picture.source).save(path)
        looks.append({"key": key, "image": str(path)})
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({"out_dir": str(tmp_path / "out"), "weights_dir": str(scene["weights"]),
                                    "generator": "fake", "options": {"detail": 2, "framing": FRAMING, **options},
                                    "looks": looks, **extra}), encoding="utf-8")
    capsys.readouterr()
    code = job_runner.run(job_runner.read_job(str(job_path)))
    lines = capsys.readouterr().out.split("\n")[:-1]
    rows = json.loads((tmp_path / "out" / "result.json").read_text())["looks"]
    return code, rows, lines, backend


def test_a_job_shapes_every_face_in_its_second_pass(tmp_path, monkeypatch, capsys, scene):
    code, rows, lines, backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 1})
    assert code == 0 and [(r["key"], r["ok"], r["relief"], r["textured"], r["note"]) for r in rows] == [
        ("l1", True, True, True, None), ("l2", True, True, True, None)]
    # the model was closed before the first depth model ran: the two never share the memory
    assert backend.events == ["load", "generate l1", "generate l2", "close"]
    assert backend.calls_at_close == 0
    assert [c[0] for c in scene["calls"]] == ["depth", "normals"] * 2
    assert [ln.split()[0] for ln in lines] == ["progress:", "made:", "made:", "look:", "progress:", "look:", "progress:"]
    assert [ln.split()[:3] for ln in lines[1:3]] == [["made:", "l1", "ok"], ["made:", "l2", "ok"]]


def test_a_job_without_the_relief_runs_no_depth_model(tmp_path, monkeypatch, capsys, scene):
    code, rows, _lines, _backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 0})
    assert code == 0 and scene["calls"] == []
    assert [(r["relief"], r["textured"], r["note"]) for r in rows] == [(False, True, None)] * 2
    # the files are not installed: no model runs, and no look carries a note about it (the app says it once)
    monkeypatch.setattr(relief, "available", lambda weights: False)
    code, rows, _lines, _backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 1})
    assert code == 0 and scene["calls"] == []
    assert [(r["relief"], r["textured"], r["note"]) for r in rows] == [(False, True, None)] * 2


def test_a_job_near_its_memory_limit_finishes_its_heads_without_the_relief(tmp_path, monkeypatch, capsys, scene):
    monkeypatch.setattr(job_runner, "_resident_mb", lambda: 4200)
    code, rows, _lines, _backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 1}, max_rss_mb=5000)
    assert code == 0 and scene["calls"] == []
    assert all(r["ok"] and r["textured"] and r["relief"] is False for r in rows)
    assert all(r["note"] == "the face was not shaped from the photo: too little memory was left for the depth models "
                            "(4200 MB held, 5000 MB allowed)" for r in rows)


def test_a_job_shapes_the_face_of_a_head_without_the_photo(tmp_path, monkeypatch, capsys, scene):
    code, rows, _lines, _backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 1, "texture": 0})
    assert code == 0 and [c[0] for c in scene["calls"]] == ["depth", "normals"] * 2
    assert [(r["relief"], r["textured"], r["texture"], r["note"]) for r in rows] == [(True, False, None, None)] * 2


def test_the_stand_in_never_runs_the_relief(tmp_path, monkeypatch, capsys, scene):
    """The stand-in is given the picture as it is: there is no cut-out to tell the person by, and its egg has no
    face. So it leaves the depth models alone also where they are installed."""
    path = tmp_path / "l1.png"
    Image.fromarray(scene["picture"].source).save(path)
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({"out_dir": str(tmp_path / "out"), "weights_dir": str(scene["weights"]),
                                    "generator": "fake", "options": {"detail": 2, "relief": 1},
                                    "looks": [{"key": "l1", "image": str(path)}]}), encoding="utf-8")
    assert job_runner.run(job_runner.read_job(str(job_path))) == 0
    row = json.loads((tmp_path / "out" / "result.json").read_text())["looks"][0]
    assert scene["calls"] == [] and row["ok"] and row["textured"] is True
    assert row["relief"] is False and row["note"] is None


def test_a_look_the_model_failed_keeps_nothing_in_memory_for_the_second_pass(tmp_path, monkeypatch, capsys, scene):
    """Its picture record (the full-size picture, up to 28 MB, and the cut-out) is let go when the look fails, not
    when the job ends: by the time the model is closed, before any head is finished, it is gone."""
    records: list = []
    alive_at_close: list = []

    def at_close():
        gc.collect()
        alive_at_close.extend(key for key, ref in records if ref() is not None)

    code, rows, lines, backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 1}, failing=("l2",),
                                          on_close=at_close, records=records)
    assert [key for key, _ in records] == ["l1", "l2"]
    assert alive_at_close == ["l1"]                                   # the look that worked waits for its second pass
    assert code == 0 and [(r["key"], r["ok"]) for r in rows] == [("l1", True), ("l2", False)]
    assert rows[1]["error"] == "RuntimeError: the model failed" and rows[0]["relief"] is True
    assert [ln.split()[:3] for ln in lines[1:3]] == [["made:", "l1", "ok"], ["made:", "l2", "error"]]
    gc.collect()
    assert all(ref() is None for _, ref in records)                  # and nothing is left once the job has ended


def test_a_stop_right_after_the_model_loaded_still_closes_it_and_leaves_a_result(tmp_path, monkeypatch, capsys, scene):
    def stop(weights):
        raise job_runner.Stopped()

    monkeypatch.setattr(relief, "available", stop)                    # the first thing after the load
    code, rows, lines, backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 1})
    assert code == job_runner.EXIT_STOPPED and rows == [] and lines == []
    assert backend.events == ["load", "close"]                        # closed once, and no look was asked for
    doc = json.loads((tmp_path / "out" / "result.json").read_text())
    assert doc["generator"] == {"name": "bust", "version": "1"} and doc["looks"] == []


def test_a_picture_that_cannot_be_read_again_costs_the_photo_not_the_head(tmp_path, monkeypatch, capsys, scene):
    """The second pass reads each picture from its file again. When that fails, or the file is no longer the picture
    the cut-out was made from, the head the model made is still written, with its own colours, and says why."""
    def spoil():
        (tmp_path / "l1.png").unlink()
        Image.new("RGB", (SIZE // 2, SIZE // 2), (10, 20, 30)).save(tmp_path / "l2.png")

    code, rows, _lines, _backend = _run_job(tmp_path, monkeypatch, capsys, scene, {"relief": 1}, on_close=spoil)
    assert code == 0 and scene["calls"] == []                         # no depth model ran without its picture
    assert [(r["ok"], r["textured"], r["relief"], r["error"]) for r in rows] == [(True, False, False, None)] * 2
    assert rows[0]["note"].startswith("the look's picture could not be read again (FileNotFoundError")
    assert rows[1]["note"] == ("the look's picture could not be read again (ValueError: it is not the picture the "
                               "cut-out was made from)")
    for key in ("l1", "l2"):
        assert (tmp_path / "out" / f"{key}.glb").read_bytes()[:4] == b"glTF"
