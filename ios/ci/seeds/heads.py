"""The 3D heads area's seed (ios/UITests/HeadsScreens.swift): "Heads Person", with four photos and a finished
model of two looks, and "Heads Newcomer", with one photo and no model.

The model is stored the way the server stores one (app/analysis/model3d.py): DATA_DIR/people3d/<person id>/ holds
`current`, naming the folder in use, and that folder holds model.json, <key>.glb and <key>.jpg. Each look names
the face it was made from, so the server lists it (the face is the person's, on a photo in the library). The heads
are drawn here with numpy as small egg-shaped heads in the generator's file layout (gen3d/glb.py): the head with
its colours, and the photo layer over its front with its texture and weights. No model code runs, and the CI
server has no generator, so the tab shows "not installed" for a new model while the stored one shows.
"""
from __future__ import annotations

import io
import json
import math
import struct

import numpy as np
from PIL import Image

STAMP = "20260415T120000"
PERSON = "Heads Person"
NEWCOMER = "Heads Newcomer"
SKIN = (226, 186, 152)
HAIR = (52, 36, 26)
BOX = (0.40, 0.24, 0.18, 0.26)     # the face in each photo: x, y, w, h as shares of the picture
LOOKS = [
    # key, age, band, hair, hair words, facial hair, facial hair words, photo, relief, note
    ("heads-young", 29.0, "25 to 34", 0.45, "short hair", 0.30, "little or no facial hair", 0, False,
     "the face was not shaped: the depth model is not installed"),
    ("heads-older", 41.0, "35 to 44", 0.45, "short hair", 0.70, "a beard", 3, True, None),
]


def library(kit) -> dict:
    person = kit.make_person(kit.conn, PERSON)
    newcomer = kit.make_person(kit.conn, NEWCOMER)
    photos = []
    for i in range(4):
        taken = f"2026-0{i + 1}-{10 + i}T10:00:00.000Z"
        img = kit.scene(5100 + i, [(BOX, SKIN, HAIR)])
        item = kit.make_item(kit.conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename=f"HEADS_{i + 1}.JPG",
                             taken_at=taken, camera="CI Camera", category="personal")
        kit.previews(item, img)
        face = kit.make_face(kit.conn, int(item["id"]), person, box=BOX, size_px=int(BOX[2] * kit.W),
                             assigned_by="user")
        kit.crop(img, BOX).save(kit.storage.face_crop_path(face), "WEBP", quality=85)
        photos.append((item, face, taken, img))
    img = kit.scene(5200, [(BOX, (190, 140, 110), (20, 20, 24))])
    item = kit.make_item(kit.conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename="HEADS_9.JPG",
                         taken_at="2026-04-20T10:00:00.000Z", camera="CI Camera", category="personal")
    kit.previews(item, img)
    face = kit.make_face(kit.conn, int(item["id"]), newcomer, box=BOX, size_px=int(BOX[2] * kit.W), assigned_by="user")
    kit.crop(img, BOX).save(kit.storage.face_crop_path(face), "WEBP", quality=85)

    folder = kit.storage.root() / "people3d" / str(person) / STAMP
    folder.mkdir(parents=True, exist_ok=True)
    looks = []
    for key, age, band, hair, hair_words, facial, facial_words, which, relief, note in LOOKS:
        item, face, taken, img = photos[which]
        picture = _picture(img, kit.W, kit.H)
        picture.save(folder / f"{key}.jpg", "JPEG", quality=88)
        buf = io.BytesIO()
        picture.save(buf, "JPEG", quality=90)
        glb, vertices, faces = head_glb(buf.getvalue(), bearded=facial > 0.5)
        (folder / f"{key}.glb").write_bytes(glb)
        looks.append({
            "key": key, "label": f"About {int(round(age))}, {hair_words}, {facial_words}", "age": age,
            "age_band": band, "hair": hair, "hair_label": hair_words, "facial_hair": facial,
            "facial_hair_label": facial_words, "photos": 2, "face_id": face, "item_id": int(item["id"]),
            "item_uid": item["uid"], "taken_at": taken, "faults": [], "box": list(BOX), "mesh": f"{key}.glb",
            "picture": f"{key}.jpg", "bytes": len(glb), "vertices": vertices, "faces": faces, "textured": True,
            "texture": [picture.width, picture.height], "overlap": 0.95, "overlap_head": 0.94, "relief": relief,
            "note": note,
        })
    (folder / "model.json").write_text(json.dumps({
        "rev": STAMP, "person_id": person, "generated_at": "2026-04-15T12:00:00.000Z", "seconds": 42.0,
        "generator": {"name": "fake", "version": "1"},
        "options": {"max_looks": 6, "detail": 2, "framing": 2.6, "smoothing": 2, "texture": 2, "relief": 1},
        "birth_year": 1997, "scanned": 4, "usable": 4, "peak_rss_mb": None, "generator_memory_mb": None,
        "looks": looks,
    }), encoding="utf-8")
    (folder.parent / "current").write_text(STAMP, encoding="ascii")
    return {"person": person, "newcomer": newcomer, "looks": len(looks)}


def _picture(img: Image.Image, width: int, height: int) -> Image.Image:
    """The look's picture: a square 2.6 face widths wide around the face, at 512 px."""
    x, y, w, h = BOX
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    half = 1.3 * w * width
    box = (int(cx - half), int(max(0, cy - half)), int(cx + half), int(min(height, cy + half)))
    return img.crop(box).resize((512, 512), Image.Resampling.LANCZOS)


def _linear16(srgb: np.ndarray) -> np.ndarray:
    c = np.asarray(srgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.rint(linear * 65535.0).astype("<u2")


def head_glb(texture: bytes, bearded: bool) -> tuple[bytes, int, int]:
    """An egg-shaped head (+Y up, +Z towards the viewer) with its colours, and the photo layer over its front, as
    binary glTF in the generator's layout. Returns the bytes, the vertex count and the triangle count."""
    seg_u, seg_v = 64, 48
    around = np.linspace(-math.pi, math.pi, seg_u + 1)     # 0 is the front
    down = np.linspace(0.0, math.pi, seg_v + 1)            # from the top
    t, p = np.meshgrid(around, down)
    up = np.cos(p)
    narrow = 0.86 + 0.14 * (up + 1) / 2                     # the jaw narrower than the crown
    x = 0.45 * np.sin(p) * np.sin(t) * narrow
    y = 0.62 * up
    z = 0.50 * np.sin(p) * np.cos(t) * narrow
    z = z + 0.07 * np.exp(-((x / 0.07) ** 2 + ((y + 0.02) / 0.10) ** 2)) * (z > 0)   # a nose
    v = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    width = seg_u + 1
    tris = []
    for j in range(seg_v):
        for i in range(seg_u):
            a, b, c, d = j * width + i, j * width + i + 1, (j + 1) * width + i, (j + 1) * width + i + 1
            tris.append((a, c, b))
            tris.append((b, c, d))
    f = np.asarray(tris, dtype=np.uint32)

    n = np.zeros_like(v)
    corners = v[f.astype(np.int64)]
    face_n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for k in range(3):
        np.add.at(n, f[:, k].astype(np.int64), face_n)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n = n.astype(np.float32)

    fy, fz = v[:, 1], v[:, 2]
    hairy = np.maximum(np.clip((fy - 0.22) / 0.10, 0, 1),
                       np.clip((-fz - 0.05) / 0.10, 0, 1) * np.clip((fy + 0.30) / 0.15, 0, 1))
    if bearded:
        hairy = np.maximum(hairy, 0.9 * np.clip((-fy - 0.22) / 0.08, 0, 1) * (fz > 0.05))
    rgb = np.asarray(SKIN, np.float32)[None, :] * (1 - hairy[:, None]) + np.asarray(HAIR, np.float32)[None, :] * hairy[:, None]
    colours = np.full((len(v), 4), 65535, dtype="<u2")
    colours[:, :3] = _linear16(np.clip(rgb, 0, 255).astype(np.uint8))

    weight = np.clip((n[:, 2] - 0.35) / 0.40, 0.0, 1.0)
    keep = (weight[f.astype(np.int64)] > 0.004).any(axis=1)
    layer = f[keep].astype(np.int64)
    ids = np.unique(layer)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids))
    layer_faces = remap[layer].astype(np.uint32)
    lv, ln = v[ids], n[ids]
    uv = np.stack([0.5 + lv[:, 0] * 0.45, 0.5 - lv[:, 1] * 0.42], axis=1).clip(0, 1).astype(np.float32)
    weights = np.full((len(ids), 4), 65535, dtype="<u2")
    weights[:, 3] = np.rint(weight[ids] * 65535).astype("<u2")

    parts: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []

    def view(data: bytes, target: int | None) -> int:
        offset = sum(len(part) for part in parts)
        parts.append(data + b"\0" * ((4 - len(data) % 4) % 4))
        entry = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            entry["target"] = target
        views.append(entry)
        return len(views) - 1

    def add(data: bytes, target: int, accessor: dict) -> int:
        accessors.append({"bufferView": view(data, target), **accessor})
        return len(accessors) - 1

    def points(arr: np.ndarray) -> dict:
        return {
            "POSITION": add(np.ascontiguousarray(arr, dtype="<f4").tobytes(), 34962, {
                "componentType": 5126, "count": int(len(arr)), "type": "VEC3",
                "min": [float(c) for c in arr.min(axis=0)], "max": [float(c) for c in arr.max(axis=0)]}),
        }

    def normals(arr: np.ndarray) -> int:
        return add(np.ascontiguousarray(arr, dtype="<f4").tobytes(), 34962,
                   {"componentType": 5126, "count": int(len(arr)), "type": "VEC3"})

    def colour(arr: np.ndarray) -> int:
        return add(np.ascontiguousarray(arr, dtype="<u2").tobytes(), 34962,
                   {"componentType": 5123, "normalized": True, "count": int(len(arr)), "type": "VEC4"})

    def indices(arr: np.ndarray) -> int:
        return add(np.ascontiguousarray(arr, dtype="<u4").tobytes(), 34963,
                   {"componentType": 5125, "count": int(arr.size), "type": "SCALAR"})

    skin = {**points(v), "NORMAL": normals(n), "COLOR_0": colour(colours)}
    skin_indices = indices(f)
    photo = {**points(lv), "NORMAL": normals(ln)}
    photo["TEXCOORD_0"] = add(np.ascontiguousarray(uv, dtype="<f4").tobytes(), 34962,
                              {"componentType": 5126, "count": int(len(uv)), "type": "VEC2"})
    photo["COLOR_0"] = colour(weights)
    photo_indices = indices(layer_faces)
    image = view(texture, None)
    blob = b"".join(parts)

    def material(name: str, **more) -> dict:
        rough = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9}
        rough.update(more.pop("pbr", {}))
        return {"name": name, "pbrMetallicRoughness": rough, **more}

    doc = {
        "asset": {"version": "2.0", "generator": "pcc-gen3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "head"}],
        "meshes": [{"name": "head", "primitives": [
            {"attributes": skin, "indices": skin_indices, "material": 0, "mode": 4},
            {"attributes": photo, "indices": photo_indices, "material": 1, "mode": 4}]}],
        "materials": [material("skin", doubleSided=True),
                      material("photo", doubleSided=False, alphaMode="BLEND", pbr={"baseColorTexture": {"index": 0}})],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": image, "mimeType": "image/jpeg"}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    text = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    text += b" " * ((4 - len(text) % 4) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    data = b"".join([struct.pack("<III", 0x46546C67, 2, total), struct.pack("<II", len(text), 0x4E4F534A), text,
                     struct.pack("<II", len(blob), 0x004E4942), blob])
    return data, int(len(v)), int(len(f))
