"""Binary glTF 2.0 for one head (numpy and struct only).

write(path, mesh): POSITION and NORMAL as float32, COLOR_0 as normalised unsigned shorts (four per vertex, the fourth
65535), uint32 indices, one rough material. A mesh's colours are sRGB bytes, as in the picture they came from; glTF
vertex colours are linear, so the writer converts them, and keeps 16 bits because 8 linear bits band in dark hair
and shadow. read(path) reads such a file back, colours as sRGB bytes again (the tests and nothing else need it).

write(path, head) for a gen3d.mesh.Head with a photo layer writes two primitives in the one mesh:
- 0, material "skin": the whole surface as above, opaque, double sided;
- 1, material "photo": the layer's triangles again (the same positions and normals, so the two lie exactly on each
  other), TEXCOORD_0 as float32 with (0, 0) at the texture's top left corner, COLOR_0 white with the layer's weight
  as its alpha, a base colour texture, alphaMode BLEND, single sided. The texture is the JPEG, stored in the file
  (images[0] points at a buffer view), with one sampler: linear, mipmaps, clamped to the edge.
A viewer draws 1 over 0, so the photo fades into the surface's own colours where its weight falls. A head without
a layer is written exactly as a plain mesh is, and files written before the layer existed read as before.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from gen3d.mesh import Head, Mesh, vertex_normals

MAGIC = 0x46546C67  # "glTF"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
GENERATOR = "pcc-gen3d"
LINEAR, LINEAR_MIPMAP_LINEAR, CLAMP_TO_EDGE = 9729, 9987, 33071
ROUGHNESS = 0.9


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def to_linear16(srgb: np.ndarray) -> np.ndarray:
    """sRGB bytes -> linear values as 16-bit integers."""
    c = np.asarray(srgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.rint(linear * 65535.0).astype("<u2")


def to_srgb8(linear16: np.ndarray) -> np.ndarray:
    """The way back: 16-bit linear values -> sRGB bytes."""
    c = np.asarray(linear16, dtype=np.float64) / 65535.0
    srgb = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def encode(mesh: Mesh | Head) -> bytes:
    if isinstance(mesh, Head):
        if mesh.photo is None or mesh.texture is None:
            return encode(mesh.base)
        return _encode_head(mesh)
    v = np.ascontiguousarray(mesh.vertices, dtype="<f4")
    f = np.ascontiguousarray(mesh.faces, dtype="<u4")
    n = np.ascontiguousarray(mesh.normals if mesh.normals is not None else vertex_normals(v, f.astype(np.int64)),
                             dtype="<f4")
    parts: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def add(data: bytes, target: int, accessor: dict[str, Any]) -> int:
        offset = sum(len(p) for p in parts)
        parts.append(_pad(data, b"\0"))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        accessors.append({"bufferView": len(views) - 1, **accessor})
        return len(accessors) - 1

    attributes = {
        "POSITION": add(v.tobytes(), ARRAY_BUFFER, {
            "componentType": FLOAT, "count": int(v.shape[0]), "type": "VEC3",
            "min": [float(x) for x in v.min(axis=0)], "max": [float(x) for x in v.max(axis=0)]}),
        "NORMAL": add(n.tobytes(), ARRAY_BUFFER, {"componentType": FLOAT, "count": int(n.shape[0]), "type": "VEC3"}),
    }
    if mesh.colors is not None:
        rgba = np.full((v.shape[0], 4), 65535, dtype="<u2")
        rgba[:, :3] = to_linear16(np.asarray(mesh.colors, dtype=np.uint8)[:, :3])
        attributes["COLOR_0"] = add(rgba.tobytes(), ARRAY_BUFFER, {
            "componentType": USHORT, "normalized": True, "count": int(v.shape[0]), "type": "VEC4"})
    indices = add(f.tobytes(), ELEMENT_ARRAY_BUFFER, {"componentType": UINT, "count": int(f.size), "type": "SCALAR"})
    blob = b"".join(parts)
    doc = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "head"}],
        "meshes": [{"name": "head", "primitives": [{"attributes": attributes, "indices": indices, "material": 0,
                                                    "mode": 4}]}],
        "materials": [{"name": "skin", "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                                "roughnessFactor": 0.9}}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    text = _pad(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(text) + 8 + len(blob)
    return b"".join([struct.pack("<III", MAGIC, 2, total), struct.pack("<II", len(text), JSON_CHUNK), text,
                     struct.pack("<II", len(blob), BIN_CHUNK), blob])


def _encode_head(head: Head) -> bytes:
    base, layer = head.base, head.photo
    assert layer is not None and head.texture is not None
    v = np.ascontiguousarray(base.vertices, dtype="<f4")
    f = np.ascontiguousarray(base.faces, dtype="<u4")
    n = np.ascontiguousarray(base.normals if base.normals is not None else vertex_normals(v, f.astype(np.int64)),
                             dtype="<f4")
    parts: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def view(data: bytes, target: int | None) -> int:
        offset = sum(len(p) for p in parts)
        parts.append(_pad(data, b"\0"))
        entry: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            entry["target"] = target
        views.append(entry)
        return len(views) - 1

    def add(data: bytes, target: int, accessor: dict[str, Any]) -> int:
        accessors.append({"bufferView": view(data, target), **accessor})
        return len(accessors) - 1

    def corners(pos: np.ndarray, nor: np.ndarray) -> dict[str, int]:
        return {
            "POSITION": add(pos.tobytes(), ARRAY_BUFFER, {
                "componentType": FLOAT, "count": int(pos.shape[0]), "type": "VEC3",
                "min": [float(x) for x in pos.min(axis=0)], "max": [float(x) for x in pos.max(axis=0)]}),
            "NORMAL": add(nor.tobytes(), ARRAY_BUFFER, {"componentType": FLOAT, "count": int(nor.shape[0]), "type": "VEC3"}),
        }

    def colour(rgba: np.ndarray) -> int:
        return add(np.ascontiguousarray(rgba, dtype="<u2").tobytes(), ARRAY_BUFFER, {
            "componentType": USHORT, "normalized": True, "count": int(rgba.shape[0]), "type": "VEC4"})

    def indices(faces: np.ndarray) -> int:
        return add(np.ascontiguousarray(faces, dtype="<u4").tobytes(), ELEMENT_ARRAY_BUFFER,
                   {"componentType": UINT, "count": int(faces.size), "type": "SCALAR"})

    skin = corners(v, n)
    if base.colors is not None:
        rgba = np.full((v.shape[0], 4), 65535, dtype="<u2")
        rgba[:, :3] = to_linear16(np.asarray(base.colors, dtype=np.uint8)[:, :3])
        skin["COLOR_0"] = colour(rgba)
    skin_indices = indices(f)

    ids = np.asarray(layer.vertex_ids, dtype=np.int64)
    photo = corners(np.ascontiguousarray(v[ids]), np.ascontiguousarray(n[ids]))
    photo["TEXCOORD_0"] = add(np.ascontiguousarray(layer.uv, dtype="<f4").tobytes(), ARRAY_BUFFER, {
        "componentType": FLOAT, "count": int(ids.shape[0]), "type": "VEC2"})
    weight = np.full((ids.shape[0], 4), 65535, dtype="<u2")
    weight[:, 3] = np.rint(np.clip(np.asarray(layer.alpha, dtype=np.float64), 0.0, 1.0) * 65535.0).astype("<u2")
    photo["COLOR_0"] = colour(weight)
    photo_indices = indices(np.asarray(layer.faces))
    image = view(bytes(head.texture), None)

    def material(name: str, **more: Any) -> dict[str, Any]:
        rough = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": ROUGHNESS}
        rough.update(more.pop("pbr", {}))
        return {"name": name, "pbrMetallicRoughness": rough, **more}

    blob = b"".join(parts)
    doc = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "head"}],
        "meshes": [{"name": "head", "primitives": [
            {"attributes": skin, "indices": skin_indices, "material": 0, "mode": 4},
            {"attributes": photo, "indices": photo_indices, "material": 1, "mode": 4}]}],
        "materials": [material("skin", doubleSided=True),
                      material("photo", doubleSided=False, alphaMode="BLEND", pbr={"baseColorTexture": {"index": 0}})],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR, "wrapS": CLAMP_TO_EDGE, "wrapT": CLAMP_TO_EDGE}],
        "images": [{"bufferView": image, "mimeType": "image/jpeg"}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    text = _pad(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(text) + 8 + len(blob)
    return b"".join([struct.pack("<III", MAGIC, 2, total), struct.pack("<II", len(text), JSON_CHUNK), text,
                     struct.pack("<II", len(blob), BIN_CHUNK), blob])


def write(path: str | os.PathLike[str], mesh: Mesh | Head) -> int:
    """Write the mesh, or the head, as a .glb (to a scratch name first, then renamed into place). Returns the byte
    count."""
    data = encode(mesh)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return len(data)


_DTYPES = {FLOAT: "<f4", UBYTE: "u1", UINT: "<u4", USHORT: "<u2"}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read(path: str | os.PathLike[str]) -> dict[str, Any]:
    """{"json", "positions", "normals", "colors" (n, 4 sRGB uint8, or None), "indices"} of a .glb written by
    write(): the whole surface. With a photo layer also "photo": {"positions", "normals", "uv", "alpha" (0..1),
    "indices"} and "texture" (the JPEG's bytes); both None without one."""
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != MAGIC or version != 2 or total != len(data):
        raise ValueError("not a binary glTF 2.0 file")
    n_json, kind = struct.unpack_from("<II", data, 12)
    if kind != JSON_CHUNK:
        raise ValueError("the first chunk is not JSON")
    doc = json.loads(data[20:20 + n_json].decode("utf-8"))
    n_bin, kind = struct.unpack_from("<II", data, 20 + n_json)
    if kind != BIN_CHUNK:
        raise ValueError("the second chunk is not binary")
    blob = data[28 + n_json:28 + n_json + n_bin]

    def accessor(index: int) -> np.ndarray:
        a = doc["accessors"][index]
        view = doc["bufferViews"][a["bufferView"]]
        width = _WIDTH[a["type"]]
        arr = np.frombuffer(blob, dtype=_DTYPES[a["componentType"]], count=a["count"] * width,
                            offset=view.get("byteOffset", 0) + a.get("byteOffset", 0))
        return arr.reshape(a["count"], width) if width > 1 else arr

    prim = doc["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    colours = None
    if "COLOR_0" in attrs:
        stored = accessor(attrs["COLOR_0"])
        colours = np.full(stored.shape, 255, dtype=np.uint8)
        colours[:, :3] = to_srgb8(stored[:, :3])
    photo, texture = None, None
    if len(doc["meshes"][0]["primitives"]) > 1:
        second = doc["meshes"][0]["primitives"][1]
        more = second["attributes"]
        photo = {
            "positions": accessor(more["POSITION"]),
            "normals": accessor(more["NORMAL"]) if "NORMAL" in more else None,
            "uv": accessor(more["TEXCOORD_0"]),
            "alpha": accessor(more["COLOR_0"])[:, 3].astype(np.float64) / 65535.0,
            "indices": accessor(second["indices"]),
        }
        held = doc["bufferViews"][doc["images"][0]["bufferView"]]
        texture = bytes(blob[held.get("byteOffset", 0):held.get("byteOffset", 0) + held["byteLength"]])
    return {
        "json": doc,
        "positions": accessor(attrs["POSITION"]),
        "normals": accessor(attrs["NORMAL"]) if "NORMAL" in attrs else None,
        "colors": colours,
        "indices": accessor(prim["indices"]),
        "photo": photo,
        "texture": texture,
    }
