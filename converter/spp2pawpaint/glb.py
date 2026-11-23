"""Write the .spp mesh out as a glTF binary (.glb) PawPaint can open.

One glTF mesh per Substance mesh part (keeping part names, which PawPaint
shows and Substance geometry masks refer to) and one glTF material per
texture set (keeping texture set names, which the importer uses to match
layer bundles to materials).
"""

import json
import struct

import numpy as np


def _pad4(b, fill=b"\x00"):
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(scene, path, scale=1.0):
    """scale multiplies every vertex position (e.g. back to metres)."""
    bin_chunks = []
    offset = 0
    buffer_views = []
    accessors = []

    def add(data, target=None):
        nonlocal offset
        raw = _pad4(data.tobytes())
        view = {"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes}
        if target:
            view["target"] = target
        buffer_views.append(view)
        bin_chunks.append(raw)
        offset += len(raw)
        return len(buffer_views) - 1

    def accessor(arr, comp_type, typ, target=None, minmax=False):
        bv = add(arr, target)
        acc = {"bufferView": bv, "componentType": comp_type, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"] = [float(x) for x in arr.min(0)]
            acc["max"] = [float(x) for x in arr.max(0)]
        accessors.append(acc)
        return len(accessors) - 1

    names = list(scene.material_names)
    used = sorted({p.material for p in scene.parts})
    for i in used:
        while len(names) <= i:
            names.append("")
    materials = [{"name": names[i] or ("Material_%d" % i),
                  "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                                           "metallicFactor": 0.0, "roughnessFactor": 0.5}}
                 for i in range(len(names))]

    meshes, nodes = [], []
    for part in scene.parts:
        if len(part.tris) == 0:
            continue
        pos = np.ascontiguousarray(part.positions * scale, np.float32)
        attrs = {"POSITION": accessor(pos, 5126, "VEC3", 34962, minmax=True)}
        if part.normals is not None:
            n = part.normals.astype(np.float32)
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0, 1, 0], np.float32))
            attrs["NORMAL"] = accessor(np.ascontiguousarray(n, np.float32), 5126, "VEC3", 34962)
        for k, uv in enumerate(part.uvs[:2]):
            g = np.ascontiguousarray(np.c_[uv[:, 0], 1.0 - uv[:, 1]], np.float32)  # glTF v points down
            attrs["TEXCOORD_%d" % k] = accessor(g, 5126, "VEC2", 34962)
        idx = part.tris.astype(np.uint32).reshape(-1)
        ia = accessor(np.ascontiguousarray(idx), 5125, "SCALAR", 34963)
        meshes.append({"name": part.name,
                       "primitives": [{"attributes": attrs, "indices": ia, "material": int(part.material)}]})
        nodes.append({"name": part.name, "mesh": len(meshes) - 1})

    gltf = {
        "asset": {"version": "2.0", "generator": "spp2pawpaint"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": offset}],
    }
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    binary = b"".join(bin_chunks)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(binary), 0x004E4942))
        f.write(binary)
