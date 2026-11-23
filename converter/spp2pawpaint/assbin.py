"""Reader for the Assimp binary dump Substance stores as paint/scene.assbin.

Substance writes a variant of Assimp's .assbin: a 512 byte header, then
chunks that start with a u64 size (counting the size field itself) instead
of a magic + u32 size, and every vertex array carries its own u32 count.
Faces are written as (u32 polygon-start flag, u32 index count, indices),
so the flag lets us rebuild the original quads/ngons from triangles.
"""

import struct
import numpy as np

HAS_POSITIONS = 0x1
HAS_NORMALS = 0x2
HAS_TANGENTS = 0x4
UV_BASE = 0x100
COLOR_BASE = 0x10000


class MeshPart:
    """One Assimp mesh (Substance keeps one per node / material pair)."""

    def __init__(self, name, material, positions, normals, uvs, tris, poly_start):
        self.name = name
        self.material = material      # index into Scene.material_names
        self.positions = positions    # (N, 3) float32
        self.normals = normals        # (N, 3) float32 or None
        self.uvs = uvs                # list of (N, 2) float32
        self.tris = tris              # (F, 3) uint32
        self.poly_start = poly_start  # (F,) bool: True where a new polygon begins


class Scene:
    def __init__(self):
        self.parts = []
        self.material_names = []
        # Uniform scale Substance applied to normalise the mesh (stored vertex =
        # original * normalize_scale), from ALG_PTV_TOTAL_TRANSFORM. None if absent.
        self.normalize_scale = None


def _read_ptv_scale(b, start, end):
    """Find node metadata 'ALG_PTV_TOTAL_TRANSFORM' and return its uniform scale.

    Metadata layout (Substance assbin): u32 nKeys, nKeys x aiString, u32 nValues,
    then per value u32 type + payload (type 6 = 4x4 float matrix, row-major).
    """
    key = b"ALG_PTV_TOTAL_TRANSFORM"
    k = b.find(key, start, end)
    if k < 0:
        return None
    # walk back to the start of the key list: try a few preceding keys
    for probe in range(k - 4, max(start, k - 400), -1):
        try:
            nkeys = struct.unpack_from("<I", b, probe)[0]
            if not 1 <= nkeys <= 16:
                continue
            o = probe + 4
            keys = []
            for _ in range(nkeys):
                n = struct.unpack_from("<I", b, o)[0]
                if not 1 <= n <= 128:
                    raise ValueError
                keys.append(b[o + 4:o + 4 + n])
                o += 4 + n
            if key not in keys:
                continue
            nvals = struct.unpack_from("<I", b, o)[0]
            o += 4
            if nvals != nkeys:
                continue
            for kk in keys:
                t = struct.unpack_from("<I", b, o)[0]
                o += 4
                if t == 6:
                    m = np.frombuffer(b, np.float32, 16, o).reshape(4, 4)
                    o += 64
                    if kk == key:
                        s = float(np.linalg.norm(m[0, :3]))
                        return s if s > 0 else None
                else:
                    raise ValueError  # unknown payload size: give up
        except (ValueError, struct.error):
            continue
    return None


def _arr(b, o, comp):
    n = struct.unpack_from("<I", b, o)[0]
    o += 4
    a = np.frombuffer(b, np.float32, n * comp, o).reshape(-1, comp)
    return a, o + n * comp * 4


def _skip_node(b, o):
    size = struct.unpack_from("<Q", b, o)[0]
    return o + size


def _read_material_names(b, o, end):
    """Material chunks hold aiMaterial properties; we only want ?mat.name."""
    names = []
    key = b"?mat.name"
    pos = o
    while True:
        k = b.find(key, pos, end)
        if k < 0:
            break
        p = k + len(key)
        # after the key: u32 semantic, u32 index, u32 type, u32 data length, data
        try:
            sem, idx, ptype, dlen = struct.unpack_from("<4I", b, p)
            data = b[p + 16:p + 16 + dlen]
            sl = struct.unpack_from("<I", data, 0)[0]
            names.append(data[4:4 + sl].decode("utf-8", "replace"))
        except struct.error:
            names.append("")
        pos = p
    return names


def parse(b):
    if not b.startswith(b"ASSIMP.binary-dump"):
        raise ValueError("not an Assimp binary dump")
    sc = Scene()
    o = 512
    scene_size = struct.unpack_from("<Q", b, o)[0]
    scene_end = o + scene_size
    o += 8
    o += 4  # scene flags
    node_end = _skip_node(b, o)  # node hierarchy
    sc.normalize_scale = _read_ptv_scale(b, o, node_end)
    o = node_end
    nmeshes = struct.unpack_from("<I", b, o)[0]
    o += 4
    for _ in range(nmeshes):
        size = struct.unpack_from("<Q", b, o)[0]
        end = o + size
        o += 8
        n = struct.unpack_from("<I", b, o)[0]
        name = b[o + 4:o + 4 + n].decode("utf-8", "replace")
        o += 4 + n
        prim, mat, flags = struct.unpack_from("<3I", b, o)
        o += 12
        pos, o = _arr(b, o, 3)
        nrm = None
        if flags & HAS_NORMALS:
            nrm, o = _arr(b, o, 3)
        if flags & HAS_TANGENTS:
            _, o = _arr(b, o, 3)
            _, o = _arr(b, o, 3)
        for k in range(8):
            if flags & (COLOR_BASE << k):
                _, o = _arr(b, o, 4)
        uvs = []
        for k in range(8):
            if flags & (UV_BASE << k):
                o += 4  # number of UV components
                uv, o = _arr(b, o, 3)
                uvs.append(np.ascontiguousarray(uv[:, :2]))
        nf = struct.unpack_from("<I", b, o)[0]
        o += 4
        faces = np.frombuffer(b, np.uint32, nf * 5, o).reshape(-1, 5)
        if nf and not np.all(faces[:, 1] == 3):
            raise ValueError("mesh %r has non-triangle faces" % name)
        sc.parts.append(MeshPart(name, mat, np.array(pos), None if nrm is None else np.array(nrm),
                                 uvs, np.ascontiguousarray(faces[:, 2:5]),
                                 faces[:, 0].astype(bool)))
        o = end
    sc.material_names = _read_material_names(b, o, scene_end)
    return sc
