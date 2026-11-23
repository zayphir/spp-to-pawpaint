"""Open a Substance 3D Painter .spp project and expose its parts.

An .spp is an HDF5 file. The pieces this converter uses:
  paint/document.bin      layer stacks for every texture set (see docbin)
  paint/scene.assbin      the mesh (see assbin)
  texture-cache/<uid>/<stackuid>/<channel>
                          LZ4 compressed GPU readbacks of each texture set's
                          final channels, keyed by sceneMaterialUID and the
                          channel bit flag
  resources/<sha>.image   embedded images (baked maps, imported bitmaps)
  resources.toc           names of the embedded resources
"""

import re
import struct

import h5py
import lz4.block
import numpy as np

from . import assbin, docbin
from .qtvariant import read_map

# Substance channel bit flags (DataChannel.type / texture-cache keys).
CH_BASECOLOR = 1 << 0
CH_HEIGHT = 1 << 1
CH_SPECULAR = 1 << 2
CH_OPACITY = 1 << 3
CH_EMISSIVE = 1 << 4
CH_DISPLACEMENT = 1 << 5
CH_GLOSSINESS = 1 << 6
CH_ROUGHNESS = 1 << 7
CH_METALLIC = 1 << 13
CH_NORMAL = 1 << 22

CHANNEL_NAMES = {
    CH_BASECOLOR: "basecolor", CH_HEIGHT: "height", CH_SPECULAR: "specular",
    CH_OPACITY: "opacity", CH_EMISSIVE: "emissive", CH_DISPLACEMENT: "displacement",
    CH_GLOSSINESS: "glossiness", CH_ROUGHNESS: "roughness", CH_METALLIC: "metallic",
    CH_NORMAL: "normal",
}

# DataAdditionalMap.usage -> name. 0, 2, 3, 4, 5 seen in real files; 1 (AO) is
# the remaining baker in Substance's order and is unverified.
BAKED_USAGE = {0: "id", 1: "ao", 2: "curvature", 3: "world_normals", 4: "position", 5: "thickness"}

# OpenGL internal formats seen in texture-cache entries.
_GL_FORMATS = {
    0x8C43: (np.uint8, 4),    # GL_SRGB8_ALPHA8
    0x8058: (np.uint8, 4),    # GL_RGBA8
    0x822B: (np.uint8, 2),    # GL_RG8
    0x8229: (np.uint8, 1),    # GL_R8
    0x822F: (np.float16, 2),  # GL_RG16F
    0x822D: (np.float16, 1),  # GL_R16F
    0x881A: (np.float16, 4),  # GL_RGBA16F
    0x8230: (np.float32, 2),  # GL_RG32F
    0x8814: (np.float32, 4),  # GL_RGBA32F
    0x822C: (np.uint16, 2),   # GL_RG16
    0x805B: (np.uint16, 4),   # GL_RGBA16
}


def decode_texture_cache(blob):
    """Return (header dict, array H x W x C) with row 0 at the TOP of the image.

    Substance stores GL readbacks, whose first row is the bottom (v = 0),
    so the array is flipped to the usual image orientation here.
    """
    h, o = read_map(blob)
    if h.get("compression") not in ("lz4", None):
        raise ValueError("unsupported texture-cache compression %r" % h.get("compression"))
    total = h["bufferSize"]
    out = bytearray()
    if h.get("compression") == "lz4":
        while len(out) < total and o < len(blob):
            n = struct.unpack_from(">I", blob, o)[0]
            o += 4
            out += lz4.block.decompress(blob[o:o + n], uncompressed_size=total - len(out))
            o += n
    else:
        out = blob[o:o + total]
    fmt = h["internalFormat"]
    if fmt not in _GL_FORMATS:
        raise ValueError("unsupported texture format 0x%x" % fmt)
    dt, comps = _GL_FORMATS[fmt]
    a = np.frombuffer(bytes(out), dt).reshape(h["height"], h["width"], comps)
    return h, np.ascontiguousarray(a[::-1])


class SppProject:
    def __init__(self, path, load_doc=True):
        """load_doc=False skips the layer document (worker processes only
        need the mesh and embedded images, and the document of a heavily
        hand-painted project can take gigabytes in memory)."""
        self.path = path
        self.h5 = h5py.File(path, "r")
        if load_doc:
            self.doc, self.parse_report = docbin.parse_with_report(self._bytes("paint/document.bin"))
        else:
            self.doc, self.parse_report = {}, {"unknown_types": [], "errors": []}
        self._scene = None

    def _bytes(self, name):
        return self.h5[name][()].tobytes()

    def has(self, name):
        return name in self.h5

    @property
    def scene(self):
        if self._scene is None:
            self._scene = assbin.parse(self._bytes("paint/scene.assbin"))
        return self._scene

    @property
    def materials(self):
        return self.doc.get("materials") or []

    def cached_channels(self, material):
        """Map channel flag -> image array for a texture set's final result.

        Returns {} when Substance did not save a texture cache for it.
        """
        uid = material.get("sceneMaterialUID")
        grp = "texture-cache/%d" % uid
        if grp not in self.h5:
            return {}
        out = {}
        for stack_key in self.h5[grp]:
            g = self.h5[grp][stack_key]
            for ch in g:
                if ch == "list":
                    continue
                try:
                    flag = int(ch)
                    _, img = decode_texture_cache(g[ch][()].tobytes())
                except (ValueError, KeyError, struct.error, lz4.block.LZ4BlockError):
                    continue
                out[flag] = img
            break  # one stack per texture set in practice
        return out

    # -- embedded resources -------------------------------------------------
    def resource_name(self, url):
        """Map a document URL like '/Body_normal_base?version=<sha>.image' to
        its HDF5 dataset name, or None (aliases, external resources)."""
        if not url or url.startswith("resourcealias:"):
            return None
        m = re.search(r"version=([0-9a-fA-F]+)\.(\w+)", url)
        if m:
            name = "resources/%s.%s" % (m.group(1), m.group(2))
            return name if name in self.h5 else None
        return None

    def resource_image(self, url):
        """Decode an embedded image resource to an array (row 0 = top), or None."""
        name = self.resource_name(url)
        if not name:
            return None
        from .images import decode_resource_image
        try:
            return decode_resource_image(self._bytes(name))
        except Exception:
            return None

    def baked_maps(self, material):
        """{usage name: url} of the mesh maps baked for a texture set."""
        out = {}
        for a in material.get("additionalMaps") or []:
            if not a:
                continue
            url = a.get("urlToBitmapRes") or ""
            usage = BAKED_USAGE.get(int(a.get("usage", -1)), "map%d" % int(a.get("usage", -1)))
            if url and self.resource_name(url):
                out[usage] = url
        nb = (material.get("normalMapBitmap") or {}).get("urlToBitmapRes") or ""
        if nb and self.resource_name(nb):
            out["normal"] = nb
        return out

    def close(self):
        self.h5.close()
