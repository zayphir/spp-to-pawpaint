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


GL_SRGB8_ALPHA8 = 0x8C43

# Document versions (third number in document.bin) up to 0x59 name a texture
# set's channels by bit flag (base colour 1, height 2, roughness 128, metallic
# 8192, normal 4194304), in DataChannel.type and as the names of the cached
# result textures. Version 0x5c (Substance 12.2) names them by the bit's
# NUMBER instead (0, 1, 7, 13, 22); layers still use the bit flags. Read as
# flags, '1' is then the height texture taken for the base colour, and the
# real base colour ('0') and everything else is unknown.
INDEX_IDS_FROM_VERSION = 0x5c


def ids_are_indices(ids, version=None):
    """Are these channel ids bit numbers (new files) rather than bit flags?
    0 and anything that isn't a power of two can only be a bit number; ids
    that could be either (1, 2, 4, 8, 16) are decided by the document version."""
    ids = [int(i) for i in ids]
    if any(i == 0 or (i & (i - 1)) for i in ids):
        return True
    return bool(ids) and version is not None and version >= INDEX_IDS_FROM_VERSION


def normalise_channel_ids(materials, version=None):
    """Rewrite DataChannel.type of a new document to the bit flags the rest of
    the converter (and older documents) use. Returns True if it did."""
    chans = [c for m in materials or [] for st in (m.get("stacks") or []) if isinstance(st, dict)
             for c in (st.get("channels") or []) if isinstance(c, dict) and isinstance(c.get("type"), int)]
    if not ids_are_indices([c["type"] for c in chans], version):
        return False
    for c in chans:
        if 0 <= c["type"] < 128:
            c["channelIndex"] = c["type"]
            c["type"] = 1 << c["type"]
    return True


# What else document version 0x5c stores differently from 0x59 and older, as
# far as the converter reads it (found by saving two older test projects with
# Substance 12.2 and comparing every object of the two documents):
#   polygon-fill hits   'projViewInvMatrix' is gone; 'viewMatrix' and
#                       'projectionMatrix' are stored instead
#   polygon-fill hits   'color' (1 = add, 0 = remove) is gone; the action has
#                       'sources' with one colour for all its hits (an action
#                       that mixed both is stored as several actions)
#   path stroke keys    'pressure' is now 'size' (and 'opacity')
#   DataBrush           is called DataBrushStamp
# Unread, every polygon fill of such a project was skipped ('action failed')
# and the masks had to be rebuilt by the correction step alone.
OLD_STYLE_UP_TO_VERSION = 0x59


def _proj_view_inverse(view16, proj16):
    """inverse(projection x view) in the stored (column-major) layout of the old 'projViewInvMatrix'."""
    v = np.asarray(view16, np.float64).reshape(4, 4).T
    p = np.asarray(proj16, np.float64).reshape(4, 4).T
    return [float(x) for x in np.linalg.inv(p @ v).T.reshape(-1)]


def normalise_document(root):
    """Give the objects of a new document (see above) the fields older
    documents have, so the rest of the converter reads both alike. Changes
    `root` in place and returns how many objects it touched."""
    touched = 0
    stack = [root]
    while stack:
        o = stack.pop()
        if isinstance(o, list):
            if o and isinstance(o[0], (dict, list)):
                stack.extend(o)
            continue
        if not isinstance(o, dict):
            continue
        cls = o.get("__class")
        if cls == "DataBrushStamp":
            o["__class"] = "DataBrush"
            touched += 1
        elif cls == "DataStroke3DKey":
            if "pressure" not in o and "size" in o:
                o["pressure"] = o["size"]
                touched += 1
        elif cls == "DataActionDecal":
            colour = None
            src = o.get("sources")
            for s in (src if isinstance(src, list) else [src]):
                if isinstance(s, dict) and s.get("color") is not None:
                    colour = list(s["color"])
                    break
            for hit in o.get("hits") or []:
                if not isinstance(hit, dict):
                    continue
                if colour is not None and hit.get("color") is None:
                    hit["color"] = colour
                    touched += 1
                if (hit.get("projViewInvMatrix") is None and hit.get("viewMatrix") is not None
                        and hit.get("projectionMatrix") is not None):
                    try:
                        hit["projViewInvMatrix"] = _proj_view_inverse(hit["viewMatrix"], hit["projectionMatrix"])
                        touched += 1
                    except (ValueError, np.linalg.LinAlgError):
                        pass
        stack.extend(v for v in o.values() if isinstance(v, (dict, list)))
    return touched


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
        self.doc_version = None
        self.channel_ids_are_indices = False
        self._cache_formats = {}
        if load_doc:
            raw = self._bytes("paint/document.bin")
            self.doc, self.parse_report = docbin.parse_with_report(raw)
            if len(raw) >= 12:
                self.doc_version = struct.unpack_from("<I", raw, 8)[0]
            self.channel_ids_are_indices = normalise_channel_ids(self.materials, self.doc_version)
            if self.doc_version is None or self.doc_version > OLD_STYLE_UP_TO_VERSION:
                normalise_document(self.doc)
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
        out, formats = {}, {}
        for stack_key in self.h5[grp]:
            g = self.h5[grp][stack_key]
            names = [ch for ch in g if ch != "list"]
            # new files name the textures by the channel's bit number (see ids_are_indices)
            by_index = (getattr(self, "channel_ids_are_indices", False)
                        or ids_are_indices([int(ch) for ch in names if ch.isdigit()]))
            if by_index:        # in the order an old file lists the same textures (by flag, as text)
                names.sort(key=lambda ch: str(1 << int(ch)) if ch.isdigit() and int(ch) < 128 else ch)
            for ch in names:
                try:
                    flag = int(ch)
                    hdr, img = decode_texture_cache(g[ch][()].tobytes())
                except (ValueError, KeyError, struct.error, lz4.block.LZ4BlockError):
                    continue
                if by_index:
                    if not 0 <= flag < 128:
                        continue
                    flag = 1 << flag
                out[flag] = img
                formats[flag] = hdr.get("internalFormat")
            break  # one stack per texture set in practice
        if not hasattr(self, "_cache_formats"):
            self._cache_formats = {}
        self._cache_formats[uid] = formats
        return out

    def cached_formats(self, material):
        """OpenGL formats of the textures cached_channels(material) returned: {channel flag: format}."""
        return dict(getattr(self, "_cache_formats", {}).get(material.get("sceneMaterialUID")) or {})

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
