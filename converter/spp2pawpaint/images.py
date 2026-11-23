"""Decode images embedded in an .spp (resources/<sha>.image).

Layout: a QVariantMap header (width, height, format, tilesCount,
compression, ...), then tiles. Each tile is
    u32 x0, u32 y0, u32 x1, u32 y1, u32 byteLength, u32 0,
followed by the compressed tile (a complete JPEG 2000 file for
compression "jp2"), padded to 4 bytes. Tiles are 512x512 for 4K maps.
"""

import io
import struct

import numpy as np
from PIL import Image

from .qtvariant import read_map

# Tile rectangles are already top-down (row 0 = v 1), unlike the texture
# cache. Verified by overlaying Body_id on the rasterised UV islands.
FLIP_VERTICAL = False
# Raw LZ4 images (LUM8 curvature/thickness): row order checked the same way.
LZ4_FLIP_VERTICAL = False

_MAGICS = (b"\x00\x00\x00\x0cjP  ", b"\xff\x4f\xff\x51", b"\x89PNG", b"\xff\xd8\xff")


def _unwrap_white(data, a):
    """Pillow converts 16-bit colour JPEG 2000 to 8-bit as (v + 128) >> 8 and
    stores the result in a byte, so values of 65408 and up (pure white) wrap
    to 0. A solid white mask shape (a test project's iris masks) came out as a thin
    outline. Decode again with the declared precision raised to 17 bits: for
    lossless data that maps v to (v + 33024) >> 9 = 64..192 with no wrap,
    which tells wrapped white (192) from real black (64)."""
    if a.dtype != np.uint8 or a.ndim != 3 or not (a == 0).any():
        return a
    s = data.find(bytes((0xFF, 0x51)))   # SIZ marker
    if s < 0 or s + 40 > len(data):
        return a
    ncomp = struct.unpack_from(">H", data, s + 38)[0]
    if ncomp < 2 or ncomp > 4 or s + 40 + 3 * ncomp > len(data):
        return a
    if any(data[s + 40 + 3 * i] != 15 for i in range(ncomp)):   # not unsigned 16-bit
        return a
    d2 = bytearray(data)
    for i in range(ncomp):
        d2[s + 40 + 3 * i] = 16
    try:
        im = Image.open(io.BytesIO(bytes(d2)))
        im.load()
        u = np.asarray(im)
    except Exception:
        return a
    if u.shape != a.shape or u.dtype != np.uint8:
        return a
    wrapped = (a == 0) & (u >= 160)
    if not wrapped.any():
        return a
    a = a.copy()
    a[wrapped] = 255
    return a


def _decode_tile(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    a = _unwrap_white(data, np.asarray(im))
    if a.ndim == 2:
        a = a[..., None]
    if a.dtype == np.uint16 or a.dtype == np.int32:
        a = (a.astype(np.float32) / 65535.0)
    elif a.dtype != np.uint8:
        a = a.astype(np.float32)
    return a


def decode_resource_image(blob):
    """Return an image array (H, W, C), row 0 = top of the texture (v = 1).

    uint8 when the data is 8-bit, float32 0..1 otherwise.
    """
    try:
        h, o = read_map(blob)
    except Exception:
        h, o = {}, 0
    payload = blob[o:]
    w, ht = int(h.get("width", 0) or 0), int(h.get("height", 0) or 0)
    tiles = int(h.get("tilesCount", 0) or 0)

    if str(h.get("compression", "")).lower() == "lz4" and w and ht:
        # Raw pixels in one LZ4 block (seen for LUM8 curvature/thickness maps).
        import lz4.block
        fmt = str(h.get("format", "")).upper()
        comps = 1 if fmt.startswith("LUM") or fmt in ("L8", "L16", "R8", "R16") else (3 if fmt.startswith("RGB") and not fmt.startswith("RGBA") else 4)
        depth = 2 if fmt.endswith("16") else (4 if fmt.endswith("32F") else 1)
        size = int(h.get("dataSizeRaw") or w * ht * comps * depth)
        raw = lz4.block.decompress(payload, uncompressed_size=size)
        dt = {1: np.uint8, 2: np.uint16, 4: np.float32}[depth]
        a = np.frombuffer(raw, dt, w * ht * comps).reshape(ht, w, comps)
        if dt == np.uint16:
            a = a.astype(np.float32) / 65535.0
        if fmt.startswith("BGR") and comps >= 3:
            # images the user imported into Substance (decals, whiskers ...)
            # are stored in Windows byte order
            a = a[..., [2, 1, 0] + list(range(3, comps))]
        a = a[::-1] if LZ4_FLIP_VERTICAL else a
        return np.ascontiguousarray(a)

    if tiles > 0 and w and ht:
        # Find tiles by signature (the stored length is not always exact) and
        # read each tile's rectangle from the 24 bytes in front of it.
        magic = next((m for m in _MAGICS if payload.find(m, 24) == 24), _MAGICS[0])
        starts = []
        k = payload.find(magic)
        while k >= 0:
            if k >= 24:
                starts.append(k)
            k = payload.find(magic, k + 1)
        # Tiles of one image can differ: Substance stores fully opaque tiles
        # with 3 components and the others with 4 (seen in a test project's RGBA16
        # images). A tile without alpha is opaque, so the alpha plane starts
        # at 1; before, such tiles were read as fully transparent (or, when
        # the first tile had no alpha, the alpha of the others was dropped).
        decoded = []
        for i, k in enumerate(starts):
            x0, y0, x1, y1, n, _z = struct.unpack_from("<6I", payload, k - 24)
            if not (x1 > x0 and y1 > y0 and x1 <= w and y1 <= ht):
                continue
            stop = starts[i + 1] - 24 if i + 1 < len(starts) else len(payload)
            decoded.append((x0, y0, x1, y1, _decode_tile(payload[k:stop])))
        if not decoded:
            raise ValueError("no tiles decoded")
        chans = max(t[4].shape[2] for t in decoded)
        dtype = decoded[0][4].dtype
        out = np.zeros((ht, w, chans), dtype)
        if chans == 4:
            out[..., 3] = 255 if dtype == np.uint8 else 1.0
        for x0, y0, x1, y1, tile in decoded:
            th, tw = y1 - y0, x1 - x0
            c = min(tile.shape[2], chans)
            out[y0:y0 + th, x0:x0 + tw, :c] = tile[:th, :tw, :c].astype(dtype, copy=False)
        return np.ascontiguousarray(out[::-1]) if FLIP_VERTICAL else out

    # Untiled: find the first known image signature and let Pillow read it.
    for m in _MAGICS:
        k = payload.find(m)
        if k >= 0:
            a = _decode_tile(payload[k:])
            return np.ascontiguousarray(a[::-1]) if (FLIP_VERTICAL and tiles) else a
    raise ValueError("unsupported embedded image (%s)" % h.get("compression"))
