"""Exact per-layer images from the Substance companion exporter.

For users who own Substance 3D Painter: substance_plugin/pawpaint_exporter.py
runs inside Substance (Adobe's official Python API) and, for every layer,
exports that layer alone composited over a black and over a white background:

    <project> (Substance export)/export.json
    <project> (Substance export)/<TextureSet>/<NNN>_black/<files>.png
    <project> (Substance export)/<TextureSet>/<NNN>_white/<files>.png

Over black: B = c·a.  Over white: W = c·a + (1 - a).  So a = 1 - (W - B) and
c = B / a, per channel, in linear space. That recovers each layer's colour and
true coverage with every mask, filter and generator already applied, so no
approximation is needed.

Measured on Substance 3D Painter 12.2 (Nov 2026), and handled below:
- Height: the black background is FLAT height (a height value of 0 is
  'no relief'; it exports as mid grey), the white one is fully raised. And
  PawPaint ignores a paint layer's alpha for height, so the layer's height is
  written as mid grey where it doesn't paint.
- Normal: exported in the project's own convention; with DirectX the green
  channel is flipped, so black and white come out as (0, 1, 0) and (1, 0, 1).
  A layer's normal does not include the baked normal map (Substance combines
  the two in its shader), and a flat normal would cover the baked normal in
  PawPaint, so flat normals are left out and the others are combined with
  the baked normal.
- Plugin 0.3.0 exports each layer at 100 % opacity with Normal blending (its
  own and its folders') and records the real ones; 0.2.0 exported them
  baked into the coverage, which then counted twice in PawPaint.
"""

import json
import os

import numpy as np
from PIL import Image

EXPORT_FORMAT = "spp2pawpaint-substance-export"

# Substance channel names in the export -> PawPaint channel ids.
CHANNEL_IDS = {"basecolor": "albedo", "roughness": "roughness", "metallic": "metallic",
               "normal": "normal", "height": "height", "emissive": "emission",
               "opacity": "opacity", "specular": "specular", "glossiness": "glossiness"}
SRGB_CHANNELS = ("basecolor", "emissive", "specular")


def find_export(spp_path, explicit=None):
    if explicit:
        return explicit if os.path.exists(os.path.join(explicit, "export.json")) else None
    stem = os.path.splitext(os.path.abspath(spp_path))[0]
    cand = stem + " (Substance export)"
    return cand if os.path.exists(os.path.join(cand, "export.json")) else None


def load_export(folder):
    with open(os.path.join(folder, "export.json"), encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != EXPORT_FORMAT:
        raise ValueError("not a PawPaint Substance export")
    return data


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _load(path, res):
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA", "L", "I;16", "I"):
        im = im.convert("RGB")
    if im.size != (res, res):
        im = im.resize((res, res), Image.BILINEAR)
    a = np.asarray(im)
    if a.dtype == np.uint16 or im.mode in ("I;16", "I"):
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32) / 255.0
    return a if a.ndim == 3 else a[..., None]


def isolate(black, white, srgb):
    """(colour, alpha) of a layer from its over-black / over-white renders."""
    if srgb:
        black, white = _srgb_to_linear(black), _srgb_to_linear(white)
    a = np.clip(1.0 - (white - black), 0, 1)
    a1 = a.mean(-1) if a.ndim == 3 else a          # one coverage per texel
    c = np.where(a1[..., None] > 1e-3, black / np.maximum(a1[..., None], 1e-3), 0)
    c = np.clip(c, 0, 1)
    if srgb:
        c = _linear_to_srgb(c)
    return c, a1


FLAT_NORMAL = np.array([0.5, 0.5, 1.0], np.float32)
FLAT_NORMAL_TOL = 3.0 / 255     # a layer normal closer than this to flat everywhere adds nothing
# Substance blend names (export.json 'substanceBlend' / 'channelBlend') -> PawPaint
BLEND_NAMES = {"Normal": "Normal", "Multiply": "Multiply", "LinearDodge": "Add", "Screen": "Screen",
               "Overlay": "Overlay", "SoftLight": "Soft Light", "HardLight": "Hard Light",
               "Replace": "Normal", "Passthrough": "Normal"}


def isolate_height(black, white):
    """(height image to write, coverage) of a layer from its over-black /
    over-white height renders. The black background is flat height, which
    exports as mid grey (127/255 in 8 bits), the white one is fully raised.
    The image is the layer's height where it paints and flat elsewhere:
    PawPaint ignores a paint layer's alpha for height."""
    b, w = black[..., 0], white[..., 0]
    top = float(w.max())
    uncovered = w >= top - 0.5 / 255
    bg = float(np.median(b[uncovered])) if uncovered.any() and top > 0.75 else 0.5
    span = max(top - bg, 1e-3)
    a = np.clip(1.0 - (w - b) / span, 0, 1)
    h = np.clip(0.5 + (b - bg), 0, 1)              # the layer's own flat is the background's level
    return h.astype(np.float32), a.astype(np.float32)


def isolate_normal(black, white, directx, baked=None):
    """(normal, coverage) of a layer from its over-black / over-white normal
    renders, or (None, coverage) when the layer's normal is flat (it then adds
    nothing: Substance combines layer normals with the baked normal map, and
    a flat normal would cover the baked one in PawPaint). The result is in
    the export's convention, combined with `baked` (same convention) if given."""
    b, w = black[..., :3].copy(), white[..., :3].copy()
    if directx:                                    # unmix in OpenGL terms: DirectX flips green
        b[..., 1], w[..., 1] = 1.0 - b[..., 1], 1.0 - w[..., 1]
    c, a = isolate(b, w, False)
    on = a > 1.0 / 255
    if not on.any() or float(np.abs(c[on] - FLAT_NORMAL).max()) < FLAT_NORMAL_TOL:
        return None, a
    if directx:
        c[..., 1] = np.where(on, 1.0 - c[..., 1], 0.0)
    if baked is not None:
        from .layers import rnm_blend
        bk = np.asarray(baked)[..., :3]
        bk = bk.astype(np.float32) / 255.0 if bk.dtype == np.uint8 else bk.astype(np.float32)
        if bk.shape[:2] != c.shape[:2]:
            bk = np.asarray(Image.fromarray((np.clip(bk, 0, 1) * 255 + 0.5).astype(np.uint8))
                            .resize((c.shape[1], c.shape[0]), Image.BILINEAR)).astype(np.float32) / 255.0
        if directx:                                # RNM in OpenGL terms
            bk = bk.copy(); bk[..., 1] = 1.0 - bk[..., 1]
            c[..., 1] = 1.0 - c[..., 1]
        c = np.where(on[..., None], rnm_blend(bk, c), 0.0)
        if directx:
            c[..., 1] = np.where(on, 1.0 - c[..., 1], 0.0)
    return c.astype(np.float32), a


def _blend_of(entry, warn, what):
    """PawPaint blend mode for a Substance blend name."""
    if entry in (None, "Normal"):
        return "Normal"
    if entry in BLEND_NAMES:
        return BLEND_NAMES[entry]
    warn("%s uses the blend mode %s, which PawPaint doesn't have; Normal is used" % (what, entry))
    return "Normal"


def build_specs(export_dir, ts_entry, mapdir, res, warn, normal_convention="opengl", baked_normal=None):
    """Turn one texture set of the export into bundle layer specs (bottom first,
    parent indices 1-based within this list). Images go to `mapdir`.
    `normal_convention`: the project's ('directx' flips green in the export).
    `baked_normal`: the texture set's baked normal map, to combine layer normals with."""
    layers = ts_entry.get("layers") or []
    # Plugin 0.3.0 exports the coverage alone and records blend modes and
    # opacities per channel; 0.2.0 baked the opacities into the coverage.
    per_channel = any("channelBlend" in L for L in layers)
    if layers and not per_channel:
        warn("this Substance export was made with an older PawPaint Exporter plugin: layer and folder "
             "opacities are already in each layer's coverage, so they are imported at 100 %. Export again "
             "with the current plugin to keep them editable")
    by_id = {L.get("id", i): L for i, L in enumerate(layers)}
    # Pass-through folders: PawPaint folders are always composited on their own,
    # so a blend layer inside one would only blend with the folder's own layers.
    # Such folders are left out (their layers go to the folder's parent), as
    # for rebuilt layers.
    dissolved = set()
    for i, L in enumerate(layers):
        if L.get("type") == "Folder" and L.get("substanceBlend") == "Passthrough":
            fid = L.get("id", i)
            for j, M in enumerate(layers):
                p, nested = M.get("parent"), False
                while p is not None:
                    if p == fid:
                        nested = True
                        break
                    p = (by_id.get(p) or {}).get("parent")
                if nested and M.get("type") != "Folder" and (M.get("substanceBlend") or "Normal") not in ("Normal", "Passthrough"):
                    dissolved.add(fid)
                    break

    def parent_of(L):
        p = L.get("parent")
        while p is not None and p in dissolved:
            p = (by_id.get(p) or {}).get("parent")
        return p

    specs = []
    index_of = {}
    directx = str(normal_convention).lower() == "directx"
    for i, L in enumerate(layers):
        if L.get("id", i) in dissolved:
            continue
        name = L.get("name") or "Layer"
        cb = L.get("channelBlend") or {}
        base = cb.get("basecolor") or {}
        opacity = float(L.get("opacity", 1.0)) if per_channel else 1.0
        mode = _blend_of(L.get("substanceBlend") or base.get("blend") or "Normal", warn, "'%s'" % name) \
            if per_channel else (L.get("blendMode") or "Normal")
        spec = {"name": name, "visible": bool(L.get("visible", True)), "opacity": opacity, "blendMode": mode,
                "spp": {"role": "exact", "tag": "#sppx%d" % (i + 1)}}
        p = parent_of(L)
        if p is not None and index_of.get(p):
            spec["parent"] = index_of[p]
        if L.get("type") == "Folder":
            spec["type"] = "Folder"
            specs.append(spec)
            index_of[L.get("id", i)] = len(specs)
            continue
        spec["type"] = "Paint"
        channels, overrides = {}, {}
        for ch, pair in (L.get("images") or {}).items():
            cid = CHANNEL_IDS.get(ch)
            if not cid:
                continue
            try:
                b = _load(os.path.join(export_dir, pair["black"]), res)
                w = _load(os.path.join(export_dir, pair["white"]), res)
            except (OSError, KeyError) as e:
                warn("exact layer '%s' %s images missing (%s)" % (name, ch, e))
                continue
            path = os.path.join(mapdir, "exact_%03d_%s.png" % (i + 1, cid))
            if ch == "height":
                h, a = isolate_height(b, w)
                if a.max() < 1.0 / 255 or float(np.abs(h - 0.5).max()) < 0.5 / 255:
                    continue                    # no height, or flat height: nothing to add
                Image.fromarray((h * 65535 + 0.5).astype(np.uint16)).save(path, compress_level=6)
            elif ch == "normal":
                c, a = isolate_normal(b, w, directx, baked_normal)
                if c is None:
                    continue
                out = np.dstack([(c * 255 + 0.5).astype(np.uint8), (a * 255 + 0.5).astype(np.uint8)])
                Image.fromarray(np.ascontiguousarray(out)).save(path, compress_level=6)
                spec["normalConvention"] = "directx" if directx else "opengl"
            else:
                c, a = isolate(b, w, ch in SRGB_CHANNELS)
                if a.max() < 1.0 / 255:
                    continue  # layer doesn't touch this channel
                rgb = c[..., :3] if c.shape[2] >= 3 else np.repeat(c[..., :1], 3, -1)
                out = np.dstack([(rgb * 255 + 0.5).astype(np.uint8), (a * 255 + 0.5).astype(np.uint8)])
                Image.fromarray(np.ascontiguousarray(out)).save(path, compress_level=6)
            channels[cid] = "maps/" + os.path.basename(path)
            # a channel whose own blend mode or opacity differs from the layer's
            own = cb.get(ch) if per_channel else None
            if own and ch != "basecolor":
                o_op = own.get("opacity")
                o_op = opacity if o_op is None else float(o_op)
                if cid in ("height", "normal"):
                    # Substance's defaults here are Add (LinearDodge) for height and
                    # NormalMapDetail for normals; PawPaint adds height and blends
                    # normals Normally (the normal is combined with the baked one above)
                    o_mode = mode
                else:
                    o_mode = _blend_of(own.get("blend") or "Normal", warn, "'%s' (%s)" % (name, ch))
                if o_mode != mode or abs(o_op - opacity) > 1e-4:
                    entry = {"opacity": o_op}
                    if cid not in ("height", "normal"):
                        entry["blendMode"] = o_mode
                    overrides[cid] = entry
        spec["channels"] = channels
        if overrides:
            spec["channelOverrides"] = overrides
        specs.append(spec)
        index_of[L.get("id", i)] = len(specs)
    return specs
