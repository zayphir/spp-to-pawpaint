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


def build_specs(export_dir, ts_entry, mapdir, res, warn):
    """Turn one texture set of the export into bundle layer specs (bottom first,
    parent indices 1-based within this list). Images go to `mapdir`."""
    specs = []
    index_of = {}
    for i, L in enumerate(ts_entry.get("layers") or []):
        spec = {"name": L.get("name") or "Layer", "visible": bool(L.get("visible", True)),
                "opacity": float(L.get("opacity", 1.0)), "blendMode": L.get("blendMode") or "Normal",
                "spp": {"role": "exact", "tag": "#sppx%d" % (i + 1)}}
        if L.get("parent") is not None:
            p = index_of.get(L["parent"])
            if p:
                spec["parent"] = p
        if L.get("type") == "Folder":
            spec["type"] = "Folder"
            specs.append(spec)
            index_of[L.get("id", i)] = len(specs)
            continue
        spec["type"] = "Paint"
        channels = {}
        for ch, pair in (L.get("images") or {}).items():
            cid = CHANNEL_IDS.get(ch)
            if not cid:
                continue
            try:
                b = _load(os.path.join(export_dir, pair["black"]), res)
                w = _load(os.path.join(export_dir, pair["white"]), res)
            except (OSError, KeyError) as e:
                warn("exact layer '%s' %s images missing (%s)" % (spec["name"], ch, e))
                continue
            c, a = isolate(b, w, ch in SRGB_CHANNELS)
            if a.max() < 1.0 / 255:
                continue  # layer doesn't touch this channel
            rgb = c[..., :3] if c.shape[2] >= 3 else np.repeat(c[..., :1], 3, -1)
            path = os.path.join(mapdir, "exact_%03d_%s.png" % (i + 1, cid))
            out = np.dstack([(rgb * 255 + 0.5).astype(np.uint8), (a * 255 + 0.5).astype(np.uint8)])
            Image.fromarray(np.ascontiguousarray(out)).save(path, compress_level=6)
            channels[cid] = "maps/" + os.path.basename(path)
        spec["channels"] = channels
        specs.append(spec)
        index_of[L.get("id", i)] = len(specs)
    return specs
