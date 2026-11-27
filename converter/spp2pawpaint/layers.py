"""Turn one Substance texture set into a PawPaint layer bundle."""

import json
import os
import re
import time

import numpy as np
from PIL import Image

from . import procedural as P
from . import sppfile as S
from .masks import BLEND_DISABLE, BLEND_NORMAL, MaskBuilder, RegionCover, _dilate, _source_url, as_full, as_list, pad

# Substance blending mode ids as stored in .spp files: the position in
# Substance's own BlendingMode list (measured 2026-11-23 with the calibration
# plugin in Substance 12.2: one layer per mode, saved, read back). Layers with
# Normal blending store no entry; folders default to Passthrough (25).
SUBSTANCE_BLEND_LABELS = {
    0: "Disable", 1: "Replace", 2: "Normal", 3: "Darken", 4: "Multiply", 5: "Color Burn",
    6: "Linear Burn", 7: "Lighten", 8: "Screen", 9: "Color Dodge", 10: "Linear Dodge (Add)",
    11: "Overlay", 12: "Soft Light", 13: "Hard Light", 14: "Vivid Light", 15: "Linear Light",
    16: "Pin Light", 17: "Difference", 18: "Exclusion", 19: "Tint", 20: "Saturation", 21: "Color",
    22: "Value", 23: "Divide", 24: "Inverse Divide", 25: "Passthrough", 26: "Normal Map Combine",
    27: "Normal Map Detail", 28: "Normal Map Inverse Detail", 29: "Subtract", 30: "Inverse Subtract",
    31: "Signed Addition",
}
# ... and the PawPaint blend mode for each one it has
BLEND_NAMES = {
    2: "Normal", 1: "Normal", 25: "Normal",
    4: "Multiply", 8: "Screen", 10: "Add", 11: "Overlay", 12: "Soft Light", 13: "Hard Light",
}

# Substance channel flag -> (PawPaint channel id, kind, custom?)
CHANNEL_MAP = {
    S.CH_BASECOLOR: ("albedo", "color", False),
    S.CH_ROUGHNESS: ("roughness", "scalar", False),
    S.CH_METALLIC: ("metallic", "scalar", False),
    S.CH_NORMAL: ("normal", "vector", False),
    S.CH_HEIGHT: ("height", "signed", False),
    S.CH_EMISSIVE: ("emission", "color", False),
    S.CH_OPACITY: ("opacity", "scalar", True),
    S.CH_SPECULAR: ("specular", "color", True),
    S.CH_GLOSSINESS: ("glossiness", "scalar", True),
    S.CH_DISPLACEMENT: ("displacement", "signed", True),
}
CUSTOM_KINDS = {cid: kind for cid, kind, custom in CHANNEL_MAP.values() if custom}
BUILTIN_CHANNELS = ("albedo", "roughness", "metallic", "normal", "height", "emission")
CUSTOM_NAMES = {"opacity": "Opacity", "specular": "Specular", "glossiness": "Glossiness",
                "displacement": "Displacement"}
COLOR_CHANNELS = ("albedo", "emission", "specular")
RGB_CHANNELS = ("albedo", "emission", "specular", "normal")
# Channels PawPaint has no fill value setter for: given as solid maps instead.
MAP_ONLY_CHANNELS = ("emission", "opacity", "specular", "glossiness", "displacement")
FLAT_NORMAL = (0.5, 0.5, 1.0)


MASK_MAX_RES = 2048
PNG_LEVEL = 1                # zlib level: lossless at any level, 1 is fastest
# Fit the rebuilt masks to Substance's own final texture (see refine.py)
FIT_TO_RESULT = True
BLACK_BASE_MIN_SHARE = 0.0005   # add a black base when this share of a set's texels isn't fully covered
HEAVY_STROKE_POINTS = 5000   # stroke points above which a mask/layer uses mb_low
PARALLEL_MIN_POINTS = 2000   # masks with more stroke points go to the worker pool


def _blend_pixels(mode, dst, src):
    """PawPaint layer blend of src over dst (before opacity), for composites."""
    if mode == "Multiply":
        return dst * src
    if mode == "Add":
        return np.minimum(dst + src, 1.0)
    if mode == "Screen":
        return 1.0 - (1.0 - dst) * (1.0 - src)
    if mode == "Overlay":
        return np.where(dst < 0.5, 2 * dst * src, 1.0 - 2 * (1.0 - dst) * (1.0 - src))
    if mode == "Soft Light":
        return (1 - 2 * src) * dst * dst + 2 * src * dst
    if mode == "Hard Light":
        return np.where(src < 0.5, 2 * dst * src, 1.0 - 2 * (1.0 - dst) * (1.0 - src))
    return src


def _has_content(actions):
    """True if a layer's actions hold anything besides empty groups/strokes."""
    for a in actions or []:
        if not a or not a.get("enabled", True):
            continue
        if a.get("__class") == "DataActionGroup":
            if _has_content((a.get("subStack") or {}).get("items")):
                return True
        elif a.get("__class") in ("DataActionPaint", "DataActionEraser"):
            if a.get("strokes") or a.get("strokes3D"):
                return True
        else:
            return True
    return False


def _stroke_points(obj):
    """Number of brush stroke points inside an action tree."""
    n, st = 0, [obj]
    while st:
        x = st.pop()
        if isinstance(x, dict):
            if x.get("__class") in ("DataStroke", "DataStroke3D"):
                n += len(x.get("points") or [])
            else:
                st.extend(v for v in x.values() if isinstance(v, (dict, list)))
        elif isinstance(x, list):
            st.extend(x)
    return n


def _has_class(obj, names):
    """True if an action tree contains an object of one of the classes."""
    st = [obj]
    while st:
        x = st.pop()
        if isinstance(x, dict):
            if x.get("__class") in names:
                return True
            st.extend(v for v in x.values() if isinstance(v, (dict, list)))
        elif isinstance(x, list):
            st.extend(x)
    return False


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rnm_blend(base, detail):
    """Reoriented normal mapping: detail normal on top of base, both 0..1 RGB."""
    t = base.astype(np.float32) * np.array([2, 2, 2], np.float32) + np.array([-1, -1, 0], np.float32)
    u = detail.astype(np.float32) * np.array([-2, -2, 2], np.float32) + np.array([1, 1, -1], np.float32)
    r = t * (t * u).sum(-1, keepdims=True) / np.maximum(t[..., 2:3], 1e-4) - u
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-6)
    return np.clip(r * 0.5 + 0.5, 0, 1)


def _safe(name):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return s or "layer"


def _u8(a):
    return a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def _f(a):
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


HEIGHT_FIX_TOL = 1.0 / 255.0      # the layers' height may be this far from Substance's before the detail fix takes over


def never_flat(h):
    """A height image (0..1, 0.5 = flat) for a layer whose height replaces
    what is below it. PawPaint's Replace leaves a texel alone when the layer's
    height there is exactly flat (within 1/65535 of 0.5), so flat texels are
    moved by 2.5/65535, far too little to see, and the layer then replaces
    everywhere."""
    h = np.clip(np.asarray(h, np.float32), 0, 1)
    return np.where(np.abs(h - 0.5) < 2.0 / 65535.0, np.float32(0.5 + 2.5 / 65535.0), h)


class MaterialConverter:
    def __init__(self, project, geo, mat_doc, scene_index, res, outdir, log, pool=None,
                 engine="cpu", lossless=False):
        self.p = project
        self.geo = geo
        self.doc = mat_doc
        self.name = mat_doc.get("sceneMaterialName") or "Material"
        self.res = res
        self.outdir = outdir
        self.mapdir = os.path.join(outdir, "maps")
        os.makedirs(self.mapdir, exist_ok=True)
        self.log = log
        self.warnings = []
        # Masks with heavy brush work are evaluated at up to MASK_MAX_RES and
        # scaled up when written (4K working buffers made hand-painted projects
        # several times slower for no visible gain). Everything else, such as
        # crisp polygon-fill masks, stays at full resolution.
        self.mask_res = res
        self.pool = pool          # parallel.MaskPool for heavy masks, or None
        self.scene_index = scene_index
        self._pmask = {}          # id(layer) -> pool key
        self.mb = MaskBuilder(geo, scene_index, res, self._warn) if scene_index is not None else None
        self.engine, self.lossless = engine, lossless
        if self.mb is not None:
            self.mb.engine, self.mb.lossless = engine, lossless
        self.mb_low = None
        # the GPU and the CPU's lossless mode keep every mask at full resolution
        if self.mb is not None and res > MASK_MAX_RES and engine == "cpu" and not lossless:
            self.mb_low = MaskBuilder(geo, scene_index, MASK_MAX_RES, self._warn)
        self._cov_full = None
        if self.mb_low is not None:
            def low_baked(usage):
                img = self.baked_map(usage)
                if img is None:
                    return None
                r = self._resize(img, MASK_MAX_RES)
                return r[..., 0] if r.ndim == 3 and r.shape[2] == 1 else r
            self.mb_low.baked_map = low_baked
            self.mb_low.resource_image = self.p.resource_image
        if self.mb is not None:
            self.mb.baked_map = self.baked_map
            self.mb.resource_image = self.p.resource_image
        stack = (mat_doc.get("stacks") or [{}])[0]
        self.channels = [c.get("type") for c in stack.get("channels") or []]
        self.items = ((stack.get("stack") or {}).get("items")) or []
        self.layers = []         # bundle layer specs (bottom first)
        self.custom = {}         # custom channel id -> (name, kind)
        self._count = 0
        self._total = 0
        self._context = ""
        self._writer = None
        self._writes = []
        self._discard_writes = False
        self._baked_urls = project.baked_maps(mat_doc)
        self._baked_cache = {}
        self.cache_albedo = None
        self.cache_albedo_alpha = None
        self.result_arrays = {}      # Substance's own channels (PawPaint space), for fitting
        self.fit_to_result = FIT_TO_RESULT
        self.fit_stats = None
        self.baked_files = {}
        self.normal_convention = "directx" if int(project.doc.get("normalMapSyscoord", 0) or 0) == 1 else "opengl"

    def _warn(self, msg):
        full = "%s / %s: %s" % (self.name, self._context, msg) if self._context else "%s: %s" % (self.name, msg)
        self._inexact = True       # the layer being converted isn't an exact rebuild
        if full not in self.warnings:
            self.warnings.append(full)

    # -- image output ---------------------------------------------------------
    def _save(self, arr, path):
        """Write a PNG on a background thread (zlib releases the GIL). PNG is
        lossless at any compression level; level 1 is just much faster."""
        if self._discard_writes:
            return
        img = Image.fromarray(np.ascontiguousarray(arr))
        if self._writer is None:
            from concurrent.futures import ThreadPoolExecutor
            self._writer = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) // 2)))
        self._writes.append(self._writer.submit(img.save, path, compress_level=PNG_LEVEL))

    def flush_writes(self):
        for f in self._writes:
            f.result()
        self._writes = []
        if self._writer is not None:
            self._writer.shutdown(wait=True)
            self._writer = None

    def _write_gray(self, arr, tag, pad_cov=None, bits=8):
        path = os.path.join(self.mapdir, "%s.png" % tag)
        if bits == 16:
            self._save((np.clip(arr, 0, 1) * 65535 + 0.5).astype(np.uint16), path)
        else:
            self._save(pad(_f(arr), pad_cov) if pad_cov is not None else _u8(arr).copy(), path)
        return "maps/%s.png" % tag

    def _write_rgb(self, arr, tag, alpha=None):
        path = os.path.join(self.mapdir, "%s.png" % tag)
        a = _u8(arr)
        self._save(np.dstack([a, _u8(alpha)]) if alpha is not None else a.copy(), path)
        return "maps/%s.png" % tag

    def _resize(self, img, res=None):
        res = res or self.res
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[0] == res and img.shape[1] == res:
            return img
        dt = img.dtype
        if dt == np.uint8:
            chans = [np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c])).resize((res, res), Image.BILINEAR))
                     for c in range(img.shape[2])]
        else:
            f = img.astype(np.float32)
            chans = [np.asarray(Image.fromarray(np.ascontiguousarray(f[..., c]), "F").resize((res, res), Image.BILINEAR))
                     for c in range(img.shape[2])]
        return np.stack(chans, -1).astype(dt)

    def _up(self, a):
        """Scale a mask-resolution float image up to the output resolution."""
        if a is None or a.shape[0] == self.res:
            return a
        return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F")
                          .resize((self.res, self.res), Image.BILINEAR))

    def coverage(self):
        """UV island coverage at the output resolution."""
        if self._cov_full is None and self.mb is not None:
            self._cov_full = self.mb.coverage()
        return self._cov_full

    def _builder_for(self, obj):
        """Full-resolution mask builder, or the capped one for heavy brush work."""
        if self.mb_low is not None and _stroke_points(obj) > HEAVY_STROKE_POINTS:
            return self.mb_low
        return self.mb

    def _custom(self, cid, kind):
        self.custom[cid] = (CUSTOM_NAMES.get(cid, cid), kind)

    # -- baked mesh maps --------------------------------------------------------
    def baked_map(self, usage):
        """Baked map resized to the import resolution (uint8/float array) or None."""
        if usage not in self._baked_cache:
            img = None
            url = self._baked_urls.get(usage)
            if url:
                raw = self.p.resource_image(url)
                if raw is not None:
                    img = self._resize(raw)
                    if img.shape[2] == 1:
                        img = img[..., 0]
                    elif usage == "normal":
                        # Substance's baked normals often carry Z (blue) only in
                        # places (0 elsewhere); where present it matches
                        # sqrt(1-x²-y²) (corr 0.999), so rebuild it everywhere.
                        n = _f(img[..., :3]) if img.dtype == np.uint8 else img[..., :3].astype(np.float32)
                        x, y = n[..., 0] * 2 - 1, n[..., 1] * 2 - 1
                        z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
                        img = _u8(np.stack([n[..., 0], n[..., 1], z * 0.5 + 0.5], -1))
            self._baked_cache[usage] = img
        return self._baked_cache[usage]

    def export_baked_maps(self):
        """Write every baked map next to the bundle as reference images."""
        if not self._baked_urls:
            return {}
        folder = os.path.join(self.outdir, "baked")
        os.makedirs(folder, exist_ok=True)
        for usage in sorted(self._baked_urls):
            img = self.baked_map(usage)
            if img is None:
                self._warn("baked %s map could not be decoded" % usage)
                continue
            self._save(_u8(img).copy(), os.path.join(folder, "%s.png" % usage))
            self.baked_files[usage] = "baked/%s.png" % usage
        return self.baked_files

    def baked_normal_layer(self):
        n = self.baked_map("normal")
        if n is None:
            return None
        rgb = n[..., :3] if n.ndim == 3 else np.repeat(n[..., None], 3, -1)
        path = self._write_rgb(rgb, "baked_normal")
        return {"type": "Fill", "name": "Baked Normal (from Substance)", "visible": True,
                "channels": {"normal": path}, "normalConvention": self.normal_convention,
                "spp": {"role": "baked_normal", "tag": "#sppbaked", "onlyChannels": ["normal"]}}

    # -- the flattened result -------------------------------------------------
    def flattened_layer(self):
        cache = self.p.cached_channels(self.doc)
        approximate = False
        if not cache:
            cache = self._composite_cache()
            approximate = True
            if not cache:
                self._warn("no saved texture cache and nothing to composite, so there is no result layer")
                return None
            self._warn("no saved texture cache (Substance had it turned off or the project was never "
                       "displayed); the 'Substance Result' layer was rebuilt from the converted layers "
                       "and is approximate")
        maps = {}
        replace = {}
        has_rough = S.CH_ROUGHNESS in cache
        for flag, img in cache.items():
            if flag == S.CH_GLOSSINESS and not has_rough:
                # spec/gloss project: PawPaint is metal/rough, so give it roughness
                g = self._resize(img)[..., 0]
                g = _f(g) if g.dtype == np.uint8 else g.astype(np.float32)
                maps["roughness"] = self._write_gray(1.0 - g, "result_roughness")
                self._warn("glossiness converted to roughness (1 - glossiness)")
            if flag not in CHANNEL_MAP:
                self._warn("channel %s has no PawPaint equivalent and was left out" % S.CHANNEL_NAMES.get(flag, flag))
                continue
            cid, kind, custom = CHANNEL_MAP[flag]
            img = self._resize(img)
            tag = "result_%s" % cid
            if flag in (S.CH_BASECOLOR, S.CH_EMISSIVE, S.CH_SPECULAR):
                rgb = img[..., :3]
                if rgb.dtype != np.uint8:
                    rgb = (linear_to_srgb(rgb.astype(np.float32)) * 255 + 0.5).astype(np.uint8) \
                        if not approximate else _u8(rgb)
                maps[cid] = self._write_rgb(rgb, tag)
                if flag == S.CH_BASECOLOR and not approximate:
                    self.cache_albedo = rgb
                    if img.shape[2] >= 4:
                        self.cache_albedo_alpha = _f(img[..., 3]) if img.dtype == np.uint8 else img[..., 3].astype(np.float32)
            elif flag == S.CH_NORMAL:
                nrm = _f(img[..., :3]) if img.dtype == np.uint8 else img[..., :3].astype(np.float32)
                # texels Substance never computed are stored as 0,0,0: they're flat
                empty = np.linalg.norm(nrm, axis=-1) < 0.1
                if empty.any():
                    nrm[empty] = FLAT_NORMAL
                self.result_arrays["normal_layers"] = nrm
                baked = self.baked_map("normal")
                if baked is not None:
                    b = _f(baked[..., :3]) if baked.dtype == np.uint8 else baked[..., :3].astype(np.float32)
                    nrm = rnm_blend(b, nrm)
                self.result_arrays[cid] = nrm
                maps[cid] = self._write_rgb(nrm, tag)
            elif flag in (S.CH_HEIGHT, S.CH_DISPLACEMENT):
                h = img[..., 0].astype(np.float32)
                h = h if approximate else 0.5 + 0.5 * h
                self.result_arrays[cid] = np.clip(h, 0, 1)
                # PawPaint ADDS a layer's height to what is below it unless
                # the layer's height is set to Replace. This layer sits on
                # layers that already give Substance's height, so without
                # Replace every bump showed twice as high while it was visible.
                maps[cid] = self._write_gray(never_flat(h), tag, bits=16)
                replace[cid] = {"blendMode": "Replace", "opacity": 1.0}
            else:
                v = img[..., 0]
                v = _f(v) if v.dtype == np.uint8 else v.astype(np.float32)
                self.result_arrays[cid] = v
                maps[cid] = self._write_gray(v, tag)
            if custom:
                self._custom(cid, kind)
        if "normal" not in maps and self.baked_map("normal") is not None:
            b = self.baked_map("normal")
            maps["normal"] = self._write_rgb(b[..., :3], "result_normal")
        name = "Substance Result (approximate)" if approximate else "Substance Result (flattened)"
        spec = {"type": "Paint", "name": name, "visible": True, "channels": maps,
                "normalConvention": self.normal_convention, "spp": {"role": "flattened"}}
        if replace:
            spec["channelOverrides"] = replace
        return spec

    # -- layer stack ------------------------------------------------------------
    def _set_channels(self):
        """PawPaint ids of the channels this texture set has."""
        cids = getattr(self, "_cids", None)
        if cids is None:
            cids = set()
            for c in ((self.doc.get("stacks") or [{}])[0].get("channels") or []):
                flag = (c or {}).get("type")
                if flag in CHANNEL_MAP:
                    cids.add(CHANNEL_MAP[flag][0])
            self._cids = cids
        return cids

    def _blend(self, layer):
        """Return (blendMode, opacity, channelOverrides, disabled channel ids,
        per-channel (blend, opacity) where a channel differs from the layer's).

        Substance stores blending per channel: a list of entries, each naming
        the channels it is for. A channel that no entry names keeps its
        default, 100 %. A PawPaint layer has one blend mode and opacity plus
        per-channel overrides, so the layer's own are the base colour's and
        every channel that differs gets an override; a channel set to
        'Disable' is switched off.

        (Until 0.6.3 the FIRST entry was taken for the whole layer. In a test
        project the main colour folder's only entries set height and normal to
        0 %, so the folder came out at 0 % opacity: everything in it was
        invisible in PawPaint and had to be painted over by the detail fix.)"""
        entries = layer.get("perChannelBlending") or []
        if not entries:
            return "Normal", 1.0, None, set(), {}
        ALL = 0xFFFFFFFFFFFFFFFF
        base = None                           # an entry for every channel, if there is one
        per = {}
        for e in entries:
            mode = int(e.get("blendingMode", BLEND_NORMAL))
            op = float(e.get("opacity", 1.0))
            bits = int(e.get("channelTypes", 0))
            if (bits & ALL) == ALL:           # 64-bit masks, or 128-bit ones (Substance 12.2+)
                base = (mode, op)
            else:
                for flag, (cid, _, _) in CHANNEL_MAP.items():
                    if bits & flag:
                        per[cid] = (mode, op)
        cids = set(self._set_channels()) | set(per) | {"albedo"}
        fallback = base if base is not None else (None, 1.0)     # None: the channel's default mode
        eff = {cid: per.get(cid, fallback) for cid in cids}
        disabled = set(cid for cid, (m, _) in eff.items() if m == BLEND_DISABLE)
        if not (cids - disabled):
            return "Normal", 0.0, None, set(c for c, _, _ in CHANNEL_MAP.values()), {}
        if base is not None and base[0] == BLEND_DISABLE:
            # 'Disable' for everything except the channels named on their own
            disabled |= set(c for c, _, _ in CHANNEL_MAP.values()) - set(per)

        def mapped(cid, m, o, warn=True):
            """Substance's mode for one channel as PawPaint's (blend, opacity)."""
            if cid == "normal":
                return "Normal", o            # PawPaint always lays normals over each other
            if cid in ("height", "displacement"):
                return ("Replace" if m == 1 else "Add"), o       # PawPaint adds height, or replaces it
            if m is not None and m not in BLEND_NAMES and warn:
                self._warn("blend mode '%s'%s is not in PawPaint, Normal used"
                           % (SUBSTANCE_BLEND_LABELS.get(m, "unknown (id %s)" % m),
                              "" if cid == "albedo" else " (%s)" % cid))
            return BLEND_NAMES.get(m, "Normal"), o

        lead = "albedo" if "albedo" not in disabled else sorted(cids - disabled)[0]
        name, op = mapped(lead, *eff[lead])
        if lead != "albedo" and lead in ("normal", "height", "displacement"):
            name = "Normal"
        ov, chan = {}, {}
        for cid in sorted(cids - disabled):
            if cid == lead:
                continue
            nm, o = mapped(cid, *eff[cid])
            if cid in ("height", "displacement"):
                differs = nm == "Replace" or o != op
            elif cid == "normal":
                differs = o != op
            else:
                differs = (nm, o) != (name, op)
            if differs:
                ov[cid] = {"blendMode": nm, "opacity": o}
                chan[cid] = (nm, o)
        return name, op, (ov or None), disabled, chan

    def _mask_entry(self, layer, tag):
        if self.mb is None:
            return None, None
        pkey = self._pmask.get(id(layer))
        m = None
        if pkey is not None:
            try:
                t_wait = time.time()
                m, msgs = self.pool.result(pkey)
                detail = getattr(self.log, "detail", None)
                if detail is not None:
                    detail("      mask from worker %s: computed in %.1f s, waited %.1f s for it" % (
                        getattr(self.pool, "last_worker", "?"), getattr(self.pool, "last_seconds", 0.0),
                        time.time() - t_wait))
                for msg in msgs:
                    self._warn(msg)
            except Exception as e:  # a worker died (e.g. out of memory): do it here instead
                self.log("    (worker failed on this mask: %s; computing it here)" % e)
                pkey = None
        if pkey is None:
            m = self._builder_for(layer.get("maskActions")).layer_mask(layer)
        if m is None:
            return None, None
        m = self._up(m)
        path = self._write_gray(m, "mask_%s" % tag, pad_cov=self.coverage())
        return [{"path": path, "name": "Mask (from Substance)"}], _u8(m)

    def _value_for(self, flag, c):
        if flag in (S.CH_BASECOLOR, S.CH_EMISSIVE, S.CH_SPECULAR):
            return [float(x) for x in linear_to_srgb(c[:3])]
        if flag == S.CH_NORMAL:
            return [float(x) for x in c[:3]]
        if flag in (S.CH_HEIGHT, S.CH_DISPLACEMENT):
            return float(np.clip(0.5 + 0.5 * c[0], 0, 1))
        return float(np.clip(c[0], 0, 1))

    def _fill_content(self, fill):
        """Per channel content of a fill action: ({cid: value entry}, {cid: image}).
        Returns None when a source can't be reproduced at all."""
        vals, maps = {}, {}
        self._fill_alpha = None      # coverage from an image source's transparency
        self._albedo_unknown = False  # a colour source existed but couldn't be rebuilt
        # A fill that doesn't repeat (uvSamplingWrap 0/1/2) only covers its
        # own UV square, moved and scaled by the fill's transform; this holds
        # for plain colours too. a test project's 'Iris Main' is a colour fill shrunk to
        # the lower part of the iris and then blurred (checked against the
        # per-layer export).
        wrap = int(fill.get("uvSamplingWrap", 3))
        if wrap != 3 and int(fill.get("projection", 0) or 0) in P.UV_LIKE_PROJECTIONS:
            u, v = P.transform_uv(*P.texel_uv(self.res), fill.get("uvTransformation"))
            inside = P.wrap_coords(u, v, wrap)[2]
            if not inside.all():
                self._fill_alpha = inside.astype(np.float32)
        for src in as_list(fill.get("sources")):
            if not src:
                continue
            flag = int(src.get("channelTypes", 0))
            if flag not in CHANNEL_MAP:
                # one source feeding several channels: a Substance material
                if isinstance(src.get("procedural"), dict):
                    self._material_source(src, flag, vals)
                continue
            cid, kind, custom = CHANNEL_MAP[flag]
            if src.get("__class") == "DataSourceUniform":
                vals[cid] = {"value": self._value_for(flag, list(src.get("color") or [0, 0, 0])),
                             "opacity": float(src.get("opacity", 1.0))}
            else:
                url = _source_url(src)
                img = self.p.resource_image(url) if url else None
                if img is None:
                    what = P.sbsar_name(src.get("procedural")) if isinstance(src.get("procedural"), dict) else "image"
                    self._warn("%s source on the %s channel cannot be reproduced; left out" % (what or "procedural", cid))
                    if cid == "albedo":
                        self._albedo_unknown = True
                    continue
                if int(fill.get("projection", 0) or 0) not in P.UV_LIKE_PROJECTIONS:
                    self._warn("fill uses a %s projection; applied in UV space instead"
                               % P.PROJECTION_NAMES.get(int(fill.get("projection")), str(fill.get("projection"))))
                g = P.sample_uv(img, self.res, fill.get("uvTransformation"), int(fill.get("uvSamplingWrap", 3)) != 0)
                if g.ndim == 3 and g.shape[2] == 4 and float(g[..., 3].min()) < 0.999:
                    # Substance only paints where the image is opaque
                    a = g[..., 3].astype(np.float32)
                    self._fill_alpha = a if self._fill_alpha is None else self._fill_alpha * a
                if cid in RGB_CHANNELS:
                    g = g if g.ndim == 3 else np.repeat(g[..., None], 3, -1)
                    g = g[..., :3]
                else:
                    g = g[..., :3].mean(-1) if g.ndim == 3 else g
                maps[cid] = _u8(g)
            if custom:
                self._custom(cid, kind)
        # A flat normal fill doesn't override the baked normal in Substance
        # (that one is combined on top in the shader), so leave it out.
        nv = vals.get("normal")
        if nv and max(abs(a - b) for a, b in zip(nv["value"], FLAT_NORMAL)) < 0.01:
            vals.pop("normal")
        return vals, maps

    # Tweak names Substance materials commonly expose, per channel.
    _MATERIAL_TWEAKS = {
        "albedo": ("color", "basecolor", "base_color", "basecolor_color", "albedo", "diffuse", "tint", "color_1"),
        "roughness": ("roughness", "roughness_value", "rough", "glossiness_inverse"),
        "metallic": ("metallic", "metalness", "metal"),
        "emission": ("emissive", "emissive_color", "emission"),
    }

    def _material_source(self, src, bits, vals):
        """Approximate a procedural Substance material (sbsar) as flat values
        taken from its own parameters. Its texture detail can't be rebuilt;
        the 'Substance Result' layer shows it exactly."""
        proc = src["procedural"]
        name = P.sbsar_name(proc) or "material"
        tw = {str(k).lower(): v for k, v in P.tweaks(proc).items()}
        got = []
        for flag, (cid, kind, custom) in CHANNEL_MAP.items():
            if not bits & flag or cid in vals:
                continue
            v = None
            for key in self._MATERIAL_TWEAKS.get(cid, ()):
                if key in tw:
                    v = tw[key]
                    break
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                c = [float(x) for x in v[:3]] + [0.0] * (3 - len(v[:3]))
            else:
                c = [float(v)] * 3
            # material colour parameters are display (sRGB) values, unlike
            # uniform fill colours which are stored linear
            v_out = [min(max(x, 0.0), 1.0) for x in c] if cid in COLOR_CHANNELS else self._value_for(flag, c)
            vals[cid] = {"value": v_out, "opacity": float(src.get("opacity", 1.0))}
            got.append(cid)
            if custom:
                self._custom(cid, kind)
        if got:
            self._warn("Substance material '%s' approximated as flat %s from its settings (its texture "
                       "detail is only in the Substance Result layer)" % (name, "/".join(got)))
        else:
            self._warn("Substance material '%s' cannot be reproduced; left out" % name)
        if bits & S.CH_BASECOLOR and "albedo" not in vals:
            self._albedo_unknown = True

    def _solid(self, value, tag):
        """A solid-colour map, for channels PawPaint has no value setter for."""
        path = os.path.join(self.mapdir, "%s.png" % tag)
        if isinstance(value, list):
            Image.new("RGB", (self.res, self.res), tuple(int(round(x * 255)) for x in value[:3])).save(path, compress_level=PNG_LEVEL)
        else:
            Image.new("L", (self.res, self.res), int(round(value * 255))).save(path, compress_level=PNG_LEVEL)
        return "maps/%s.png" % tag

    def _convert_color_layer(self, L, spec, tag):
        actions = [a for a in ((L.get("actions") or {}).get("items") or []) if a and a.get("enabled", True)]
        fills = [a for a in actions if a.get("__class") == "DataActionFill"]
        if len(actions) == 1 and fills:
            vals, maps = self._fill_content(fills[0])
            spec["type"] = "Fill"
            channels = {}
            for cid in MAP_ONLY_CHANNELS:
                if cid in vals:
                    channels[cid] = self._solid(vals[cid]["value"], "fill_%s_%s" % (tag, cid))
            for cid, img in maps.items():
                channels[cid] = (self._write_rgb(img, "fill_%s_%s" % (tag, cid)) if img.ndim == 3
                                 else self._write_gray(img, "fill_%s_%s" % (tag, cid)))
            if channels:
                spec["channels"] = channels
            if maps and "normal" in maps:
                spec["normalConvention"] = self.normal_convention
            spec["spp"]["fill"] = {k: v for k, v in vals.items() if k not in MAP_ONLY_CHANNELS}
            spec["spp"]["mapChannels"] = sorted(channels)
            alpha = self._fill_alpha
            if alpha is not None:
                path = self._write_gray(alpha, "alpha_%s" % tag)
                masks = spec.get("masks") or []
                # masks are listed top-first; the top one multiplies the rest
                spec["masks"] = [{"path": path, "name": "Image transparency",
                                  "blendMode": "Multiply" if masks else "Max"}] + masks
            return {"kind": "fill", "values": vals, "maps": maps, "alpha": alpha,
                    "albedo_unknown": self._albedo_unknown}
        # Anything else becomes a paint layer rendered from its actions.
        spec["type"] = "Paint"
        rendered = self._render_paint(actions)
        if not rendered:
            spec["channels"] = {}
            if _has_content(actions):     # an empty paint layer is simply empty
                self._warn("layer content could not be reproduced, imported empty")
            return None
        channels = {}
        for cid, (val, alpha) in rendered.items():
            rgb = val if val.ndim == 3 else np.repeat(val[..., None], 3, -1)
            channels[cid] = self._write_rgb(rgb, "paint_%s_%s" % (tag, cid), alpha)
            if cid in CUSTOM_KINDS:
                self._custom(cid, CUSTOM_KINDS[cid])
        spec["channels"] = channels
        return {"kind": "paint", "channels": {c: (_u8(v), _u8(a)) for c, (v, a) in rendered.items()}}

    def _content_cover(self, act):
        """Brush coverage of one paint/eraser action for the layer's content:
        (coverage, (row slice, column slice)) with the coverage cut to the part
        of the texture the strokes touch, or (None, None) without strokes.
        Outside that part the coverage is 0 and blending changes nothing, so
        _render_paint only blends there; it used to blend every action over
        the whole texture (a 4K paint layer of 43 actions took half a minute)."""
        c = self._builder_for(act)._stroke_cover(act)
        if c is None:
            return None, None
        if isinstance(c, RegionCover) and c.res == self.res:
            return c.data, c.slices()
        full = self._up(as_full(c))
        ys = np.nonzero(full.any(1))[0]
        xs = np.nonzero(full.any(0))[0]
        if len(ys) == 0:
            return full[:0, :0], (slice(0, 0), slice(0, 0))
        sl = (slice(int(ys[0]), int(ys[-1]) + 1), slice(int(xs[0]), int(xs[-1]) + 1))
        return full[sl], sl

    def _render_paint(self, actions):
        """Render a layer's own actions per channel -> {cid: (value HxWxC or HxW, alpha)}."""
        if self.mb is None:
            return None
        out = {}
        flat = []

        def collect(items):
            for a in items or []:
                if not a or not a.get("enabled", True):
                    continue
                if a.get("__class") == "DataActionGroup":
                    collect((a.get("subStack") or {}).get("items"))
                else:
                    flat.append(a)
        collect(actions)
        for act in flat:
            cls = act.get("__class")
            maps = {}
            sl = (slice(None), slice(None))       # the part of the texture `cov` is for
            if cls == "DataActionFill":
                vals, maps = self._fill_content(act)
                cov = (self._fill_alpha if self._fill_alpha is not None
                       else np.ones((self.res, self.res), np.float32))
            elif cls == "DataActionPaint":
                vals = {}
                for src in as_list(act.get("sourceColor")):
                    if src.get("__class") != "DataSourceUniform":
                        continue
                    flag = int(src.get("channelTypes", 0))
                    if flag in CHANNEL_MAP:
                        vals[CHANNEL_MAP[flag][0]] = {"value": self._value_for(flag, list(src.get("color") or [0, 0, 0])),
                                                      "opacity": float(act.get("opacity", 1.0))}
                if not vals:
                    # Nothing this action paints can be used (e.g. a brush whose
                    # colour comes from a Substance material): don't replay its
                    # strokes. A layer of several hundred such actions took over
                    # a minute at 4K for a result that was thrown away.
                    continue
                cov, sl = self._content_cover(act)
                if cov is None:
                    continue
            elif cls == "DataActionEraser":
                if not out:
                    continue        # nothing painted yet, so nothing to erase
                cov, sl = self._content_cover(act)
                if cov is not None:
                    for cid in out:
                        al = out[cid][1]
                        al[sl] = al[sl] * (1 - cov)
                continue
            elif cls == "DataActionFilterLevels":
                bits = int(act.get("channelTypes", 0))
                for cid in list(out):
                    flag = next((f for f, (c, _, _) in CHANNEL_MAP.items() if c == cid), 0)
                    if bits & flag:
                        out[cid] = (P.apply_levels(out[cid][0], act.get("levels") or {}), out[cid][1])
                continue
            elif isinstance(act.get("procedural"), dict) and P.sbsar_name(act["procedural"]) == "blur":
                # a blur on the layer's content softens its coverage and colours
                inten = P.tweaks(act["procedural"]).get("intensity", 0.5)
                for cid in list(out):
                    val, al = out[cid]
                    if val.ndim == 3:
                        bl = [P.blur(val[..., k], al, inten, self.res) for k in range(val.shape[2])]
                        out[cid] = (np.stack([b[0] for b in bl], -1).astype(np.float32), bl[0][1].astype(np.float32))
                    else:
                        v2, a2 = P.blur(val, al, inten, self.res)
                        out[cid] = (v2.astype(np.float32), a2.astype(np.float32))
                continue
            elif isinstance(act.get("procedural"), dict):
                self._warn("effect '%s' inside a paint layer cannot be reproduced and was skipped"
                           % P.sbsar_name(act["procedural"]))
                continue
            else:
                self._warn("layer content action %s was skipped" % cls)
                continue
            entries = list(vals.items()) + [(c, {"image": m, "opacity": 1.0}) for c, m in maps.items()]
            for cid, e in entries:
                if "image" in e:
                    v = _f(e["image"])
                else:
                    v = np.asarray(e["value"], np.float32)
                k = cov * e["opacity"]
                if cid not in out:
                    shape = (self.res, self.res, 3) if (v.ndim == 1 or v.ndim == 3) else (self.res, self.res)
                    out[cid] = (np.zeros(shape, np.float32), np.zeros((self.res, self.res), np.float32))
                val, al = out[cid]
                kk = k[..., None] if val.ndim == 3 else k
                vs = v[sl] if "image" in e else v
                val[sl] += (vs - val[sl]) * kk
                al[sl] += (1 - al[sl]) * k
        return out or None

    def _walk(self, items, parent_idx, path):
        for L in items:
            if not L:
                continue
            self._count += 1
            name = L.get("label") or "Layer"
            self._context = "/".join(path + [name])
            # progress, so big hand-painted projects never look frozen
            self.log("    layer %d/%d: %s" % (self._count, self._total, self._context))
            builders = [b for b in (self.mb, self.mb_low) if b is not None]
            for b in builders:
                b.enter(L)
            try:
                self._convert_one(L, name, parent_idx, path)
            except Exception as e:  # one broken layer must not stop the rest
                self._warn("layer could not be converted (%s); left out" % e)
            finally:
                for b in builders:
                    b.leave()
        self._context = ""

    def _submit_heavy_masks(self):
        """Queue masks with lots of brush strokes on the worker pool, heaviest
        first, so they compute on other cores while we walk the stack."""
        if self.pool is None or self.mb is None:
            return
        from . import parallel
        jobs = []

        def walk(items, ancestors):
            for L in items:
                if not L:
                    continue
                if L.get("maskEnabled") and L.get("maskActions"):
                    n = _stroke_points(L["maskActions"])
                    if n > PARALLEL_MIN_POINTS:
                        jobs.append((n, L, list(ancestors)))
                if L.get("subStack"):
                    walk(L["subStack"].get("items") or [], ancestors + [L])
        walk(self.items, [])
        for n, L, anc in sorted(jobs, key=lambda j: -j[0]):
            res = self.mb_low.res if (self.mb_low is not None and n > HEAVY_STROKE_POINTS) else self.res
            key = parallel.mask_key(L, anc)
            self._pmask[id(L)] = self.pool.submit(key, self.scene_index, res, self._baked_urls, anc, L,
                                                  lossless=self.lossless, engine=self.engine)
        if jobs:
            self.log("  %d hand-painted masks are being computed by %d worker processes%s"
                     % (len(jobs), self.pool.workers, " (each using the GPU)" if self.engine == "gpu" else ""))

    def _convert_one(self, L, name, parent_idx, path):
        tag = "%03d_%s" % (self._count, _safe(name))
        mode, op, ov, disabled, chan = self._blend(L)
        spec = {"name": name, "visible": bool(L.get("enabled", True)), "opacity": op,
                "blendMode": mode, "spp": {"uid": L.get("uid"), "tag": "#spp%d" % self._count}}
        if parent_idx:
            spec["parent"] = parent_idx
        if ov:
            spec["channelOverrides"] = ov
        if disabled:
            spec["spp"]["disabledChannels"] = sorted(disabled)
        if L.get("__partial__"):
            self._warn("this layer uses Substance data the converter doesn't know yet; some settings may be missing")
        cls = L.get("__class")
        if cls not in ("DataLayerGroup", "DataLayerColor"):
            self._warn("layer type %s is not supported and was skipped" % cls)
            return
        self._inexact = False
        masks, mask_arr = self._mask_entry(L, tag)
        if masks:
            spec["masks"] = masks
        node = {"mask": mask_arr, "op": op if spec["visible"] else 0.0, "disabled": disabled, "mode": mode,
                "spec": spec, "tag": tag,
                # channels whose blend or opacity is not the layer's own (see _blend)
                "chan": {c: (nm, o if spec["visible"] else 0.0) for c, (nm, o) in chan.items()}}
        # 'exact': the mask comes only from steps that are rebuilt exactly
        # (fills, images, projections, polygon fills), with no brush strokes
        # and no warnings. The fitter leaves such layers alone (refine.py).
        # Polygon fills don't count: their picks depend on what was visible
        # (one test project's tail: 3.75% of texels off, and trusting them cost the
        # editable layers 3.3%).
        node["exact"] = (not self._inexact and _stroke_points(L.get("maskActions")) == 0
                         and not _has_class(L.get("maskActions"), ("DataActionDecal",)))
        if cls == "DataLayerGroup":
            spec["type"] = "Folder"
            self.layers.append(spec)
            idx = len(self.layers)
            if (L.get("actions") or {}).get("items"):
                self._warn("effects placed directly on a folder were skipped")
            node["children"] = []
            self._stack.append(node["children"])
            try:
                self._walk(((L.get("subStack") or {}).get("items")) or [], idx, path + [name])
            finally:
                self._stack.pop()
            self._stack[-1].append(node)
        else:
            self._inexact = False
            content = self._convert_color_layer(L, spec, tag)
            node["exact"] = (node["exact"] and not self._inexact and _stroke_points(L.get("actions")) == 0
                             and not _has_class(L.get("actions"), ("DataActionDecal",)))
            if disabled and spec["type"] == "Fill":
                for cid in disabled:
                    spec["spp"].get("fill", {}).pop(cid, None)
                    if "channels" in spec:
                        spec["channels"].pop(cid, None)
            self.layers.append(spec)
            node["content"] = content
            self._stack[-1].append(node)

    def convert(self, include_layers=True, include_flat=True, include_baked=True, exact=None):
        """Write manifest.json + maps. `exact` (optional) is a list of layer
        specs from the Substance companion exporter that replaces the rebuilt
        layers for this texture set."""
        self._stack = [[]]
        if include_baked:
            self.export_baked_maps()
            bn = self.baked_normal_layer()
            if bn:
                self.layers.append(bn)
        if exact is not None:
            base = len(self.layers)
            for s in exact:
                s = dict(s)
                if s.get("parent"):
                    s["parent"] += base
                self.layers.append(s)
        elif include_layers:
            def n_layers(items):
                return sum(1 + n_layers(((L.get("subStack") or {}).get("items")) or []) for L in items if L)
            self._total = n_layers(self.items)
            self._submit_heavy_masks()
            self._walk(self.items, 0, [])
            if self.pool is not None and self._pmask:
                self.pool.release()     # every mask is in: free the workers' memory for the fit
        flat = None
        if include_flat:
            flat = self.flattened_layer()
        elif exact is None and include_layers and self.fit_to_result and self.mb is not None:
            # No result layer wanted, but the masks are still fitted to Substance's
            # result: read it without writing its images or declaring its channels.
            # (Before 0.6.2, leaving the result layer out silently skipped the fit
            # and the detail fix, so the layers came out as the raw replay.)
            custom, notes = dict(self.custom), list(self.warnings)
            self._discard_writes = True
            try:
                self.flattened_layer()
            finally:
                self._discard_writes = False
                self.custom, self.warnings = custom, notes      # notes about a layer that isn't written
        if exact is None and include_layers and self.fit_to_result and self.mb is not None:
            self._add_black_base()
            self._normal_where_nothing_below(self._stack[0])
            if self.cache_albedo is not None:
                self._fit_masks()
        if flat:
            flat["spp"]["tag"] = "#sppflat"
            self.layers.append(flat)
        self.flush_writes()
        # Every channel a layer names must be declared, or PawPaint refuses the
        # whole texture set ("unknown channel").
        for s in self.layers:
            for cid in list((s.get("channels") or {})) + list((s.get("channelOverrides") or {})):
                if cid not in BUILTIN_CHANNELS and cid not in self.custom:
                    self._custom(cid, CUSTOM_KINDS.get(cid, "scalar"))
        manifest = {"format": "pawpaint-layer-bundle", "version": 1,
                    "channels": [{"id": c, "name": n, "kind": k} for c, (n, k) in sorted(self.custom.items())],
                    "layers": self.layers}
        with open(os.path.join(self.outdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        return manifest

    def _add_black_base(self):
        """Where no layer fully covers a texel, Substance shows the layers over
        black (a test project's eyes: the Blush glow sits on an island with nothing under
        it). A black fill layer at the bottom makes PawPaint show the same,
        whatever it does with partly covered texels on its own."""
        a = self.cache_albedo_alpha
        if a is None or not self._stack[0]:
            return
        inside = self.coverage() > 0.5
        n_in = int(inside.sum())
        if n_in == 0 or int((inside & (a < 0.99)).sum()) <= BLACK_BASE_MIN_SHARE * n_in:
            return
        pos = 0
        while pos < len(self.layers) and (self.layers[pos].get("spp") or {}).get("role") == "baked_normal":
            pos += 1
        for sp in self.layers:
            if sp.get("parent", 0) > pos:
                sp["parent"] += 1
        black = {"value": [0.0, 0.0, 0.0], "opacity": 1.0}
        spec = {"name": "Black base (as in Substance)", "type": "Fill", "visible": True, "opacity": 1.0,
                "blendMode": "Normal",
                "spp": {"uid": None, "tag": "#sppbase", "role": "black_base", "fill": {"albedo": dict(black)},
                        "mapChannels": []}}
        self.layers.insert(pos, spec)
        self._stack[0].insert(0, {"mask": None, "op": 1.0, "disabled": set(), "mode": "Normal", "spec": spec,
                                  "tag": "000_base", "exact": True,
                                  "content": {"kind": "fill", "values": {"albedo": dict(black)}, "maps": {},
                                              "alpha": None}})
        self._warn("some areas have no fully opaque layer; Substance shows them over black, so a "
                   "'Black base (as in Substance)' layer was added at the bottom")

    # -- fitting to Substance's own result --------------------------------------
    def _normal_where_nothing_below(self, nodes):
        """PawPaint blends a Multiply/Screen/... layer against an empty
        background as if it were black; Substance shows it as is. A blend layer
        or folder with nothing under it in its own folder becomes Normal."""
        seen = False
        for n in nodes:
            if "children" in n:
                self._normal_where_nothing_below(n["children"])
            has = "children" in n or bool(n.get("content"))
            if has and not seen and n.get("mode", "Normal") != "Normal":
                n["mode"] = "Normal"
                n["spec"]["blendMode"] = "Normal"
                self._warn("'%s' blends with nothing beneath it; imported as Normal so it shows like in "
                           "Substance" % n["spec"].get("name"))
            seen = seen or has

    def _fit_masks(self):
        from . import refine
        self.log("    fitting masks to Substance's result...")
        new, baked, resid, st = refine.fit(self, self.cache_albedo, self.coverage(), self.log)
        n = max(st["texels"], 1)
        self.fit_stats = st
        self.log("    masks fitted: layers match Substance on %.2f%% of texels (was %.2f%%), %.2f%% need the "
                 "detail fix" % (100 - 100.0 * st["mismatch_after"] / n, 100 - 100.0 * st["mismatch_before"] / n,
                                 100.0 * st["fixed"] / n))
        if st.get("baked"):
            self._warn("textured in Substance, so its exact texture was baked into the layer: %s"
                       % ", ".join(st["baked"]))

        def walk(nodes):
            for nd in nodes:
                if "children" in nd:
                    nd["mask"] = None
                    nd["spec"].pop("masks", None)
                    walk(nd["children"])
                    continue
                m = new.get(id(nd))
                spec = nd.get("spec")
                if m is None or spec is None:
                    continue
                c = nd.get("content") or {}
                tex = baked.get(id(nd))
                if tex is not None:
                    # its content isn't a flat colour in Substance: use the exact texture
                    if c.get("kind") == "fill":
                        c.setdefault("maps", {})["albedo"] = tex
                        (c.get("values") or {}).pop("albedo", None)
                        spec.setdefault("channels", {})["albedo"] = self._write_rgb(tex, "tex_%s_albedo" % nd["tag"])
                        spec["spp"].get("fill", {}).pop("albedo", None)
                        spec["spp"]["mapChannels"] = sorted(set(spec["spp"].get("mapChannels") or []) | {"albedo"})
                    elif c.get("kind") == "paint" and "albedo" in (c.get("channels") or {}):
                        a = c["channels"]["albedo"][1]
                        c["channels"]["albedo"] = (tex, a)
                        spec.setdefault("channels", {})["albedo"] = self._write_rgb(tex, "tex_%s_albedo" % nd["tag"], a)
                est = (st.get("estimated") or {}).get(id(nd))
                if est is not None and tex is None and c.get("kind") == "fill":
                    # its material couldn't be rebuilt: use the colour Substance shows
                    e = {"value": [float(x) for x in est[:3]], "opacity": 1.0}
                    c.setdefault("values", {})["albedo"] = e
                    spec["spp"].setdefault("fill", {})["albedo"] = dict(e)
                if c.get("kind") == "fill":
                    c["alpha"] = None              # baked into the mask
                    for e in (c.get("values") or {}).values():
                        e["opacity"] = 1.0         # PawPaint has no per-source opacity
                if int(m.min()) >= 255:
                    nd["mask"] = None
                    spec.pop("masks", None)
                    continue
                nd["mask"] = _u8(m)
                path = self._write_gray(m, "fit_%s" % nd["tag"])
                spec["masks"] = [{"path": path, "name": "Mask (fitted to Substance)"}]
        walk(self._stack[0])
        self._apply_scalar_fits(refine.fit_scalars(
            self, new, self.coverage(),
            {c: self.result_arrays[c] for c in ("roughness", "metallic") if c in self.result_arrays}))
        self._dissolve_passthrough_folders()
        fix = {}
        if resid is not None:
            fix["albedo"] = (resid[..., :3], resid[..., 3])
        fix.update(self._channel_fixes())
        if fix:
            channels, content, over = {}, {}, {}
            for cid, (v, a) in fix.items():
                if cid in ("height", "displacement"):
                    # Height replaces what is below wherever it isn't exactly
                    # flat (PawPaint ignores a layer's coverage for height), so
                    # it needs no alpha and can keep 16 bits: an 8-bit height
                    # shows as fine steps once PawPaint lights it.
                    channels[cid] = self._write_gray(v, "detail_fix_%s" % cid, bits=16)
                    over[cid] = {"blendMode": "Replace", "opacity": 1.0}
                else:
                    rgbv = v if v.ndim == 3 else np.repeat(_u8(v)[..., None], 3, -1)
                    channels[cid] = self._write_rgb(rgbv, "detail_fix_%s" % cid, a)
                content[cid] = (_u8(v), a)
            spec = {"type": "Paint", "name": "Detail fix (from Substance)", "visible": True,
                    "opacity": 1.0, "blendMode": "Normal", "channels": channels,
                    "spp": {"tag": "#sppfix", "role": "detail_fix"}}
            if over:
                spec["channelOverrides"] = over
            self.layers.append(spec)
            self._stack[0].append({"mask": None, "op": 1.0, "mode": "Normal", "disabled": set(),
                                   "content": {"kind": "paint", "channels": content}})
            parts = []
            for cid, (v, a) in fix.items():
                parts.append("%s %.2f%%" % (cid, 100.0 * float((a > 0).mean())))
            self._warn("texels the layers can't reproduce (procedural textures, filters, blend modes PawPaint "
                       "doesn't have) are in the 'Detail fix (from Substance)' layer: %s" % ", ".join(parts))

    def _apply_scalar_fits(self, fits):
        """Set roughness/metallic of layers to what Substance shows where they're on top."""
        if not fits:
            return

        def walk(nodes):
            for nd in nodes:
                if "children" in nd:
                    walk(nd["children"])
                    continue
                f = fits.get(id(nd))
                if not f:
                    continue
                c, spec = nd.get("content") or {}, nd["spec"]
                for cid, v in f.items():
                    if c.get("kind") == "fill":
                        if np.isscalar(v) or np.ndim(v) == 0:
                            e = {"value": float(v), "opacity": 1.0}
                            c.setdefault("values", {})[cid] = e
                            (c.get("maps") or {}).pop(cid, None)
                            spec["spp"].setdefault("fill", {})[cid] = dict(e)
                            if cid in (spec.get("channels") or {}):
                                spec["channels"].pop(cid)
                                spec["spp"]["mapChannels"] = sorted(set(spec["spp"].get("mapChannels") or []) - {cid})
                        else:
                            c.setdefault("maps", {})[cid] = _u8(v)
                            (c.get("values") or {}).pop(cid, None)
                            spec["spp"].get("fill", {}).pop(cid, None)
                            spec.setdefault("channels", {})[cid] = self._write_gray(v, "tex_%s_%s" % (nd["tag"], cid))
                            spec["spp"]["mapChannels"] = sorted(set(spec["spp"].get("mapChannels") or []) | {cid})
                    elif c.get("kind") == "paint" and cid in (c.get("channels") or {}):
                        a = c["channels"][cid][1]
                        vv = np.full(a.shape, float(v), np.float32) if np.ndim(v) == 0 else v
                        c["channels"][cid] = (_u8(vv), a)
                        spec.setdefault("channels", {})[cid] = self._write_rgb(
                            np.repeat(_u8(vv)[..., None], 3, -1), "tex_%s_%s" % (nd["tag"], cid), a)
        walk(self._stack[0])

    def _channel_fixes(self):
        """Roughness, metallic, height and normal: where the fitted layers
        don't give Substance's value, the detail fix layer holds it (per
        channel, so it never hides another channel)."""
        out = {}
        cov = self.coverage() > 0.5
        for cid in ("roughness", "metallic", "height"):
            target = self.result_arrays.get(cid)
            if target is None:
                continue
            comp = self.composite(cid)
            if comp is None:
                comp = np.full(target.shape, 0.5 if cid == "height" else 0.0, np.float32)
            # Height is lit from its slope, so the edge of a corrected patch
            # shows as a ridge as high as the tolerance: 8/255 gave 'random
            # bumps' around every patch (reported on a project with embossed
            # lines). It gets a tolerance PawPaint's lighting can't show.
            tol = HEIGHT_FIX_TOL if cid == "height" else 8.0 / 255.0
            bad = cov & (np.abs(comp - target) > tol)
            if bad.any():
                bad = _dilate(bad.astype(np.uint8), 1) > 0
                v = np.where(bad, target, 0.5 if cid == "height" else 0.0).astype(np.float32)
                if cid == "height":
                    # Replace only acts where the value isn't exactly flat
                    v = np.where(bad, never_flat(v), np.float32(0.5))
                out[cid] = (v, (bad * 255).astype(np.uint8))
        target = self.result_arrays.get("normal")
        detail = self.result_arrays.get("normal_layers")
        if target is not None and detail is not None:
            comp = self.composite("normal")
            has_detail = np.abs(detail - np.array(FLAT_NORMAL, np.float32)).max(-1) > 0.02
            differs = has_detail if comp is None else (has_detail | (np.abs(comp - detail).max(-1) > 0.03))
            bad = cov & differs
            if bad.any():
                bad = _dilate(bad.astype(np.uint8), 1) > 0
                out["normal"] = (np.where(bad[..., None], target, np.array(FLAT_NORMAL, np.float32)).astype(np.float32),
                                 (bad * 255).astype(np.uint8))
        return out

    def _dissolve_passthrough_folders(self):
        """Substance folders usually pass through: a blend layer inside one
        blends with everything beneath the folder. PawPaint folders composite
        their children on their own first, so a Multiply/Overlay/... layer in a
        folder would only blend with that folder's other layers. Such folders
        are dissolved (their layers move up a level, in order), which is what
        pass-through means. Folder masks are already baked into the layers."""
        def has_blend(nodes):
            return any(has_blend(n["children"]) if "children" in n else
                       (n.get("mode", "Normal") != "Normal" and n.get("content")) for n in nodes)

        index = {id(sp): i for i, sp in enumerate(self.layers)}
        parent_of = {id(sp): (self.layers[sp["parent"] - 1] if sp.get("parent") else None) for sp in self.layers}
        gone = set()

        def mute(nodes, off):
            for k in nodes:
                k["disabled"] = set(k.get("disabled") or ()) | off
                for c in off:
                    (k.get("chan") or {}).pop(c, None)
                sp = k.get("spec")
                if sp is not None:
                    info = sp.setdefault("spp", {})
                    info["disabledChannels"] = sorted(set(info.get("disabledChannels") or ()) | off)
                    ovr = sp.get("channelOverrides")
                    if ovr:
                        for c in off:
                            ovr.pop(c, None)
                        if not ovr:
                            sp.pop("channelOverrides", None)
                if "children" in k:
                    mute(k["children"], off)

        def walk(nodes, parent_spec):
            out = []
            for n in nodes:
                if "children" not in n:
                    out.append(n)
                    continue
                spec = n.get("spec")
                dissolve = (spec is not None and n.get("mode", "Normal") == "Normal"
                            and float(n.get("op", 1.0)) >= 0.999 and has_blend(n["children"]))
                kids = walk(n["children"], parent_spec if dissolve else spec)
                if dissolve:
                    gone.add(id(spec))
                    # the folder goes, so what it switched off for everything in
                    # it (a channel at 0 %, or 'Disable') is switched off in each
                    # of its layers instead
                    off = set(n.get("disabled") or ()) | set(c for c, (_, o) in (n.get("chan") or {}).items() if o <= 0)
                    if off:
                        mute(kids, off)
                    for k in kids:
                        if k.get("spec") is not None and parent_of.get(id(k["spec"])) is spec:
                            parent_of[id(k["spec"])] = parent_spec
                    out.extend(kids)
                else:
                    n["children"] = kids
                    out.append(n)
            return out

        self._stack[0][:] = walk(self._stack[0], None)
        if not gone:
            return
        names = [sp.get("name") for sp in self.layers if id(sp) in gone]
        keep = [sp for sp in self.layers if id(sp) not in gone]
        pos = {id(sp): i + 1 for i, sp in enumerate(keep)}
        for sp in keep:
            par = parent_of.get(id(sp))
            while par is not None and id(par) in gone:
                par = parent_of.get(id(par))
            if par is None:
                sp.pop("parent", None)
            else:
                sp["parent"] = pos[id(par)]
        self.layers[:] = keep
        self._warn("folders with blend layers inside were dissolved so the blends work like in Substance "
                   "(pass-through): %s" % ", ".join(str(x) for x in names))

    # -- compositing (validation + fallback result) -----------------------------
    def composite(self, cid):
        """Render the rebuilt stack's `cid` channel (PawPaint space, float 0..1),
        or None if no layer touches it. Normal blending only."""
        res = self.res
        rgb = cid in RGB_CHANNELS
        touched = [False]

        def content_of(n):
            c = n.get("content")
            if not c or cid in n.get("disabled", ()):
                return None
            if c["kind"] == "paint":
                if cid not in c["channels"]:
                    return None
                v, a = c["channels"][cid]
                return _f(v), _f(a)
            alpha = c.get("alpha")
            if cid in c["maps"]:
                v = _f(c["maps"][cid])
                return v, (alpha if alpha is not None else np.ones((res, res), np.float32))
            e = c["values"].get(cid)
            if e is None:
                return None
            v = np.asarray(e["value"], np.float32)
            v = np.broadcast_to(v, (res, res, 3) if rgb else (res, res))
            a = np.full((res, res), e["opacity"], np.float32)
            return v, (a * alpha if alpha is not None else a)

        signed = cid in ("height", "displacement")

        def blend_of(n):
            """(blend, opacity) of node n for this channel."""
            own = (n.get("chan") or {}).get(cid)
            return own if own is not None else (n.get("mode", "Normal"), n["op"])

        def render(nodes):
            if signed:
                # PawPaint: h = clamp(acc + (content - 0.5) * opacity * mask)
                h = np.full((res, res), 0.5, np.float32)
                for n in nodes:
                    if "children" in n:
                        if cid in (n.get("disabled") or ()):
                            continue
                        c = render(n["children"])
                        a = np.ones((res, res), np.float32)
                    else:
                        r = content_of(n)
                        if r is None:
                            continue
                        c, a = r
                        touched[0] = True
                    k = a * blend_of(n)[1]
                    if n["mask"] is not None:
                        k = k * _f(n["mask"])
                    h = np.clip(h + (c - 0.5) * k, 0, 1)
                return h
            col = np.zeros((res, res, 3) if rgb else (res, res), np.float32)
            al = np.zeros((res, res), np.float32)
            for n in nodes:
                if "children" in n:
                    if cid in (n.get("disabled") or ()):
                        continue
                    c, a = render(n["children"])
                else:
                    r = content_of(n)
                    if r is None:
                        continue
                    c, a = r
                    touched[0] = True
                mode, op_c = blend_of(n)
                k = a * op_c
                if n["mask"] is not None:
                    k = k * _f(n["mask"])
                # PawPaint's merge (layer_merge.frag), on stored values
                b = _blend_pixels(mode, col, c) if cid != "normal" else c
                a_out = k + al * (1 - k)
                safe = np.maximum(a_out, 1e-12)
                if rgb:
                    nc = (b * k[..., None] + col * (al * (1 - k))[..., None]) / safe[..., None]
                    col = np.where((a_out > 0)[..., None], nc, col)
                else:
                    nc = (b * k + col * al * (1 - k)) / safe
                    col = np.where(a_out > 0, nc, col)
                al = a_out
            return col, al

        out = render(self._stack[0])
        if not touched[0]:
            return None
        return out if signed else out[0]

    def composite_albedo(self):
        c = self.composite("albedo")
        return c if c is not None else np.zeros((self.res, self.res, 3), np.float32)

    def _composite_cache(self):
        """Fallback 'cache' made from our own composites, keyed by channel flag."""
        out = {}
        for flag, (cid, _, _) in CHANNEL_MAP.items():
            c = self.composite(cid)
            if c is None:
                continue
            if cid == "normal":
                c = np.where(np.linalg.norm(c, axis=-1, keepdims=True) < 1e-6, np.array(FLAT_NORMAL, np.float32), c)
            out[flag] = c if c.ndim == 3 else c[..., None]
        return out
