"""Approximations of Substance procedural effects, in numpy only.

Substance runs filters and generators (.sbsar graphs) inside its own engine;
we can't run those. What we can do is approximate the common ones from their
recorded parameters and, for generators, the mesh maps baked into the .spp
(curvature, AO, position, thickness, world-space normals). Every function
here works on float32 images in [0, 1], H x W, row 0 = top (v = 1).
"""

import math

import numpy as np
from PIL import Image, ImageFilter

# Measured in Substance 12.2 (calibration plugin, starter 'blur' on a hard
# mask edge): the result is a Gaussian, the same in UV units at 512/1024/2048,
# with sigma = 1.125 px at 512 per unit of intensity from intensity ~1.5 up
# (2 -> 2.35, 4 -> 4.5, 8 -> 9.0, 16 -> 18.0 px). Below that the kernel is
# relatively wider until it vanishes at 0.3 (table below, px at 512).
BLUR_SIGMA_PER_INTENSITY = 2.25    # pixels at 1024 for intensity 1.0 (linear part)
BLUR_LOW_TABLE = ((0.3, 0.0), (0.4, 0.33), (0.5, 0.69), (0.75, 1.13), (1.0, 1.31), (1.5, 1.85), (2.0, 2.35))
# bevel() is kept for experiments but NOT used: no scale in {0.02..0.25}
# beat skipping the filter (Eye 94.35 skipped vs <=94.32), so its real
# behaviour is unknown.
BEVEL_DISTANCE_SCALE = 0.05
USE_BEVEL = False


def sbsar_name(proc):
    """'resource://starter_assets/blur/blur?version=..' -> 'blur'."""
    url = (proc or {}).get("urlToSbsRes") or ""
    return url.split("?")[0].rstrip("/").split("/")[-1].lower()


def tweaks(proc):
    return {t.get("identifier"): t.get("value") for t in (proc or {}).get("tweaks") or [] if t}


# -- blur ---------------------------------------------------------------------
def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(sigma * 3)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def _blur_separable(a, sigma):
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    p = np.pad(a, ((0, 0), (r, r)), mode="wrap")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * p[:, i:i + a.shape[1]]
    p = np.pad(out, ((r, r), (0, 0)), mode="wrap")
    out2 = np.zeros_like(a)
    for i, w in enumerate(k):
        out2 += w * p[i:i + a.shape[0], :]
    return out2


def gaussian_blur(a, sigma):
    """Gaussian blur with texture wrap. Large sigmas run on a downsampled copy."""
    a = np.asarray(a, np.float32)
    if sigma < 0.3:
        return a.copy()
    h, w = a.shape
    f = 1
    while sigma / f > 6 and min(h, w) // (f * 2) >= 64:
        f *= 2
    if f == 1:
        return _blur_separable(a, sigma)
    small = np.asarray(Image.fromarray(a, "F").resize((w // f, h // f), Image.BOX))
    small = _blur_separable(small, sigma / f)
    return np.asarray(Image.fromarray(small, "F").resize((w, h), Image.BILINEAR))


def blur_sigma_px(intensity, res):
    """Gaussian sigma in pixels at `res` for a Substance blur intensity."""
    i = max(0.0, float(intensity))
    xs, ys = zip(*BLUR_LOW_TABLE)
    s512 = float(np.interp(i, xs, ys)) if i <= xs[-1] else i * BLUR_SIGMA_PER_INTENSITY / 2.0
    return s512 * res / 512.0


def blur(v, alpha, intensity, res):
    """Blur a (value, alpha) buffer the way a filter on a mask stack does."""
    sigma = blur_sigma_px(intensity, res)
    va = gaussian_blur(v * alpha, sigma)
    aa = gaussian_blur(alpha, sigma)
    vv = np.where(aa > 1e-4, va / np.maximum(aa, 1e-4), v)
    return np.clip(vv, 0, 1), np.clip(aa, 0, 1)


# -- bevel --------------------------------------------------------------------
def _distance_inside(mask, max_px):
    """Approximate distance (px) from each inside texel to the mask edge."""
    img = Image.fromarray((mask * 255).astype(np.uint8))
    dist = np.zeros(mask.shape, np.float32)
    cur = img
    steps = int(min(max_px, 256))
    for d in range(1, steps + 1):
        cur = cur.filter(ImageFilter.MinFilter(3))
        a = np.asarray(cur) > 127
        dist += a
        if not a.any():
            break
    return dist


def bevel(v, distance, smoothing, res):
    """Ramp the inside of a mask from its edge inward (distance < 0 = inward)."""
    px = max(1.0, abs(float(distance)) * res * BEVEL_DISTANCE_SCALE)
    scale = 1
    while px / scale > 48 and res // (scale * 2) >= 128:
        scale *= 2
    m = v > 0.5
    if scale > 1:
        sm = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((res // scale, res // scale), Image.BOX)) > 127
        d = _distance_inside(sm.astype(np.float32), px / scale + 1) * scale
        d = np.asarray(Image.fromarray(d.astype(np.float32), "F").resize((res, res), Image.BILINEAR))
    else:
        d = _distance_inside(m.astype(np.float32), px + 1)
    t = np.clip(d / px, 0, 1)
    s = float(np.clip(smoothing, 0, 1))
    ramp = t * (1 - s) + (t * t * (3 - 2 * t)) * s
    return np.where(m, ramp, 0.0).astype(np.float32)


# -- images in UV space ---------------------------------------------------------
def transform_uv(u, v, transform=None):
    """Apply a fill's Data2DTransformation (scale = tiling, rotation in
    degrees, translation) to texture coordinates (v up)."""
    tr = transform or {}
    sx, sy = (tr.get("scaleFactors") or [1.0, 1.0])[:2]
    rot = math.radians(float(tr.get("rotation") or 0.0))
    tx, ty = (tr.get("translation") or [0.0, 0.0])[:2]
    px, py = (tr.get("GUIPivot") or [0.5, 0.5])[:2]
    if (sx, sy, rot, tx, ty) == (1.0, 1.0, 0.0, 0.0, 0.0):
        return u, v
    # Measured on a test project's iris fills (colour fills shrunk to a window, four
    # layers, window edges within 0.0001 of Substance's per-layer export):
    # the translation is applied first, then the scale about the pivot.
    # (Before: scale first, then translate, which put the windows elsewhere.)
    c, s = math.cos(rot), math.sin(rot)
    u0, v0 = u - px - tx, v - py - ty
    return (c * u0 - s * v0) * sx + px, (s * u0 + c * v0) * sy + py


def texel_uv(res):
    """Texture coordinates of every texel centre (v up, row 0 = top)."""
    t = (np.arange(res, dtype=np.float32) + 0.5) / res
    return np.repeat(t[None, :], res, 0), np.repeat((1.0 - t)[:, None], res, 1)


def sample_at(img, u, v):
    """Nearest-texel lookup of an image at texture coordinates in [0, 1] (v up)."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., None]
    h, w = a.shape[:2]
    ix = np.clip((np.asarray(u) * w).astype(np.int64), 0, w - 1)
    iy = np.clip(((1.0 - np.asarray(v)) * h).astype(np.int64), 0, h - 1)
    out = a[iy, ix]
    return out[..., 0] if out.shape[-1] == 1 else out


def sample_uv(img, res, transform=None, wrap=True):
    """Resample an image onto a res x res UV-space texture through a fill's
    Data2DTransformation."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., None]
    identity = transform_uv(0.25, 0.75, transform) == (0.25, 0.75)
    if identity and a.shape[:2] == (res, res):
        return a[..., 0] if a.shape[2] == 1 else a
    u, v = transform_uv(*texel_uv(res), transform)
    if wrap:
        u, v = np.mod(u, 1.0), np.mod(v, 1.0)
    else:
        u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
    return sample_at(a, u, v)


# -- Levels filter (DataActionFilterLevels / DataLevels) ---------------------------------
def apply_levels(x, lv):
    """Substance's Levels: input range -> gamma -> output range. `x` is HxW or
    HxWx3 (0..1); per-channel settings come as RGB triples (the mean is used
    for single-channel data)."""
    def get(k, d):
        v = lv.get(k, d)
        return np.asarray(v if isinstance(v, (list, tuple)) else [v] * 3, np.float32)[:3]
    lo, hi = get("inputMinimum", 0.0), get("inputMaximum", 1.0)
    g, olo, ohi = get("gamma", 1.0), get("outputMinimum", 0.0), get("outputMaximum", 1.0)
    if x.ndim == 2:
        lo, hi, g, olo, ohi = (float(a.mean()) for a in (lo, hi, g, olo, ohi))
    y = np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
    y = y ** (1.0 / np.maximum(g, 1e-3))
    return (olo + y * (ohi - olo)).astype(np.float32)


# -- UV-space patterns ----------------------------------------------------------------
def gradient_value(u, v, tw):
    """Substance's starter 'Gradient Linear 1' at texture coordinates (u, v),
    v up. Measured in Substance 12.2 to within 8-bit rounding:
      rotation (quarter turns): 0 -> v, 1 -> u, 2 -> 1-v, 3 -> 1-u
      Tiling T: frac((t - 0.5) * T + 0.5)
      Position p: p < .5 -> clip((t - (1-2p)) / 2p); p > .5 -> clip(t / (2(1-p)))
      Contrast c (after Position): clip((t - .5) / (1-c) + .5)
      invert: 1 - t"""
    rot = int(tw.get("rotation", 0) or 0) % 4
    t = [v, u, 1.0 - v, 1.0 - u][rot]
    tiles = max(int(tw.get("Tiling", tw.get("tiling", 1)) or 1), 1)
    if tiles != 1:
        t = np.mod((t - 0.5) * tiles + 0.5, 1.0)
    pos = float(tw.get("Position", tw.get("position", 0.5)))
    if pos < 0.5 - 1e-6:
        t = np.clip((t - (1.0 - 2.0 * pos)) / max(2.0 * pos, 1e-6), 0.0, 1.0)
    elif pos > 0.5 + 1e-6:
        t = np.clip(t / max(2.0 * (1.0 - pos), 1e-6), 0.0, 1.0)
    con = float(tw.get("Contrast", tw.get("contrast", 0.0)))
    if con > 0:
        t = np.clip((t - 0.5) / max(1.0 - con, 1e-4) + 0.5, 0.0, 1.0)
    if tw.get("invert") in (True, 1):
        t = 1.0 - t
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def gradient_linear(tw, res):
    """'Gradient Linear 1' as a UV-projected res x res texture (row 0 = top)."""
    t = (np.arange(res, dtype=np.float32) + 0.5) / res
    u = np.repeat(t[None, :], res, 0)
    v = np.repeat((1.0 - t)[:, None], res, 1)
    return gradient_value(u, v, tw)


# -- 3D projections ---------------------------------------------------------------
# Projection ids as stored in DataActionFill.projection, measured in Substance
# 12.2 (one fill per sp.layerstack.ProjectionMode, read back from the .spp).
PROJ_UV, PROJ_PLANAR, PROJ_TRIPLANAR, PROJ_SPHERICAL, PROJ_CYLINDRICAL, PROJ_FILL, PROJ_WARP = 0, 1, 2, 3, 4, 6, 7
PROJECTION_NAMES = {0: "UV", 1: "planar", 2: "tri-planar", 3: "spherical", 4: "cylindrical", 6: "fill", 7: "warp"}
UV_LIKE_PROJECTIONS = (PROJ_UV, PROJ_FILL)


def scene_frame(V):
    """Substance's normalised scene space: centred on the bounding box and
    scaled by half its largest extent (measured: projection size 1 spans the
    whole calibration scene)."""
    bmin, bmax = np.min(V, 0), np.max(V, 0)
    return (bmin + bmax) / 2.0, max(float(np.max(bmax - bmin)) / 2.0, 1e-9)


def euler_matrix(e):
    """Rotation for dddEulerAngles (radians). Single-axis rotations are
    measured; the order for combined rotations (Z*Y*X) is a guess."""
    ax, ay, az = (float(a) for a in (list(e or []) + [0, 0, 0])[:3])
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def warp_grid_is_flat(act):
    """True when a warp projection's grid has not been deformed (it is then a
    planar projection)."""
    g = act.get("dddGrid") or {}
    for vtx in g.get("vertices") or []:
        p, uv = vtx.get("position3D") or [0, 0, 0], vtx.get("positionUV") or [0, 0]
        if abs(p[0] - (uv[0] * 2 - 1)) > 1e-4 or abs(p[1] - (uv[1] * 2 - 1)) > 1e-4 or abs(p[2]) > 1e-4:
            return False
    return True


def planar_coords(pos, act, frame):
    """Projection coordinates of 3D points for a planar (or flat warp) fill.

    Measured: local = R^T (p_norm - dddPosition) / dddSize, and the texture
    coordinates are (local.xy + 1) / 2 (v up). Returns (u, v, depth) where
    depth = local z (larger is nearer the projector)."""
    center, half = frame
    off = np.asarray((list(act.get("dddPosition") or []) + [0, 0, 0])[:3], np.float64)
    size = np.asarray((list(act.get("dddSize") or []) + [1, 1, 1])[:3], np.float64)
    size[np.abs(size) < 1e-9] = 1e-9
    q = ((np.asarray(pos, np.float64) - center) / half - off) @ euler_matrix(act.get("dddEulerAngles"))
    q = q / size
    return (q[..., 0] + 1.0) / 2.0, (q[..., 1] + 1.0) / 2.0, q[..., 2]


def projector_depth(V, T, act, frame, cells=256, far=-1.0):
    """Depth buffer as seen by a planar/warp projector: for each cell of the
    projection square, the largest local z (nearest the projector) of any
    surface in front of `far`. Triangles are point-sampled densely enough
    for the cell size. Returns (zbuf, cells); cells nobody covers are -inf."""
    u, v, z = planar_coords(V, act, frame)
    tu, tv, tz = u[T], v[T], z[T]                         # (n, 3)
    keep = ((tu.max(1) >= 0) & (tu.min(1) <= 1) & (tv.max(1) >= 0) & (tv.min(1) <= 1) & (tz.max(1) >= far))
    tu, tv, tz = tu[keep], tv[keep], tz[keep]
    zbuf = np.full((cells, cells), -np.inf)
    if not len(tu):
        return zbuf, cells
    ext = np.maximum(tu.max(1) - tu.min(1), tv.max(1) - tv.min(1)) * cells
    steps = np.clip(np.ceil(ext * 1.5), 1, 64).astype(int)
    for n in np.unique(steps):
        sel = steps == n
        a = np.arange(n + 1)
        i, j = np.meshgrid(a, a)
        m = (i + j) <= n
        w1, w2 = (i[m] / n), (j[m] / n)
        w0 = 1 - w1 - w2
        pu = tu[sel] @ np.stack([w0, w1, w2])              # (k, samples)
        pv = tv[sel] @ np.stack([w0, w1, w2])
        pz = tz[sel] @ np.stack([w0, w1, w2])
        ok = (pu >= 0) & (pu < 1) & (pv >= 0) & (pv < 1)
        ix = (pu[ok] * cells).astype(np.int64)
        iy = (pv[ok] * cells).astype(np.int64)
        np.maximum.at(zbuf, (iy, ix), pz[ok])
    return zbuf, cells


DEPTH_CULL_TOLERANCE = 0.08   # local z units; Substance's hardness 0.75 edge is not modelled further


def wrap_coords(u, v, wrap):
    """Apply uvSamplingWrap (0 none, 1 horizontal, 2 vertical, 3 both).
    Returns (u, v, inside) where inside is False where nothing is sampled."""
    wrap = int(wrap)
    inside = np.ones(np.shape(u), bool)
    if wrap in (1, 3):
        u = np.mod(u, 1.0)
    else:
        inside &= (u >= 0) & (u <= 1)
    if wrap in (2, 3):
        v = np.mod(v, 1.0)
    else:
        inside &= (v >= 0) & (v <= 1)
    return u, v, inside


# -- generators (approximate) ----------------------------------------------------------
def _gray(m):
    m = np.asarray(m, np.float32)
    if m.dtype == np.uint8:
        m = m / 255.0
    return m.mean(-1) if m.ndim == 3 else m


def _levels(x, tw):
    lvl = tw.get("level", tw.get("levels", tw.get("balance")))
    con = tw.get("contrast")
    if isinstance(lvl, (int, float)):
        x = np.clip(x + (float(lvl) - 0.5), 0, 1)
    if isinstance(con, (int, float)):
        c = float(con)
        x = np.clip((x - 0.5) * (1 + 4 * c) + 0.5, 0, 1)
    if tw.get("invert") in (True, 1):
        x = 1 - x
    return x


def generator(name, tw, baked):
    """Approximate a Substance generator from baked maps.

    `baked(usage)` returns a float image for 'curvature', 'ao', 'position',
    'thickness' or 'world_normals' (or None). Returns (mask, label) or
    (None, reason)."""
    n = name.lower()
    curv = lambda: baked("curvature")  # noqa: E731
    if "edge" in n or "wear" in n:
        c = curv()
        if c is None:
            return None, "needs a baked curvature map"
        x = np.clip((_gray(c) - 0.5) * 4.0, 0, 1)
        return _levels(x, tw), "edge wear (from curvature)"
    if "dirt" in n or "cavity" in n or "grime" in n:
        ao, c = baked("ao"), curv()
        if ao is None and c is None:
            return None, "needs baked AO or curvature"
        x = np.ones_like(_gray(ao if ao is not None else c))
        if ao is not None:
            x *= 1 - _gray(ao)
        if c is not None:
            x *= np.clip((0.5 - _gray(c)) * 4.0, 0, 1) * 0.5 + 0.5
        return _levels(np.clip(x * 1.5, 0, 1), tw), "dirt (from AO/curvature)"
    if "curvature" in n:
        c = curv()
        return (None, "needs a baked curvature map") if c is None else (_levels(_gray(c), tw), "curvature")
    if "occlusion" in n or n in ("ao", "ambient_occlusion"):
        ao = baked("ao")
        return (None, "needs a baked AO map") if ao is None else (_levels(1 - _gray(ao), tw), "ambient occlusion")
    if "thickness" in n:
        t = baked("thickness")
        return (None, "needs a baked thickness map") if t is None else (_levels(_gray(t), tw), "thickness")
    if "position" in n or "gradient" in n:
        p = baked("position")
        if p is None:
            return None, "needs a baked position map"
        p = np.asarray(p, np.float32)
        if p.max() > 1.5:
            p = p / 255.0
        g = p[..., 1] if p.ndim == 3 else p
        return _levels(g, tw), "position gradient (vertical)"
    if "light" in n:
        wn = baked("world_normals")
        if wn is None:
            return None, "needs a baked world-space normal map"
        wn = np.asarray(wn, np.float32)
        if wn.max() > 1.5:
            wn = wn / 255.0
        y = wn[..., 1] if wn.ndim == 3 else wn
        return _levels(np.clip(y, 0, 1), tw), "light (from world normals, light from above)"
    return None, "unknown generator"
