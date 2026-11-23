"""Substance's brush stamp model, measured with the calibration project
(hand-painted brush tests in Substance 3D Painter 12.2, 2026-11-23).

Measured facts (Substance 3D Painter 12.2, round 'shape' alpha):

* Falloff of one stamp at distance d from its centre, radius R and alpha
  hardness h:  f = 1 - smoothstep(h, 1, d / R).  (Hardness 0 and 0.5 dabs
  match to 0.01 over the whole profile.)
* Stamps are laid along the stroke every  spacing x 2R  (spacing is a
  fraction of the brush DIAMETER) and stack like layers: within one stroke
  alpha = 1 - prod(1 - flow * f_i).  A flow 0.1 / spacing 0.2 / hardness 0.5
  stroke reaches 0.326 on its centre line; this model gives 0.325, and its
  cross-sections match to 0.004 RMS in both the 3D and the UV view.
* Stroke opacity multiplies the stroke's alpha; separate strokes composite
  'over' each other (two 50 % strokes overlap to 0.745).
* Every stamp is a sphere around a point on the surface, also in the 2D UV
  view (which only picks the surface point under the cursor): on a region
  whose UV is stretched 2:1 a UV-view dab comes out as a 1:2 ellipse in UV
  and round in 3D. Brush size spaces (brushSizeSpace) set the sphere's radius:
    - 1 Object:   R = size / 2 in normalised scene units (both views).
    - 2 Texture:  R = size / 2 / rho, rho = sqrt(UV area / scene area) of the
      triangle under the stamp, so the dab covers the UV area of a circle of
      radius size / 2 (43 x 86 texels on the 2:1 region, 61.5 x 61.5 on the
      square one; predicted 43 x 87 and 61.4).
    - 0 Viewport: R = size in NDC units in the 3D view (the viewport is 2
      high); in the UV view it behaved like Texture.

So that long strokes stay fast, the stamps of a straight segment are
integrated instead of summed one by one: -ln(transmittance) of a texel is
the sum over segments of  (G(p, t) - G(p, t - L)) / (2 spacing)  in units of
R, where G is a precomputed cumulative integral of the stamp's
-ln(1 - flow f). Single clicks are one exact stamp, and each stroke end adds
half a stamp, so the stamp count matches Substance's on average.
"""

import numpy as np

G_CAP = 16.0            # -ln(1 - a) of one stamp is capped (a = 1 - 1e-7)
_NP, _NX, _FINE = 129, 257, 8
_tables = {}

SPACE_VIEWPORT, SPACE_OBJECT, SPACE_TEXTURE = 0, 1, 2


def falloff(q, h):
    """Stamp alpha at normalised distance q = d / R for hardness h."""
    q = np.asarray(q, np.float64)
    if h >= 1.0:
        return (q <= 1.0).astype(np.float64)
    t = np.clip((q - h) / (1.0 - h), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def stamp_logt(q, flow, h):
    """-ln(1 - flow * falloff): one stamp's contribution to a stroke."""
    a = np.clip(flow, 0.0, 1.0) * falloff(q, h)
    return np.minimum(-np.log(np.maximum(1.0 - a, 1e-12)), G_CAP)


def edge_hardness(h, r_px):
    """Hardness with at least about one texel of anti-aliased edge."""
    if r_px <= 0:
        return min(h, 0.999)
    return float(min(h, max(0.0, 1.0 - 1.0 / r_px), 0.999))


class Table:
    """Cumulative stamp integral G(p, x) for p in [0, 1], x in [-1, 1]."""

    def __init__(self, flow, h):
        self.flow, self.h = flow, h
        p = np.linspace(0.0, 1.0, _NP)
        xf = np.linspace(-1.0, 1.0, (_NX - 1) * _FINE + 1)
        g = stamp_logt(np.hypot(p[:, None], xf[None, :]), flow, h)
        dx = xf[1] - xf[0]
        cum = np.concatenate([np.zeros((_NP, 1)), np.cumsum((g[:, 1:] + g[:, :-1]) * 0.5 * dx, axis=1)], 1)
        self.G = np.ascontiguousarray(cum[:, ::_FINE], np.float32)          # (NP, NX)
        self.g = np.ascontiguousarray(stamp_logt(np.linspace(0, 1, _NX), flow, h), np.float32)

    def G_at(self, p, x):
        fp = np.clip(np.asarray(p, np.float32), 0.0, 1.0) * (_NP - 1)
        fx = (np.clip(np.asarray(x, np.float32), -1.0, 1.0) + 1.0) * 0.5 * (_NX - 1)
        i0 = np.minimum(fp.astype(np.int32), _NP - 2)
        j0 = np.minimum(fx.astype(np.int32), _NX - 2)
        a, b = fp - i0, fx - j0
        G = self.G
        return ((G[i0, j0] * (1 - b) + G[i0, j0 + 1] * b) * (1 - a)
                + (G[i0 + 1, j0] * (1 - b) + G[i0 + 1, j0 + 1] * b) * a)

    def stamp(self, q):
        fq = np.clip(np.asarray(q, np.float32), 0.0, 1.0) * (_NX - 1)
        i0 = np.minimum(fq.astype(np.int32), _NX - 2)
        b = fq - i0
        v = self.g[i0] * (1 - b) + self.g[i0 + 1] * b
        return np.where(np.asarray(q) >= 1.0, 0.0, v)


def table(flow, h):
    key = (round(float(flow), 3), round(float(h), 3))
    t = _tables.get(key)
    if t is None:
        if len(_tables) > 64:
            _tables.clear()
        t = _tables[key] = Table(*key)
    return t


def segment_logt(tab, p, t, L, r, spacing):
    """-ln(transmittance) from the stamps of a straight segment of length L.
    p: distance to the segment's line, t: position along it from its start,
    r: brush radius (all the same unit)."""
    r = np.maximum(r, 1e-12)
    inv = 1.0 / (2.0 * max(float(spacing), 1e-3))
    pr = p / r
    return (tab.G_at(pr, t / r) - tab.G_at(pr, (t - L) / r)) * inv


def coverage(logt):
    return 1.0 - np.exp(-np.asarray(logt, np.float64))


def uv_density(geo, tris):
    """sqrt(UV area / scene area) of a set of triangles (normalised scene units)."""
    T = geo.T[tris]
    V = geo.V[T]
    UV = geo.UV[T]
    aw = 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1).sum()
    e1, e2 = UV[:, 1] - UV[:, 0], UV[:, 2] - UV[:, 0]
    au = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    return float(np.sqrt(au / aw)) if aw > 0 and au > 0 else 1.0


def tri_density(geo):
    """Per-triangle sqrt(UV area / scene area) for the whole mesh (cached on geo)."""
    d = getattr(geo, "_tri_density", None)
    if d is None:
        V = geo.V[geo.T]
        UV = geo.UV[geo.T]
        aw = 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1)
        e1, e2 = UV[:, 1] - UV[:, 0], UV[:, 2] - UV[:, 0]
        au = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        d = np.sqrt(au / np.maximum(aw, 1e-20))
        d = np.where((aw > 0) & (au > 0), d, np.nan)
        geo._tri_density = d
    return d


def brush_params(action):
    """(size, space, flow, spacing, hardness) of a paint/eraser action."""
    from .masks import as_list
    b = action.get("brush") or {}
    em = b.get("stampsEmitter") or {}
    hard = 1.0
    for src in as_list(action.get("sourceTransparent")):
        for tw in ((src.get("procedural") or {}).get("tweaks") or []):
            if tw.get("identifier") == "hardness":
                hard = float(tw.get("value", 1.0))
    flow = float(b.get("flow", 1.0))
    return (float(b.get("brushSize", 0.01)), int(b.get("brushSizeSpace", SPACE_OBJECT)),
            flow, float(em.get("spacing", 0.15)), hard)
