"""Rebuild Substance layer masks (and paint coverage) as UV-space images.

Substance never stores mask pixels; it stores the actions that made them:
polygon-fill picks, rectangle picks, brush/path strokes, fills, erasers and
procedural filters. This module replays the ones we can reproduce:

  * polygon fill (triangle / polygon / mesh / UV-chunk), point and rectangle,
    including symmetry duplicates          -> exact
  * uniform fills                          -> exact
  * geometry masks (enable per mesh part)  -> exact
  * brush, path and eraser strokes         -> close approximation
  * procedural filters / generators        -> skipped, reported
"""

import numpy as np
from PIL import Image, ImageFilter

from . import brushmodel as BM
from . import procedural as P
from . import strokes2d as S2
from .strokes2d import stroke_pressure as _stroke_pressure

try:  # multi-threaded per-pixel maths (identical results, several times faster)
    import numexpr as _ne
except ImportError:  # pragma: no cover
    _ne = None

BLEND_NORMAL = 25        # 25 is really Passthrough; for single actions it acts as Normal (2)
BLEND_PASSTHROUGH = 25   # folders with no blending entry default to it as well
BLEND_REPLACE = 1
# Isolated mask folders start from opaque black (see _eval_action).
MASK_GROUP_BLACK = True
UV_SAME_ISLAND = True            # UV-view stamps only reach the UV island under the cursor
UV_MIRROR = True                 # apply stroke symmetry to UV-view strokes (via the 3D surface)
# Polygon-fill picks and brush strokes only see the meshes of the layer's own
# texture set: other texture sets never block them. Measured against
# Substance's per-layer export of a test project: the fur cards (texture set 'fluff') sit
# in front of the chest and belly, and with them blocking, White's clicks and
# strokes never reached the body (White off on 5.07% of texels, 2.46% without).
OWN_SET_ONLY = True
# Depth culling of projected fills (hide what the projector can't see) was
# modelled with a z-buffer, inferred from a test project when its projection frame was
# still wrong. Against Substance's per-layer export, a test project's Shading gradient is
# off on 1.82% of texels with that model and 0.17% without it (the inner
# thighs get the gradient although the belly is in front of them), so it is
# off. The depth slab (|local z| <= depth) is what keeps the back clean.
PROJECTOR_DEPTH_CULL = False
PROJECTOR_OWN_SET_ONLY = False   # (only with PROJECTOR_DEPTH_CULL) only the own texture set occludes
UV_BRUSH_RADIUS_PER_SIZE = 0.5   # stamp sphere radius = brushSize / 2 (scene units; / local UV density in Texture space)
BLEND_MULTIPLY = 4      # ids as stored in .spp files (see layers.BLEND_NAMES)
BLEND_DISABLE = 0
# Other blend modes used in mask stacks (ids = Substance's BlendingMode order)
MASK_BLENDS = {
    3: np.minimum,                                              # Darken
    7: np.maximum,                                              # Lighten
    8: lambda d, s: 1.0 - (1.0 - d) * (1.0 - s),                # Screen
    10: lambda d, s: np.minimum(d + s, 1.0),                    # Linear Dodge (Add)
    17: lambda d, s: np.abs(d - s),                             # Difference
    29: lambda d, s: np.maximum(d - s, 0.0),                    # Subtract
    30: lambda d, s: np.maximum(s - d, 0.0),                    # Inverse Subtract
    6: lambda d, s: np.maximum(d + s - 1.0, 0.0),               # Linear Burn
    11: lambda d, s: np.where(d < 0.5, 2 * d * s, 1 - 2 * (1 - d) * (1 - s)),   # Overlay
}

# Substance polygon-fill granularity -> Geometry.select code
# (Geometry: 0 triangle, 1 polygon, 2 mesh island, 3 UV island).
GRANULARITY = {0: 0, 1: 1, 2: 2, 3: 3}

# Substance stores brush size as a fraction of the mesh bounding-box diagonal.
# memory for reusing identical brush work (colour-variant copies of layers)
STROKE_CACHE_BYTES = 1536 * 2 ** 20


def _mat(m16):
    """Substance matrices are column-major 4x4."""
    return np.asarray(m16, np.float64).reshape(4, 4).T


def _unproject_ray(inv_pv, x, y):
    def un(z):
        q = inv_pv @ np.array([2 * x - 1, 2 * y - 1, z, 1.0])
        return q[:3] / q[3] if abs(q[3]) > 1e-12 else None
    a, b = un(-1.0), un(1.0)
    if a is None or b is None:
        return None
    d = b - a
    n = np.linalg.norm(d)
    if n == 0:
        return None
    return a, d / n


def _bezier_points(keys, step):
    """Sample a Substance path: anchors are type 0/1, handles 3 (out) and 2 (in)."""
    pts = [(np.asarray(k["position"], np.float64), k["type"], k.get("pressure", 1.0),
            np.asarray(k.get("normal") or [0, 0, 0], np.float64)) for k in keys]
    anchors = [i for i, p in enumerate(pts) if p[1] in (0, 1)]
    out = []
    for a, b in zip(anchors, anchors[1:]):
        p0, p3 = pts[a][0], pts[b][0]
        h = [pts[i] for i in range(a + 1, b)]
        p1 = next((q[0] for q in h if q[1] == 3), p0)
        p2 = next((q[0] for q in h if q[1] == 2), p3)
        length = (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2))
        n = max(2, int(np.ceil(length / max(step, 1e-9))) + 1)
        t = np.linspace(0, 1, n)[:, None]
        seg = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3
        pr = np.linspace(pts[a][2] or 1.0, pts[b][2] or 1.0, n)
        na, nb = pts[a][3], pts[b][3]
        nrm = (1 - t) * na + t * nb
        out.extend(zip(seg, pr, nrm))
    if not out and pts:
        out = [(pts[0][0], 1.0, pts[0][3])]
    return out



def as_list(x):
    """Source fields are lists in most files, but Substance 12 stores a single
    source (e.g. a brush's sourceTransparent) as one object."""
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    return list(x)

class RegionCover:
    """Brush coverage that only touches part of the texture: a sub-image at
    (y0, x0). Keeps per-stroke work proportional to what the brush touched,
    which matters at 4K with thousands of brush actions."""

    __slots__ = ("y0", "x0", "data", "res")

    def __init__(self, y0, x0, data, res):
        self.y0, self.x0, self.data, self.res = int(y0), int(x0), data, int(res)

    def slices(self):
        h, w = self.data.shape
        return (slice(self.y0, self.y0 + h), slice(self.x0, self.x0 + w))

    def full(self):
        out = np.zeros((self.res, self.res), np.float32)
        out[self.slices()] = self.data
        return out

    @staticmethod
    def from_full(a):
        ys, xs = np.nonzero(a > 0)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return RegionCover(y0, x0, np.ascontiguousarray(a[y0:y1, x0:x1]), a.shape[0])


def as_full(cover):
    if cover is None or not isinstance(cover, RegionCover):
        return cover
    return cover.full()


class Buffer:
    """A grayscale layer with coverage: value v and alpha a in [0, 1]."""

    def __init__(self, res, value=0.0, alpha=0.0):
        self.v = np.full((res, res), value, np.float32)
        self.a = np.full((res, res), alpha, np.float32)

    def paint(self, cover, value, opacity=1.0):
        """Normal-blend a constant value through a coverage map (full image
        or RegionCover)."""
        if isinstance(cover, RegionCover):
            sl = cover.slices()
            v, a, c = self.v[sl], self.a[sl], cover.data
        else:
            v, a, c = self.v, self.a, cover
        val, op = np.float32(value), np.float32(opacity)
        if _ne is not None:
            _ne.evaluate("v + (val - v) * (c * op)", out=v, casting="same_kind")
            _ne.evaluate("a + (1 - a) * (c * op)", out=a, casting="same_kind")
            return
        k = c * op
        v += (val - v) * k
        a += (1.0 - a) * k

    def erase(self, cover, opacity=1.0):
        if isinstance(cover, RegionCover):
            a, c = self.a[cover.slices()], cover.data
        else:
            a, c = self.a, cover
        op = np.float32(opacity)
        if _ne is not None:
            _ne.evaluate("a * (1 - c * op)", out=a, casting="same_kind")
            return
        a *= 1.0 - c * op

    def blend_onto(self, dst, mode, opacity):
        if mode == BLEND_DISABLE:
            return
        sv, sa, dv, da, op = self.v, self.a, dst.v, dst.a, np.float32(opacity)
        other = MASK_BLENDS.get(mode)
        if other is not None:   # Lighten, Darken, Screen, Add, Subtract, ... (ids measured in Substance)
            k = sa * op
            dv += (other(dv, sv) - dv) * k
            da += (1.0 - da) * k
            return
        if _ne is not None:
            if mode == BLEND_MULTIPLY:
                _ne.evaluate("dv * (1 + (sv - 1) * (sa * op))", out=dv, casting="same_kind")
            else:
                _ne.evaluate("dv + (sv - dv) * (sa * op)", out=dv, casting="same_kind")
            _ne.evaluate("da + (1 - da) * (sa * op)", out=da, casting="same_kind")
            return
        k = sa * op
        if mode == BLEND_MULTIPLY:
            dv *= 1.0 + (sv - 1.0) * k
        else:  # Normal, Replace and everything else
            dv += (sv - dv) * k
        da += (1.0 - da) * k


class MaskBuilder:
    def __init__(self, geo, material_index, res, warn):
        self.geo = geo
        self.mat = material_index
        self.res = res
        self.warn = warn
        self.tris = np.nonzero(geo.material == material_index)[0]
        self._in_mat = np.zeros(len(geo.T), bool)
        self._in_mat[self.tris] = True
        self._sel_cache = {}
        self._posmap = None
        self._frame = None       # Substance's normalised scene frame (centre, half extent)
        self._coverage = None
        # Mesh parts each nested layer is limited to (geometry masks). Picks
        # pass through parts a layer is masked off from, like in Substance.
        self._allowed = [self._in_mat.copy() if OWN_SET_ONLY else None]
        # Set by the caller: usage -> float image at self.res (baked maps),
        # and url -> image array (embedded resources).
        self.baked_map = lambda usage: None
        self.resource_image = None
        self._screen = None  # ScreenStrokePainter / GPUStrokePainter, created on first 2D stroke
        self.engine = "cpu"   # "cpu" or "gpu" (set by the converter)
        self.lossless = False # CPU: full resolution everywhere

    def _parts_enabled(self, layer):
        if not layer.get("enabledGeometryMask", True) or int(layer.get("geometryMaskType", 0)) != 1:
            return None
        listed = {e.get("value") for e in layer.get("enabledMeshList") or []}
        default_on = layer.get("enabledMeshDefault", True)
        keep = np.zeros(len(self.geo.T), bool)
        for pi, p in enumerate(self.geo.scene.parts):
            if (p.name not in listed) if default_on else (p.name in listed):
                keep |= self.geo.part == pi
        return keep

    def enter(self, layer):
        """Call before converting a layer (and its children); pair with leave()."""
        cur = self._allowed[-1]
        own = self._parts_enabled(layer)
        if own is None:
            self._allowed.append(cur)
        else:
            self._allowed.append(own if cur is None else (own & cur))

    def leave(self):
        self._allowed.pop()

    @property
    def allowed(self):
        return self._allowed[-1]

    # -- helpers -----------------------------------------------------------
    def coverage(self):
        """Texels covered by this texture set's UV islands."""
        if self._coverage is None:
            img = self.geo.rasterize(self.tris, self.res)
            self._coverage = np.asarray(img, np.float32) / 255.0
        return self._coverage

    def _raster(self, tris):
        """Coverage of a set of triangles, as the part of the texture they touch
        (the same texels as a full image; a selection used to be a whole
        res x res float image, to build, to keep and to blend)."""
        tris = tris[self._in_mat[tris]]
        if len(tris) == 0:
            return None
        img = self.geo.rasterize(tris, self.res)
        box = img.getbbox()
        if box is None:
            return None
        return RegionCover(box[1], box[0], np.asarray(img.crop(box), np.float32) / 255.0, self.res)

    def _selection_cover(self, tri, gran):
        g = self.geo
        if gran == 1:
            key = ("p", int(g.poly[tri]))
        elif gran == 2:
            key = ("m", int(g.mesh_island[tri]))
        elif gran == 3:
            key = ("u", int(g.uv_island[tri]))
        else:
            key = ("t", int(tri))
        if key not in self._sel_cache:
            self._sel_cache[key] = self._raster(g.select(tri, gran))
        return self._sel_cache[key]

    def _pick(self, inv_pv, x, y, dup=None):
        r = _unproject_ray(inv_pv, x, y)
        if r is None:
            return None
        o, d = r
        if dup is not None:
            o = (dup @ np.append(o, 1.0))[:3]
            d = dup[:3, :3] @ d
        return self.geo.raycast(o, d, self.allowed)

    def _hit_matrices(self, hit):
        mats = [None]
        if hit.get("duplicateCount", 0) >= 1:
            mats.append(_mat(hit["duplicateMatrix"]))
        return mats

    # -- polygon fill --------------------------------------------------------
    def _apply_decal(self, action, buf):
        for hit in action.get("hits") or []:
            cls = hit.get("__class")
            value = float((hit.get("color") or [1, 1, 1])[0])
            gran = GRANULARITY.get(int(hit.get("granularity", 0)), 0)
            picks = []
            if int(hit.get("viewType", 1)) == 0:
                # picked in the 2D UV view: positions are UV coordinates
                if cls == "DataDecalHitPunctual":
                    picks = self._uv_point_pick(*hit["position"])
                elif cls == "DataDecalHitRect":
                    picks = self._uv_rect_picks(hit["position0"], hit["position1"])
                else:
                    self.warn("unsupported polygon fill hit %s" % cls)
                    continue
                if hit.get("duplicateCount", 0) >= 1 and len(picks):
                    picks = np.concatenate([picks, self.geo.mirror_tris(picks, _mat(hit["duplicateMatrix"]), self.tris)])
            else:
                inv_pv = _mat(hit["projViewInvMatrix"])
                if cls == "DataDecalHitPunctual":
                    x, y = hit["position"]
                    for dm in self._hit_matrices(hit):
                        t = self._pick(inv_pv, x, y, dm)
                        if t is not None:
                            picks.append(t)
                elif cls == "DataDecalHitRect":
                    picks.extend(self._rect_picks(hit, inv_pv))
                else:
                    self.warn("unsupported polygon fill hit %s" % cls)
                    continue
            picks = np.unique(np.asarray(picks, np.int64))
            picks = picks[self._in_mat[picks]] if len(picks) else picks
            if len(picks) == 0:
                continue
            if gran == 0 or len(picks) > 64:
                sel = picks if gran == 0 else np.unique(np.concatenate(
                    [self.geo.select(t, gran) for t in self._unique_groups(picks, gran)]))
                cov = self._raster(sel)
                if cov is not None:
                    buf.paint(cov, value)
            else:
                for t in self._unique_groups(picks, gran):
                    cov = self._selection_cover(t, gran)
                    if cov is not None:
                        buf.paint(cov, value)

    def _unique_groups(self, picks, gran):
        """One representative triangle per selected polygon / island."""
        g = self.geo
        key = {1: g.poly, 2: g.mesh_island, 3: g.uv_island}.get(gran)
        if key is None:
            return picks
        _, first = np.unique(key[picks], return_index=True)
        return picks[first]

    def _uv_point_pick(self, x, y):
        uv = self.geo.UV[self.geo.T[self.tris]]
        a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
        p = np.array([x, y])
        v0, v1, v2 = b - a, c - a, p - a
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        ok = np.abs(den) > 1e-18
        den[~ok] = 1
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        inside = ok & (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        return self.tris[inside][:1]

    def _uv_rect_picks(self, p0, p1):
        """Triangles overlapping a rectangle drawn in the UV view."""
        (x0, y0), (x1, y1) = p0, p1
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        c = self.__dict__.get("_uv_boxes")
        if c is None:   # the triangles' UVs and their bounding boxes, once per texture set
            uv_all = self.geo.UV[self.geo.T[self.tris]]
            c = self._uv_boxes = (uv_all, uv_all[..., 0].min(1), uv_all[..., 0].max(1),
                                  uv_all[..., 1].min(1), uv_all[..., 1].max(1))
        uv_all, u_min, u_max, v_min, v_max = c
        # only triangles whose box meets the rectangle can overlap it; the exact
        # test below is the same per triangle as it was over all of them
        cand = np.nonzero((u_max >= lo_x) & (u_min <= hi_x) & (v_max >= lo_y) & (v_min <= hi_y))[0]
        if len(cand) == 0:
            return self.tris[cand]
        uv = uv_all[cand]
        ok = np.ones(len(cand), bool)
        corners = np.array([[lo_x, lo_y], [hi_x, lo_y], [hi_x, hi_y], [lo_x, hi_y]])
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = uv[:, b] - uv[:, a]
            n = np.stack([-e[:, 1], e[:, 0]], 1)
            side = ((uv[:, 3 - a - b] - uv[:, a]) * n).sum(1)
            pc = ((corners[None] - uv[:, a][:, None]) * n[:, None]).sum(-1)
            separated = np.where(side[:, None] >= 0, pc < 0, pc > 0).all(1)
            ok &= ~separated
        return self.tris[cand[ok]]

    def _rect_picks(self, hit, inv_pv):
        (x0, y0), (x1, y1) = hit["position0"], hit["position1"]
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        picks = []
        try:
            pv0 = np.linalg.inv(inv_pv)
        except np.linalg.LinAlgError:
            pv0 = np.linalg.pinv(inv_pv)
        for dm in self._hit_matrices(hit):
            pv = pv0
            if dm is not None:
                pv = pv @ np.linalg.inv(dm)
            C = np.c_[self.geo.centroid[self.tris], np.ones(len(self.tris))] @ pv.T
            w = C[:, 3]
            ok = w > 1e-9
            sx = (C[:, 0] / np.where(ok, w, 1) + 1) / 2
            sy = (C[:, 1] / np.where(ok, w, 1) + 1) / 2
            inside = ok & (sx >= lo_x) & (sx <= hi_x) & (sy >= lo_y) & (sy <= hi_y)
            cand = self.tris[inside]
            # keep only triangles that are actually visible from the camera
            vis = []
            budget = 400
            inv_dm = inv_pv if dm is None else dm @ inv_pv
            q = inv_dm @ np.array([0, 0, -1.0, 1.0])
            cam = q[:3] / q[3] if abs(q[3]) > 1e-12 else None
            if cam is None:
                vis.extend(cand[:budget])
            else:
                dirs = []
                for t in cand[:budget]:
                    d = self.geo.centroid[t] - cam
                    d /= np.linalg.norm(d)
                    dirs.append(d)
                hits = self.geo.raycast_many(cam, dirs, self.allowed)
                vis.extend(t for t, h in zip(cand[:budget], hits) if h == t)
            if len(cand) > budget:
                vis.extend(cand[budget:].tolist())
            # A tiny drag acts like a click on the triangle under it. A drag
            # with zero width or height selects nothing in Substance.
            if not vis and hi_x > lo_x and hi_y > lo_y:
                t = self._pick(inv_pv, (x0 + x1) / 2, (y0 + y1) / 2, dm)
                if t is not None:
                    vis.append(t)
            picks.extend(vis)
        return picks

    # -- strokes ---------------------------------------------------------------
    def _positions(self):
        if self._posmap is None:
            pos, tid = self.geo.position_map(self.tris, self.res)
            ys, xs = np.nonzero(tid >= 0)
            self._posmap = (pos[ys, xs].astype(np.float64), ys, xs)
        return self._posmap

    def _stamp_cover(self, centers, radius, hardness, cull=None, flow=1.0, weight=1.0):
        """Coverage of brush stamps on the surface, in UV space, with
        Substance's stamp model (brushmodel.py): within one action's stamps,
        alpha = 1 - prod(1 - flow * falloff).

        Works triangle by triangle, so mesh parts that share (overlap) UV
        space each get the paint that landed on them. `weight` scales each
        stamp (used when stamps are placed further apart than Substance's
        spacing, to keep the same total)."""
        cover = np.zeros((self.res, self.res), np.float32)
        if len(centers) == 0:
            return cover
        C = np.array([c[0] for c in centers], np.float64)
        R = radius * np.maximum(np.array([c[1] for c in centers], np.float64), 0.05)
        N = np.array([c[2] if len(c) > 2 and c[2] is not None else (0, 0, 0) for c in centers], np.float64)
        nl = np.linalg.norm(N, axis=1)
        N = np.where(nl[:, None] > 1e-9, N / np.maximum(nl[:, None], 1e-9), 0)
        # Substance's brush culling: skip surfaces facing away from the stroke
        cos_thr = None if cull is None else float(np.cos(np.clip(cull, 0, 1) * np.pi / 2))
        g = self.geo
        tris = self.tris
        if self.allowed is not None:
            tris = tris[self.allowed[tris]]
        r_px = float(np.median(R)) * BM.uv_density(g, self.tris) * self.res
        tab = BM.table(flow, BM.edge_hardness(hardness, r_px))
        V = g.V[g.T[tris]]                          # (n, 3, 3)
        FN = g.normal[tris]
        FN = FN / np.maximum(np.linalg.norm(FN, axis=1, keepdims=True), 1e-12)
        cen = V.mean(1)
        trad = np.sqrt(((V - cen[:, None]) ** 2).sum(-1)).max(1)
        lo, hi = C.min(0) - R.max(), C.max(0) + R.max()
        near = np.all((cen + trad[:, None] >= lo) & (cen - trad[:, None] <= hi), axis=1)
        tris, V, cen, trad, FN = tris[near], V[near], cen[near], trad[near], FN[near]
        if len(tris) == 0:
            return cover
        # stamps per triangle: bounding-sphere overlap
        uvp = g.uv_pixels(tris, self.res)
        for k in range(len(tris)):
            dc = np.sqrt(((C - cen[k]) ** 2).sum(1))
            sel = dc < R + trad[k]
            if cos_thr is not None:
                facing = N @ FN[k]
                sel &= (nl < 1e-9) | (facing >= cos_thr)
            if not sel.any():
                continue
            c_k, r_k = C[sel], R[sel]
            t = uvp[k]
            x0 = max(int(np.floor(t[:, 0].min())), 0)
            x1 = min(int(np.ceil(t[:, 0].max())), self.res - 1)
            y0 = max(int(np.floor(t[:, 1].min())), 0)
            y1 = min(int(np.ceil(t[:, 1].max())), self.res - 1)
            if x1 < x0 or y1 < y0:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)
            a, b, c = t[0], t[1], t[2]
            v0, v1, v2 = b - a, c - a, px - a
            d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
            den = d00 * d11 - d01 * d01
            if abs(den) < 1e-12:
                continue
            d20, d21 = v2 @ v0, v2 @ v1
            w1 = (d11 * d20 - d01 * d21) / den
            w2 = (d00 * d21 - d01 * d20) / den
            w0 = 1 - w1 - w2
            pad_ = 0.75 / max(np.sqrt(abs(den)) ** 0.5, 1e-6)
            inside = (w0 >= -pad_) & (w1 >= -pad_) & (w2 >= -pad_)
            if not inside.any():
                continue
            w = np.clip(np.stack([w0, w1, w2], 1)[inside], 0, 1)
            w /= w.sum(1, keepdims=True)
            pos = w @ V[k]
            dist = np.sqrt(((pos[:, None, :] - c_k[None]) ** 2).sum(-1))  # (texels, stamps)
            lt = (tab.stamp(dist / r_k[None]) * weight).sum(1)
            if not (lt > 0).any():
                continue
            f = BM.coverage(lt)
            ty = yy.ravel()[inside]
            tx = xx.ravel()[inside]
            cover[ty, tx] = np.maximum(cover[ty, tx], f.astype(np.float32))
        return cover

    def _stroke_cover(self, action):
        """Coverage of an action's strokes. Identical stroke work (common in
        projects with colour variants that copy layers) is computed once."""
        key = self._stroke_key(action)
        cache = self.__dict__.setdefault("_stroke_cache", {})
        if key in cache:
            cache[key] = cache.pop(key)          # most recently used last
            return cache[key]
        cov = self._stroke_cover_uncached(action)
        if cov is not None and not isinstance(cov, RegionCover):
            cov = RegionCover.from_full(cov)
        if cov is not None and cov.data.dtype != np.float32:
            cov.data = cov.data.astype(np.float32)
        budget = STROKE_CACHE_BYTES
        used = self.__dict__.get("_stroke_cache_bytes", 0) + (0 if cov is None else cov.data.nbytes)
        while cache and used > budget:
            old_key = next(iter(cache))
            old = cache.pop(old_key)
            used -= 0 if old is None else old.data.nbytes
        cache[key] = cov
        self._stroke_cache_bytes = used
        return cov

    def _stroke_key(self, action):
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        h.update(repr(sorted((k, v) for k, v in (action.get("brush") or {}).items()
                             if not isinstance(v, (dict, list)) and k != "uid")).encode())
        em = (action.get("brush") or {}).get("stampsEmitter") or {}
        h.update(repr(sorted((k, v) for k, v in em.items() if not isinstance(v, (dict, list)) and k != "uid")).encode())
        for src in as_list(action.get("sourceTransparent")):
            for tw in ((src.get("procedural") or {}).get("tweaks") or []):
                h.update(repr((tw.get("identifier"), tw.get("value"))).encode())
        for s in action.get("strokes") or []:
            h.update(np.asarray([p["position"] for p in s.get("points") or []], np.float32).tobytes())
            h.update(np.asarray(_stroke_pressure(s), np.float32).tobytes())
            h.update(np.asarray(s.get("viewMatrix") or [], np.float32).tobytes())
            h.update(np.asarray(s.get("projectionMatrix") or [], np.float32).tobytes())
            h.update(repr((s.get("duplicateCount"), s.get("duplicateMatrix"))).encode())
        for s in action.get("strokes3D") or []:
            h.update(np.asarray([p["position"] for p in s.get("points") or []], np.float32).tobytes())
            h.update(repr((s.get("duplicateCount"), [p.get("type") for p in s.get("points") or []])).encode())
        al = self.allowed
        h.update(b"all" if al is None else np.packbits(al).tobytes())
        return h.hexdigest()

    def _uv_density(self):
        if self.__dict__.get("_density") is None:
            self._density = BM.uv_density(self.geo, self.tris)
        return self._density

    def _stroke_cover_uncached(self, action):
        brush = action.get("brush") or {}
        size, space, flow, spacing, hardness = BM.brush_params(action)
        radius = size * S2.BRUSH_RADIUS_PER_SIZE          # Object space, scene units (see strokes2d)
        if space == BM.SPACE_TEXTURE:
            radius /= self._uv_density()
        # Substance lays stamps every spacing x diameter; place them at least
        # every 0.1 radius and weight them so the total stays the same.
        true_step = max(2 * radius * spacing, 1e-9)
        step = max(true_step, radius * 0.1)
        weight = true_step / step
        centers = []
        for s in action.get("strokes3D") or []:
            pts = _bezier_points(s.get("points") or [], step)
            mats = [None]
            if s.get("duplicateCount", 0) >= 1:
                mats.append(_mat(s["duplicateMatrix"]))
            for dm in mats:
                for p, pr, nrm in pts:
                    if dm is not None:
                        p = (dm @ np.append(p, 1.0))[:3]
                        nrm = dm[:3, :3] @ nrm
                    centers.append((p, 1.0, nrm))
        cull = float(brush.get("cullingAngle", 0.5)) if brush.get("cullingEnabled", True) else None
        cover = None
        if centers:
            cover = self._stamp_cover(centers, radius, hardness, cull, flow, weight)
        strokes_all = action.get("strokes") or []
        strokes_uv = [st for st in strokes_all if int(st.get("viewType", 1)) == 0]
        strokes2d = [st for st in strokes_all if int(st.get("viewType", 1)) != 0]
        if strokes_uv:
            c_uv = self._uv_stroke_cover(strokes_uv, brush, hardness, flow, spacing, space)
            if c_uv is not None:
                cover = c_uv if cover is None else np.maximum(as_full(cover), c_uv)
        if strokes2d:
            # Screen-space strokes: brush size is relative to the viewport.
            c2 = self._painter().cover(strokes2d, brush, hardness, self.allowed,
                                    cull=brush.get("cullingEnabled", True), flow=flow, spacing=spacing,
                                    space=space, density=self._uv_density())
            if c2 is not None:
                if cover is None:
                    cover = c2
                else:
                    cover = np.maximum(cover, as_full(c2))
        return cover

    def _painter(self):
        if self._screen is None:
            if self.engine == "gpu":
                from .gpu import GPUStrokePainter
                self._screen = GPUStrokePainter(self.geo, self.tris, self.res, self.warn)
            else:
                from .strokes2d import ScreenStrokePainter
                self._screen = ScreenStrokePainter(self.geo, self.tris, self.res, self.warn,
                                                   lossless=self.lossless)
        return self._screen

    def _uv_to_surface(self, uv):
        """Surface point, normal and triangle under UV points (NaN / -1 where
        a point is off the texture set's islands)."""
        if self.__dict__.get("_uvlut") is None:
            r = min(max(self.res, 1024), 2048)
            pos, tid = self.geo.position_map(self.tris, r)
            self._uvlut = (r, pos, tid)
        r, pos, tid = self._uvlut
        x = np.clip((uv[:, 0] * r).astype(int), 0, r - 1)
        y = np.clip(((1 - uv[:, 1]) * r).astype(int), 0, r - 1)
        t = tid[y, x]
        p = pos[y, x].astype(np.float64)
        ok = (t >= 0) & ~np.isnan(p[:, 0])
        n = np.zeros_like(p)
        if ok.any():
            fn = self.geo.normal[t[ok]]
            n[ok] = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        return p, n, np.where(ok, t, -1), ok

    def _uv_stroke_cover(self, strokes, brush, hardness, flow=1.0, spacing=0.15, space=1):
        """Strokes painted in the 2D UV view (viewType 0): positions are UV
        coordinates. Substance stamps a sphere around the surface point under
        each UV position, exactly like in the 3D view (measured: dabs are
        round in 3D and stretched in UV where the UV is stretched), so the
        points are lifted onto the surface and go through the 3D stamp
        engine. Radius by size space (brushmodel.py); Viewport behaves like
        Texture in the UV view."""
        size = float(brush.get("brushSize", 0.01))
        emitter = brush.get("stampsEmitter") or {}
        press_on = bool(emitter.get("pressureSizeEnabled", True))
        press_min = float(emitter.get("pressureSizeMinimum", 0.0))
        dens_all = BM.tri_density(self.geo)
        groups = {}     # UV island (or None) -> stamp segments
        for st in strokes:
            uv = np.array([k["position"] for k in st.get("points") or []], np.float64)
            if not len(uv):
                continue
            pr = _stroke_pressure(st)
            scale = np.ones(len(uv))
            if press_on and len(pr) == len(uv):
                scale = press_min + (1 - press_min) * np.clip(pr, 0, 1)
            p, n, t, ok = self._uv_to_surface(uv)
            if space == BM.SPACE_OBJECT:
                rad = size * UV_BRUSH_RADIUS_PER_SIZE * scale
            else:
                d = np.where(t >= 0, dens_all[np.maximum(t, 0)], np.nan)
                d = np.where(np.isfinite(d), d, self._uv_density())
                rad = size * UV_BRUSH_RADIUS_PER_SIZE * scale / np.maximum(d, 1e-9)
            # thin to a quarter of the radius (the stamp integral handles the rest)
            keep = np.arange(len(p))
            if len(p) > 2:
                pp = np.where(ok[:, None], p, 0)
                trav = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))]
                bucket = np.floor(trav / max(0.25 * float(np.median(rad)), 1e-9)).astype(np.int64)
                keep = np.r_[0, np.nonzero(np.diff(bucket) | np.diff(ok.astype(np.int64)))[0] + 1]
                if keep[-1] != len(p) - 1:
                    keep = np.r_[keep, len(p) - 1]
            p, n, ok, rad, t = p[keep], n[keep], ok[keep], rad[keep], t[keep]
            isl = np.where(t >= 0, self.geo.uv_island[np.maximum(t, 0)], -1)
            mats = [None]
            if st.get("duplicateCount", 0) >= 1 and UV_MIRROR:
                mats.append(_mat(st["duplicateMatrix"]))
            for dm in mats:
                pm, nm = p, n
                if dm is not None:
                    pm = (np.c_[p, np.ones(len(p))] @ dm.T)[:, :3]
                    nm = n @ dm[:3, :3].T
                # culling: a camera far out along each point's surface normal
                cams = np.where(ok[:, None], pm + nm * 1e3, 0)
                if UV_SAME_ISLAND and dm is None:
                    # one run per island: the stroke breaks where the cursor
                    # moves onto another island
                    for i_ in np.unique(isl[ok]):
                        sel = ok & (isl == i_)
                        groups.setdefault(int(i_), []).extend(
                            S2.stroke_segments(np.where(sel[:, None], pm, 0), rad, sel, cams, spacing))
                else:
                    groups.setdefault(None, []).extend(
                        S2.stroke_segments(np.where(ok[:, None], pm, 0), rad, ok, cams, spacing))
        if not groups:
            return None
        total = None
        for i_, segs in groups.items():
            allowed = self.allowed
            if i_ is not None:
                m = self.geo.uv_island == i_
                allowed = m if allowed is None else (allowed & m)
            cov = self._painter().cover_segments(segs, hardness, flow, spacing, allowed,
                                                 cull=brush.get("cullingEnabled", True))
            if cov is None:
                continue
            cov = as_full(cov)
            total = cov if total is None else 1 - (1 - total) * (1 - cov)
        return total

    # -- stacks ----------------------------------------------------------------
    @staticmethod
    def _blending(action):
        for b in action.get("perChannelBlending") or []:
            return int(b.get("blendingMode", BLEND_NORMAL)), float(b.get("opacity", 1.0))
        return BLEND_NORMAL, 1.0

    def eval_actions(self, items, base=None):
        """Evaluate a list of mask actions (bottom first) into a Buffer.

        One action failing (unexpected data) only skips that action."""
        buf = base or Buffer(self.res)
        for act in items or []:
            if not act or not act.get("enabled", True):
                continue
            try:
                self._eval_action(act, buf)
            except Exception as e:  # keep going: a broken action must not lose the mask
                self.warn("mask action %s could not be reproduced (%s)" % (act.get("__class"), e))
        return buf

    def _eval_action(self, act, buf):
        cls = act.get("__class")
        if cls == "DataActionGroup":
            items = (act.get("subStack") or {}).get("items")
            mode, op = self._blending(act)
            if mode == BLEND_PASSTHROUGH:
                # Pass-through folders (the default) are evaluated on their own
                # and laid over the mask below. Applying their effects straight
                # onto the mask below scored slightly worse on another test project (body
                # layers alone 99.80 vs 99.81), so this stays.
                sub = self.eval_actions(items)
                sub.blend_onto(buf, BLEND_NORMAL, op)
                return
            # A folder with its own blend mode is a greyscale mask of its own
            # that starts black (a test project's Shading mask paints white strokes into a
            # Multiply folder to keep only the painted part of a gradient).
            sub = self.eval_actions(items, base=Buffer(self.res, 0.0, 1.0) if MASK_GROUP_BLACK else None)
            sub.blend_onto(buf, mode, op)
        elif cls == "DataActionDecal":
            self._apply_decal(act, buf)
        elif cls == "DataActionFill":
            img = self._fill_source_image(act)
            if img is None:
                return
            mode, op = self._blending(act)
            tmp = Buffer(self.res)
            v, a = img
            tmp.v[:] = v
            tmp.a[:] = a
            tmp.blend_onto(buf, mode, op)
        elif cls == "DataActionPaint":
            cov = self._stroke_cover(act)
            if cov is not None:
                col = (as_list(act.get("sourceColor")) or [{}])[0].get("color", [1, 1, 1])
                buf.paint(cov, float(col[0]), float(act.get("opacity", 1.0)))
        elif cls == "DataActionEraser":
            cov = self._stroke_cover(act)
            if cov is not None:
                buf.erase(cov, float(act.get("opacity", 1.0)))
        elif cls == "DataActionFilterLevels":
            buf.v[:] = P.apply_levels(buf.v, act.get("levels") or {})
        elif isinstance(act.get("procedural"), dict):
            self._apply_procedural(act, buf)
        else:
            self.warn("mask action %s was skipped" % cls)

    def _apply_procedural(self, act, buf):
        """Filters (blur, bevel) and generators placed in a mask stack."""
        proc = act["procedural"]
        name = P.sbsar_name(proc) or "effect"
        tw = P.tweaks(proc)
        mode, op = self._blending(act)
        if name == "blur":
            v, a = P.blur(buf.v, buf.a, tw.get("intensity", 0.5), self.res)
            buf.v += (v - buf.v) * op
            buf.a += (a - buf.a) * op
            self.warn("blur filter approximated")
            return
        if name == "bevel" and P.USE_BEVEL:
            v = P.bevel(buf.v, tw.get("distance", -0.02), tw.get("smoothing", 0.5), self.res)
            buf.v += (v - buf.v) * op
            self.warn("bevel filter approximated")
            return
        if name == "bevel":
            self.warn("bevel filter cannot be reproduced yet and was skipped (soft edges may differ)")
            return
        mask, label = P.generator(name, tw, self.baked_map)
        if mask is None:
            self.warn("procedural effect '%s' cannot be reproduced (%s) and was skipped" % (name, label))
            return
        tmp = Buffer(self.res)
        tmp.v[:] = mask
        tmp.a[:] = 1.0
        tmp.blend_onto(buf, mode, op)
        self.warn("generator '%s' approximated as %s" % (name, label))

    def _fill_source_image(self, act):
        """(value, alpha) of a fill action in a mask, or None if unsupported."""
        full = np.ones((self.res, self.res), np.float32)
        for src in as_list(act.get("sources")):
            if not src:
                continue
            if src.get("__class") == "DataSourceUniform":
                return np.full((self.res, self.res), float(src["color"][0]), np.float32), full * float(src.get("opacity", 1.0))
            url = _source_url(src)
            if url and self.resource_image is not None:
                img = self.resource_image(url)
                if img is not None:
                    # an image's own transparency limits where the fill acts
                    got = self._project(act, lambda u, v: P.sample_at(img, u, v))
                    if got is not None:
                        g, a = got
                        if g.ndim == 3:
                            if g.shape[2] == 4:
                                a = a * g[..., 3].astype(np.float32)
                            g = g[..., :3].mean(-1)
                        return g.astype(np.float32), a * float(src.get("opacity", 1.0))
                    g = P.sample_uv(img, self.res, act.get("uvTransformation"), int(act.get("uvSamplingWrap", 3)) != 0)
                    a = full
                    if g.ndim == 3:
                        if g.shape[2] == 4:
                            a = g[..., 3].astype(np.float32)
                        g = g[..., :3].mean(-1)
                    return g.astype(np.float32), a * float(src.get("opacity", 1.0))
            if isinstance(src.get("procedural"), dict):
                name = P.sbsar_name(src["procedural"])
                if "gradient_linear" in (name or ""):
                    tw = P.tweaks(src["procedural"])
                    got = self._project(act, lambda u, v: P.gradient_value(u, v, tw))
                    if got is None:
                        u, v = P.transform_uv(*P.texel_uv(self.res), act.get("uvTransformation"))
                        u, v, inside = P.wrap_coords(u, v, act.get("uvSamplingWrap", 3))
                        got = P.gradient_value(u, v, tw), inside.astype(np.float32)
                    return got[0], got[1] * float(src.get("opacity", 1.0))
                mask, label = P.generator(name, P.tweaks(src["procedural"]), self.baked_map)
                if mask is not None:
                    self.warn("mask fill '%s' approximated as %s" % (name, label))
                    return mask, full * float(src.get("opacity", 1.0))
                self.warn("mask fill '%s' cannot be reproduced (%s) and was skipped" % (name, label))
                return None
        self.warn("mask fill with an unsupported source was skipped")
        return None

    def _project(self, act, sample):
        """(value, alpha) of a fill seen through a planar or flat warp
        projection, or None for UV-like projections. Other 3D projections are
        approximated in UV space (with a warning)."""
        proj = int(act.get("projection", 0) or 0)
        if proj in P.UV_LIKE_PROJECTIONS:
            return None
        if proj not in (P.PROJ_PLANAR, P.PROJ_WARP):
            self.warn("fill uses a %s projection; applied in UV space instead" % P.PROJECTION_NAMES.get(proj, str(proj)))
            return None
        if proj == P.PROJ_WARP and not P.warp_grid_is_flat(act):
            self.warn("deformed warp projection approximated as planar")
        if self._frame is None:
            # Projection boxes are in the .spp's stored mesh coordinates. Those
            # are normalised when the project is created (centre 0, half
            # extent 1), but NOT renormalised when the mesh is updated later:
            # a test project's stored bounding box is off-centre and 5.6 long, and its
            # Shading gradient only lands where Substance shows it with the
            # stored coordinates as they are (checked against the per-layer
            # export). Deriving a frame from the bounding box moved it to the
            # chest.
            self._frame = (np.zeros(3), 1.0)
        warp = proj == P.PROJ_WARP
        depth = float((act.get("dddGrid") or {}).get("depth", 1.0) or 1.0) if warp else 1.0
        # Meshes that share UV space (display copies) all write the same
        # texels. Draw the triangles the projection reaches last, so a texel
        # takes its position from a copy the projection actually covers.
        cu, cv, cz = P.planar_coords(self.geo.centroid[self.tris], act, self._frame)
        reach = np.abs(cz) <= depth * 1.05
        if warp or int(act.get("dddWrap", 1)) == 0:
            reach &= (cu > -0.05) & (cu < 1.05) & (cv > -0.05) & (cv < 1.05)
        order = np.concatenate([self.tris[~reach], self.tris[reach]])
        pos = self.geo.position_map(order, self.res)[0]
        on = ~np.isnan(pos[..., 0])
        u = np.zeros((self.res, self.res), np.float64)
        v = np.zeros_like(u)
        z = np.full_like(u, np.inf)
        u[on], v[on], z[on] = P.planar_coords(pos[on], act, self._frame)
        # Measured: a warp projection is always cropped to its square (repeat
        # and crop settings make no difference) and only reaches surfaces with
        # |local z| <= its depth (dddGrid.depth); planar repeats unless
        # cropped (dddWrap 0) and reaches |local z| <= 1.
        square = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        if PROJECTOR_DEPTH_CULL and act.get("depthCulling", True) and (square & on).any():
            # depth culling (on by default): hide surfaces the projector can't
            # see; every mesh occludes
            occ = self.geo.T[self._in_mat] if PROJECTOR_OWN_SET_ONLY else self.geo.T
            zbuf, cells = P.projector_depth(self.geo.V, occ, act, self._frame, far=-depth)
            sq = square & on
            ix = np.clip((u[sq] * cells).astype(np.int64), 0, cells - 1)
            iy = np.clip((v[sq] * cells).astype(np.int64), 0, cells - 1)
            hidden = np.zeros_like(square)
            hidden[sq] = z[sq] < zbuf[iy, ix] - P.DEPTH_CULL_TOLERANCE
            on = on & ~hidden
        u, v = P.transform_uv(u, v, act.get("uvTransformation"))
        u, v, inside = P.wrap_coords(u, v, 3 if warp else act.get("uvSamplingWrap", 3))
        if warp or int(act.get("dddWrap", 1)) == 0:
            inside &= square
        inside &= np.abs(z) <= depth + 1e-6
        val = np.asarray(sample(u, v), np.float32)
        alpha = (inside & on).astype(np.float32)
        return val, alpha

    def geometry_mask(self, layer):
        """Coverage of the mesh parts a layer is limited to, or None."""
        keep = self._parts_enabled(layer)
        if keep is None:
            return None
        keep = keep & self._in_mat
        if keep.all() or (keep == self._in_mat).all():
            return None
        return np.asarray(self.geo.rasterize(np.nonzero(keep)[0], self.res), np.float32) / 255.0

    def layer_mask(self, layer):
        """Final mask for a layer or folder as float [0,1], or None for no mask."""
        mask = None
        if layer.get("maskEnabled", False) and layer.get("maskActions"):
            init = 1.0 if int(layer.get("maskInitial", 1)) == 1 else 0.0
            base = Buffer(self.res, value=init, alpha=1.0)
            mask = self.eval_actions(layer["maskActions"].get("items"), base).v
        geo = self.geometry_mask(layer)
        if geo is not None:
            mask = geo if mask is None else mask * geo
        return mask


def _source_url(src, depth=0):
    """Resource URL of an image source, whatever the class calls the field
    (also looks inside nested objects such as a DataBitmap)."""
    if not isinstance(src, dict) or depth > 3:
        return None
    for k, v in src.items():
        if isinstance(v, str) and ("?version=" in v or k.lower().startswith("urltobitmap")) and ".sbsar" not in v:
            return v
    for k, v in src.items():
        if isinstance(v, dict) and k != "procedural":
            u = _source_url(v, depth + 1)
            if u:
                return u
    return None


def _dilate(a, r):
    """Square max filter of radius r (separable; window doubling), same as
    PIL's MaxFilter(2r+1) including at the borders."""
    if r <= 0:
        return a.copy()
    w = 2 * r + 1
    out = np.pad(a, r, mode="constant", constant_values=0)
    for axis in (0, 1):
        m = out
        span = 1
        while span < w:
            step = min(span, w - span)
            if axis == 0:
                m = np.maximum(m[:-step], m[step:])
            else:
                m = np.maximum(m[:, :-step], m[:, step:])
            span += step
        out = m   # each entry = max over a window of w along this axis
    return out


def pad(img, coverage, iterations=4):
    """Bleed values past UV island borders so seams don't show. Only texels
    outside the islands change."""
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    filled = coverage > 0.5
    grown = _dilate(out, iterations)
    return np.where(filled, out, grown)
