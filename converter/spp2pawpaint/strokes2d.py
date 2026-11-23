"""Fast replay of screen-space brush strokes (DataStroke) onto a texture set.

Heavily hand-painted projects can hold millions of stroke points (one sample:
9,151 strokes, 4.6 M points, 2,249 camera views), so this works in bulk:

1. Points are thinned to the brush spacing, then turned into rays with the
   stroke's own camera, and all rays of a stroke are tested against the
   triangles under the stroke's screen box at once (vectorised Moller-Trumbore).
2. The brush size of a screen stroke is a fraction of the viewport, so the
   radius on the surface is computed per hit from its depth.
3. Coverage is the swept brush (capsules between consecutive hits) tested
   against per-mesh-part surface maps (texel -> 3D position + normal), so
   parts that share UV space each get their own paint and back faces
   (relative to the stroke's camera) are skipped.
4. Each action is evaluated at a resolution matched to its brush size (the
   brush stays several texels wide), then scaled up.
"""

import math

import numpy as np
from PIL import Image

from . import brushmodel as BM

# Brush size by brushSizeSpace (measured in Substance 12.2, see brushmodel.py):
#   0 Viewport: radius = brushSize in NDC units (the viewport is 2 units high)
#   1 Object:   radius = brushSize / 2 in normalised scene units
#   2 Texture:  radius = brushSize / 2 in UV units (scene radius = that / UV density)
# The Object rule was first found by scoring four sample files (one test project's body
# 22.7 -> 75.4, another project's body 69.0 -> 92.2) and then confirmed exactly by the
# calibration dabs (235.4 vs 234.9 texels).
SCREEN_SIZE_SCALE = 1.0
BRUSH_RADIUS_PER_SIZE = 0.5
# Stamps wrap over the surface from the point under the cursor ('Tangent |
# Wrap' alignment): they only reach triangles connected to the hit triangle
# through the mesh inside the stamp's sphere. Without this a stroke on the
# chest also painted the arm next to it (measured on a test project's White mask against
# Substance's per-layer export).
WRAP_CONNECTED = False


def stroke_radius(size, scale, w, fy, ortho, space=1, density=1.0):
    """World-space (normalised scene) brush radius per hit (arrays). For
    Texture space `density` is the UV density of the triangle under each hit."""
    if space == 0:
        return size * SCREEN_SIZE_SCALE * scale * (1.0 if ortho else w) / fy
    if space == 2:
        return size * BRUSH_RADIUS_PER_SIZE * scale / np.maximum(density, 1e-9)
    return size * BRUSH_RADIUS_PER_SIZE * scale


def thin_step_ndc(size, cam, center, fy, space=1, density=1.0):
    """Brush radius in NDC (approximate) for point thinning."""
    if space == 0:
        return size * SCREEN_SIZE_SCALE
    d = max(float(np.linalg.norm(cam - center)), 1e-6)
    return float(np.median(stroke_radius(size, 1.0, d, fy, False, space, density))) * fy / d


def stroke_pressure(stroke):
    """Per-point pen pressure. Older files store one value per point; Substance
    9.1 stores a timeline of DataTimelineKey {x: time, value} instead."""
    pr = stroke.get("pressure") or []
    if pr and isinstance(pr[0], dict):
        xs = np.array([float(k.get("x", 0.0)) for k in pr])
        vs = np.array([float(k.get("value", 1.0)) for k in pr])
        t = np.array([float(p.get("time", 0.0)) for p in stroke.get("points") or []])
        order = np.argsort(xs, kind="stable")
        return np.interp(t, xs[order], vs[order])
    return np.asarray(pr, np.float64)


MAX_POINTS_PER_STROKE = 400
MIN_INTERNAL_RES = 256
TARGET_RADIUS_PX = 6.0


def _mat(m16):
    return np.asarray(m16, np.float64).reshape(4, 4).T



def wrap_reach(geo, hit_tris, segs, allowed=None):
    """Bool per triangle: what the stamps of `segs` can reach by wrapping over
    the surface from the triangles the cursor hit (`hit_tris`)."""
    if not segs or len(hit_tris) == 0:
        return allowed
    A, B, RA, RB = (np.concatenate([s[k] for s in segs]) for k in range(4))
    R = np.maximum(RA, RB).astype(np.float64)
    rmax = float(R.max())
    cen, trad = geo.centroid, geo.tri_radius
    lo = np.minimum(A, B).min(0) - rmax
    hi = np.maximum(A, B).max(0) + rmax
    ok = ((cen + trad[:, None] >= lo) & (cen - trad[:, None] <= hi)).all(1)
    if allowed is not None:
        ok &= allowed
    cand = np.nonzero(ok)[0]
    if len(cand) == 0:
        return np.zeros(len(geo.T), bool)
    # triangles whose bounding sphere touches a stamp segment
    near = np.zeros(len(cand), bool)
    pc, pr = cen[cand], trad[cand]
    step = max(1, 6_000_000 // max(len(cand), 1))
    for s0 in range(0, len(A), step):
        a, b, r = A[s0:s0 + step], B[s0:s0 + step], R[s0:s0 + step]
        ab = b - a
        den = np.maximum((ab * ab).sum(1), 1e-18)
        t = np.clip(((pc[:, None, :] - a[None]) * ab[None]).sum(-1) / den[None], 0, 1)
        d = pc[:, None, :] - (a[None] + t[..., None] * ab[None])
        near |= (np.sqrt((d * d).sum(-1)) <= r[None] + pr[:, None]).any(1)
    cand = cand[near]
    out = np.zeros(len(geo.T), bool)
    if len(cand) == 0:
        return out
    _, loc = np.unique(geo.welded[geo.T[cand]], return_inverse=True)
    loc = loc.reshape(-1, 3)
    reached = np.isin(cand, np.asarray(hit_tris, np.int64))
    seen_v = np.zeros(int(loc.max()) + 1, bool)
    n = -1
    while int(reached.sum()) != n:
        n = int(reached.sum())
        seen_v[loc[reached].reshape(-1)] = True
        reached = seen_v[loc].any(1)
    out[cand[reached]] = True
    return out


class ScreenStrokePainter:
    def __init__(self, geo, tris, res, warn=None, lossless=False):
        self.lossless = lossless   # True: always work at full resolution
        self.geo = geo
        self.tris = np.asarray(tris)
        self.res = res
        self.warn = warn or (lambda m: None)
        self._cam_cache = {}
        self._maps = {}
        self._layers = None
        self._grids = {}
        V = geo.V[geo.T[self.tris]]
        UV = geo.UV[geo.T[self.tris]]
        a3 = 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1)
        e1, e2 = UV[:, 1] - UV[:, 0], UV[:, 2] - UV[:, 0]
        auv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        # UV units per world unit for this texture set (texel density)
        self.uv_per_world = math.sqrt(max(auv.sum(), 1e-12) / max(a3.sum(), 1e-12))

    # -- cameras ------------------------------------------------------------------
    def _camera(self, pv):
        key = pv.round(6).tobytes()
        c = self._cam_cache.get(key)
        if c is None:
            g = self.geo
            if not hasattr(self, "_V32"):
                self._V32 = g.V.astype(np.float32)
                self._T = g.T
            m = pv.astype(np.float32)
            V = self._V32
            w = V @ m[3, :3] + m[3, 3]
            front = w > 1e-9
            ws = np.where(front, w, 1.0)
            x = (V @ m[0, :3] + m[0, 3]) / ws
            y = (V @ m[1, :3] + m[1, 3]) / ws
            T = self._T
            xt, yt = x[T], y[T]
            lo = np.stack([np.minimum(np.minimum(xt[:, 0], xt[:, 1]), xt[:, 2]),
                           np.minimum(np.minimum(yt[:, 0], yt[:, 1]), yt[:, 2])], 1)
            hi = np.stack([np.maximum(np.maximum(xt[:, 0], xt[:, 1]), xt[:, 2]),
                           np.maximum(np.maximum(yt[:, 0], yt[:, 1]), yt[:, 2])], 1)
            ft = front[T]
            c = (lo, hi, ft[:, 0] & ft[:, 1] & ft[:, 2])
            if len(self._cam_cache) > 6:
                self._cam_cache.pop(next(iter(self._cam_cache)))
            self._cam_cache[key] = c
        return c

    def _hits(self, pv, inv_pv, P, ndc, allowed, dm=None):
        """Nearest triangle and 3D point per NDC point (vectorised). Returns
        (points (K,3), w (K,), valid (K,) bool)."""
        g = self.geo
        K = len(ndc)
        one = np.ones(K)
        near = np.c_[ndc, -one, one] @ inv_pv.T
        far = np.c_[ndc, one, one] @ inv_pv.T
        ok = (np.abs(near[:, 3]) > 1e-12) & (np.abs(far[:, 3]) > 1e-12)
        o = near[:, :3] / np.where(ok, near[:, 3], 1)[:, None]
        f = far[:, :3] / np.where(ok, far[:, 3], 1)[:, None]
        d = f - o
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        pv_cam = pv
        if dm is not None:
            o = (np.c_[o, one] @ dm.T)[:, :3]
            d = d @ dm[:3, :3].T
            pv_cam = pv @ np.linalg.inv(dm)
        lo, hi, front = self._camera(pv_cam)
        pad = 0.002
        bmin, bmax = ndc.min(0) - pad, ndc.max(0) + pad
        cand = front & (hi[:, 0] >= bmin[0]) & (lo[:, 0] <= bmax[0]) & (hi[:, 1] >= bmin[1]) & (lo[:, 1] <= bmax[1])
        if allowed is not None:
            cand &= allowed
        idx = np.nonzero(cand)[0]
        best_t = np.full(K, np.inf)
        best_i = np.full(K, -1, np.int64)
        if len(idx):
            A, E1, E2 = g._A[idx], g._E1[idx], g._E2[idx]
            step = max(1, 3_000_000 // max(len(idx), 1))
            for s in range(0, K, step):
                dd, oo = d[s:s + step], o[s:s + step]
                P_ = np.cross(dd[:, None, :], E2[None])                 # (k, m, 3)
                det = np.einsum("mj,kmj->km", E1, P_)
                inv = np.where(np.abs(det) > 1e-14, 1.0 / np.where(det == 0, 1, det), 0)
                S = oo[:, None, :] - A[None]
                u = np.einsum("kmj,kmj->km", S, P_) * inv
                Q = np.cross(S, E1[None])
                v = np.einsum("kj,kmj->km", dd, Q) * inv
                t = np.einsum("mj,kmj->km", E2, Q) * inv
                hit = (inv != 0) & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (t > 0)
                t = np.where(hit, t, np.inf)
                j = np.argmin(t, axis=1)
                tt = t[np.arange(len(j)), j]
                best_t[s:s + step] = tt
                best_i[s:s + step] = np.where(np.isfinite(tt), idx[j], -1)
        valid = np.isfinite(best_t) & ok
        p = o + d * np.where(valid, best_t, 0)[:, None]
        w = (np.c_[p, one] @ pv_cam.T)[:, 3]
        return p, np.abs(w), valid, best_i

    # -- surface maps -------------------------------------------------------------------
    def _surface(self, r_int):
        """Concatenated per-part texel maps at r_int: ys, xs, pos, nrm, tri."""
        m = self._maps.get(r_int)
        if m is not None:
            return m
        g = self.geo
        ys_l, xs_l, pos_l, nrm_l, tri_l = [], [], [], [], []
        for tp in self._uv_layers():
            pos, tid = g.position_map(tp, r_int)
            yy, xx = np.nonzero(tid >= 0)
            t = tid[yy, xx]
            ys_l.append(yy.astype(np.int32))
            xs_l.append(xx.astype(np.int32))
            pos_l.append(pos[yy, xx].astype(np.float32))
            n = g.normal[t]
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            nrm_l.append(n.astype(np.float32))
            tri_l.append(t)
        m = (np.concatenate(ys_l), np.concatenate(xs_l), np.concatenate(pos_l),
             np.concatenate(nrm_l), np.concatenate(tri_l))
        self._maps[r_int] = m
        return m

    def _uv_layers(self):
        """Triangle sets that don't overlap in UV. One texel can belong to
        several surfaces: mesh parts that share UV space, and copies of the
        model inside one part (a test project's body holds a posed display copy on the
        same UVs; with one position per texel the stamps looked up the far
        copy and painted nothing). Mesh islands are packed greedily into
        layers whose low-resolution footprints don't overlap."""
        if self._layers is None:
            g = self.geo
            R = 128
            layers = []          # [occupancy (R,R) bool, [triangle arrays]]
            for p in np.unique(g.part[self.tris]):
                tp = self.tris[g.part[self.tris] == p]
                isl = g.mesh_island[tp]
                ids, counts = np.unique(isl, return_counts=True)
                for i in ids[np.argsort(-counts)]:
                    ti = tp[isl == i]
                    uv = g.UV[g.T[ti]]
                    lo = np.clip(np.floor(uv.min(1) * R).astype(int), 0, R - 1)
                    hi = np.clip(np.floor(uv.max(1) * R).astype(int), 0, R - 1)
                    fp = np.zeros((R, R), bool)
                    for (x0, y0), (x1, y1) in zip(lo, hi):
                        fp[y0:y1 + 1, x0:x1 + 1] = True
                    n = int(fp.sum())
                    for lay in layers:
                        if int((lay[0] & fp).sum()) <= 0.05 * n or len(layers) >= 8 and lay is layers[-1]:
                            lay[0] |= fp
                            lay[1].append(ti)
                            break
                    else:
                        layers.append([fp, [ti]])
            self._layers = [np.concatenate(l[1]) for l in layers]
        return self._layers

    def _grid(self, r_int, cell):
        key = (r_int, cell)
        gr = self._grids.get(key)
        if gr is None:
            ys, xs, pos, nrm, tri = self._surface(r_int)
            k = np.floor(pos / cell).astype(np.int64)
            h = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)
            order = np.argsort(h, kind="stable")
            gr = (h[order], order)
            if len(self._grids) > 8:
                self._grids.pop(next(iter(self._grids)))
            self._grids[key] = gr
        return gr

    def _candidates(self, gr, cell, pts, rmax):
        """Texel indices whose grid cell is near any of the sample points."""
        hs, order = gr
        k = np.floor(pts / cell).astype(np.int64)
        reach = int(math.ceil(rmax / cell))
        offs = np.arange(-reach, reach + 1)
        oo = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3)
        keys = (k[:, None, :] + oo[None]).reshape(-1, 3)
        keys = np.unique(keys, axis=0)
        h = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
        h = np.unique(h)
        lo = np.searchsorted(hs, h, "left")
        hi = np.searchsorted(hs, h, "right")
        n = hi - lo
        m = n > 0
        if not m.any():
            return np.zeros(0, np.int64)
        lo, n = lo[m], n[m]
        rep = np.repeat(lo - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
        return order[np.arange(n.sum()) + rep]

    # -- public -------------------------------------------------------------------------------
    def cover(self, strokes, brush, hardness, allowed=None, cull=True, flow=1.0, spacing=0.15,
              space=1, density=1.0):
        """Coverage image (res x res, float32) of a list of DataStroke dicts,
        with Substance's stamp model (brushmodel.py)."""
        size = float(brush.get("brushSize", 0.01))
        emitter = brush.get("stampsEmitter") or {}
        press_on = bool(emitter.get("pressureSizeEnabled", True))
        press_min = float(emitter.get("pressureSizeMinimum", 0.0))
        center = (self.geo.V.min(0) + self.geo.V.max(0)) / 2
        segs = []   # (a, b, ra, rb, cam, wa, wb)
        hit_tris = []
        for s in strokes:
            pts = s.get("points") or []
            if not pts:
                continue
            P = _mat(s["projectionMatrix"])
            V = _mat(s["viewMatrix"])
            pv = P @ V
            try:
                inv_pv = np.linalg.inv(pv)
                cam = np.linalg.inv(V)[:3, 3]
            except np.linalg.LinAlgError:
                continue
            ortho = abs(P[3, 3] - 1.0) < 1e-6
            fy = abs(P[1, 1]) if abs(P[1, 1]) > 1e-9 else 1.0
            xy = np.array([p["position"] for p in pts], np.float64)
            pr = stroke_pressure(s)
            scale = np.ones(len(xy))
            if press_on and len(pr) == len(xy):
                scale = press_min + (1 - press_min) * np.clip(pr, 0, 1)
            ndc = xy * 2 - 1
            # thin to a quarter (lossless) or half (fast) of the brush radius on screen
            r_ndc = max(thin_step_ndc(size, cam, center, fy, space, density), 1e-5)
            step = max(r_ndc * (0.25 if self.lossless else 0.5), 1e-4)
            travelled = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ndc, axis=0), axis=1))]
            bucket = np.floor(travelled / step).astype(np.int64)
            keep = np.r_[0, np.nonzero(np.diff(bucket))[0] + 1]
            if keep[-1] != len(ndc) - 1:
                keep = np.r_[keep, len(ndc) - 1]
            if len(keep) > MAX_POINTS_PER_STROKE and not self.lossless:
                keep = keep[np.linspace(0, len(keep) - 1, MAX_POINTS_PER_STROKE).astype(int)]
            ndc, scale = ndc[keep], scale[keep]
            mats = [None]
            if s.get("duplicateCount", 0) >= 1:
                mats.append(_mat(s["duplicateMatrix"]))
            for dm in mats:
                p, w, valid, ti = self._hits(pv, inv_pv, P, ndc, allowed, dm)
                dens = local_density(self.geo, ti, density) if space == 2 else density
                r = stroke_radius(size, scale, w, fy, ortho, space, dens)
                c = cam if dm is None else (dm @ np.append(cam, 1.0))[:3]
                segs.extend(stroke_segments(p, r, valid, c, spacing))
                hit_tris.append(np.asarray(ti)[np.asarray(valid, bool)])
        if WRAP_CONNECTED and hit_tris:
            allowed = wrap_reach(self.geo, np.unique(np.concatenate(hit_tris)), segs, allowed)
        return self.cover_segments(segs, hardness, flow, spacing, allowed, cull)

    def cover_segments(self, segs, hardness, flow, spacing, allowed=None, cull=True):
        """Coverage of 3D stamp segments (from stroke_segments)."""
        if not segs:
            return None
        A, B, RA, RB, C, WA, WB = (np.concatenate([s[k] for s in segs]) for k in range(7))
        rmed = float(np.median(np.r_[RA, RB]))
        # internal resolution: keep the brush about TARGET_RADIUS_PX texels wide
        r_int = self.res
        while (not self.lossless and r_int > MIN_INTERNAL_RES
               and rmed * self.uv_per_world * r_int > TARGET_RADIUS_PX * 2):
            r_int //= 2
        ys, xs, pos, nrm, tri = self._surface(r_int)
        logt = np.zeros(len(ys), np.float32)          # per surface entry (parts may share texels)
        rmax = float(max(RA.max(), RB.max()))
        cell = 2.0 ** math.ceil(math.log2(max(rmax, 1e-6)))
        gr = self._grid(r_int, cell)
        tab = BM.table(flow, BM.edge_hardness(hardness, rmed * self.uv_per_world * r_int))
        # process segments in batches near each other to keep matrices small
        order = np.argsort((A[:, 0] / cell).astype(np.int64) * 1000003 + (A[:, 1] / cell).astype(np.int64))
        for s0 in range(0, len(order), 64):
            sel = order[s0:s0 + 64]
            a, b, ra, rb, cc, wa, wb = A[sel], B[sel], RA[sel], RB[sel], C[sel], WA[sel], WB[sel]
            samples = np.concatenate([a, b, (a + b) / 2])
            idx = self._candidates(gr, cell, samples, rmax)
            if len(idx) == 0:
                continue
            if allowed is not None:
                idx = idx[allowed[tri[idx]]]
                if len(idx) == 0:
                    continue
            chunk = max(1, 4_000_000 // max(len(sel), 1))
            for c0 in range(0, len(idx), chunk):
                ii = idx[c0:c0 + chunk]
                v = segments_logt(tab, pos[ii], a, b, ra, rb, wa, wb, spacing)
                if cull:
                    vdir = pos[ii][:, None, :] - cc[None].astype(np.float32)
                    facing = -np.einsum("nk,nsk->ns", nrm[ii], vdir)
                    v = np.where(facing > 0, v, 0)
                logt[ii] += v.sum(1).astype(np.float32)
        hit = logt > 1e-6
        if not hit.any():
            return None
        cov = np.zeros((r_int, r_int), np.float32)
        np.maximum.at(cov, (ys[hit], xs[hit]), BM.coverage(logt[hit]).astype(np.float32))
        # return only the touched region (scaled up to the output resolution)
        from .masks import RegionCover
        ys_, xs_ = np.nonzero(cov > 0)
        if len(ys_) == 0:
            return None
        pad_ = 1 if r_int != self.res else 0
        y0, y1 = max(ys_.min() - pad_, 0), min(ys_.max() + 1 + pad_, r_int)
        x0, x1 = max(xs_.min() - pad_, 0), min(xs_.max() + 1 + pad_, r_int)
        crop = np.clip(cov[y0:y1, x0:x1], 0, 1)
        if r_int != self.res:
            f = self.res // r_int
            crop = np.asarray(Image.fromarray(np.ascontiguousarray(crop), "F")
                              .resize(((x1 - x0) * f, (y1 - y0) * f), Image.BILINEAR))
            y0, x0 = y0 * f, x0 * f
        return RegionCover(y0, x0, np.clip(crop, 0, 1).astype(np.float32), self.res)


def stroke_segments(p, r, valid, cam, spacing=None):
    """Split one stroke's surface hits into runs of consecutive valid hits and
    return segment arrays (a, b, ra, rb, cam, wa, wb). wa / wb are the extra
    stamp weights at a / b: half a stamp at each end of a run, one whole
    stamp for an isolated hit (a click).

    With `spacing`, a run is cut where Substance's last stamp falls: stamps
    sit every spacing x diameter from the start and the release point itself
    gets none (measured: strokes end floor(L / step) steps after the start)."""
    out = []
    v = np.asarray(valid, bool)
    if not v.any():
        return out
    cam = np.asarray(cam, np.float64)
    cams = cam if cam.ndim == 2 else np.repeat(cam[None], len(p), 0)   # one camera per point
    idx = np.nonzero(v)[0]
    breaks = np.nonzero(np.diff(idx) != 1)[0] + 1
    for run in np.split(idx, breaks):
        pa, ra_, ca = p[run], r[run], cams[run]
        if len(run) > 1 and spacing:
            seg = np.linalg.norm(np.diff(pa, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            step = 2.0 * float(np.mean(ra_)) * float(spacing)
            if step > 0:
                Lt = np.floor(cum[-1] / step + 1e-6) * step
                if Lt <= 0:
                    pa, ra_ = pa[:1], ra_[:1]
                else:
                    k = int(np.searchsorted(cum, Lt, "left"))          # cum[k-1] < Lt <= cum[k]
                    k = min(max(k, 1), len(cum) - 1)
                    f = (Lt - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-12)
                    end_p = pa[k - 1] + (pa[k] - pa[k - 1]) * f
                    end_r = ra_[k - 1] + (ra_[k] - ra_[k - 1]) * f
                    pa = np.concatenate([pa[:k], end_p[None]])
                    ra_ = np.r_[ra_[:k], end_r]
                    ca = np.concatenate([ca[:k], ca[k - 1:k]])
        if len(pa) == 1:
            out.append((pa[:1], pa[:1], ra_[:1], ra_[:1], ca[:1], np.ones(1), np.zeros(1)))
            continue
        n = len(pa) - 1
        wa = np.zeros(n)
        wb = np.zeros(n)
        wa[0] = 0.5
        wb[-1] = 0.5
        out.append((pa[:-1], pa[1:], ra_[:-1], ra_[1:], ca[:-1], wa, wb))
    return out


def local_density(geo, tri_ids, fallback):
    """UV density of the triangle under each hit (Texture size space)."""
    from . import brushmodel as _BM
    d = _BM.tri_density(geo)
    ti = np.asarray(tri_ids)
    out = np.full(len(ti), float(np.mean(fallback)) if np.size(fallback) else 1.0)
    ok = ti >= 0
    if ok.any():
        v = d[ti[ok]]
        out[ok] = np.where(np.isfinite(v), v, out[ok])
    return out


def segments_logt(tab, q, a, b, ra, rb, wa, wb, spacing):
    """-ln(transmittance) at points q (n, D) from segments a->b (s, D):
    the integrated stamps along each segment plus the end stamps. (n, s)"""
    q = np.asarray(q, np.float32)
    a32 = np.asarray(a, np.float32)
    ab = np.asarray(b, np.float32) - a32
    L = np.sqrt((ab * ab).sum(1))                              # (s,)
    u = ab / np.maximum(L, 1e-12)[:, None]
    rel = q[:, None, :] - a32[None]                            # (n, s, D)
    t = np.einsum("nsk,sk->ns", rel, u)                        # along the segment
    d2 = np.einsum("nsk,nsk->ns", rel, rel)
    p = np.sqrt(np.maximum(d2 - t * t, 0.0))                  # to the segment's line
    ra32 = np.asarray(ra, np.float32)
    rb32 = np.asarray(rb, np.float32)
    frac = np.clip(t / np.maximum(L, 1e-12), 0.0, 1.0)
    rr = ra32 + (rb32 - ra32) * frac
    v = np.where(L > 0, BM.segment_logt(tab, p, t, L, rr, spacing), 0.0)
    wa = np.asarray(wa, np.float32)
    wb = np.asarray(wb, np.float32)
    if wa.any():
        v = v + wa * tab.stamp(np.sqrt(d2) / np.maximum(ra32, 1e-12))
    if wb.any():
        db = q[:, None, :] - np.asarray(b, np.float32)[None]
        v = v + wb * tab.stamp(np.sqrt(np.einsum("nsk,nsk->ns", db, db)) / np.maximum(rb32, 1e-12))
    return v
