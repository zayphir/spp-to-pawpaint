"""Mesh helpers: global triangle tables, picking and selection islands."""

import numpy as np
from PIL import Image, ImageDraw


def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def _components(n_items, pairs_a, pairs_b):
    """Connected components of items linked by (a, b) pairs; returns labels."""
    parent = list(range(n_items))
    for a, b in zip(pairs_a.tolist(), pairs_b.tolist()):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[ra] = rb
    roots = [_find(parent, i) for i in range(n_items)]
    _, labels = np.unique(np.array(roots), return_inverse=True)
    return labels


class Geometry:
    """All mesh parts merged into one triangle soup with lookup tables."""

    def __init__(self, scene):
        self.scene = scene
        V, UV, T, part, poly = [], [], [], [], []
        off = 0
        poly_off = 0
        for pi, p in enumerate(scene.parts):
            V.append(p.positions.astype(np.float64))
            UV.append(p.uvs[0] if p.uvs else np.zeros((len(p.positions), 2), np.float32))
            T.append(p.tris.astype(np.int64) + off)
            part.append(np.full(len(p.tris), pi, np.int32))
            pid = np.cumsum(p.poly_start.astype(np.int64)) - 1 + poly_off
            pid[pid < poly_off] = poly_off
            poly.append(pid)
            poly_off = (pid.max() + 1) if len(pid) else poly_off
            off += len(p.positions)
        self.V = np.concatenate(V)
        self.UV = np.concatenate(UV).astype(np.float64)
        self.T = np.concatenate(T)
        self.part = np.concatenate(part)
        self.poly = np.concatenate(poly)
        self.material = np.array([scene.parts[i].material for i in range(len(scene.parts))], np.int32)[self.part]
        A = self.V[self.T[:, 0]]
        self._A = A
        self._E1 = self.V[self.T[:, 1]] - A
        self._E2 = self.V[self.T[:, 2]] - A
        self.normal = np.cross(self._E1, self._E2)
        self.centroid = A + (self._E1 + self._E2) / 3.0
        self._uv_island = None
        self._mesh_island = None
        bmin, bmax = self.V.min(0), self.V.max(0)
        self.diagonal = float(np.linalg.norm(bmax - bmin))

    # -- islands -----------------------------------------------------------
    @property
    def uv_island(self):
        """Triangles connected through shared vertices (seams split vertices)."""
        if self._uv_island is None:
            T = self.T
            vlab = _components(len(self.V), np.concatenate([T[:, 0], T[:, 1]]),
                               np.concatenate([T[:, 1], T[:, 2]]))
            self._uv_island = vlab[T[:, 0]]
        return self._uv_island

    @property
    def welded(self):
        """Per vertex: id shared by all vertices at the same position in the
        same mesh part (UV seams split vertices; this joins them again)."""
        if getattr(self, "_welded", None) is None:
            key = np.round(self.V / max(self.diagonal, 1e-9) * 1e5).astype(np.int64)
            _, weld = np.unique(key, axis=0, return_inverse=True)
            weld = weld.reshape(-1)
            # keep parts separate: Substance's mesh fill never crosses meshes
            pkey = np.zeros(len(self.V), np.int64)
            for pi in np.unique(self.part):
                pkey[np.unique(self.T[self.part == pi])] = pi
            _, weld = np.unique(np.stack([weld, pkey], 1), axis=0, return_inverse=True)
            self._welded = weld.reshape(-1)
        return self._welded

    @property
    def tri_radius(self):
        """Per triangle: largest distance from its centroid to a corner."""
        if getattr(self, "_tri_radius", None) is None:
            d = self.V[self.T] - self.centroid[:, None, :]
            self._tri_radius = np.sqrt((d * d).sum(-1)).max(1)
        return self._tri_radius

    @property
    def mesh_island(self):
        """Triangles connected in 3D (vertices welded by position)."""
        if self._mesh_island is None:
            weld = self.welded
            T = weld[self.T]
            vlab = _components(int(weld.max()) + 1, np.concatenate([T[:, 0], T[:, 1]]),
                               np.concatenate([T[:, 1], T[:, 2]]))
            self._mesh_island = vlab[T[:, 0]]
        return self._mesh_island

    # -- picking -----------------------------------------------------------
    def _moller(self, idx, o, d):
        """Ray/triangle distances for triangle subset `idx` (inf = miss)."""
        E1, E2, A = self._E1[idx], self._E2[idx], self._A[idx]
        P = np.cross(d, E2)
        det = np.einsum("ij,ij->i", E1, P)
        ok = np.abs(det) > 1e-14
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        S = o - A
        u = np.einsum("ij,ij->i", S, P) * inv
        Q = np.cross(S, E1)
        v = (Q @ d) * inv
        t = np.einsum("ij,ij->i", E2, Q) * inv
        hit = ok & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (t > 0)
        return np.where(hit, t, np.inf)

    def raycast(self, origin, direction, allowed=None, candidates=None):
        """Nearest triangle hit by the ray, or None. Two-sided.

        `allowed` (bool per triangle) makes other triangles transparent;
        `candidates` limits the test to a subset of triangle ids.
        """
        d = np.asarray(direction, np.float64)
        o = np.asarray(origin, np.float64)
        idx = np.arange(len(self.T)) if candidates is None else np.asarray(candidates, np.int64)
        if allowed is not None:
            idx = idx[allowed[idx]]
        if len(idx) == 0:
            return None
        t = self._moller(idx, o, d)
        k = int(np.argmin(t))
        return None if not np.isfinite(t[k]) else int(idx[k])

    def screen_grid(self, pv, cells=96):
        """Bin triangles by screen position for a camera (pv = proj @ view).

        Returns a function (x, y in 0..1) -> candidate triangle ids, so many
        rays from one camera don't each test every triangle.
        """
        Vh = np.c_[self.V, np.ones(len(self.V))] @ pv.T
        w = Vh[:, 3]
        front = w > 1e-9
        ndc = Vh[:, :2] / np.where(front, w, 1)[:, None]
        tri_front = front[self.T].all(1)
        xy = ndc[self.T]
        lo = np.floor((xy.min(1) + 1) / 2 * cells).astype(np.int64)
        hi = np.floor((xy.max(1) + 1) / 2 * cells).astype(np.int64)
        always = np.nonzero(~tri_front)[0]
        ok = tri_front & (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < cells) & (lo[:, 1] < cells)
        lo = np.clip(lo, 0, cells - 1)
        hi = np.clip(hi, 0, cells - 1)
        span = hi - lo + 1
        big = ok & (span[:, 0] * span[:, 1] > 64)
        always = np.concatenate([always, np.nonzero(big)[0]])
        small = np.nonzero(ok & ~big)[0]
        keys, owners = [], []
        for dx in range(8):
            for dy in range(8):
                m = (span[small, 0] > dx) & (span[small, 1] > dy)
                ids = small[m]
                keys.append((lo[ids, 1] + dy) * cells + (lo[ids, 0] + dx))
                owners.append(ids)
        keys = np.concatenate(keys)
        owners = np.concatenate(owners)
        order = np.argsort(keys, kind="stable")
        keys, owners = keys[order], owners[order]

        def lookup(x, y):
            cx = min(max(int(x * cells), 0), cells - 1)
            cy = min(max(int(y * cells), 0), cells - 1)
            k = cy * cells + cx
            a = np.searchsorted(keys, k, "left")
            b = np.searchsorted(keys, k, "right")
            return np.concatenate([owners[a:b], always])
        return lookup

    def select(self, tri, granularity):
        """Triangle ids covered by a polygon-fill pick on `tri`.

        granularity: 0 triangle, 1 polygon, 2 mesh island, 3 UV island.
        """
        if granularity == 0:
            return np.array([tri])
        if granularity == 1:
            return np.nonzero(self.poly == self.poly[tri])[0]
        if granularity == 2:
            return np.nonzero(self.mesh_island == self.mesh_island[tri])[0]
        return np.nonzero(self.uv_island == self.uv_island[tri])[0]

    def mirror_tris(self, tris, matrix, candidates):
        """Triangles (from `candidates`) nearest to the mirror image of `tris`."""
        tris = np.asarray(tris)
        if len(tris) == 0:
            return tris
        cand = np.asarray(candidates)
        key = (matrix.tobytes(), len(cand), int(cand[0]) if len(cand) else -1)
        cache = self.__dict__.setdefault("_mirror_grid", {})
        if key not in cache:
            cc = self.centroid[cand]
            cell = max(float(np.median(np.linalg.norm(self._E1[cand], axis=1))), 1e-9)
            q = np.floor(cc / cell).astype(np.int64)
            grid = {}
            for i, k in enumerate(map(tuple, q.tolist())):
                grid.setdefault(k, []).append(i)
            cache[key] = (cell, grid, cc)
        cell, grid, cc = cache[key]
        c = self.centroid[tris]
        mc = (np.c_[c, np.ones(len(c))] @ matrix.T)[:, :3]
        out = np.empty(len(mc), np.int64)
        for j, p in enumerate(mc):
            base = np.floor(p / cell).astype(np.int64).tolist()
            best = -1
            for r in (1, 2, 4):
                ids = []
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        for dz in range(-r, r + 1):
                            ids.extend(grid.get((base[0] + dx, base[1] + dy, base[2] + dz), ()))
                if ids:
                    ids = np.array(ids)
                    best = int(ids[np.argmin(((cc[ids] - p) ** 2).sum(1))])
                    break
            if best < 0:
                best = int(np.argmin(((cc - p) ** 2).sum(1)))
            out[j] = cand[best]
        return out

    # -- rasterising -------------------------------------------------------
    def uv_pixels(self, tris, res):
        """UV coords of the given triangles in pixel space (image top = v 1)."""
        uv = self.UV[self.T[tris]]  # (n, 3, 2)
        px = np.empty_like(uv)
        px[..., 0] = uv[..., 0] * res
        px[..., 1] = (1.0 - uv[..., 1]) * res
        return px

    def rasterize(self, tris, res, value=255, img=None):
        """Draw triangles into an 'L' PIL image (created if not given)."""
        if img is None:
            img = Image.new("L", (res, res), 0)
        if len(tris) == 0:
            return img
        draw = ImageDraw.Draw(img)
        for t in self.uv_pixels(tris, res):
            draw.polygon([(t[0, 0], t[0, 1]), (t[1, 0], t[1, 1]), (t[2, 0], t[2, 1])], fill=value, outline=value)
        return img

    def triangle_id_map(self, tris, res):
        """int32 map of which triangle covers each texel (-1 for none)."""
        img = Image.new("RGB", (res, res), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        px = self.uv_pixels(tris, res)
        for k, t in enumerate(px):
            code = k + 1
            col = (code & 255, (code >> 8) & 255, (code >> 16) & 255)
            draw.polygon([(t[0, 0], t[0, 1]), (t[1, 0], t[1, 1]), (t[2, 0], t[2, 1])], fill=col, outline=col)
        a = np.asarray(img, np.int64)
        code = a[..., 0] | (a[..., 1] << 8) | (a[..., 2] << 16)
        out = np.full(code.shape, -1, np.int64)
        m = code > 0
        out[m] = np.asarray(tris)[code[m] - 1]
        return out

    def position_map(self, tris, res):
        """Per-texel 3D surface position for a texture set (NaN where empty)."""
        tid = self.triangle_id_map(tris, res)
        ys, xs = np.nonzero(tid >= 0)
        t = tid[ys, xs]
        uvp = self.uv_pixels(t, res)  # (n, 3, 2)
        p = np.stack([xs + 0.5, ys + 0.5], 1)
        a, b, c = uvp[:, 0], uvp[:, 1], uvp[:, 2]
        v0, v1, v2 = b - a, c - a, p - a
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        den[np.abs(den) < 1e-12] = 1e-12
        w1 = np.clip((d11 * d20 - d01 * d21) / den, 0, 1)
        w2 = np.clip((d00 * d21 - d01 * d20) / den, 0, 1)
        w0 = np.clip(1 - w1 - w2, 0, 1)
        # edge texels (drawn by the polygon outline) can fall just outside the
        # triangle; renormalise so they stay on its surface
        wsum = np.maximum(w0 + w1 + w2, 1e-12)
        w0, w1, w2 = w0 / wsum, w1 / wsum, w2 / wsum
        P = self.V[self.T[t]]
        pos = P[:, 0] * w0[:, None] + P[:, 1] * w1[:, None] + P[:, 2] * w2[:, None]
        out = np.full((res, res, 3), np.nan, np.float32)
        out[ys, xs] = pos
        return out, tid
