"""Fit the rebuilt layer masks to Substance's own final texture.

Replaying Substance's brush strokes gets masks close, but never pixel-exact
(stamp shapes, Substance-only filters such as its fur brush, procedural
materials). Every .spp also stores Substance's exact final textures (the
texture cache). Most layers are a colour (or image) through a mask, so the
masks can be solved from that final image: at every texel, find which layer
Substance shows (and how two layers blend at an anti-aliased edge, and how
strongly any Multiply shading layer darkens it), then set the masks so that
PawPaint's own compositing reproduces the texel.

Texels the replayed masks already get right are left alone. The layers stay
editable: recolouring a layer recolours exactly the area Substance showed it
in. What can't be explained by the stack (texture detail of procedural
materials, blend modes PawPaint doesn't have) goes into an optional
'Detail fix (from Substance)' paint layer, so the visible result is exact.

PawPaint's compositing (from its shaders, layer_merge.frag):
    a       = layer opacity * coverage * mask
    blended = blend(acc, top)                    (Normal: top, Multiply: acc*top, ...)
    aOut    = a + aAcc * (1 - a)
    acc     = (blended * a + acc * aAcc * (1 - a)) / aOut
on the stored sRGB values. Folders composite their children from empty, then
merge like a layer.
"""

import numpy as np

from .masks import _dilate

TOL = 8.0 / 255.0          # a texel "matches" when every channel is within this
BLEND_MODES = ("Multiply", "Screen", "Overlay", "Add", "Soft Light", "Hard Light")
# Texels the fitted layers explain within this are left to the layers even if
# not exact: anti-aliased edges blend in linear light in Substance but in sRGB
# in PawPaint, so a half-covered edge texel can't match to the last bit. Putting
# such texels in the fix layer would leave dotted outlines when a layer is
# recoloured later.
TOL_FIX = 24.0 / 255.0
TILE = 128                 # rows per processing tile
EVIDENCE_RADIUS = 8        # texels: how far a replayed mask counts as evidence
# How strongly to prefer layers the replayed masks put there. Two layers can
# look the same on screen (Lavender over Pink vs Lavender over a light-pink
# Shading layer at an anti-aliased edge) and only differ once someone
# recolours a layer, so the replay decides when colours can't.
EVIDENCE_WEIGHT = (12.0 / 255.0) ** 2
PAIR_CANDIDATES = 4        # edge texels: try every pair among this many best layers
# 'One-layer edit': explain a mismatched texel by changing ONE layer's coverage
# and keeping every other layer as replayed. Costs EDIT_WEIGHT per unit of
# change. Measured against Substance's per-layer export: where the replayed
# recipe is right, the single/pair explanations below rewrote layers that were
# already correct (one test project's eyes, Pupil Right: 0.05% of texels off replayed, 2.80%
# fitted), because they zero every layer above the one they pick.
ONE_EDIT = True
# Layers with an exact recipe (no brush strokes, no warnings; node['exact'])
# are trusted: the fitter doesn't change them, as long as the replayed picture
# is right on at least 1 - TRUST_MAX_BAD of the texels where they show. What
# they can't explain goes to the detail fix. Without this the fitter 'fixed'
# mismatches caused by other layers through layers that were right (one test project's eyes:
# 3.3 wrong layer-texels per 100 replayed, 7.2 fitted).
TRUST_EXACT = True
OVER_BLACK = True
TRUST_MAX_BAD = 0.10
EDIT_WEIGHT = (12.0 / 255.0) ** 2
UNDER_WEIGHT = 1.25        # evidence for the layer right under a see-through layer (see solve)
MULTIPLY_PRIOR = (4.0 / 255.0) ** 2    # pull towards the replayed shading strength


def _f(a):
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def blend(mode, base, top):
    if mode == "Multiply":
        return base * top
    if mode == "Add":
        return np.minimum(base + top, 1.0)
    if mode == "Screen":
        return 1.0 - (1.0 - base) * (1.0 - top)
    if mode == "Overlay":
        return np.where(base < 0.5, 2 * base * top, 1.0 - 2 * (1.0 - base) * (1.0 - top))
    if mode == "Soft Light":
        d = np.where(base <= 0.25, ((16 * base - 12) * base + 4) * base, np.sqrt(np.maximum(base, 0)))
        return np.where(top <= 0.5, base - (1 - 2 * top) * base * (1 - base), base + (2 * top - 1) * (d - base))
    if mode == "Hard Light":
        return np.where(top < 0.5, 2 * base * top, 1.0 - 2 * (1.0 - base) * (1.0 - top))
    return top


def merge(acc, acc_a, top, a, mode):
    """PawPaint's layer merge (RGB arrays (n,3), coverage (n,))."""
    b = blend(mode, acc, top)
    a_out = a + acc_a * (1.0 - a)
    w = np.where(a_out > 0, 1.0 / np.maximum(a_out, 1e-12), 0.0)
    c = (b * a[:, None] + acc * (acc_a * (1.0 - a))[:, None]) * w[:, None]
    c = np.where((a_out > 0)[:, None], c, acc)
    return c, a_out


class Leaf:
    """One paint/fill layer as PawPaint will see it, for the albedo channel."""

    def __init__(self, node, ancestors, res):
        self.exact = False
        self.node = node
        self.spec = node.get("spec")
        self.ancestors = ancestors
        self.mode = node.get("mode", "Normal")
        self.op = float(node.get("op", 1.0))
        c = node.get("content") or {}
        self.kind = c.get("kind")
        self.color = None          # (3,) or (res,res,3) uint8/float
        self.pp_cov = None         # coverage PawPaint applies itself (paint alpha), or None = 1
        self.src = None            # what our model multiplies in besides the mask (fill opacity * image alpha)
        self.anc_op = 1.0
        self.fittable = False
        disabled = node.get("disabled") or ()
        if "albedo" in disabled:
            return
        if self.kind == "paint" and "albedo" in (c.get("channels") or {}):
            v, a = c["channels"]["albedo"]
            self.color = v
            self.pp_cov = a
        elif self.kind == "fill":
            if "albedo" in (c.get("maps") or {}):
                self.color = c["maps"]["albedo"]
                self.src = c.get("alpha")
            elif "albedo" in (c.get("values") or {}):
                e = c["values"]["albedo"]
                self.color = np.asarray(e["value"], np.float32)[:3]
                op = float(e.get("opacity", 1.0))
                a = c.get("alpha")
                self.src = op if a is None else a * op
        anc_op = 1.0
        for f in ancestors:
            anc_op *= float(f.get("op", 1.0))
        self.anc_op = anc_op
        self.exact = bool(node.get("exact")) and all(bool(f.get("exact")) for f in ancestors)
        self.fittable = (self.color is not None and self.op > 0 and anc_op > 0
                         and self.mode in ("Normal",) + BLEND_MODES
                         and all(f.get("mode", "Normal") == "Normal" for f in ancestors))

    # -- per-tile accessors -------------------------------------------------------------
    def rgb(self, y0, y1):
        c = self.color
        if c.ndim == 1:
            return None
        return _f(c[y0:y1]).reshape(-1, c.shape[-1])[:, :3]

    def _part(self, a, y0, y1, n):
        if a is None:
            return np.ones(n, np.float32)
        if np.isscalar(a) or np.ndim(a) == 0:
            return np.full(n, float(a), np.float32)
        return _f(a[y0:y1]).reshape(-1)

    def base_mask(self, y0, y1, n):
        """The layer's own mask times its folders' masks (what gets baked)."""
        m = self._part(self.node.get("mask"), y0, y1, n)
        for f in self.ancestors:
            if f.get("mask") is not None:
                m = m * self._part(f["mask"], y0, y1, n)
        return m

    def k0(self, y0, y1, n):
        """Replayed effective coverage (PawPaint terms) before fitting."""
        return (self.op * self.anc_op * self._part(self.pp_cov, y0, y1, n)
                * self._part(self.src, y0, y1, n) * self.base_mask(y0, y1, n))

    def max_k(self, y0, y1, n):
        """Most coverage this layer can reach (paint layers only where painted)."""
        return self.op * self.anc_op * self._part(self.pp_cov, y0, y1, n)


def collect(stack):
    """Leaves bottom -> top with their folder chain."""
    out = []

    def walk(nodes, anc):
        for n in nodes:
            if "children" in n:
                walk(n["children"], anc + [n])
            else:
                out.append((n, anc))
    walk(stack, [])
    return out


BAKE_SHARE = 0.10          # bake a layer's texture when this share of its own area is off
BAKE_MIN_TEXELS = 256


def _cols(leaves, y0, y1, n, baked):
    cols = []
    for L in leaves:
        b = baked.get(id(L.node))
        if b is not None:
            cols.append(_f(b[y0:y1]).reshape(-1, 3))
        elif L.color is None:
            cols.append(None)
        elif L.color.ndim == 1:
            cols.append(np.broadcast_to(L.color.astype(np.float32), (n, 3)))
        else:
            cols.append(L.rgb(y0, y1))
    return cols


def _final_k(leaves, new, y0, y1, n):
    return np.stack([
        (L.op * L.anc_op * L._part(L.pp_cov, y0, y1, n) * _f(new[id(L.node)][y0:y1]).reshape(-1))
        if L.color is not None else np.zeros(n, np.float32) for L in leaves], 0)


def fit(mc, cache_rgb, coverage, log=print):
    """Fit masks of MaterialConverter `mc` to `cache_rgb` (res x res x 3 uint8,
    sRGB). Returns (new_masks {id(node): float mask}, baked {id(node): uint8
    colour map}, residual_rgba or None, stats). Masks returned are what each
    layer's single PawPaint mask must be (folder masks and image alpha baked
    in). `baked` holds layers whose content isn't a flat colour in Substance
    (procedural materials, filtered content): their exact texture, taken from
    the final image where the layer shows."""
    res = mc.res
    leaves = [Leaf(n, anc, res) for n, anc in collect(mc._stack[0])]
    fit_idx = [i for i, L in enumerate(leaves) if L.fittable]
    normal = [i for i in fit_idx if leaves[i].mode == "Normal"]
    # layers whose Substance material couldn't be reproduced at all: estimate
    # their colour from where they show, so they can be fitted (and baked)
    estimated = {}
    for L in leaves:
        # only layers whose colour source existed but couldn't be rebuilt; a
        # height-only layer (e.g. 'Muscles') must not get a colour
        if (L.color is None and L.kind == "fill" and (L.node.get("content") or {}).get("albedo_unknown")
                and "albedo" not in (L.node.get("disabled") or ())
                and L.op > 0 and L.anc_op > 0 and L.mode == "Normal"
                and all(f.get("mode", "Normal") == "Normal" for f in L.ancestors)):
            ys = []
            for y0 in range(0, res, 256):
                y1 = min(res, y0 + 256)
                k = L.base_mask(y0, y1, (y1 - y0) * res)
                sel = np.nonzero((k >= 0.5) & (coverage[y0:y1].reshape(-1) > 0.5))[0]
                if len(sel):
                    ys.append(cache_rgb[y0:y1].reshape(-1, 3)[sel[:: max(1, len(sel) // 2000)]])
            if ys:
                L.color = np.median(np.concatenate(ys), 0).astype(np.float32) / 255.0
                L.src = None
                L.fittable = True
                estimated[id(L.node)] = L.color
    fit_idx = [i for i, L in enumerate(leaves) if L.fittable]
    normal = [i for i in fit_idx if leaves[i].mode == "Normal"]
    multiply = [i for i in fit_idx if leaves[i].mode in BLEND_MODES]   # blend layers with a fitted strength
    new = {id(L.node): np.zeros((res, res), np.uint8) for L in leaves}
    # evidence: where each replayed mask is (dilated), for tie-breaking
    # Evidence is where the replay makes a layer VISIBLE, not just where its
    # mask is: a layer completely covered by another one in the replay (e.g. a
    # base layer with the same colour as the layer above it) gets no credit
    # there, so recolouring the upper layer later recolours those texels too.
    evidence = {i: np.zeros((res, res), np.uint8) for i in normal}
    for y0 in range(0, res, 256):
        y1 = min(res, y0 + 256)
        n = (y1 - y0) * res
        T = np.ones(n, np.float32)
        for i in sorted(normal, reverse=True):
            k = np.clip(leaves[i].k0(y0, y1, n), 0, 1)
            evidence[i][y0:y1] = (k * T * 255 + 0.5).astype(np.uint8).reshape(y1 - y0, res)
            T = T * (1 - k)
    for i in normal:
        evidence[i] = _dilate(evidence[i], EVIDENCE_RADIUS)
    # blend layers (Multiply, Overlay, ...) may only act where the replay has
    # them: otherwise e.g. a paw-bean Overlay highlight could 'explain' a white
    # chest by brightening the pink under it
    for i in multiply:
        e = np.zeros((res, res), np.uint8)
        for y0 in range(0, res, 256):
            y1 = min(res, y0 + 256)
            e[y0:y1] = (np.clip(leaves[i].k0(y0, y1, (y1 - y0) * res), 0, 1) * 255 + 0.5).astype(np.uint8).reshape(y1 - y0, res)
        evidence[i] = _dilate(e, EVIDENCE_RADIUS)
    n_bad0 = n_in = 0
    covm = coverage > 0.5
    dom_n = np.zeros(len(leaves), np.int64)
    dom_bad = np.zeros(len(leaves), np.int64)
    # which exact layers does the replayed picture confirm?
    trusted = set()
    if TRUST_EXACT and any(L.exact for L in leaves):
        tn = np.zeros(len(leaves), np.int64)
        tb = np.zeros(len(leaves), np.int64)
        for y0 in range(0, res, TILE):
            y1 = min(res, y0 + TILE)
            n = (y1 - y0) * res
            C = _f(cache_rgb[y0:y1]).reshape(-1, 3)
            inside = covm[y0:y1].reshape(-1)
            K = np.stack([np.clip(L.k0(y0, y1, n), 0, 1) for L in leaves], 0)
            cols = _cols(leaves, y0, y1, n, {})
            err = np.abs(forward_flat(leaves, cols, K, n) - C).max(1)
            dom, vis = dominant(leaves, K, n)
            solid = inside & (vis >= 0.97) & (dom >= 0)
            np.add.at(tn, dom[solid], 1)
            np.add.at(tb, dom[solid & (err > TOL)], 1)
        for i, L in enumerate(leaves):
            if L.exact and L.fittable and (L.mode != "Normal" or tb[i] <= TRUST_MAX_BAD * max(int(tn[i]), 1)):
                trusted.add(i)
    # pass 1: fit the masks
    for y0 in range(0, res, TILE):
        y1 = min(res, y0 + TILE)
        n = (y1 - y0) * res
        C = _f(cache_rgb[y0:y1]).reshape(-1, 3)
        inside = covm[y0:y1].reshape(-1)
        K = np.stack([L.k0(y0, y1, n) for L in leaves], 0)          # (nl, n)
        cols = _cols(leaves, y0, y1, n, {})
        pred = forward_flat(leaves, cols, K, n)
        bad = inside & (np.abs(pred - C).max(1) > TOL)
        n_in += int(inside.sum())
        n_bad0 += int(bad.sum())
        if bad.any() and normal:
            K = solve(leaves, cols, K, C, bad, normal, multiply, evidence, y0, y1, n, trusted)
        for li, L in enumerate(leaves):
            if L.color is None or not L.fittable:
                # not part of the fit: bake folder masks / image alpha as replayed
                m = L.base_mask(y0, y1, n) * L._part(L.src, y0, y1, n)
            else:
                den = L.op * L.anc_op * L._part(L.pp_cov, y0, y1, n)
                m = np.where(den > 1e-4, K[li] / np.maximum(den, 1e-4), 0.0)
            new[id(L.node)][y0:y1] = (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(y1 - y0, res)
        Kf = _final_k(leaves, new, y0, y1, n)
        err = np.abs(forward_flat(leaves, cols, Kf, n) - C).max(1)
        dom, vis = dominant(leaves, Kf, n)
        solid = inside & (vis >= 0.97) & (dom >= 0)
        np.add.at(dom_n, dom[solid], 1)
        np.add.at(dom_bad, dom[solid & (err > TOL)], 1)
    # layers whose own area is consistently off: their content isn't flat
    bake_ids = [li for li in normal
                if dom_n[li] >= BAKE_MIN_TEXELS and dom_bad[li] > BAKE_SHARE * dom_n[li]]
    baked = {}
    for li in bake_ids:
        L = leaves[li]
        if L.color.ndim == 1:
            b = np.empty((res, res, 3), np.uint8)
            b[:] = (np.clip(L.color[:3], 0, 1) * 255 + 0.5).astype(np.uint8)
        else:
            b = np.array(L.color[..., :3] if L.color.dtype == np.uint8 else
                         (np.clip(L.color[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))
        baked[id(L.node)] = b
    # pass 2: bake those textures, then put what's left in the detail fix
    resid = np.zeros((res, res, 4), np.uint8)
    n_bad1 = n_fix = 0
    err_img = np.zeros((res, res), np.uint8)
    vis_img = np.zeros((res, res), np.uint8)
    for y0 in range(0, res, TILE):
        y1 = min(res, y0 + TILE)
        n = (y1 - y0) * res
        C = _f(cache_rgb[y0:y1]).reshape(-1, 3)
        inside = covm[y0:y1].reshape(-1)
        Kf = _final_k(leaves, new, y0, y1, n)
        if baked:
            dom, vis = dominant(leaves, Kf, n)
            mts = [_cols([leaves[mi]], y0, y1, n, {})[0] for mi in multiply]
            for li in bake_ids:
                sel = inside & (dom == li) & (vis >= 0.97)
                if sel.any():
                    chain = [(leaves[mi].mode, mts[q][sel], Kf[mi][sel]) for q, mi in enumerate(multiply) if mi > li]
                    blk = baked[id(leaves[li].node)][y0:y1].reshape(-1, 3)
                    blk[sel] = (unapply(C[sel], chain) * 255 + 0.5).astype(np.uint8)
        cols = _cols(leaves, y0, y1, n, baked)
        err = np.abs(forward_flat(leaves, cols, Kf, n) - C).max(1)
        n_bad1 += int((inside & (err > TOL)).sum())
        err_img[y0:y1] = np.minimum(err * 255.0, 255).astype(np.uint8).reshape(y1 - y0, res)
        vis_img[y0:y1] = (np.clip(max_visibility(leaves, Kf, n), 0, 1) * 255).astype(np.uint8).reshape(y1 - y0, res)
    # The edge allowance (TOL_FIX) is for anti-aliased edges between layers:
    # partly covered texels right next to fully covered ones. Wide soft areas
    # (glows, blurred shapes) get the strict tolerance; they used to pass as
    # 'edges' and stayed visibly off.
    solid = vis_img >= 247
    edge = ~solid & (_dilate(solid.astype(np.uint8) * 255, 2) > 0)
    tol8, fix8 = TOL * 255.0, TOL_FIX * 255.0
    still_img = covm & ((err_img > fix8) | ((err_img > tol8) & ~edge))
    n_fix = int(still_img.sum())
    if n_fix:
        resid[still_img, :3] = cache_rgb[..., :3][still_img]
        resid[still_img, 3] = 255
    stats = {"texels": n_in, "mismatch_before": n_bad0, "mismatch_after": n_bad1, "fixed": n_fix,
             "baked": [leaves[li].spec.get("name") for li in bake_ids if leaves[li].spec]}
    if n_fix:
        # grow the fix by a texel so bilinear sampling never shows a seam
        a = _dilate(resid[..., 3], 1)
        grow = (a > 0) & (resid[..., 3] == 0)
        if grow.any():
            resid[grow, :3] = cache_rgb[..., :3][grow]
            resid[grow, 3] = 255
    stats["estimated"] = estimated
    return new, baked, (resid if n_fix else None), stats


def dominant(leaves, K, n):
    """Per texel: index of the most visible Normal layer (-1 if none) and how
    much of the texel it covers."""
    T = np.ones(n, np.float32)
    best = np.zeros(n, np.float32)
    idx = np.full(n, -1, np.int64)
    for li in range(len(leaves) - 1, -1, -1):
        L = leaves[li]
        if L.color is None or L.mode != "Normal":
            continue
        w = K[li] * T
        better = w > best
        best = np.where(better, w, best)
        idx = np.where(better, li, idx)
        T = T * (1 - K[li])
    return idx, best


def max_visibility(leaves, K, n):
    """Per texel, how much the most visible Normal layer shows (1 = one layer
    covers the texel completely; less = an edge between layers)."""
    T = np.ones(n, np.float32)
    best = np.zeros(n, np.float32)
    for L, k in zip(reversed(leaves), K[::-1]):
        if L.color is None or L.mode != "Normal":
            continue
        best = np.maximum(best, k * T)
        T = T * (1 - k)
    return best


def forward_flat(leaves, cols, K, n):
    acc = np.zeros((n, 3), np.float32)
    acc_a = np.zeros(n, np.float32)
    for L, c, k in zip(leaves, cols, K):
        if c is None:
            continue
        acc, acc_a = merge(acc, acc_a, c, k, L.mode if L.mode in ("Normal", "Multiply", "Add", "Screen",
                                                              "Overlay", "Soft Light", "Hard Light") else "Normal")
    # Where the layers don't cover a texel completely, Substance's result is
    # the colour over black (a test project's Blush glow sits on an island with nothing
    # under it: cache 147,0,0 at coverage 0.29, not 255,0,0).
    return acc * acc_a[:, None] if OVER_BLACK else acc


def apply_chain(base, chain):
    """Apply blend layers [(mode, colour (m,3), strength (m,))] in order."""
    v = base
    for mode, t, k in chain:
        v = v + (blend(mode, v, t) - v) * k[:, None]
    return v


def unapply(C, chain, iters=4):
    """Base colour that the blend chain turns into C (fixed point, clipped)."""
    base = C.copy()
    for _ in range(iters):
        base = np.clip(base + (C - apply_chain(base, chain)), 0, 1)
    return base


def _fit_strengths(base, Cb, chain_specs, k0):
    """Closed-form strengths of the blend layers above `base` (two passes of
    coordinate descent; each layer's effect is ~linear in its strength)."""
    ks = [np.array(k, np.float32) for k in k0]
    for _ in range(2):
        for q in range(len(chain_specs)):
            lo = [(md, t, (ks[r] if r != q else np.zeros_like(ks[q]))) for r, (md, t) in enumerate(chain_specs)]
            hi = [(md, t, (ks[r] if r != q else np.ones_like(ks[q]))) for r, (md, t) in enumerate(chain_specs)]
            F0, F1 = apply_chain(base, lo), apply_chain(base, hi)
            d = F1 - F0
            num = ((Cb - F0) * d).sum(1) + MULTIPLY_PRIOR * k0[q]
            den = (d * d).sum(1) + MULTIPLY_PRIOR
            ks[q] = np.clip(num / np.maximum(den, 1e-12), 0, 1)
    return ks


def solve(leaves, cols, K, C, bad, normal, blends, evidence, y0, y1, n, trusted=()):
    """Re-explain the mismatched texels `bad` (bool, n)."""
    idx = np.nonzero(bad)[0]
    Cb = C[idx]
    m = len(idx)
    specs_all = [(leaves[q].mode, cols[q][idx]) for q in blends]
    k0_all = [K[q][idx] for q in blends]
    best_cost = np.full(m, np.inf, np.float32)
    best = np.full(m, -1, np.int64)
    best_k = [np.array(k, np.float32) for k in k0_all]
    cand_cost = []
    evs = []
    for j in normal:
        L = leaves[j]
        cj = cols[j][idx]
        reach = L.max_k(y0, y1, n)[idx]
        above = [r for r, q in enumerate(blends) if q > j]       # blend layers that act on j
        ks = [np.array(k, np.float32) for k in k0_all]
        if above:
            sub = _fit_strengths(cj, Cb, [specs_all[r] for r in above], [k0_all[r] for r in above])
            for r, k in zip(above, sub):
                ks[r] = k
        F = apply_chain(cj, [(specs_all[r][0], specs_all[r][1], ks[r]) for r in above])
        err = ((Cb - F) ** 2).sum(1)
        ev = evidence[j][y0:y1].reshape(-1)[idx] / 255.0
        evs.append(ev)
        cost = err + EVIDENCE_WEIGHT * (1 - ev)
        for r in above:                       # blend strength beyond what the replay has
            evb = evidence[blends[r]][y0:y1].reshape(-1)[idx] / 255.0
            cost = cost + 4 * EVIDENCE_WEIGHT * np.maximum(ks[r] - evb, 0)
        cost = np.where(reach > 0.5, cost, np.inf)
        cand_cost.append(cost)
        better = cost < best_cost
        best_cost = np.where(better, cost, best_cost)
        best = np.where(better, j, best)
        for q in range(len(ks)):
            best_k[q] = np.where(better, ks[q], best_k[q])
    # soft edges: try every pair among the best few layers (stack order decides
    # which one is on top); evidence counts for both layers of a pair
    cc = np.stack(cand_cost, 0)                                  # (nn, m)
    ev_all = np.stack(evs, 0)                                    # (nn, m)
    nn = cc.shape[0]
    nm = np.array(normal)
    allc = np.stack([cols[i][idx] for i in normal], 0)          # (nn, m, 3)
    reach_all = np.stack([leaves[i].max_k(y0, y1, n)[idx] for i in normal], 0)
    ar = np.arange(m)
    # Evidence for the LOWER layer of a pair is also how much the replay has it
    # right under the upper layer (nothing opaque in between), not only how
    # visible it is in the replay. A soft gradient layer (a test project's 'Shading') is
    # replayed too strong, which hides the White layer under it; without this,
    # 'Shading over White' and 'a white-coloured layer higher up (Eyebrows, Ear
    # Floof) over a solid Shading' cost the same, the choice flipped from texel
    # to texel, and hiding Shading in PawPaint showed stripes and holes.
    k_raw = np.stack([np.clip(K[i][idx], 0, 1) for i in normal], 0)            # (nn, m)
    pre = np.concatenate([np.zeros((1, m), np.float32),
                          np.cumsum(np.log(np.maximum(1.0 - k_raw, 1e-6)), 0)], 0)  # (nn+1, m)
    kk = min(PAIR_CANDIDATES, nn)
    top = np.argsort(cc, 0)[:kk]                                 # ranks into `normal`
    pair_cost = np.full(m, np.inf, np.float32)
    upper = np.full(m, -1, np.int64)
    lower = np.full(m, -1, np.int64)
    alpha = np.zeros(m, np.float32)
    pair_replayed = np.zeros(m, bool)        # the pair keeps the replayed blend strengths
    # Blend layers above a pair: try both the strengths fitted for the best
    # single layer and the replayed ones. The single layer's strengths belong
    # to a different explanation (a test project: a solid 'Shading' brightened by the
    # 'Bean Higlights' Overlay), and forcing them onto 'Shading over White'
    # made the right pair lose and left holes in White.
    strength_sets = ((best_k, False), (k0_all, True))
    for a in range(kk):
        for b in range(a + 1, kk):
            ra, rb = top[a], top[b]
            ia, ib = nm[ra], nm[rb]
            u_r = np.where(ia > ib, ra, rb)
            l_r = np.where(ia > ib, rb, ra)
            up, lo = nm[u_r], nm[l_r]
            cu, cl = allc[u_r, ar], allc[l_r, ar]
            between = np.where(u_r > l_r + 1, pre[u_r, ar] - pre[np.minimum(l_r + 1, nn), ar], 0.0)
            under = UNDER_WEIGHT * k_raw[l_r, ar] * np.exp(between)
            ev_vis = ev_all[l_r, ar]
            for kset, replayed in strength_sets:
                acts = [np.asarray(blends[r]) > up for r in range(len(blends))]
                chain_u = [(md, t, np.where(acts[r], kset[r], 0.0).astype(np.float32))
                           for r, (md, t) in enumerate(specs_all)]
                F0 = apply_chain(cl, chain_u)
                F1 = apply_chain(cu, chain_u)
                dv = F1 - F0
                al = np.clip(((Cb - F0) * dv).sum(1) / np.maximum((dv * dv).sum(1), 1e-8), 0, 1)
                d = Cb - apply_chain(al[:, None] * cu + (1 - al[:, None]) * cl, chain_u)
                err = (d ** 2).sum(1)
                # the 'right under' evidence only breaks ties between pairs that
                # already reproduce the texel; it never makes the result worse
                ev_lo = np.where(np.abs(d).max(1) <= TOL, np.maximum(ev_vis, under), ev_vis)
                ev_cost = EVIDENCE_WEIGHT * (2 - ev_all[u_r, ar] - ev_lo) * 0.5
                # (no blend-evidence penalty for pairs: the singles' weight dropped
                # one test project's eyes from 98.8 to 83.6% layers alone, 0.1 of it to 95.5%)
                cost = err + ev_cost
                ok = (up != lo) & (reach_all[u_r, ar] > 0.5) & (cost < pair_cost)
                pair_cost = np.where(ok, cost, pair_cost)
                upper = np.where(ok, up, upper)
                lower = np.where(ok, lo, lower)
                alpha = np.where(ok, al, alpha)
                pair_replayed = np.where(ok, replayed, pair_replayed)
    use_pair = (pair_cost + 1e-7 < best_cost) & (upper >= 0)
    # one-layer edits (see ONE_EDIT)
    edit_cost = np.full(m, np.inf, np.float32)
    edit_j = np.full(m, -1, np.int64)
    edit_k = np.zeros(m, np.float32)
    if ONE_EDIT:
        Ks = np.stack([np.asarray(K[i][idx], np.float32) for i in range(len(leaves))], 0)
        cs = [None if c is None else c[idx] for c in cols]
        below = np.zeros(m, np.float32)        # coverage of the Normal layers under j
        under = {}
        for j in sorted(normal):
            under[j] = below.copy()
            below = below + (1 - below) * np.clip(Ks[j], 0, 1)
        for j in list(normal) + list(blends):
            kmax = leaves[j].max_k(y0, y1, n)[idx]
            ev = evidence[j][y0:y1].reshape(-1)[idx] / 255.0
            live = (kmax > 0) & ((Ks[j] > 0) | (ev > 0))
            if j in under:
                # a fully-on layer with nothing under it is not thinned out:
                # that would 'explain' a darker texture by showing the black
                # behind it (textures are baked instead)
                live &= (under[j] > 0.5) | (Ks[j] < 0.98 * kmax)
            if j in trusted:
                # a trusted layer only gets its SOFT parts re-weighted:
                # Substance mixes in linear light, PawPaint in sRGB, so a
                # half-covered texel needs another opacity to look the same.
                # Where it is fully on or off it stays as replayed.
                live &= (Ks[j] > 0) & (Ks[j] < 0.98 * kmax)
            if not live.any():
                continue
            sub = np.nonzero(live)[0]
            K1 = Ks[:, sub].copy()
            csub = [None if c is None else c[sub] for c in cs]
            K1[j] = 0.0
            F0 = forward_flat(leaves, csub, K1, len(sub))
            K1[j] = kmax[sub]
            F1 = forward_flat(leaves, csub, K1, len(sub))
            dv = F1 - F0
            t = np.clip(((Cb[sub] - F0) * dv).sum(1) / np.maximum((dv * dv).sum(1), 1e-8), 0, 1)
            knew = t * kmax[sub]
            K1[j] = knew
            dd = Cb[sub] - forward_flat(leaves, csub, K1, len(sub))
            cost = (dd ** 2).sum(1) + EDIT_WEIGHT * np.abs(knew - Ks[j][sub])
            if leaves[j].mode != "Normal":      # blend strength beyond what the replay has
                cost = cost + 4 * EVIDENCE_WEIGHT * np.maximum(knew - ev[sub], 0)
            better = cost < edit_cost[sub]
            w = sub[better]
            edit_cost[w] = cost[better]
            edit_j[w] = j
            edit_k[w] = knew[better]
    use_edit = (edit_j >= 0) & (edit_cost <= np.minimum(best_cost, np.where(upper >= 0, pair_cost, np.inf)))
    # write the solution
    top = np.where(use_pair, upper, best)
    low = np.where(use_pair, lower, best)
    prop = {}
    for j in normal:
        sel_top = top == j
        sel_low = (low == j) & use_pair
        above = j > top
        between = (j > low) & (j < top) & use_pair
        new_k = K[j][idx].copy()
        new_k[above | between] = 0.0
        new_k[sel_low] = 1.0
        new_k[sel_top] = np.where(use_pair[sel_top], alpha[sel_top], 1.0)
        prop[j] = np.minimum(new_k, leaves[j].max_k(y0, y1, n)[idx])
    blend_new = [np.where(use_pair & pair_replayed, k0_all[q], best_k[q]) for q in range(len(blends))]
    # a single/pair explanation that would rewrite a trusted layer is dropped:
    # use the one-layer edit if there is one, else keep the replay as it is
    veto = np.zeros(m, bool)
    for j in normal:
        if j in trusted:
            veto |= np.abs(prop[j] - K[j][idx]) > 0.02
    for q, mi in enumerate(blends):
        if mi in trusted:
            veto |= np.abs(blend_new[q] - k0_all[q]) > 0.02
    keep = use_edit | veto
    use_edit = (use_edit | veto) & (edit_j >= 0)
    for q, mi in enumerate(blends):
        K[mi][idx] = np.where(keep, k0_all[q], blend_new[q])
    for j in normal:
        K[j][idx] = np.where(keep, K[j][idx], prop[j])
    for j in np.unique(edit_j[use_edit]):
        sel = use_edit & (edit_j == j)
        K[j][idx[sel]] = edit_k[sel]
    return K


SCALAR_SPREAD = 3.0 / 255.0   # a layer's value is uniform when its texels vary less than this


def fit_scalars(mc, new, coverage, targets):
    """Per layer and scalar channel (roughness, metallic): the value Substance
    shows where that layer is the one on top. Only layers that carry the
    channel in Substance are changed. Returns {id(node): {cid: float or
    (res,res) float32 map}}."""
    res = mc.res
    leaves = [Leaf(n, anc, res) for n, anc in collect(mc._stack[0])]
    covm = coverage > 0.5
    out = {}
    carriers = {}
    for cid in targets:
        idx = []
        for li, L in enumerate(leaves):
            c = L.node.get("content") or {}
            if cid in (L.node.get("disabled") or ()):
                continue
            has = (cid in (c.get("values") or {}) or cid in (c.get("maps") or {})
                   or cid in (c.get("channels") or {}))
            if has and L.op > 0 and L.anc_op > 0:
                idx.append(li)
        carriers[cid] = idx
    for cid, T in targets.items():
        # who is on top for this channel: only layers that carry it count
        doms = np.full((res, res), -1, np.int32)
        for y0 in range(0, res, TILE):
            y1 = min(res, y0 + TILE)
            n = (y1 - y0) * res
            Kf = {}
            for li in carriers[cid]:
                L = leaves[li]
                c = L.node.get("content") or {}
                cov = c["channels"][cid][1] if L.kind == "paint" else None
                Kf[li] = L.op * L.anc_op * L._part(cov, y0, y1, n) * _f(new[id(L.node)][y0:y1]).reshape(-1)
            Tr = np.ones(n, np.float32)
            best = np.zeros(n, np.float32)
            dom = np.full(n, -1, np.int64)
            for li in sorted(carriers[cid], reverse=True):
                w = Kf[li] * Tr
                better = w > best
                best = np.where(better, w, best)
                dom = np.where(better, li, dom)
                Tr = Tr * (1 - Kf[li])
            dom = np.where(covm[y0:y1].reshape(-1) & (best >= 0.97), dom, -1)
            doms[y0:y1] = dom.reshape(y1 - y0, res)
        for li in carriers[cid]:
            sel = doms == li
            cnt = int(sel.sum())
            if cnt < BAKE_MIN_TEXELS:
                continue
            vals = T[sel]
            med = float(np.median(vals))
            spread = float(np.mean(np.abs(vals - med)))
            L = leaves[li]
            if spread <= SCALAR_SPREAD:
                out.setdefault(id(L.node), {})[cid] = med
            else:
                m = np.full((res, res), med, np.float32)
                m[sel] = T[sel]
                out.setdefault(id(L.node), {})[cid] = m
    return out
