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

import atexit
import os
import time

import numpy as np

TOL = 8.0 / 255.0          # a texel "matches" when every channel is within this
BLEND_MODES = ("Multiply", "Screen", "Overlay", "Add", "Soft Light", "Hard Light")
# Texels the fitted layers explain within this are left to the layers even if
# not exact: anti-aliased edges blend in linear light in Substance but in sRGB
# in PawPaint, so a half-covered edge texel can't match to the last bit. Putting
# such texels in the fix layer would leave dotted outlines when a layer is
# recoloured later.
TOL_FIX = 24.0 / 255.0
# The fit works through the texture in strips of rows. Every step is per texel,
# so the result doesn't depend on the strip height; it only decides speed.
# Small strips keep the working arrays inside the CPU's cache (a 4K strip of
# 128 rows was 524,288 texels: every one of the many small steps then streamed
# megabytes through memory), and strips are what the worker processes share out.
TILE_TEXELS = 65536        # texels per strip
TILE_MAX_ROWS = 128
# Big fits run their strips in worker processes (threads don't help: the steps
# are many small numpy calls and Python runs one thread at a time between them).
WORKERS = 0                # 0 = one per CPU core (cli sets it from --workers); 1 = this process only
MAX_WORKERS = 32
STRIPS_PER_WORKER = 4
POOL_MIN_WORK = 12e6       # fitted layers x texels from which the worker processes pay off
PROGRESS_SECONDS = 10.0    # how often a long fit says how far it is
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
# Joint edit: explain a mismatched texel by nudging the coverage of several
# layers at once, each as little as possible, instead of rewriting the texel
# as 'one layer' or 'one layer over another'. Where soft layers overlap (three
# airbrushed colours over each other) the replay is usually nearly right, and a
# rewrite as a pair throws away the third layer and sets the lower one to full.
JOINT_EDIT = True
JOINT_RIDGE = 1e-3         # keeps the joint step small where the layers' colours are nearly alike
JOINT_STEPS = 1            # steps of the joint edit (the layers' directions are kept from the first)
# What a single-layer or pair explanation pays per unit of coverage it changes
# against the replay, as a share of EDIT_WEIGHT (the edits pay 1). 0 = nothing,
# as before the joint edit existed.
REWRITE_WEIGHT = 1.0


def _f(a):
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def tile_rows(res):
    return max(1, min(TILE_MAX_ROWS, TILE_TEXELS // max(int(res), 1)))


def worker_count():
    n = int(WORKERS) if WORKERS else (os.cpu_count() or 1)
    n = max(1, min(n, MAX_WORKERS))
    if n > 1:
        try:        # a worker holds numpy and one strip: a few hundred MB at most
            from .settings import available_memory_gb
            n = max(1, min(n, int(available_memory_gb() * 0.6 / 0.3)))
        except Exception:
            pass
    return n


def _duration(seconds):
    seconds = int(seconds + 0.5)
    if seconds < 90:
        return "%d s" % max(seconds, 1)
    if seconds < 5400:
        return "%d min" % ((seconds + 30) // 60)
    return "%.1f h" % (seconds / 3600.0)


class _Progress:
    """Says how far a long step is, at most every PROGRESS_SECONDS."""

    def __init__(self, log, what, steps):
        self.log, self.what, self.steps = log, what, max(int(steps), 1)
        self.t0 = self.last = time.time()

    def __call__(self, done):
        now = time.time()
        if now - self.last >= PROGRESS_SECONDS and done < self.steps:
            self.last = now
            left = (now - self.t0) * (self.steps - done) / max(done, 1)
            self.log("      %s: %d%% (about %s left)" % (self.what, 100 * done // self.steps, _duration(left)))


def on_threads(fn, items):
    """fn(item) for every item, on a few threads. For steps made of a few
    numpy calls on whole images: there numpy does the work without holding
    Python's lock, so threads help (unlike in the strips, see WORKERS)."""
    items = list(items)
    n = min(8, len(items), os.cpu_count() or 1) if WORKERS != 1 else 1
    if n <= 1:
        for it in items:
            fn(it)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(fn, items))


_POOL = {"ex": None, "n": 0}


def _pool(n):
    if _POOL["ex"] is not None and _POOL["n"] == n:
        return _POOL["ex"]
    close_pool()
    import concurrent.futures as cf
    import multiprocessing as mp
    _POOL["ex"] = cf.ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))
    _POOL["n"] = n
    return _POOL["ex"]


def close_pool():
    """Stop the fit's worker processes (cli does this when a conversion ends)."""
    ex, _POOL["ex"], _POOL["n"] = _POOL["ex"], None, 0
    if ex is not None:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


atexit.register(close_pool)


def run_strips(fn, jobs, workers=1, progress=None, log=None):
    """fn(job) for every job, in worker processes when workers > 1. Returns
    the results in job order. Strips don't depend on each other, so the
    results are the same either way; if the processes can't be used, the work
    is done here."""
    out = [None] * len(jobs)
    if workers > 1 and len(jobs) > 1:
        try:
            from concurrent.futures import as_completed
            ex = _pool(workers)
            futs = {ex.submit(fn, job): i for i, job in enumerate(jobs)}
            done = 0
            for f in as_completed(futs):
                out[futs[f]] = f.result()
                done += 1
                if progress:
                    progress(done)
            return out
        except Exception as e:      # e.g. the pool could not start, or a worker was killed
            close_pool()
            if log:
                log("      (worker processes failed: %s; fitting on one core)" % e)
            out = [None] * len(jobs)
    for i, job in enumerate(jobs):
        out[i] = fn(job)
        if progress:
            progress(i + 1)
    return out


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

    def strip(self, y0, y1, seen):
        """This layer for rows y0:y1 only (its rows start at 0), small enough
        to send to a worker process. `seen` shares arrays between the layers
        of one strip (folder masks), so each is cut and sent once."""
        def cut(a):
            if isinstance(a, np.ndarray) and a.ndim >= 2:
                if id(a) not in seen:
                    seen[id(a)] = a[y0:y1]
                return seen[id(a)]
            return a
        s = Leaf.__new__(Leaf)
        s.exact, s.mode, s.op, s.kind = self.exact, self.mode, self.op, self.kind
        s.anc_op, s.fittable, s.spec = self.anc_op, self.fittable, None
        s.node = {"mask": cut(self.node.get("mask"))}
        s.ancestors = [{"mask": cut(f.get("mask"))} for f in self.ancestors]
        s.color, s.pp_cov, s.src = cut(self.color), cut(self.pp_cov), cut(self.src)
        return s


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
    """Colour of every layer for rows y0:y1. `baked`: {layer index: texture} replaces a layer's own colour."""
    cols = []
    for i, L in enumerate(leaves):
        b = baked.get(i)
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
    """Coverage of every layer with the fitted masks `new` (one per layer, by index)."""
    return np.stack([
        (L.op * L.anc_op * L._part(L.pp_cov, y0, y1, n) * _f(new[i][y0:y1]).reshape(-1))
        if L.color is not None else np.zeros(n, np.float32) for i, L in enumerate(leaves)], 0)


# -- the fit's three passes, one strip of rows at a time -----------------------------
# A job is a dict with everything a strip needs, cut to its rows (see fit), so
# the same function runs here or in a worker process.

def _trust_strip(job):
    """How often is each layer the one on top, and how often is the replayed picture wrong there?"""
    leaves, res = job["leaves"], job["res"]
    h = job["cache"].shape[0]
    n = h * res
    tn = np.zeros(len(leaves), np.int64)
    tb = np.zeros(len(leaves), np.int64)
    C = _f(job["cache"]).reshape(-1, 3)
    inside = job["inside"].reshape(-1)
    K = np.stack([np.clip(L.k0(0, h, n), 0, 1) for L in leaves], 0)
    colsT = turn_cols(_cols(leaves, 0, h, n, {}))
    err = np.abs(forward_turned(leaves, colsT, K, n) - _turned(C)).max(0)
    dom, vis = dominant(leaves, K, n)
    solid = inside & (vis >= 0.97) & (dom >= 0)
    np.add.at(tn, dom[solid], 1)
    np.add.at(tb, dom[solid & (err > TOL)], 1)
    return tn, tb


def _fit_strip(job):
    """Pass 1: the fitted mask of every layer for this strip (uint8, layers x rows x width) and its counts."""
    leaves, res = job["leaves"], job["res"]
    h = job["cache"].shape[0]
    n = h * res
    nl = len(leaves)
    dn = np.zeros(nl, np.int64)
    db = np.zeros(nl, np.int64)
    C = _f(job["cache"]).reshape(-1, 3)
    inside = job["inside"].reshape(-1)
    K = np.stack([L.k0(0, h, n) for L in leaves], 0)          # (nl, n)
    cols = _cols(leaves, 0, h, n, {})
    colsT = turn_cols(cols)
    CT = _turned(C)
    bad = inside & (np.abs(forward_turned(leaves, colsT, K, n) - CT).max(0) > TOL)
    if bad.any() and job["normal"]:
        K = solve(leaves, cols, K, C, bad, job["normal"], job["blends"], job["evidence"], 0, h, n, job["trusted"],
                  colsT)
    new = np.zeros((nl, h, res), np.uint8)
    for li, L in enumerate(leaves):
        if L.color is None or not L.fittable:
            # not part of the fit: bake folder masks / image alpha as replayed
            m = L.base_mask(0, h, n) * L._part(L.src, 0, h, n)
        else:
            den = L.op * L.anc_op * L._part(L.pp_cov, 0, h, n)
            m = np.where(den > 1e-4, K[li] / np.maximum(den, 1e-4), 0.0)
        new[li] = (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(h, res)
    Kf = _final_k(leaves, new, 0, h, n)
    err = np.abs(forward_turned(leaves, colsT, Kf, n) - CT).max(0)
    dom, vis = dominant(leaves, Kf, n)
    solid = inside & (vis >= 0.97) & (dom >= 0)
    np.add.at(dn, dom[solid], 1)
    np.add.at(db, dom[solid & (err > TOL)], 1)
    return new, int(inside.sum()), int(bad.sum()), dn, db


def _check_strip(job):
    """Pass 2: fill in the textures of baked layers, then measure what the fitted layers still get wrong."""
    leaves, res, baked, blends = job["leaves"], job["res"], job["baked"], job["blends"]
    h = job["cache"].shape[0]
    n = h * res
    C = _f(job["cache"]).reshape(-1, 3)
    inside = job["inside"].reshape(-1)
    Kf = _final_k(leaves, job["new"], 0, h, n)
    if baked:
        dom, vis = dominant(leaves, Kf, n)
        mts = [_cols([leaves[mi]], 0, h, n, {})[0] for mi in blends]
        for li in sorted(baked):
            sel = inside & (dom == li) & (vis >= 0.97)
            if sel.any():
                chain = [(leaves[mi].mode, mts[q][sel], Kf[mi][sel]) for q, mi in enumerate(blends) if mi > li]
                if not baked[li].flags.writeable:
                    baked[li] = baked[li].copy()
                blk = baked[li].reshape(-1, 3)
                blk[sel] = (unapply(C[sel], chain) * 255 + 0.5).astype(np.uint8)
    cols = _cols(leaves, 0, h, n, baked)
    err = np.abs(forward_turned(leaves, turn_cols(cols), Kf, n) - _turned(C)).max(0)
    err_rows = np.minimum(err * 255.0, 255).astype(np.uint8).reshape(h, res)
    vis_rows = (np.clip(max_visibility(leaves, Kf, n), 0, 1) * 255).astype(np.uint8).reshape(h, res)
    return int((inside & (err > TOL)).sum()), err_rows, vis_rows, baked


def fit(mc, cache_rgb, coverage, log=print):
    """Fit the masks (see _fit). The worker processes of a big fit are stopped
    again afterwards: idle, they would hold memory the next texture set's mask
    painting needs."""
    cache_rgb = np.asarray(cache_rgb)
    if cache_rgb.ndim == 2:
        cache_rgb = cache_rgb[..., None]
    if cache_rgb.shape[2] < 3:               # greyscale (with or without alpha): the fit works on colours
        cache_rgb = np.repeat(cache_rgb[..., :1], 3, -1)
    elif cache_rgb.shape[2] > 3:
        cache_rgb = cache_rgb[..., :3]
    try:
        return _fit(mc, cache_rgb, coverage, log)
    finally:
        close_pool()


def _fit(mc, cache_rgb, coverage, log=print):
    """Fit masks of MaterialConverter `mc` to `cache_rgb` (res x res x 3 uint8,
    sRGB). Returns (new_masks {id(node): float mask}, baked {id(node): uint8
    colour map}, residual_rgba or None, stats). Masks returned are what each
    layer's single PawPaint mask must be (folder masks and image alpha baked
    in). `baked` holds layers whose content isn't a flat colour in Substance
    (procedural materials, filtered content): their exact texture, taken from
    the final image where the layer shows."""
    from .masks import _dilate
    res = mc.res
    t_start = time.time()
    timing = []

    def lap(what):      # where a long fit spends its time, for the log file
        timing.append("%s %.1f s" % (what, time.time() - lap.t))
        lap.t = time.time()
    lap.t = t_start
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
    masks = [np.zeros((res, res), np.uint8) for L in leaves]      # the fitted mask of every layer
    new = {id(L.node): m for L, m in zip(leaves, masks)}
    # evidence: where each replayed mask is (dilated), for tie-breaking
    # Evidence is where the replay makes a layer VISIBLE, not just where its
    # mask is: a layer completely covered by another one in the replay (e.g. a
    # base layer with the same colour as the layer above it) gets no credit
    # there, so recolouring the upper layer later recolours those texels too.
    # A big fit runs in worker processes (see WORKERS). They take a few seconds
    # to start, so start them now, while the evidence is worked out here.
    workers = worker_count() if len(fit_idx) * res * res >= POOL_MIN_WORK else 1
    if workers == 1 and len(fit_idx) * res * res >= POOL_MIN_WORK and (os.cpu_count() or 1) > 1 and WORKERS != 1:
        log("      (too little free memory for worker processes; fitting on one core)")
    rows = tile_rows(res)
    if workers > 1:     # several strips per worker, so that busy and empty rows even out
        rows = max(1, min(rows, res // (workers * STRIPS_PER_WORKER)))
    spans = [(y0, min(res, y0 + rows)) for y0 in range(0, res, rows)]
    workers = min(workers, len(spans))
    if workers > 1:
        try:
            ex = _pool(workers)
            for _ in range(workers):
                ex.submit(int)
        except Exception:
            close_pool()
    evidence = {i: np.zeros((res, res), np.uint8) for i in normal}
    down = sorted(normal, reverse=True)

    def visible_rows(y0):
        y1 = min(res, y0 + 256)
        n = (y1 - y0) * res
        T = np.ones(n, np.float32)
        for i in down:
            k = np.clip(leaves[i].k0(y0, y1, n), 0, 1)
            evidence[i][y0:y1] = (k * T * 255 + 0.5).astype(np.uint8).reshape(y1 - y0, res)
            T = T * (1 - k)
    on_threads(visible_rows, range(0, res, 256))

    def spread(i):
        evidence[i] = _dilate(evidence[i], EVIDENCE_RADIUS)
    on_threads(spread, normal)

    # blend layers (Multiply, Overlay, ...) may only act where the replay has
    # them: otherwise e.g. a paw-bean Overlay highlight could 'explain' a white
    # chest by brightening the pink under it
    def blend_evidence(i):
        e = np.zeros((res, res), np.uint8)
        for y0 in range(0, res, 256):
            y1 = min(res, y0 + 256)
            e[y0:y1] = (np.clip(leaves[i].k0(y0, y1, (y1 - y0) * res), 0, 1) * 255 + 0.5).astype(np.uint8).reshape(y1 - y0, res)
        evidence[i] = _dilate(e, EVIDENCE_RADIUS)
    for i in multiply:
        evidence[i] = None
    on_threads(blend_evidence, multiply)
    covm = coverage > 0.5
    lap("evidence")

    def jobs(**more):
        """One job per strip: the layers, Substance's picture and the island mask, cut to its rows."""
        out = []
        for y0, y1 in spans:
            job = {"res": res, "leaves": [L.strip(y0, y1, seen) for seen in [{}] for L in leaves],
                   "cache": cache_rgb[y0:y1], "inside": covm[y0:y1]}
            for k, v in more.items():
                job[k] = v(y0, y1) if callable(v) else v
            out.append(job)
        return out

    # which exact layers does the replayed picture confirm?
    trusted = set()
    if TRUST_EXACT and any(L.exact for L in leaves):
        parts = run_strips(_trust_strip, jobs(), workers, log=log)
        tn = sum(p[0] for p in parts)
        tb = sum(p[1] for p in parts)
        for i, L in enumerate(leaves):
            if L.exact and L.fittable and (L.mode != "Normal" or tb[i] <= TRUST_MAX_BAD * max(int(tn[i]), 1)):
                trusted.add(i)
    lap("exact layers check")
    # pass 1: fit the masks
    if workers > 1:
        log("      (%d layers to fit on %d CPU cores)" % (len(fit_idx), workers))
    parts = run_strips(_fit_strip, jobs(normal=normal, blends=multiply, trusted=trusted,
                                        evidence=lambda y0, y1: {i: e[y0:y1] for i, e in evidence.items()}),
                       workers, _Progress(log, "fitting masks", len(spans)), log)
    n_in = n_bad0 = 0
    dom_n = np.zeros(len(leaves), np.int64)
    dom_bad = np.zeros(len(leaves), np.int64)
    for (y0, y1), (strip, a, b, dn, db) in zip(spans, parts):
        for li in range(len(leaves)):
            masks[li][y0:y1] = strip[li]
        n_in += a
        n_bad0 += b
        dom_n += dn
        dom_bad += db
    del parts
    lap("fit")
    # layers whose own area is consistently off: their content isn't flat
    bake_ids = [li for li in normal
                if dom_n[li] >= BAKE_MIN_TEXELS and dom_bad[li] > BAKE_SHARE * dom_n[li]]
    baked = {}
    baked_i = {}                                                 # the same textures by layer index
    for li in bake_ids:
        L = leaves[li]
        if L.color.ndim == 1:
            b = np.empty((res, res, 3), np.uint8)
            b[:] = (np.clip(L.color[:3], 0, 1) * 255 + 0.5).astype(np.uint8)
        else:
            b = np.array(L.color[..., :3] if L.color.dtype == np.uint8 else
                         (np.clip(L.color[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))
        baked[id(L.node)] = b
        baked_i[li] = b
    # pass 2: bake those textures, then put what's left in the detail fix
    resid = np.zeros((res, res, 4), np.uint8)
    n_bad1 = n_fix = 0
    err_img = np.zeros((res, res), np.uint8)
    vis_img = np.zeros((res, res), np.uint8)
    parts = run_strips(_check_strip, jobs(blends=multiply, new=lambda y0, y1: [m[y0:y1] for m in masks],
                                          baked=lambda y0, y1: {li: b[y0:y1] for li, b in baked_i.items()}),
                       workers, _Progress(log, "checking the fitted layers", len(spans)), log)
    for (y0, y1), (bad1, err_rows, vis_rows, bk) in zip(spans, parts):
        n_bad1 += bad1
        err_img[y0:y1] = err_rows
        vis_img[y0:y1] = vis_rows
        for li, rows_ in bk.items():
            baked_i[li][y0:y1] = rows_
    del parts
    lap("check")
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
    lap("detail fix")
    detail = getattr(log, "detail", None)
    if detail is not None and time.time() - t_start >= 2.0:
        detail("      fit timing: %s (%d strips, %d workers)" % (", ".join(timing), len(spans), workers))
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


def turn_cols(cols):
    """A list of layer colours ((n, 3) arrays or None, as _cols gives them) for
    forward_turned: (3, n) arrays, and a (3, 1) column for a layer that has one
    colour everywhere, which numpy then handles without touching n values."""
    out = []
    for c in cols:
        if c is None:
            out.append(None)
        elif c.strides[0] == 0:                 # np.broadcast_to of one colour
            out.append(np.array(c[0], np.float32).reshape(3, 1))
        else:
            out.append(_turned(c))
    return out


def take_cols(colsT, idx):
    """Turned colours for a subset of the texels."""
    return [None if c is None else (c if c.shape[1] == 1 else c[:, idx]) for c in colsT]


def _merge_turned(acc, acc_a, top, a, mode):
    """merge() for (3, n) colours; `top` may be a (3, 1) column."""
    b = blend(mode, acc, top)
    a_out = a + acc_a * (1.0 - a)
    w = np.where(a_out > 0, 1.0 / np.maximum(a_out, 1e-12), 0.0)
    c = (b * a + acc * (acc_a * (1.0 - a))) * w
    c = np.where(a_out > 0, c, acc)
    return c, a_out


def forward_turned(leaves, colsT, K, n):
    """The picture the layers give with coverages K, as (3, n). Same numbers
    as merging (n, 3) arrays layer by layer, computed channel by channel
    (see _turned)."""
    acc = np.zeros((3, n), np.float32)
    acc_a = np.zeros(n, np.float32)
    for L, c, k in zip(leaves, colsT, K):
        if c is None:
            continue
        acc, acc_a = _merge_turned(acc, acc_a, c, k,
                                   L.mode if L.mode in ("Normal", "Multiply", "Add", "Screen",
                                                        "Overlay", "Soft Light", "Hard Light") else "Normal")
    # Where the layers don't cover a texel completely, Substance's result is
    # the colour over black (a test project's Blush glow sits on an island with nothing
    # under it: cache 147,0,0 at coverage 0.29, not 255,0,0).
    return acc * acc_a if OVER_BLACK else acc


def forward_flat(leaves, cols, K, n):
    """forward_turned for (n, 3) colours, returning (n, 3)."""
    return np.ascontiguousarray(forward_turned(leaves, turn_cols(cols), K, n).T)


def _step(v, mode, t, k, nz=None):
    """`v` after one blend layer (colour `t`, strength `k` per texel). Texels
    where the strength is 0 are skipped: v + (blend - v) * 0 is v, so the
    result is the same to the last bit as blending every texel. Returns a new
    array, or `v` itself when the layer acts nowhere. `nz`: np.flatnonzero(k)
    if the caller already has it."""
    if nz is None:
        nz = np.flatnonzero(k)
    if len(nz) == 0:
        return v
    if len(nz) * 3 > len(k) * 2:
        return v + (blend(mode, v, t) - v) * k[:, None]
    out = v.copy()
    vs = v[nz]
    out[nz] = vs + (blend(mode, vs, t[nz]) - vs) * k[nz, None]
    return out


def apply_chain(base, chain):
    """Apply blend layers [(mode, colour (m,3), strength (m,))] in order."""
    v = base
    for mode, t, k in chain:
        v = _step(v, mode, t, k)
    return v


def unapply(C, chain, iters=4):
    """Base colour that the blend chain turns into C (fixed point, clipped)."""
    base = C.copy()
    for _ in range(iters):
        base = np.clip(base + (C - apply_chain(base, chain)), 0, 1)
    return base


def _turned(a):
    """(m, 3) colours as a contiguous (3, m) array. numpy runs a step over
    'every texel of one channel' several times faster than one over 'the three
    channels of every texel' (which is what a per-texel strength times an
    (m, 3) array is), and the numbers are the same."""
    return np.ascontiguousarray(a.T)


def _step_turned(v, mode, t, k, nz):
    """_step for (3, m) colours; `nz` = np.flatnonzero(k). Picking out some
    texels costs more per texel than a pass over all of them, so that is only
    done when the layer acts on few (measured: under a tenth)."""
    if len(nz) == 0:
        return v
    if len(nz) * 10 > len(k):
        return v + (blend(mode, v, t) - v) * k
    out = v.copy()
    vs = v[:, nz]
    out[:, nz] = vs + (blend(mode, vs, t[:, nz]) - vs) * k[nz]
    return out


def _fit_strengths(base, Cb, chain_specs, k0, turned=None):
    """Closed-form strengths of the blend layers above `base` (two passes of
    coordinate descent; each layer's effect is ~linear in its strength).

    For layer q the chain is evaluated with q off (F0) and fully on (F1), all
    other layers at their current strength. The part of the chain below q is
    the same for both and for the next q, so it is carried along in `v`
    instead of being recomputed, and layers above q are skipped where their
    strength is 0. Both give exactly the values of running the whole chain
    every time, which is what this did before and what made texture sets with
    many blend layers take hours (the work grew with the square of their
    number, on every texel).

    `turned`: (_turned(Cb), [_turned(colour) for the chain]) when the caller
    has them already (solve calls this once per candidate layer)."""
    CbT, ts = turned if turned is not None else (_turned(Cb), [_turned(t) for _, t in chain_specs])
    v0 = _turned(base)
    ks = [np.array(k, np.float32) for k in k0]
    nzs = [np.flatnonzero(k) for k in ks]
    nb = len(chain_specs)
    for _ in range(2):
        v = v0                             # the chain below q, at the strengths fitted so far
        for q in range(nb):
            bv = blend(chain_specs[q][0], v, ts[q]) - v
            F0 = v
            F1 = v + bv
            for r in range(q + 1, nb):
                mr = chain_specs[r][0]
                F0 = _step_turned(F0, mr, ts[r], ks[r], nzs[r])
                F1 = _step_turned(F1, mr, ts[r], ks[r], nzs[r])
            d = F1 - F0
            num = ((CbT - F0) * d).sum(0) + MULTIPLY_PRIOR * k0[q]
            den = (d * d).sum(0) + MULTIPLY_PRIOR
            ks[q] = np.clip(num / np.maximum(den, 1e-12), 0, 1)
            nzs[q] = np.flatnonzero(ks[q])
            v = v + bv * ks[q]
    return ks


def solve(leaves, cols, K, C, bad, normal, blends, evidence, y0, y1, n, trusted=(), colsT=None):
    """Re-explain the mismatched texels `bad` (bool, n). `colsT`: turn_cols(cols), if the caller has it."""
    idx = np.nonzero(bad)[0]
    Cb = C[idx]
    m = len(idx)
    specs_all = [(leaves[q].mode, cols[q][idx]) for q in blends]
    k0_all = [K[q][idx] for q in blends]
    CbT = _turned(Cb)
    specs_T = [_turned(t) for _, t in specs_all]
    best_cost = np.full(m, np.inf, np.float32)
    best_err = np.full(m, np.inf, np.float32)      # largest channel error of the best single layer
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
            sub = _fit_strengths(cj, Cb, [specs_all[r] for r in above], [k0_all[r] for r in above],
                                 (CbT, [specs_T[r] for r in above]))
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
        best_err = np.where(better, np.abs(Cb - F).max(1), best_err)
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
    pair_err = np.full(m, np.inf, np.float32)
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
                pair_err = np.where(ok, np.abs(d).max(1), pair_err)
                upper = np.where(ok, up, upper)
                lower = np.where(ok, lo, lower)
                alpha = np.where(ok, al, alpha)
                pair_replayed = np.where(ok, replayed, pair_replayed)
    use_pair = (pair_cost + 1e-7 < best_cost) & (upper >= 0)
    # one-layer edits (see ONE_EDIT)
    edit_cost = np.full(m, np.inf, np.float32)
    edit_err = np.full(m, np.inf, np.float32)
    edit_j = np.full(m, -1, np.int64)
    edit_k = np.zeros(m, np.float32)
    if ONE_EDIT:
        Ks = np.stack([np.asarray(K[i][idx], np.float32) for i in range(len(leaves))], 0)
        cs = take_cols(colsT if colsT is not None else turn_cols(cols), idx)

        below = np.zeros(m, np.float32)        # coverage of the Normal layers under j
        under = {}
        for j in sorted(normal):
            under[j] = below.copy()
            below = below + (1 - below) * np.clip(Ks[j], 0, 1)
        parts = []                             # for the joint edit: (layer, texels, effect per unit coverage, top, evidence)
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
            csub = take_cols(cs, sub)
            Cs = CbT[:, sub]
            K1[j] = 0.0
            F0 = forward_turned(leaves, csub, K1, len(sub))
            K1[j] = kmax[sub]
            F1 = forward_turned(leaves, csub, K1, len(sub))
            dv = F1 - F0
            if JOINT_EDIT:
                parts.append((j, sub, (dv / kmax[sub]).astype(np.float32), kmax[sub],
                              ev[sub] if leaves[j].mode != "Normal" else None))
            t = np.clip(((Cs - F0) * dv).sum(0) / np.maximum((dv * dv).sum(0), 1e-8), 0, 1)
            knew = t * kmax[sub]
            K1[j] = knew
            dd = Cs - forward_turned(leaves, csub, K1, len(sub))
            e2 = (dd ** 2).sum(0)
            cost = e2 + EDIT_WEIGHT * np.abs(knew - Ks[j][sub])
            if leaves[j].mode != "Normal":      # blend strength beyond what the replay has
                cost = cost + 4 * EVIDENCE_WEIGHT * np.maximum(knew - ev[sub], 0)
            better = cost < edit_cost[sub]
            w = sub[better]
            edit_cost[w] = cost[better]
            edit_err[w] = np.abs(dd).max(0)[better]
            edit_j[w] = j
            edit_k[w] = knew[better]
    # joint edit (see JOINT_EDIT): the smallest change of all live layers together
    # that explains the texel, from one Gauss-Newton step. The picture is linear
    # in each layer's coverage, so `d` (the effect of a layer per unit coverage)
    # is exact for that layer alone; the step is checked with the real picture.
    joint_cost = np.full(m, np.inf, np.float32)
    joint_max = np.full(m, np.inf, np.float32)     # largest channel error after the joint edit
    Kj = None
    if ONE_EDIT and JOINT_EDIT and parts:
        A = np.zeros((m, 3, 3), np.float32)
        for j, sub, d, _, _ in parts:
            A[sub] += d.T[:, :, None] * d.T[:, None, :]
        A += np.float32(JOINT_RIDGE) * np.eye(3, dtype=np.float32)
        Kj = Ks.copy()
        Kc = Ks.copy()                         # the iterate; Kj keeps each texel's best so far
        dj = CbT - forward_turned(leaves, cs, Kc, m)                    # (3, m)
        joint_err = np.full(m, np.inf, np.float32)
        for _ in range(JOINT_STEPS):
            y = np.linalg.solve(A, dj.T[:, :, None])[:, :, 0]          # (m, 3)
            for j, sub, d, top_k, _ in parts:
                Kc[j][sub] = np.clip(Kc[j][sub] + (d * y[sub].T).sum(0), 0, top_k)
            dj = CbT - forward_turned(leaves, cs, Kc, m)
            e2 = (dj ** 2).sum(0).astype(np.float32)
            better = e2 < joint_err
            if not better.any():
                break
            joint_err = np.where(better, e2, joint_err)
            joint_max = np.where(better, np.abs(dj).max(0), joint_max)
            Kj[:, better] = Kc[:, better]
            if (joint_max <= np.float32(TOL) * 0.5).all():
                break
        change = np.zeros(m, np.float32)
        extra = np.zeros(m, np.float32)
        for j, sub, d, top_k, evb in parts:
            change[sub] += np.abs(Kj[j][sub] - Ks[j][sub])
            if evb is not None:                 # blend strength beyond what the replay has
                extra[sub] += 4 * EVIDENCE_WEIGHT * np.maximum(Kj[j][sub] - evb, 0)
        joint_cost = (joint_err + EDIT_WEIGHT * change + extra).astype(np.float32)
        joint_cost = np.where(change > 0, joint_cost, np.inf)
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
    # a single / pair explanation rewrites the texel; the edits only nudge it
    rewrite_cost = np.minimum(best_cost, np.where(upper >= 0, pair_cost, np.inf))
    if REWRITE_WEIGHT > 0:
        moved = np.zeros(m, np.float32)
        for j in normal:
            moved += np.abs(prop[j] - K[j][idx])
        for q in range(len(blends)):
            moved += np.abs(blend_new[q] - k0_all[q])
        rewrite_cost = rewrite_cost + np.float32(REWRITE_WEIGHT * EDIT_WEIGHT) * moved
    if JOINT_EDIT:
        # An explanation that reproduces the texel beats one that doesn't,
        # whatever it costs; among those that do, the cheapest wins. So paying
        # for rewrites never trades the picture for tidier layers.
        tol = np.float32(TOL)
        rewrite_err = np.where(use_pair, pair_err, best_err)
        costs = [rewrite_cost, np.where(edit_j >= 0, edit_cost, np.inf), joint_cost]
        fine = [(rewrite_err <= tol), (edit_err <= tol) & (edit_j >= 0), (joint_max <= tol) & np.isfinite(joint_cost)]
        any_fine = fine[0] | fine[1] | fine[2]
        costs = [np.where(any_fine & ~f, np.inf, c) for c, f in zip(costs, fine)]
        use_edit = (edit_j >= 0) & (costs[1] <= costs[0]) & np.isfinite(costs[1])
        use_joint = (costs[2] < np.minimum(costs[0], costs[1]))
    else:
        use_edit = (edit_j >= 0) & (edit_cost <= rewrite_cost)
        use_joint = np.zeros(m, bool)
    # a single/pair explanation that would rewrite a trusted layer is dropped:
    # use the one-layer edit if there is one, else keep the replay as it is
    veto = np.zeros(m, bool)
    for j in normal:
        if j in trusted:
            veto |= np.abs(prop[j] - K[j][idx]) > 0.02
    for q, mi in enumerate(blends):
        if mi in trusted:
            veto |= np.abs(blend_new[q] - k0_all[q]) > 0.02
    keep = use_edit | veto | use_joint
    use_edit = (use_edit | veto) & (edit_j >= 0) & ~use_joint
    for q, mi in enumerate(blends):
        K[mi][idx] = np.where(keep, k0_all[q], blend_new[q])
    for j in normal:
        K[j][idx] = np.where(keep, K[j][idx], prop[j])
    for j in np.unique(edit_j[use_edit]):
        sel = use_edit & (edit_j == j)
        K[j][idx[sel]] = edit_k[sel]
    if use_joint.any():
        for j, sub, _, _, _ in parts:
            sel = sub[use_joint[sub]]
            K[j][idx[sel]] = Kj[j][sel]
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
        for y0 in range(0, res, tile_rows(res)):
            y1 = min(res, y0 + tile_rows(res))
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
