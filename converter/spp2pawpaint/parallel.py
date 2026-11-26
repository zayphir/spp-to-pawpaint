"""Evaluate heavy hand-painted layer masks on several CPU cores.

Masks with lots of brush strokes dominate conversion time on hand-painted
projects. They are independent of each other, so a pool of worker processes
evaluates them while the main process carries on with everything else.

Each worker opens the .spp itself but loads only the mesh and the embedded
images, never the layer document (that can take gigabytes for such
projects). A task carries just the layer's mask actions plus the
mesh-part limits (geometry masks) of the layer and its parent folders, so
the worker reproduces exactly what the main process would compute.
Identical masks (colour-variant copies) are submitted once.
"""

import hashlib
import os
import pickle

import numpy as np

# worker-process globals
_W = {}

GEO_KEYS = ("enabledGeometryMask", "geometryMaskType", "enabledMeshList", "enabledMeshDefault")


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


def geo_spec(layer):
    return {k: layer.get(k) for k in GEO_KEYS}


def mask_key(layer, ancestors):
    """Hash of everything a layer's mask depends on (uids ignored)."""
    h = hashlib.blake2b(digest_size=16)

    def feed(o):
        if isinstance(o, dict):
            if o.get("__class") in ("DataStroke", "DataStroke3D"):
                h.update(np.asarray([p.get("position") for p in o.get("points") or []], np.float32).tobytes())
                for k in sorted(o):
                    if k not in ("points", "uid", "label"):
                        feed(k)
                        feed(o[k])
                return
            for k in sorted(o):
                if k in ("uid", "label"):
                    continue
                h.update(k.encode())
                feed(o[k])
        elif isinstance(o, list):
            h.update(b"[%d" % len(o))
            for x in o:
                feed(x)
        else:
            h.update(repr(o).encode())

    feed(layer.get("maskActions"))
    feed([layer.get("maskInitial"), layer.get("maskEnabled")])
    feed([geo_spec(a) for a in ancestors] + [geo_spec(layer)])
    return h.hexdigest()


def _init(path):
    from .geometry import Geometry
    from .sppfile import SppProject
    proj = SppProject(path, load_doc=False)
    _W.clear()
    _W.update(proj=proj, geo=Geometry(proj.scene), builders={}, baked_cache={})


def _baked(baked_urls, usage, res):
    key = (baked_urls.get(usage), res)
    cache = _W["baked_cache"]
    if key not in cache:
        img = None
        url = baked_urls.get(usage)
        if url:
            raw = _W["proj"].resource_image(url)
            if raw is not None:
                from PIL import Image
                chans = []
                for c in range(raw.shape[2] if raw.ndim == 3 else 1):
                    a = raw[..., c] if raw.ndim == 3 else raw
                    mode = None if a.dtype == np.uint8 else "F"
                    im = Image.fromarray(np.ascontiguousarray(a if mode is None else a.astype(np.float32)), mode)
                    chans.append(np.asarray(im.resize((res, res), Image.BILINEAR)))
                img = chans[0] if len(chans) == 1 else np.stack(chans, -1)
        cache[key] = img
    return cache[key]


def _builder(mat, res, baked_urls, lossless=False, engine="cpu"):
    b = _W["builders"].get((mat, res, lossless, engine))
    if b is None:
        from .masks import MaskBuilder
        msgs = []
        b = MaskBuilder(_W["geo"], mat, res, msgs.append)
        b.lossless = lossless
        if engine == "gpu":
            from . import gpu
            b.engine = "gpu" if gpu.available()[0] else "cpu"
        b.baked_map = lambda usage, r=res, u=baked_urls: _baked(u, usage, r)
        b.resource_image = _W["proj"].resource_image
        b._msgs = msgs
        _W["builders"][(mat, res, lossless, engine)] = b
    return b


def _task(payload):
    mat, res, baked_urls, ancestors, layer, lossless, engine = pickle.loads(payload)
    b = _builder(mat, res, baked_urls, lossless, engine)
    b._msgs.clear()
    for a in ancestors:
        b.enter(a)
    b.enter(layer)
    try:
        m = b.layer_mask(layer)
    finally:
        for _ in range(len(ancestors) + 1):
            b.leave()
    out = None if m is None else m.astype(np.float16)
    return out, list(b._msgs)


class MaskPool:
    """Submit masks; results are fetched (blocking) when the converter needs them."""

    def __init__(self, path, workers=None):
        self.path = path
        self.workers = workers or default_workers()
        self.ex = None
        self.futures = {}   # key -> future
        self._start()

    def _start(self):
        import concurrent.futures as cf
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.ex = cf.ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx,
                                         initializer=_init, initargs=(self.path,))

    def submit(self, key, mat, res, baked_urls, ancestors, layer, lossless=False, engine="cpu"):
        key = (mat, res, lossless, engine, key)
        if key in self.futures:
            return key
        if self.ex is None:
            self._start()
        payload = pickle.dumps((mat, res, dict(baked_urls), [geo_spec(a) for a in ancestors], layer, lossless,
                                engine), protocol=pickle.HIGHEST_PROTOCOL)
        self.futures[key] = self.ex.submit(_task, payload)
        return key

    def result(self, key):
        f = self.futures.get(key)
        if f is None:
            return None
        m, msgs = f.result()
        return (None if m is None else m.astype(np.float32)), msgs

    def release(self):
        """Stop the workers and drop their results once a texture set's masks
        are all in. At 4K each worker holds gigabytes (the mesh's surface maps,
        a GPU context) that it will never use again for this texture set, and
        the mask fit that follows needs that memory. The workers start again
        when the next texture set submits masks."""
        ex, self.ex = self.ex, None
        self.futures = {}
        if ex is not None:
            try:
                ex.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass

    def close(self):
        ex, self.ex = self.ex, None
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)
