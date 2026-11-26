"""GPU replay of screen-space brush strokes (OpenGL 4.3 via moderngl).

Same result as strokes2d.ScreenStrokePainter, always at full resolution:

1. Picking: the mesh is rendered from each stroke's camera into a float
   buffer holding world position + triangle id (depth-tested, two-sided,
   geometry-masked parts discarded). Every brush point's surface hit is a
   lookup instead of a ray cast.
2. Coverage: the texture set's triangles are drawn in UV space at the output
   resolution; a fragment shader sums Substance's stamp model (brushmodel.py:
   integrated stamps along each segment, end stamps, flow, measured falloff,
   back-face culling from the stroke camera) from each fragment's
   interpolated world position, as -ln(transmittance). Each batch is drawn
   with MAX blending (parts that share UVs) into a scratch texture and then
   added to the total. Only the touched region is read back.

PawPaint itself requires OpenGL 4.5, so its users have a capable GPU. If no
context can be created, the converter falls back to the CPU path.
"""

import math

import numpy as np

from . import brushmodel as BM
from . import strokes2d as S2

PICK_RES = 2048
SEG_BATCH = 256
import threading

_TLS = threading.local()   # an OpenGL context belongs to the thread that made it


def context():
    if not getattr(_TLS, "tried", False):
        _TLS.tried = True
        _TLS.ctx, _TLS.error = None, None
        try:
            import moderngl
            _TLS.ctx = moderngl.create_standalone_context(require=430)
        except Exception as e:  # no GPU / driver / headless
            _TLS.error = str(e)
    return _TLS.ctx


def available():
    """(True, renderer name) or (False, reason), for the calling thread."""
    ctx = context()
    if ctx is None:
        return False, getattr(_TLS, "error", None) or "no OpenGL 4.3 context"
    return True, ctx.info.get("GL_RENDERER", "GPU")


SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "swiftshader", "software rasterizer",
                      "microsoft basic render", "gdi generic")


def is_software(renderer):
    """True for OpenGL drawn by the processor (e.g. Mesa's llvmpipe on a Linux
    system without a graphics driver). It gives the same result as a graphics
    card but is slower than the CPU engine, so 'auto' doesn't use it."""
    r = str(renderer or "").lower()
    return any(s in r for s in SOFTWARE_RENDERERS)


PICK_VS = """
#version 430
in vec3 in_pos;
uniform mat4 pv;
out vec3 v_pos;
void main() { v_pos = in_pos; gl_Position = pv * vec4(in_pos, 1.0); }
"""
PICK_FS = """
#version 430
in vec3 v_pos;
uniform usampler2D allowed;
uniform int use_allowed;
out vec4 o;
void main() {
    int id = gl_PrimitiveID;
    if (use_allowed == 1) {
        ivec2 t = ivec2(id % 4096, id / 4096);
        if (texelFetch(allowed, t, 0).r == 0u) discard;
    }
    o = vec4(v_pos, float(id + 1));
}
"""
COV_VS = """
#version 430
in vec2 in_uv;
in vec3 in_pos;
in vec3 in_n;
out vec3 v_pos;
flat out vec3 v_n;
void main() {
    v_pos = in_pos; v_n = in_n;
    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
}
"""
COV_FS = """
#version 430
in vec3 v_pos;
flat in vec3 v_n;
layout(std140, binding = 0) uniform Segs { vec4 A[%(N)d]; vec4 B[%(N)d]; vec4 C[%(N)d]; vec4 W[%(N)d]; };
uniform int nseg;
uniform int cull;
uniform float inv2sp;
uniform sampler2D gtab;   // G(p, x): x along width (-1..1), p along height (0..1)
uniform sampler2D stab;   // stamp -ln(1 - flow f(q)), q along width (0..1)
out float f;
float G(float p, float x) {
    vec2 uv = vec2((clamp(x, -1.0, 1.0) + 1.0) * 0.5, clamp(p, 0.0, 1.0));
    vec2 sz = vec2(textureSize(gtab, 0));
    return texture(gtab, (uv * (sz - 1.0) + 0.5) / sz).r;
}
float S(float q) {
    if (q >= 1.0) return 0.0;
    float sz = float(textureSize(stab, 0).x);
    return texture(stab, vec2((q * (sz - 1.0) + 0.5) / sz, 0.5)).r;
}
void main() {
    float total = 0.0;
    for (int i = 0; i < nseg; i++) {
        if (cull == 1 && -dot(v_n, v_pos - C[i].xyz) <= 0.0) continue;
        vec3 a = A[i].xyz; vec3 b = B[i].xyz; vec3 ab = b - a;
        float L = length(ab);
        vec3 rel = v_pos - a;
        float d2 = dot(rel, rel);
        float v = 0.0;
        if (L > 0.0) {
            vec3 u = ab / L;
            float t = dot(rel, u);
            float p = sqrt(max(d2 - t * t, 0.0));
            float r = max(mix(A[i].w, B[i].w, clamp(t / L, 0.0, 1.0)), 1e-12);
            v = (G(p / r, t / r) - G(p / r, (t - L) / r)) * inv2sp;
        }
        if (C[i].w > 0.0) v += C[i].w * S(sqrt(d2) / max(A[i].w, 1e-12));
        if (W[i].x > 0.0) v += W[i].x * S(length(v_pos - b) / max(B[i].w, 1e-12));
        total += v;
    }
    if (total <= 0.0) discard;
    f = total;
}
"""
ADD_VS = """
#version 430
in vec2 in_xy;
void main() { gl_Position = vec4(in_xy, 0.0, 1.0); }
"""
ADD_FS = """
#version 430
uniform sampler2D src;
out float f;
void main() { f = texelFetch(src, ivec2(gl_FragCoord.xy), 0).r; }
"""


class GPUStrokePainter:
    def __init__(self, geo, tris, res, warn=None):
        import moderngl
        self.mgl = moderngl
        self.ctx = context()
        if self.ctx is None:
            raise RuntimeError("no GPU context")
        self.geo = geo
        self.tris = np.asarray(tris)
        self.res = res
        self.warn = warn or (lambda m: None)
        ctx = self.ctx
        # -- picking: whole scene (everything can occlude) ------------------------
        self.pick_prog = ctx.program(vertex_shader=PICK_VS, fragment_shader=PICK_FS)
        vbo = ctx.buffer(geo.V.astype("f4").tobytes())
        ibo = ctx.buffer(geo.T.astype("i4").tobytes())
        self.pick_vao = ctx.vertex_array(self.pick_prog, [(vbo, "3f", "in_pos")], index_buffer=ibo)
        self.pick_tex = ctx.texture((PICK_RES, PICK_RES), 4, dtype="f4")
        self.pick_depth = ctx.depth_renderbuffer((PICK_RES, PICK_RES))
        self.pick_fbo = ctx.framebuffer([self.pick_tex], self.pick_depth)
        n = len(geo.T)
        self._allow_h = (n + 4095) // 4096
        self.allow_tex = ctx.texture((4096, self._allow_h), 1, dtype="u1")
        self.allow_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._allow_key = None
        self._pick_key = None
        # -- coverage: this texture set's triangles in UV space --------------------
        T = geo.T[self.tris]
        uv = geo.UV[T].astype("f4")                       # (n,3,2)
        pos = geo.V[T].astype("f4")                       # (n,3,3)
        fn = geo.normal[self.tris]
        fn = (fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)).astype("f4")
        data = np.concatenate([uv, pos, np.repeat(fn[:, None, :], 3, 1)], -1).reshape(-1, 8)
        self.cov_prog = ctx.program(vertex_shader=COV_VS, fragment_shader=COV_FS % {"N": SEG_BATCH})
        self.cov_vbo = ctx.buffer(data.tobytes())
        self.cov_ibo = ctx.buffer(reserve=len(self.tris) * 3 * 4, dynamic=True)
        self.cov_vao = ctx.vertex_array(self.cov_prog, [(self.cov_vbo, "2f 3f 3f", "in_uv", "in_pos", "in_n")],
                                        index_buffer=self.cov_ibo, index_element_size=4)
        self.seg_ubo = ctx.buffer(reserve=SEG_BATCH * 4 * 16, dynamic=True)
        self.cov_tex = ctx.texture((res, res), 1, dtype="f4")      # total -ln(T)
        self.cov_fbo = ctx.framebuffer([self.cov_tex])
        self.tmp_tex = ctx.texture((res, res), 1, dtype="f4")      # one batch
        self.tmp_fbo = ctx.framebuffer([self.tmp_tex])
        self.add_prog = ctx.program(vertex_shader=ADD_VS, fragment_shader=ADD_FS)
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.add_vao = ctx.vertex_array(self.add_prog, [(quad, "2f", "in_xy")])
        self._tabs = {}
        # per-triangle bounding spheres for batch culling
        self.cen = pos.mean(1).astype(np.float64)
        self.trad = np.sqrt(((pos - self.cen[:, None, :].astype("f4")) ** 2).sum(-1)).max(1).astype(np.float64)
        self.uvpx = uv * res
        self.uv_density = BM.uv_density(geo, self.tris)
        self._tri_ids = self.tris   # global triangle ids, for geometry-mask filtering

    # -- picking ------------------------------------------------------------------------
    def _set_allowed(self, allowed):
        key = None if allowed is None else hash(np.packbits(allowed).tobytes())
        if key == self._allow_key:
            return
        self._allow_key = key
        if allowed is not None:
            buf = np.zeros(4096 * self._allow_h, np.uint8)
            buf[:len(allowed)] = allowed.astype(np.uint8)
            self.allow_tex.write(buf.tobytes())
        self._pick_key = None

    def _render_pick(self, pv, allowed):
        key = (pv.round(7).tobytes(), self._allow_key)
        if key == self._pick_key:
            return
        self._pick_key = key
        ctx = self.ctx
        self.pick_fbo.use()
        ctx.viewport = (0, 0, PICK_RES, PICK_RES)
        ctx.disable(self.mgl.BLEND | self.mgl.CULL_FACE)
        ctx.enable(self.mgl.DEPTH_TEST)
        self.pick_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.pick_prog["pv"].write(pv.T.astype("f4").tobytes())
        self.pick_prog["use_allowed"].value = 0 if allowed is None else 1
        if allowed is not None:
            self.allow_tex.use(0)
            self.pick_prog["allowed"].value = 0
        self.pick_vao.render(self.mgl.TRIANGLES)

    def _hits(self, pv, ndc, allowed, dm):
        pv_cam = pv if dm is None else pv @ np.linalg.inv(dm)
        self._set_allowed(allowed)
        self._render_pick(pv_cam, allowed)
        px = np.clip(((ndc * 0.5 + 0.5) * PICK_RES).astype(np.int64), 0, PICK_RES - 1)
        x0, y0 = px.min(0)
        x1, y1 = px.max(0) + 1
        raw = self.pick_fbo.read(viewport=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                                 components=4, dtype="f4", attachment=0)
        img = np.frombuffer(raw, np.float32).reshape(int(y1 - y0), int(x1 - x0), 4)
        s = img[px[:, 1] - y0, px[:, 0] - x0]
        valid = s[:, 3] > 0.5
        p = s[:, :3].astype(np.float64)
        w = (np.c_[p, np.ones(len(p))] @ pv_cam.T)[:, 3]
        tri = np.where(valid, np.round(s[:, 3]).astype(np.int64) - 1, -1)
        return p, np.abs(w), valid, tri

    # -- public ---------------------------------------------------------------------------------
    def cover(self, strokes, brush, hardness, allowed=None, cull=True, flow=1.0, spacing=0.15,
              space=1, density=1.0):
        from .masks import RegionCover
        size = float(brush.get("brushSize", 0.01))
        emitter = brush.get("stampsEmitter") or {}
        press_on = bool(emitter.get("pressureSizeEnabled", True))
        press_min = float(emitter.get("pressureSizeMinimum", 0.0))
        center = (self.geo.V.min(0) + self.geo.V.max(0)) / 2
        segs = []
        hit_tris = []
        for s in strokes:
            pts = s.get("points") or []
            if not pts:
                continue
            P = S2._mat(s["projectionMatrix"])
            V = S2._mat(s["viewMatrix"])
            pv = P @ V
            try:
                cam = np.linalg.inv(V)[:3, 3]
            except np.linalg.LinAlgError:
                continue
            ortho = abs(P[3, 3] - 1.0) < 1e-6
            fy = abs(P[1, 1]) if abs(P[1, 1]) > 1e-9 else 1.0
            xy = np.array([p["position"] for p in pts], np.float64)
            pr = S2.stroke_pressure(s)
            scale = np.ones(len(xy))
            if press_on and len(pr) == len(xy):
                scale = press_min + (1 - press_min) * np.clip(pr, 0, 1)
            ndc = xy * 2 - 1
            r_ndc = max(S2.thin_step_ndc(size, cam, center, fy, space, density), 1e-5)
            step = max(r_ndc * 0.25, 1e-4)   # denser than the CPU fast path
            travelled = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ndc, axis=0), axis=1))]
            bucket = np.floor(travelled / step).astype(np.int64)
            keep = np.r_[0, np.nonzero(np.diff(bucket))[0] + 1]
            if keep[-1] != len(ndc) - 1:
                keep = np.r_[keep, len(ndc) - 1]
            ndc, scale = ndc[keep], scale[keep]
            mats = [None]
            if s.get("duplicateCount", 0) >= 1:
                mats.append(S2._mat(s["duplicateMatrix"]))
            for dm in mats:
                p, w, valid, ti = self._hits(pv, ndc, allowed, dm)
                dens = S2.local_density(self.geo, ti, density) if space == 2 else density
                r = S2.stroke_radius(size, scale, w, fy, ortho, space, dens)
                c = cam if dm is None else (dm @ np.append(cam, 1.0))[:3]
                segs.extend(S2.stroke_segments(p, r, valid, c, spacing))
                hit_tris.append(np.asarray(ti)[np.asarray(valid, bool)])
        if S2.WRAP_CONNECTED and hit_tris:
            allowed = S2.wrap_reach(self.geo, np.unique(np.concatenate(hit_tris)), segs, allowed)
        return self.cover_segments(segs, hardness, flow, spacing, allowed, cull)

    def cover_segments(self, segs, hardness, flow, spacing, allowed=None, cull=True):
        """Coverage of 3D stamp segments (from strokes2d.stroke_segments)."""
        from .masks import RegionCover
        if not segs:
            return None
        A, B, RA, RB, C, WA, WB = (np.concatenate([x[k] for x in segs]) for k in range(7))
        rmed = float(np.median(np.r_[RA, RB]))
        tab = BM.table(flow, BM.edge_hardness(hardness, rmed * self.uv_density * self.res))
        return self._coverage(A, B, RA, RB, C, WA, WB, tab, spacing, allowed, cull, RegionCover)

    def _table_textures(self, tab):
        key = (tab.flow, tab.h)
        t = self._tabs.get(key)
        if t is None:
            gt = self.ctx.texture((tab.G.shape[1], tab.G.shape[0]), 1, tab.G.tobytes(), dtype="f4")
            st = self.ctx.texture((len(tab.g), 1), 1, tab.g.tobytes(), dtype="f4")
            for x in (gt, st):
                x.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
                x.repeat_x = x.repeat_y = False
            if len(self._tabs) > 16:
                for old in self._tabs.values():
                    old[0].release()
                    old[1].release()
                self._tabs.clear()
            t = self._tabs[key] = (gt, st)
        return t

    def _coverage(self, A, B, RA, RB, C, WA, WB, tab, spacing, allowed, cull, RegionCover):
        ctx, mgl = self.ctx, self.mgl
        self.cov_fbo.scissor = None
        self.tmp_fbo.scissor = None
        self.cov_fbo.use()
        ctx.viewport = (0, 0, self.res, self.res)
        ctx.disable(mgl.DEPTH_TEST | mgl.CULL_FACE)
        self.cov_fbo.clear(0.0)
        gt, st = self._table_textures(tab)
        gt.use(1)
        st.use(2)
        self.cov_prog["gtab"].value = 1
        self.cov_prog["stab"].value = 2
        self.cov_prog["inv2sp"].value = 1.0 / (2.0 * max(float(spacing), 1e-3))
        self.cov_prog["cull"].value = 1 if cull else 0
        self.seg_ubo.bind_to_uniform_block(0)
        tri_ok = None if allowed is None else allowed[self._tri_ids]
        rmax = float(max(RA.max(), RB.max()))
        cell = max(rmax, 1e-6) * 2
        order = np.argsort((A[:, 0] / cell).astype(np.int64) * 1000003 + (A[:, 1] / cell).astype(np.int64) * 7919
                           + (A[:, 2] / cell).astype(np.int64))
        lo_px = np.array([self.res, self.res])
        hi_px = np.array([0, 0])
        drew = False
        for s0 in range(0, len(order), SEG_BATCH):
            sel = order[s0:s0 + SEG_BATCH]
            a, b, ra, rb, cc = A[sel], B[sel], RA[sel], RB[sel], C[sel]
            r = max(ra.max(), rb.max())
            bmin = np.minimum(a, b).min(0) - r
            bmax = np.maximum(a, b).max(0) + r
            near = np.all((self.cen + self.trad[:, None] >= bmin) & (self.cen - self.trad[:, None] <= bmax), axis=1)
            if tri_ok is not None:
                near &= tri_ok
            k = np.nonzero(near)[0]
            if len(k) == 0:
                continue
            idx = (k[:, None] * 3 + np.arange(3)).astype(np.int32).ravel()
            self.cov_ibo.write(idx.tobytes())
            block = np.zeros((4, SEG_BATCH, 4), np.float32)
            n = len(sel)
            block[0, :n, :3], block[0, :n, 3] = a, ra
            block[1, :n, :3], block[1, :n, 3] = b, rb
            block[2, :n, :3], block[2, :n, 3] = cc, WA[sel]
            block[3, :n, 0] = WB[sel]
            self.seg_ubo.write(block.tobytes())
            self.cov_prog["nseg"].value = n
            uvk = self.uvpx[k].reshape(-1, 2)
            b_lo = np.clip(np.floor(uvk.min(0)).astype(int) - 1, 0, self.res)
            b_hi = np.clip(np.ceil(uvk.max(0)).astype(int) + 1, 0, self.res)
            if (b_hi <= b_lo).any():
                continue
            box = (int(b_lo[0]), int(b_lo[1]), int(b_hi[0] - b_lo[0]), int(b_hi[1] - b_lo[1]))
            # this batch -> scratch (MAX over triangles that share UVs) ...
            self.tmp_fbo.scissor = box
            self.cov_fbo.scissor = box
            self.tmp_fbo.use()
            self.tmp_fbo.clear(0.0)
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.ONE, mgl.ONE
            ctx.blend_equation = mgl.MAX
            gt.use(1)
            st.use(2)
            self.cov_vao.render(mgl.TRIANGLES, vertices=len(idx))
            # ... then added to the total
            self.cov_fbo.use()
            ctx.blend_equation = mgl.FUNC_ADD
            self.tmp_tex.use(0)
            self.add_prog["src"].value = 0
            self.add_vao.render(mgl.TRIANGLE_STRIP)
            lo_px = np.minimum(lo_px, b_lo)
            hi_px = np.maximum(hi_px, b_hi)
            drew = True
        ctx.disable(mgl.BLEND)
        ctx.blend_equation = mgl.FUNC_ADD
        self.cov_fbo.scissor = None
        self.tmp_fbo.scissor = None
        self.cov_fbo.use()
        if not drew:
            return None
        x0, y0 = np.clip(lo_px, 0, self.res)
        x1, y1 = np.clip(hi_px, 0, self.res)
        if x1 <= x0 or y1 <= y0:
            return None
        raw = self.cov_fbo.read(viewport=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), components=1, dtype="f4")
        img = np.frombuffer(raw, np.float32).reshape(int(y1 - y0), int(x1 - x0))[::-1]
        drawn = img > 0
        if not drawn.any():
            return None
        # The box read back is the bounding box of every triangle near the
        # strokes, and most of it is empty. Keep the part that was drawn on, and
        # turn -ln(T) into coverage only there: where nothing was drawn it is
        # exactly 0 either way (1 - exp(-0)), so the mask is the same.
        ys = np.nonzero(drawn.any(1))[0]
        xs = np.nonzero(drawn.any(0))[0]
        r0, r1, c0, c1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
        img, drawn = img[r0:r1, c0:c1], drawn[r0:r1, c0:c1]
        cov = np.zeros(img.shape, np.float32)
        cov[drawn] = np.clip(BM.coverage(img[drawn]).astype(np.float32), 0, 1)
        # GL rows start at v = 0; our images start at the top (v = 1)
        return RegionCover(self.res - int(y1) + r0, int(x0) + c0, cov, self.res)
