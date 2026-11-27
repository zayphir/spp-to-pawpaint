"""Command line / drag-and-drop entry point.

    SppToPawPaint MyProject.spp [--out FOLDER] [--resolution 4096] ...

Writes a folder PawPaint's "Substance Importer" plugin reads:

    MyProject (PawPaint import)/
        import.json         what to import: mesh, texture sets, resolution
        model.glb           the mesh, rebuilt from the .spp
        report.txt          what was converted and what could not be
        materials/<Texture set>/manifest.json + maps/*.png (+ baked/*.png)

Started with no arguments, or with a file dropped on the exe, it opens a
small window instead (see gui.py).
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback

import numpy as np

from . import __version__
from . import exact as X
from . import install as I
from . import vram
from .geometry import Geometry
from .glb import write_glb
from .layers import MaterialConverter
from .sppfile import SppProject

ALLOWED_RES = (512, 1024, 2048, 4096)
UNIT_FACTORS = {"cm": 0.01, "m": 1.0, "mm": 0.001, "in": 0.0254}


def _folder_name(name, taken):
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .") or "Material"
    out, n = base, 2
    while out.lower() in taken:
        out = "%s (%d)" % (base, n)
        n += 1
    taken.add(out.lower())
    return out


def pick_resolution(materials, requested=None):
    if requested:
        return min(ALLOWED_RES, key=lambda r: abs(r - requested))
    best = 0
    for m in materials:
        lg = (m.get("resolutionLog2") or [11, 11])[0]
        best = max(best, 2 ** int(lg))
    best = min(max(best, 512), 4096)
    return min(ALLOWED_RES, key=lambda r: abs(r - best))


def native_size(material):
    """A texture set's own size in Substance, as a size PawPaint has (512 to 4096)."""
    lg = (material.get("resolutionLog2") or [11, 11])[0]
    own = min(max(2 ** int(lg), 512), 4096)
    return min(ALLOWED_RES, key=lambda r: abs(r - own))


def set_sizes(materials, requested=None):
    """The size each texture set is converted at: its own. A resolution that
    was asked for is the largest any of them gets; a set that is smaller in
    Substance keeps its size, since enlarging it adds nothing. (PawPaint 1.28
    gives each material its own texture size. Older versions open the project
    at the largest and scale the smaller images up themselves.)"""
    top = min(ALLOWED_RES, key=lambda r: abs(r - requested)) if requested else None
    return [min(native_size(m), top) if top else native_size(m) for m in materials]


def convert(spp_path, out_dir=None, resolution=None, layers=True, flat=True, baked=True, check=False,
            mesh_units="cm", substance_export=None, pawpaint_dir=None, handoff=True,
            log=print, progress=None, workers=0, engine="auto", detail="fast"):
    """Convert one .spp. Returns the output folder.

    progress(done, total, label) is called per texture set, if given."""
    t0 = time.time()
    spp_path = os.path.abspath(spp_path)
    stem = os.path.splitext(os.path.basename(spp_path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(spp_path), "%s (PawPaint import)" % stem)
    os.makedirs(out_dir, exist_ok=True)
    # A conversion left here by an earlier run is being overwritten from now on.
    for old in ("import.json", "report.txt"):
        try:
            os.remove(os.path.join(out_dir, old))
        except OSError:
            pass
    log = RunLog(out_dir, log)
    try:
        return _convert(spp_path, out_dir, resolution, layers, flat, baked, check, mesh_units, substance_export,
                        pawpaint_dir, handoff, log, progress, workers, engine, detail, t0)
    except BaseException:
        log.detail(traceback.format_exc())
        raise
    finally:
        log.close()


class RunLog:
    """The run's log function, which also keeps everything in <output folder>/log.txt.
    The file is written line by line, so it is there to send along when a run
    is slow or fails. log.detail(text) goes to the file only."""

    def __init__(self, out_dir, log):
        self.log = log
        self.t0 = time.time()
        self.path = os.path.join(out_dir, "log.txt")
        try:
            self.f = open(self.path, "w", encoding="utf-8")
        except OSError:
            self.f = None
        self.detail("SppToPawPaint %s, Python %s, %s" % (__version__, sys.version.split()[0], _system_line()))

    def detail(self, text):
        if self.f is None:
            return
        try:
            for line in str(text).splitlines() or [""]:
                self.f.write("[%8.1f s] %s\n" % (time.time() - self.t0, line))
            self.f.flush()
        except (OSError, ValueError):
            pass

    def __call__(self, msg):
        self.detail(msg)
        self.log(msg)

    def close(self):
        if self.f is not None:
            try:
                self.f.close()
            except OSError:
                pass
            self.f = None


def _system_line():
    """Operating system, CPU cores and memory, for the log."""
    import platform
    parts = ["%s %s" % (platform.system(), platform.release()), "%d CPU cores" % (os.cpu_count() or 1)]
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                      "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                      "ullAvailExtendedVirtual")]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                parts.append("%.0f GB memory" % (ms.ullTotalPhys / 2.0 ** 30))
        else:
            parts.append("%.0f GB memory" % (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2.0 ** 30))
    except Exception:
        pass
    return ", ".join(parts)


def _convert(spp_path, out_dir, resolution, layers, flat, baked, check, mesh_units, substance_export,
             pawpaint_dir, handoff, log, progress, workers, engine, detail, t0):
    log.detail("options: resolution %s, layers %s, result layer %s, baked maps %s, engine %s, detail %s, workers %s, "
               "compare %s, mesh units %s" % (resolution or "auto", layers, flat, baked, engine, detail, workers or "all",
                                              check, mesh_units))
    log("Reading %s" % os.path.basename(spp_path))
    proj = SppProject(spp_path)
    warnings = []
    rep = proj.parse_report
    if rep["unknown_types"] or rep["errors"]:
        warnings.append("This .spp was saved by a newer Substance version with data this converter doesn't "
                        "know yet (%d spots); some layer settings may be missing. Please report it."
                        % max(len(rep["errors"]), 1))
    if int(proj.doc.get("colorManagementWorkflow", 0) or 0) != 0:
        warnings.append("The project uses OCIO/ACE colour management; colours may look slightly different.")
    scene = proj.scene
    geo = Geometry(scene)
    mats = [m for m in proj.materials if m]
    log("  %d texture sets, %d mesh parts, %d triangles" % (len(mats), len(scene.parts), len(geo.T)))
    if any(m.get("useUVTiles") for m in mats):
        warnings.append("This is a UDIM project; only UV tile 1001 is converted for now.")

    # mesh, back at its original size when Substance recorded it
    scale = 1.0
    if mesh_units != "raw" and scene.normalize_scale:
        scale = UNIT_FACTORS.get(mesh_units, 0.01) / scene.normalize_scale
    log("Writing mesh: model.glb" + ("" if scale == 1.0 else " (original size, %s)" % mesh_units))
    write_glb(scene, os.path.join(out_dir, "model.glb"), scale)

    # exact layers from the Substance companion exporter, if present
    export_dir = X.find_export(spp_path, substance_export)
    export = None
    if export_dir:
        try:
            export = X.load_export(export_dir)
            log("Using exact layers from %s" % os.path.basename(export_dir))
        except Exception as e:
            warnings.append("Substance export folder could not be read (%s); rebuilt layers used instead." % e)

    # Processing engine: the GPU when one works (always full detail), else the CPU.
    use = "cpu"
    if layers and engine in ("auto", "gpu"):
        from . import gpu
        ok, name = gpu.available()
        if ok and engine == "auto" and gpu.is_software(name):
            log("  engine: CPU (the only OpenGL here is software, %s, which is slower)" % name)
        elif ok:
            use = "gpu"
            log("  engine: GPU (%s), full detail" % name)
        else:
            log("  engine: CPU (%s)" % ("no usable GPU: " + name if engine == "gpu" else "no usable GPU"))
            if engine == "gpu":
                warnings.append("The GPU engine was requested but no usable GPU was found (%s); the CPU was used." % name)
        if gpu.tried_text():
            log("  OpenGL asked for: %s" % gpu.tried_text())
    if hasattr(log, "detail") and getattr(proj, "doc_version", None) is not None:
        log.detail("  document version 0x%x, channels named by %s" % (
            proj.doc_version, "bit number (Substance 12.2 or newer)" if proj.channel_ids_are_indices else "bit flag"))
    lossless = detail == "lossless"
    if use == "cpu" and layers:
        log("  engine: CPU, %s" % ("full detail (lossless, slower)" if lossless else "fast detail"))

    # The size: the project's own, unless its layers can't be imported at that
    # size on this PC's graphics card (see vram.py).
    sizes, memory, size_note = _plan_sizes(mats, scene, resolution, use, layers, log)
    in_model = [s for m, s in zip(mats, sizes) if (m.get("sceneMaterialName") or "Material") in scene.material_names]
    res = max(in_model or sizes or [pick_resolution(mats, resolution)])     # the project's size: its largest material
    if len(set(sizes)) > 1:
        log("  importing at up to %dx%d, each texture set at its own size (%s)"
            % (res, res, ", ".join("%s %d" % (m.get("sceneMaterialName") or "Material", s)
                                   for m, s in zip(mats, sizes))))
    else:
        log("  importing at %dx%d" % (res, res))

    # Heavy brush masks run in worker processes: on the CPU route one per core,
    # on the GPU route a few (each with its own GPU context) so the CPU-side
    # work around the GPU also runs in parallel.
    pool = None
    if layers and workers != 1 and export is None:
        from . import layers as LY, parallel
        from .settings import cpu_workers, gpu_workers
        heavy = sum(1 for m in mats for _ in _heavy_masks(m, LY))
        if heavy >= 2:
            try:
                n = gpu_workers(workers) if use == "gpu" else cpu_workers(workers, res, lossless)
                pool = parallel.MaskPool(spp_path, n)
                log("  %d hand-painted masks found; using %d worker processes%s"
                    % (heavy, pool.workers, " with the GPU" if use == "gpu" else " (CPU cores)"))
            except Exception as e:  # fall back to one core
                log("  could not start worker processes (%s); using one core" % e)
                pool = None

    # Big mask fits run in their own worker processes (see refine.py).
    from . import refine
    refine.WORKERS = max(int(workers or 0), 0)

    taken = set()
    entries = []

    def save_progress():
        """Leave what is finished so far importable, in case the run is stopped."""
        left = len(mats) - len(entries)
        if left > 0:
            note = ["The conversion was stopped early: %d of %d texture sets are in this folder. "
                    "Convert again for the rest." % (len(entries), len(mats))]
            _write_import(spp_path, out_dir, res, mesh_units, scale, entries, note + warnings, partial=len(mats),
                          size_note=size_note)

    try:
        _convert_sets(proj, geo, scene, mats, sizes, out_dir, entries, warnings, layers, flat, baked, check,
                      export, export_dir, pool, log, progress, taken, use, lossless, save_progress)
    finally:
        if pool is not None:
            pool.close()
        refine.close_pool()
    if progress:
        progress(len(mats), len(mats), "done")
    return _finish(proj, spp_path, out_dir, res, mesh_units, scale, entries, warnings, handoff, pawpaint_dir, log, t0,
                   memory, size_note)


def _plan_sizes(mats, scene, requested, use, layers, log):
    """(the size of each texture set, video memory of this PC, note). Every
    texture set is converted at its own size. Says what PawPaint will need for
    the layers; when no size was asked for and they can't all be imported on
    this graphics card, the sets that need the most are converted a size
    smaller until they can, which also converts several times faster."""
    sizes = set_sizes(mats, requested)
    if not layers:
        return sizes, None, None
    try:
        memory = vram.video_memory(use_opengl=(use == "gpu"))
    except Exception:
        memory = None
    # the texture sets the model uses, in the order the plugin imports them (the model's materials)
    order = list(scene.material_names)

    def name(i):
        return mats[i].get("sceneMaterialName") or "Material"
    used = sorted((i for i in range(len(mats)) if name(i) in order), key=lambda i: order.index(name(i))) \
        or list(range(len(mats)))

    def needs(at):
        return [vram.guess_mb(mats[i], at[k]) for k, i in enumerate(used)]

    own = [sizes[i] for i in used]
    if memory:
        log("  graphics card: %s" % vram.describe(memory))
    top = max(own)
    log("  PawPaint will need roughly %s of video memory for these layers at %s (%s a size smaller)"
        % (vram.gb(sum(needs(own))), "%dx%d" % (top, top) if len(set(own)) == 1 else "their own sizes, up to %dx%d" % (top, top),
           vram.gb(sum(needs([s // 2 for s in own])))))
    budget = vram.budget_mb(memory)
    if budget is None or vram.fits(needs(own), budget):
        return sizes, memory, None
    if requested:
        log("  That is more than this graphics card can give it. The PawPaint plugin will then import what doesn't "
            "fit at a smaller size; converting at that size (Resolution: Auto) would also be several times faster.")
        return sizes, memory, None
    fitted = vram.auto_sizes(own, lambda k, s: vram.guess_mb(mats[used[k]], s), budget)
    lowered = [(used[k], own[k], fitted[k]) for k in range(len(used)) if fitted[k] != own[k]]
    if not lowered:
        return sizes, memory, None
    for i, _, new in lowered:
        sizes[i] = new
    what = ", ".join("'%s' at %dx%d instead of %dx%d" % (name(i), new, new, old, old) for i, old, new in lowered)
    note = ("Converted %s: at %s own size PawPaint needs roughly %s of video memory for these layers and this "
            "graphics card can give it about %s, so it could not import them. %sTo convert at the project's own "
            "sizes anyway, choose a resolution instead of Auto."
            % (what, "its" if len(lowered) == 1 else "their", vram.gb(sum(needs(own))), vram.gb(budget),
               "The other texture sets keep their sizes. " if len(lowered) < len(used) else ""))
    log("  " + note)
    return sizes, memory, note


def _heavy_masks(m, LY):
    st = [((m.get("stacks") or [{}])[0].get("stack") or {}).get("items") or []]
    while st:
        for L in st.pop():
            if not L:
                continue
            if L.get("maskEnabled") and L.get("maskActions") and                     LY._stroke_points(L["maskActions"]) > LY.PARALLEL_MIN_POINTS:
                yield L
            if L.get("subStack"):
                st.append(L["subStack"].get("items") or [])


def _convert_sets(proj, geo, scene, mats, sizes, out_dir, entries, warnings, layers, flat, baked, check,
                  export, export_dir, pool, log, progress, taken, engine="cpu", lossless=False, saved=None):
    """`sizes`: the size of each texture set (see set_sizes), or one size for all."""
    if not isinstance(sizes, (list, tuple)):
        sizes = [sizes] * len(mats)
    for i, m in enumerate(mats):
        res = sizes[i]
        name = m.get("sceneMaterialName") or "Material"
        if progress:
            progress(i, len(mats), name)
        folder = _folder_name(name, taken)
        scene_index = scene.material_names.index(name) if name in scene.material_names else None
        if scene_index is None:
            warnings.append("%s: this texture set isn't used by any part of the model in the .spp; "
                            "it is skipped when importing onto that model" % name)
        log("Texture set '%s'%s" % (name, "" if len(set(sizes)) == 1 else " (%dx%d)" % (res, res)))
        if hasattr(log, "detail"):
            try:
                from .settings import available_memory_gb
                log.detail("  free memory: %.1f GB" % available_memory_gb())
            except Exception:
                pass
        t = time.time()
        entry = {"name": name, "bundle": "materials/%s" % folder, "layers": 0, "match": None,
                 "nativeResolution": 2 ** int((m.get("resolutionLog2") or [11])[0]),
                 "resolution": res, "inMesh": scene_index is not None}
        try:
            mc = MaterialConverter(proj, geo, m, scene_index, res, os.path.join(out_dir, "materials", folder), log, pool,
                                   engine=engine, lossless=lossless)
            exact_specs = None
            if export:
                ts = next((t_ for t_ in export.get("textureSets", []) if t_.get("name") == name), None)
                if ts:
                    exact_specs = X.build_specs(export_dir, ts, mc.mapdir, res, mc._warn)
            manifest = mc.convert(include_layers=layers, include_flat=flat, include_baked=baked, exact=exact_specs)
            score = None
            if check and layers and exact_specs is None and mc.mb is not None and mc.cache_albedo is not None:
                try:
                    comp = mc.composite_albedo()
                    cov = mc.coverage() > 0.99
                    ref = mc.cache_albedo.astype(np.float32) / 255.0
                    if cov.any():
                        score = float((np.abs(comp - ref).max(-1)[cov] < 10 / 255).mean())
                except MemoryError:
                    score = None
            entry.update(layers=len(manifest["layers"]), match=score, exact=exact_specs is not None,
                         baked=sorted(mc.baked_files), videoMemoryMB=int(round(vram.need_mb(manifest, res))))
            how = ("exact layers from Substance" if exact_specs is not None else
                   "" if score is None else "rebuilt layers match Substance on %.1f%% of pixels" % (score * 100))
            log("  %d layers%s (%s)" % (entry["layers"], ", " + how if how else "", _took(time.time() - t)))
            warnings.extend(mc.warnings)
            del mc
        except Exception as e:
            warnings.append("%s: could not be converted (%s)" % (name, e))
            log("  FAILED: %s" % e)
            traceback.print_exc()
            if hasattr(log, "detail"):
                log.detail(traceback.format_exc())
        entries.append(entry)
        # A finished texture set leaves reference cycles holding gigabytes of
        # images at 4K; collect them now so the next set's workers get the memory.
        mc = None
        gc.collect()
        if saved:
            saved()


def _took(seconds):
    return "%.1fs" % seconds if seconds < 120 else "%d min %02d s" % (seconds // 60, seconds % 60)


def _write_import(spp_path, out_dir, res, mesh_units, scale, entries, warnings, partial=0, size_note=None):
    """import.json and report.txt. `partial`: the number of texture sets the
    project has, when the run isn't finished yet (the plugin then says so).
    `size_note`: why the size is not the project's own, if it isn't."""
    info = {"format": "spp2pawpaint-import", "version": 1, "converter": __version__,
            "source": os.path.basename(spp_path), "resolution": res, "mesh": "model.glb",
            "meshUnits": mesh_units if scale != 1.0 else "raw",
            "materials": entries, "warnings": warnings}
    if partial:
        info["partial"] = True
        info["setsTotal"] = int(partial)
    if size_note:
        info["sizeNote"] = size_note
    # written next to it and swapped in, so a reader never sees half a file
    path = os.path.join(out_dir, "import.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    os.replace(path + ".tmp", path)
    _write_report(out_dir, spp_path, res, entries, warnings, size_note)


def _set_needs(entries, order=None):
    """Video memory (MB) PawPaint needs for each converted texture set that the
    model uses; in the order of the model's materials `order`, if given."""
    used = [e for e in entries if e.get("videoMemoryMB") and e.get("inMesh", True)]
    if order:
        used.sort(key=lambda e: order.index(e["name"]) if e["name"] in order else len(order))
    return [e["videoMemoryMB"] for e in used]


def _finish(proj, spp_path, out_dir, res, mesh_units, scale, entries, warnings, handoff, pawpaint_dir, log, t0,
            memory=None, size_note=None):
    _write_import(spp_path, out_dir, res, mesh_units, scale, entries, warnings, size_note=size_note)
    needs = _set_needs(entries, list(proj.scene.material_names))
    proj.close()

    if handoff:
        _handoff(out_dir, spp_path, entries, pawpaint_dir, log)
    if needs:
        budget = vram.budget_mb(memory)
        own = {e.get("resolution", res) for e in entries if e.get("videoMemoryMB") and e.get("inMesh", True)}
        line = "PawPaint needs %s of video memory for these layers at %s" % (
            vram.gb(sum(needs)), "%dx%d" % (res, res) if own <= {res} else "their own sizes (up to %dx%d)" % (res, res))
        if budget is not None and not vram.fits(needs, budget):
            line += (" and this graphics card can give it about %s, so the plugin will import what doesn't fit at a "
                     "smaller size (%s with every texture set a size smaller)" % (vram.gb(budget), vram.gb(sum(needs) / 4.0)))
        log(line + ".")
    log("Done in %s -> %s" % (_took(time.time() - t0), out_dir))
    if warnings:
        log("%d notes about things that could not be converted exactly, see report.txt" % len(warnings))
    return out_dir


def _write_report(out_dir, spp_path, res, entries, warnings, size_note=None):
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("SppToPawPaint %s - %s\n\n" % (__version__, os.path.basename(spp_path)))
        f.write("Import resolution: %d\n" % res)
        if size_note:
            f.write("%s\n" % size_note)
        f.write("\n")
        for e in entries:
            how = ("exact layers from Substance" if e.get("exact") else
                   "" if e.get("match") is None else "rebuilt layers match Substance on %.1f%% of pixels" % (e["match"] * 100))
            size = e.get("resolution")
            if size and size != res:
                how = ("%dx%d   " % (size, size) + how).rstrip()
            f.write("%-28s %3d layers   %s\n" % (e["name"], e["layers"], how))
            if e.get("baked"):
                f.write("%-28s baked maps: %s (in %s/baked/)\n" % ("", ", ".join(e["baked"]), e["bundle"]))
        needs = _set_needs(entries)
        if needs:
            f.write("\nVideo memory PawPaint needs to import the layers (it keeps every layer on the graphics card):\n")
            for e in entries:
                if e.get("videoMemoryMB") and e.get("inMesh", True):
                    own = e.get("resolution", res)
                    f.write("  %-26s %8s at %d, %s at %d\n" % (e["name"], vram.gb(e["videoMemoryMB"]), own,
                                                                vram.gb(e["videoMemoryMB"] / 4.0), own // 2))
            same = {e.get("resolution", res) for e in entries if e.get("videoMemoryMB") and e.get("inMesh", True)} <= {res}
            if same:
                f.write("  %-26s %8s at %d, %s at %d\n" % ("all of them", vram.gb(sum(needs)), res,
                                                            vram.gb(sum(needs) / 4.0), res // 2))
            else:
                f.write("  %-26s %8s at these sizes, %s a size smaller\n" % ("all of them", vram.gb(sum(needs)),
                                                                             vram.gb(sum(needs) / 4.0)))
            f.write("  When a texture set doesn't fit, the PawPaint plugin imports it at the next smaller size\n"
                    "  instead (its 'Size' option), which keeps every layer. Before PawPaint 1.28 it has to do\n"
                    "  that for the whole project.\n")
        f.write("\nNotes (%d):\n" % len(warnings))
        for w in warnings:
            f.write("  - %s\n" % w)
        f.write("\nEvery texture set gets a 'Substance Result' layer on top that is an exact copy of what\n"
                "Substance shows. Hide it to see and edit the rebuilt layers underneath.\n"
                "Baked mesh maps (normal, ID, curvature, ...) are in each texture set's 'baked' folder;\n"
                "the baked normal is also imported as the 'Baked Normal (from Substance)' layer.\n")


def _handoff(out_dir, spp_path, entries, pawpaint_dir, log):
    pp_dir = I.find_pawpaint(pawpaint_dir)
    shown = I.pawpaint_path(pp_dir, out_dir)  # differs when PawPaint runs through Proton
    if pp_dir:
        try:
            I.write_pending(pp_dir, out_dir, spp_path, [e["name"] for e in entries])
            log("PawPaint will offer this conversion in Plugins > Substance Importer.")
        except OSError as e:
            log("Could not leave a note for the PawPaint plugin (%s)." % e)
    if shown != os.path.abspath(out_dir):
        log("PawPaint runs through Proton here. Inside PawPaint the folder is: %s" % shown)
    if I.copy_to_clipboard(shown):
        log("The folder path is also on your clipboard.")


def maybe_install_plugin(pawpaint_dir=None, assume_yes=False, log=print, ask=None):
    """Install/update the PawPaint plugin if needed. Returns status text."""
    pp_dir = I.find_pawpaint(pawpaint_dir)
    if not pp_dir:
        return ("PawPaint not found (install it from Steam, or give its folder with --pawpaint-dir "
                "or the PAWPAINT_DIR environment variable)")
    status = I.plugin_status(pp_dir)
    if status == "current":
        return "PawPaint plugin is up to date"
    if not assume_yes:
        if ask is None:
            return "PawPaint plugin is %s" % status
        if not ask("The PawPaint plugin is %s. Install it now?" % status):
            return "PawPaint plugin is %s (skipped)" % status
    try:
        I.install_plugin(pp_dir, log)
        return "PawPaint plugin installed - restart PawPaint once to load it"
    except PermissionError:
        return ("No permission to write to %s. Copy the files from 'PawPaint scripts' into "
                "its 'scripts' folder by hand." % os.path.join(pp_dir, "scripts"))


def _console_ask(question):
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return True
        return input(question + " [Y/n] ").strip().lower() in ("", "y", "yes")
    except EOFError:
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="SppToPawPaint",
                                 description="Convert a Substance 3D Painter .spp project for PawPaint.")
    ap.add_argument("spp", nargs="*", help=".spp project file(s)")
    ap.add_argument("--out", help="output folder (default: next to the .spp)")
    ap.add_argument("--resolution", type=int, help="512, 1024, 2048 or 4096 (default: largest texture set)")
    ap.add_argument("--flat-only", action="store_true", help="only the flattened result, no layers")
    ap.add_argument("--layers-only", action="store_true", help="only rebuilt layers, no flattened result")
    ap.add_argument("--no-baked", action="store_true", help="don't export baked mesh maps / baked normal layer")
    ap.add_argument("--mesh-units", choices=["cm", "m", "mm", "in", "raw"], default="cm",
                    help="units of the original model (default cm; 'raw' keeps Substance's normalised size)")
    ap.add_argument("--substance-export", help="folder written by the PawPaint Exporter Substance plugin")
    ap.add_argument("--pawpaint-dir", help="PawPaint install folder (default: found through Steam)")
    ap.add_argument("--install-plugin", action="store_true", help="install/update the PawPaint plugin and exit")
    ap.add_argument("--no-install", action="store_true", help="never offer to install the PawPaint plugin")
    ap.add_argument("--yes", action="store_true", help="answer yes to questions")
    ap.add_argument("--no-handoff", action="store_true", help="don't tell the PawPaint plugin about the result")
    ap.add_argument("--engine", choices=["auto", "gpu", "cpu"], default="auto",
                    help="auto (default): the GPU if one works, else the CPU")
    ap.add_argument("--detail", choices=["fast", "lossless"], default="fast",
                    help="CPU only: 'lossless' works at full resolution everywhere (slower). The GPU is always full detail")
    ap.add_argument("--workers", type=int, default=0,
                    help="CPU cores for hand-painted masks (default 0 = all available; 1 = no worker processes)")
    ap.add_argument("--check", action="store_true", help="compare rebuilt layers with Substance's result (slower)")
    ap.add_argument("--no-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-pause", action="store_true", help="don't wait for Enter at the end")
    ap.add_argument("--cli", action="store_true", help="never open the window, even with no arguments")
    ap.add_argument("--version", action="version", version="SppToPawPaint %s" % __version__)
    args = ap.parse_args(argv)
    frozen = getattr(sys, "frozen", False)

    if args.install_plugin:
        print(maybe_install_plugin(args.pawpaint_dir, assume_yes=True))
        return 0

    files = list(args.spp)
    if not args.cli and (not files or (frozen and len(files) == 1 and len(sys.argv) == 2)):
        # double-clicked, or a file dropped onto the exe: friendly window
        try:
            from .gui import run_gui
            return run_gui(files[0] if files else None)
        except Exception:
            traceback.print_exc()
            if not files:
                ap.print_help()
                return 2

    if not args.no_install:
        print(maybe_install_plugin(args.pawpaint_dir, assume_yes=args.yes, ask=_console_ask))
    rc = 0
    for f in files:
        try:
            convert(f, args.out if len(files) == 1 else None, args.resolution,
                    layers=not args.flat_only, flat=not args.layers_only, baked=not args.no_baked,
                    check=args.check and not args.no_check, mesh_units=args.mesh_units,
                    engine=args.engine, detail=args.detail,
                    substance_export=args.substance_export, pawpaint_dir=args.pawpaint_dir,
                    handoff=not args.no_handoff, workers=args.workers)
        except Exception as e:  # keep going with the next file, but say what broke
            rc = 1
            print("\nCould not convert %s: %s" % (f, e))
            traceback.print_exc()
    if frozen and not args.no_pause:
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
