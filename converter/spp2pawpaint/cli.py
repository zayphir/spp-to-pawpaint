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
    res = pick_resolution(mats, resolution)
    log("  %d texture sets, %d mesh parts, %d triangles; importing at %dx%d"
        % (len(mats), len(scene.parts), len(geo.T), res, res))
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
        if ok:
            use = "gpu"
            log("  engine: GPU (%s), full detail" % name)
        else:
            log("  engine: CPU (%s)" % ("no usable GPU: " + name if engine == "gpu" else "no usable GPU"))
            if engine == "gpu":
                warnings.append("The GPU engine was requested but no usable GPU was found (%s); the CPU was used." % name)
    lossless = detail == "lossless"
    if use == "cpu" and layers:
        log("  engine: CPU, %s" % ("full detail (lossless, slower)" if lossless else "fast detail"))

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

    taken = set()
    entries = []
    try:
        _convert_sets(proj, geo, scene, mats, res, out_dir, entries, warnings, layers, flat, baked, check,
                      export, export_dir, pool, log, progress, taken, use, lossless)
    finally:
        if pool is not None:
            pool.close()
    if progress:
        progress(len(mats), len(mats), "done")
    return _finish(proj, spp_path, out_dir, res, mesh_units, scale, entries, warnings, handoff, pawpaint_dir, log, t0)


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


def _convert_sets(proj, geo, scene, mats, res, out_dir, entries, warnings, layers, flat, baked, check,
                  export, export_dir, pool, log, progress, taken, engine="cpu", lossless=False):
    for i, m in enumerate(mats):
        name = m.get("sceneMaterialName") or "Material"
        if progress:
            progress(i, len(mats), name)
        folder = _folder_name(name, taken)
        scene_index = scene.material_names.index(name) if name in scene.material_names else None
        if scene_index is None:
            warnings.append("%s: this texture set isn't used by any part of the model in the .spp; "
                            "it is skipped when importing onto that model" % name)
        log("Texture set '%s'" % name)
        t = time.time()
        entry = {"name": name, "bundle": "materials/%s" % folder, "layers": 0, "match": None,
                 "nativeResolution": 2 ** int((m.get("resolutionLog2") or [11])[0]),
                 "inMesh": scene_index is not None}
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
                         baked=sorted(mc.baked_files))
            how = ("exact layers from Substance" if exact_specs is not None else
                   "" if score is None else "rebuilt layers match Substance on %.1f%% of pixels" % (score * 100))
            log("  %d layers%s (%.1fs)" % (entry["layers"], ", " + how if how else "", time.time() - t))
            warnings.extend(mc.warnings)
            del mc
        except Exception as e:
            warnings.append("%s: could not be converted (%s)" % (name, e))
            log("  FAILED: %s" % e)
            traceback.print_exc()
        entries.append(entry)


def _finish(proj, spp_path, out_dir, res, mesh_units, scale, entries, warnings, handoff, pawpaint_dir, log, t0):
    info = {"format": "spp2pawpaint-import", "version": 1, "converter": __version__,
            "source": os.path.basename(spp_path), "resolution": res, "mesh": "model.glb",
            "meshUnits": mesh_units if scale != 1.0 else "raw",
            "materials": entries, "warnings": warnings}
    with open(os.path.join(out_dir, "import.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    _write_report(out_dir, spp_path, res, entries, warnings)
    proj.close()

    if handoff:
        _handoff(out_dir, spp_path, entries, pawpaint_dir, log)
    log("Done in %.0fs -> %s" % (time.time() - t0, out_dir))
    if warnings:
        log("%d notes about things that could not be converted exactly, see report.txt" % len(warnings))
    return out_dir


def _write_report(out_dir, spp_path, res, entries, warnings):
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("SppToPawPaint %s - %s\n\n" % (__version__, os.path.basename(spp_path)))
        f.write("Import resolution: %d\n\n" % res)
        for e in entries:
            how = ("exact layers from Substance" if e.get("exact") else
                   "" if e.get("match") is None else "rebuilt layers match Substance on %.1f%% of pixels" % (e["match"] * 100))
            f.write("%-28s %3d layers   %s\n" % (e["name"], e["layers"], how))
            if e.get("baked"):
                f.write("%-28s baked maps: %s (in %s/baked/)\n" % ("", ", ".join(e["baked"]), e["bundle"]))
        f.write("\nNotes (%d):\n" % len(warnings))
        for w in warnings:
            f.write("  - %s\n" % w)
        f.write("\nEvery texture set gets a 'Substance Result' layer on top that is an exact copy of what\n"
                "Substance shows. Hide it to see and edit the rebuilt layers underneath.\n"
                "Baked mesh maps (normal, ID, curvature, ...) are in each texture set's 'baked' folder;\n"
                "the baked normal is also imported as the 'Baked Normal (from Substance)' layer.\n")


def _handoff(out_dir, spp_path, entries, pawpaint_dir, log):
    pp_dir = I.find_pawpaint(pawpaint_dir)
    if pp_dir:
        try:
            I.write_pending(pp_dir, out_dir, spp_path, [e["name"] for e in entries])
            log("PawPaint will offer this conversion in Plugins > Substance Importer.")
        except OSError as e:
            log("Could not leave a note for the PawPaint plugin (%s)." % e)
    if I.copy_to_clipboard(out_dir):
        log("The folder path is also on your clipboard.")


def maybe_install_plugin(pawpaint_dir=None, assume_yes=False, log=print, ask=None):
    """Install/update the PawPaint plugin if needed. Returns status text."""
    pp_dir = I.find_pawpaint(pawpaint_dir)
    if not pp_dir:
        return "PawPaint not found (install it from Steam, or pass --pawpaint-dir)"
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
