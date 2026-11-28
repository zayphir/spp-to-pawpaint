"""PawPaint Exporter – a Substance 3D Painter plugin (optional, for Substance owners).

Exports every layer of the open project on its own, so SppToPawPaint can
import each layer exactly (procedural filters, generators and smart masks
included). It uses only Adobe's official Substance 3D Painter Python API.

Install: copy this file to
    Documents/Adobe/Adobe Substance 3D Painter/python/plugins/
then in Substance: Python > Reload Plugins Folder, and enable
"pawpaint_exporter". Use File > Export for PawPaint...

How it works, per layer and texture set:
  1. hide every other layer (keep the layer's parent folders visible),
  2. set the blend modes of the layer and its folders to Normal and their
     opacities to 100 %, so the export holds the layer's own coverage only
     (the blend modes and opacities are written to export.json and put back
     on in PawPaint),
  3. put a temporary fill layer at the bottom, black, export all channels,
     then make it white and export again,
  4. restore everything exactly as it was.
SppToPawPaint recovers colour and coverage from each black/white pair.

Nothing is saved: your project is restored and never written to disk by
this plugin. Tested end to end on Substance 3D Painter 12.2 (Steam): the
export, the restored layer stack, and the import in PawPaint 1.28.
"""

import json
import os
import traceback

import substance_painter as sp

try:  # newer Painter versions ship Qt 6 (QAction lives in QtGui there)
    from PySide6 import QtWidgets
    from PySide6.QtGui import QAction
except ImportError:  # older Painter versions ship Qt 5
    from PySide2 import QtWidgets
    from PySide2.QtWidgets import QAction

EXPORT_FORMAT = "spp2pawpaint-substance-export"
PLUGIN_VERSION = "0.3.0"
CHANNEL_NAMES = {
    "BaseColor": "basecolor", "Roughness": "roughness", "Metallic": "metallic", "Normal": "normal",
    "Height": "height", "Emissive": "emissive", "Opacity": "opacity", "Specular": "specular",
    "Glossiness": "glossiness",
}
# Substance BlendingMode names -> PawPaint layer blend modes
BLEND_TO_PAWPAINT = {"Normal": "Normal", "Multiply": "Multiply", "LinearDodge": "Add", "Screen": "Screen",
                     "Overlay": "Overlay", "SoftLight": "Soft Light", "HardLight": "Hard Light",
                     "Replace": "Normal", "Passthrough": "Normal"}

_actions = []


# -- helpers --------------------------------------------------------------------
def _channel_types(stack):
    """{substance ChannelType: export name} for the channels this stack has.

    Matched by the enum member, not by its name: Substance 12.2 renamed
    several channels (Roughness is SpecularRoughness, Metallic BaseMetalness,
    Emissive EmissionColor) and keeps the old names as aliases of the same
    member, so ct.name gives the new one. Matching by name skipped those
    three channels without a word."""
    CT = sp.textureset.ChannelType
    wanted = [(getattr(CT, old), name) for old, name in CHANNEL_NAMES.items() if hasattr(CT, old)]
    out = {}
    for ct in stack.all_channels().keys():
        for member, name in wanted:
            if ct == member:
                out[ct] = name
                break
    return out


def _children(node):
    for attr in ("sub_layers", "get_sub_layers"):
        fn = getattr(node, attr, None)
        if fn:
            try:
                return list(fn())
            except Exception:
                return []
    return []


def _is_group(node):
    return node.get_type() == sp.layerstack.NodeType.GroupLayer


def _walk(nodes, parent, out, chain):
    """Flatten the stack bottom-first: (node, parent_id, ancestors)."""
    for node in reversed(list(nodes)):  # root lists are top-first
        nid = len(out)
        out.append((node, parent, list(chain)))
        if _is_group(node):
            _walk(_children(node), nid, out, chain + [node])


def _all_nodes(nodes):
    for n in nodes:
        yield n
        if _is_group(n):
            yield from _all_nodes(_children(n))


def _export(ts_name, channels, folder, size_log2):
    os.makedirs(folder, exist_ok=True)
    maps = []
    for ct, name in channels.items():
        comps = "RGB" if name in ("basecolor", "normal", "emissive", "specular") else "L"
        maps.append({"fileName": name,
                     "channels": [{"destChannel": c, "srcChannel": c if comps == "RGB" else "L",
                                   "srcMapType": "documentMap", "srcMapName": name}
                                  for c in (("R", "G", "B") if comps == "RGB" else ("L",))],
                     "parameters": {"fileFormat": "png", "bitDepth": "16" if name in ("normal", "height") else "8",
                                    "dithering": False, "paddingAlgorithm": "infinite"}})
    # No bitDepth here: these parameters override every map's own, and
    # normal and height must stay 16-bit (with "8" here they came out 8-bit).
    params = {"fileFormat": "png", "dithering": False, "paddingAlgorithm": "infinite"}
    if size_log2:
        params["sizeLog2"] = size_log2
    config = {"exportShaderParams": False, "exportPath": folder,
              "defaultExportPreset": "pawpaint_isolate",
              "exportPresets": [{"name": "pawpaint_isolate", "maps": maps}],
              "exportList": [{"rootPath": ts_name}],
              "exportParameters": [{"parameters": params}]}
    result = sp.export.export_project_textures(config)
    status = getattr(result, "status", None)
    if status is not None and status != sp.export.ExportStatus.Success:
        raise RuntimeError("export failed: %s" % getattr(result, "message", status))
    files = {}
    for paths in (getattr(result, "textures", {}) or {}).values():
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0].lower()
            for name in channels.values():
                if base.endswith(name):
                    files[name] = p
    return files


def _blend_name(node, ct):
    try:
        return node.get_blending_mode(ct).name
    except Exception:
        return None


def _opacity(node, ct):
    try:
        return float(node.get_opacity(ct))
    except Exception:
        return None


def _neutral(nodes, channels):
    """Set Normal blending and 100 % opacity on these nodes for every channel.
    Returns what to put back: [(node, channel, blend mode or None, opacity or None)]."""
    saved = []
    for n in nodes:
        for ct in channels:
            mode = opac = None
            try:
                mode = n.get_blending_mode(ct)
            except Exception:
                pass
            try:
                opac = n.get_opacity(ct)
            except Exception:
                pass
            saved.append((n, ct, mode, opac))
            try:
                if mode is not None and mode != sp.layerstack.BlendingMode.Normal:
                    n.set_blending_mode(sp.layerstack.BlendingMode.Normal, ct)
            except Exception:
                pass
            try:
                if opac is not None and float(opac) != 1.0:
                    n.set_opacity(1.0, ct)
            except Exception:
                pass
    return saved


def _restore(saved):
    for n, ct, mode, opac in reversed(saved):
        try:
            if mode is not None and n.get_blending_mode(ct) != mode:
                n.set_blending_mode(mode, ct)
        except Exception:
            pass
        try:
            if opac is not None and n.get_opacity(ct) != opac:
                n.set_opacity(opac, ct)
        except Exception:
            pass


# -- the export ------------------------------------------------------------------
def export_for_pawpaint(out_dir, size_log2=None, progress=None):
    if not sp.project.is_open():
        raise RuntimeError("Open a project first.")
    project = sp.project.file_path() or "project.spp"
    data = {"format": EXPORT_FORMAT, "version": 1, "plugin": PLUGIN_VERSION,
            "project": os.path.basename(project), "textureSets": []}
    sets = sp.textureset.all_texture_sets()
    jobs = []
    for ts in sets:
        stack = ts.get_stack()
        flat = []
        _walk(sp.layerstack.get_root_layer_nodes(stack), None, flat, [])
        jobs.append((ts, stack, flat))
    total = sum(sum(1 for n, _, _ in flat if not _is_group(n)) for _, _, flat in jobs) * 2
    done = 0

    for ts, stack, flat in jobs:
        ts_name = ts.name()
        channels = _channel_types(stack)
        roots = sp.layerstack.get_root_layer_nodes(stack)
        everything = list(_all_nodes(roots))
        saved_vis = {n.uid(): n.is_visible() for n in everything}
        entry = {"name": ts_name, "resolution": ts.get_resolution().width,
                 "channels": sorted(channels.values()), "layers": []}
        probe = None
        try:
            # temporary background at the very bottom
            bottom = roots[-1] if roots else None
            pos = (sp.layerstack.InsertPosition.below_node(bottom) if bottom is not None
                   else sp.layerstack.InsertPosition.from_textureset_stack(stack))
            probe = sp.layerstack.insert_fill(pos)
            probe.set_name("PawPaint export background (temporary)")
            probe.active_channels = set(channels.keys())
            base_ct = next((ct for ct, name in channels.items() if name == "basecolor"), next(iter(channels), None))
            for node, parent, ancestors in flat:
                info = {"id": len(entry["layers"]), "name": node.get_name(),
                        "type": "Folder" if _is_group(node) else "Layer",
                        "parent": parent, "visible": saved_vis.get(node.uid(), True),
                        "opacity": 1.0, "blendMode": "Normal", "substanceBlend": None,
                        # every channel's own blend mode and opacity: the export holds the
                        # coverage alone, PawPaint puts these back on (0.3.0)
                        "channelBlend": {}}
                if base_ct is not None:
                    mode = _blend_name(node, base_ct)
                    opac = _opacity(node, base_ct)
                    info["substanceBlend"] = mode
                    info["blendMode"] = BLEND_TO_PAWPAINT.get(mode, "Normal")
                    if opac is not None:
                        info["opacity"] = opac
                for ct, name in channels.items():
                    info["channelBlend"][name] = {"blend": _blend_name(node, ct), "opacity": _opacity(node, ct)}
                if _is_group(node):
                    entry["layers"].append(info)
                    continue
                keep = {node.uid()} | {a.uid() for a in ancestors} | {probe.uid()}
                for n in everything:
                    n.set_visible(n.uid() in keep)
                probe.set_visible(True)
                saved_blend = _neutral([node] + list(ancestors), list(channels))
                images = {}
                try:
                    pairs = {}
                    for shade, value in (("black", 0.0), ("white", 1.0)):
                        for ct in channels:
                            probe.set_source(ct, sp.colormanagement.Color(value, value, value))
                        rel = os.path.join(ts_name, "%03d_%s" % (info["id"], shade))
                        files = _export(ts_name, channels, os.path.join(out_dir, rel), size_log2)
                        pairs[shade] = {k: os.path.relpath(v, out_dir) for k, v in files.items()}
                        done += 1
                        if progress and not progress(done, total, "%s / %s" % (ts_name, info["name"])):
                            raise KeyboardInterrupt
                    for name in channels.values():
                        if name in pairs["black"] and name in pairs["white"]:
                            images[name] = {"black": pairs["black"][name], "white": pairs["white"][name]}
                finally:
                    _restore(saved_blend)
                info["images"] = images
                entry["layers"].append(info)
        finally:
            for n in everything:
                if n.uid() in saved_vis:
                    try:
                        n.set_visible(saved_vis[n.uid()])
                    except Exception:
                        pass
            if probe is not None:
                try:
                    sp.layerstack.delete_node(probe)
                except Exception:
                    traceback.print_exc()
        data["textureSets"].append(entry)

    with open(os.path.join(out_dir, "export.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return out_dir


# -- UI ----------------------------------------------------------------------------
def _run():
    main = sp.ui.get_main_window()
    if not sp.project.is_open():
        QtWidgets.QMessageBox.information(main, "Export for PawPaint", "Open a project first.")
        return
    proj = sp.project.file_path() or os.path.join(os.path.expanduser("~"), "project.spp")
    out_dir = os.path.splitext(proj)[0] + " (Substance export)"
    choice, ok = QtWidgets.QInputDialog.getItem(
        main, "Export for PawPaint",
        "Each layer is exported twice, so this can take a while.\nResolution:",
        ["Project resolution", "2048", "1024"], 1, False)
    if not ok:
        return
    size_log2 = {"2048": 11, "1024": 10}.get(choice)
    dlg = QtWidgets.QProgressDialog("Exporting layers for PawPaint...", "Cancel", 0, 100, main)
    dlg.setWindowTitle("Export for PawPaint")
    dlg.setMinimumDuration(0)

    def progress(done, total, label):
        dlg.setMaximum(max(total, 1))
        dlg.setValue(done)
        dlg.setLabelText("Exporting %s (%d/%d)" % (label, done, total))
        QtWidgets.QApplication.processEvents()
        return not dlg.wasCanceled()

    try:
        export_for_pawpaint(out_dir, size_log2, progress)
        dlg.close()
        QtWidgets.QMessageBox.information(
            main, "Export for PawPaint",
            "Done:\n%s\n\nNow run SppToPawPaint on the .spp. It picks this folder up automatically "
            "and imports every layer exactly." % out_dir)
    except KeyboardInterrupt:
        dlg.close()
        QtWidgets.QMessageBox.information(main, "Export for PawPaint", "Cancelled. Your project was restored.")
    except Exception as e:
        dlg.close()
        traceback.print_exc()
        QtWidgets.QMessageBox.warning(main, "Export for PawPaint",
                                      "Export failed: %s\n\nYour project was restored. See the Python log." % e)


def start_plugin():
    action = QAction("Export for PawPaint...")
    action.triggered.connect(_run)
    sp.ui.add_action(sp.ui.ApplicationMenu.File, action)
    _actions.append(action)


def close_plugin():
    for a in _actions:
        try:
            sp.ui.delete_ui_element(a)
        except Exception:
            pass
    _actions.clear()


if __name__ == "__main__":
    start_plugin()
