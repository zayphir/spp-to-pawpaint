# Substance Painter (.spp) → PawPaint importer

Brings a Substance 3D Painter project into PawPaint: mesh, every texture set,
and the layer stack (folders, fill layers, masks), plus an exact copy of what
Substance shows.

> **Community project by Zayphir.** This is an independent, unofficial tool.
> It is not made, endorsed or supported by PawWorks (the makers of PawPaint)
> or by Adobe. "Substance" and "Substance 3D Painter" are trademarks of Adobe.
> See [Disclaimer](#disclaimer).

> **Status: early release, still being worked on.** The picture you get in
> PawPaint matches Substance very closely, because each texture set includes
> an exact copy of Substance's result. The editable layers underneath are
> rebuilt, and **not every layer comes out 100% perfect**:
> - hand-painted masks can differ slightly at their edges;
> - Substance-only effects (fur filters, generators, smart masks, anchor
>   points, some blend modes) can't be rebuilt as separate layers. Their look
>   is kept in the result and 'Detail fix' layers;
> - it has been tested on a handful of projects, so yours may use something
>   it hasn't seen yet. `report.txt` in the converted folder lists what
>   couldn't be converted exactly.
>
> Keep your original `.spp`, and please report problems as an issue on this
> project.

It works for **everyone who has a `.spp`**, whether or not they own Substance
(for example, people who bought an avatar base that came with the creator's
`.spp`). Substance owners can optionally export exact layers.

| Part | What it does |
|---|---|
| `SppToPawPaint.exe` (`converter/`) | Reads the `.spp` and writes a folder PawPaint can import. Also installs the PawPaint plugin for you |
| `spp_importer.lua` (`plugin/`) | PawPaint plugin: opens the mesh and imports every texture set's layers |
| `pawpaint_exporter.py` (`substance_plugin/`, optional) | Substance plugin for owners: exports every layer exactly |

There are two parts because PawPaint plugins (Lua) can't run programs or read
binary files.

## For users

1. **Drag your `.spp` onto `SppToPawPaint.exe`** (or double-click the exe and
   pick the file). A window shows the progress.
   - The first time, it offers to **install the PawPaint plugin**: it finds PawPaint through Steam, and never overwrites your own `autoload.lua`.
   - The result is a folder named `<project> (PawPaint import)` next to the `.spp`. That takes a few minutes at 4K (about 6 minutes for a hand-painted character with three texture sets, on a graphics card).
2. **Open PawPaint** (restart it once after the first install) and go to
   **Plugins › Substance Importer**. Your latest conversion is already filled
   in. Click **Import**.
3. When it says *Done*, use **File › Save As**. Plugins can't save projects.

**What you get, per texture set**
- **Substance Result**: a paint layer on top with every channel exactly as
  Substance showed it. Hide it to work with the layers below.
- **Rebuilt layers**: the Substance stack as PawPaint layers. Folders stay
  folders, fill layers keep their colours and values, and each layer gets one
  mask image ('Mask (fitted to Substance)'). The masks are rebuilt from the
  project's own recipe (fills, images, projections, polygon fills, brush
  strokes) and then corrected against Substance's final textures, so the
  layers on their own look like Substance too. Recolour or hide a layer and
  the rest stays as it was.
- **Detail fix (from Substance)**: a small paint layer under the result layer
  with the few texels no layer can explain (usually well under 1%).
- **Black base (as in Substance)**: only where a texture set has areas no
  layer fully covers. Substance shows those over black; this makes PawPaint
  do the same.
- **Baked Normal (from Substance)**: the baked normal map, at the bottom. The
  other baked maps (ID, curvature, position, thickness, world-space normals)
  are in `materials/<set>/baked/`, ready to drop onto PawPaint fills or masks.
- `report.txt` gives a match score per texture set and lists anything that
  could not be reproduced exactly.
- `log.txt` is the whole run, line by line with times, written as it goes.
  If a conversion is slow or fails, attach it to your GitHub issue.

**Stopping early.** Every texture set is saved when it is finished. If you
close the converter before the end, the folder can still be imported with the
texture sets that were done, and the plugin tells you how many are missing.

**Mesh choices in the plugin**
- *Mesh from the .spp* (default). Rebuilt at its original size (`--mesh-units`, default cm).
- *My own model file*. Your FBX, OBJ or GLB, which keeps shape keys (Substance doesn't store them).
- *The project open now*. Open your model in PawPaint first; the layers are added on top.

Texture sets are matched to materials by name, ignoring case, spaces, `_`,
`-` and `.001`. Anything left over can be matched by hand in the panel.

**Video memory.** PawPaint (1.27 and later) refuses an import that needs more
video memory than is free. The plugin then tries again with fewer layers and,
if it must, imports only the flattened result, and says so in the panel.
Closing other programs that use the graphics card (Substance, games,
browsers) or converting at a lower `--resolution` avoids that.

**Own Substance?** Install the optional Substance plugin (see
`substance_plugin/README.md`) and choose **File › Export for PawPaint...**,
then convert as usual. Every layer then comes over exactly, including blur,
bevel, generators and smart masks.

**Speed and detail settings** (in the window's *Processing* box; remembered between runs):
- **Engine.** Auto (default) uses your graphics card when one is found, and is always full detail. GPU or CPU can be forced.
- **CPU detail.** *Fast* computes masks with lots of brush strokes at up to 2K and scales them up. *Full detail (lossless, slower)* works at full resolution everywhere.
- **CPU cores.** 0 uses every core (limited only by free memory). This also sets how many cores correct the masks against Substance's result, which is the longest step on texture sets with many layers; that step always runs on the CPU and reports its progress.
- **Compare with Substance's result.** Writes match percentages to report.txt. Off by default because it's slower.

**Command line:** `SppToPawPaint.exe file.spp --cli` plus any of:
- `--resolution 512|1024|2048|4096`
- `--flat-only`, `--layers-only`, `--no-baked`
- `--mesh-units cm|m|mm|in|raw`
- `--engine auto|gpu|cpu`, `--detail fast|lossless` (CPU), `--workers N` (0 = all cores), `--check`
- `--substance-export FOLDER`
- `--install-plugin`, `--pawpaint-dir DIR`
- `--out FOLDER`

Run it with `--help` for the full list.

### Linux

There is no Linux download yet. Run the converter with Python 3.10 or newer:

```sh
git clone https://github.com/zayphir/spp-to-pawpaint
cd spp-to-pawpaint/converter
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m spp2pawpaint /path/to/project.spp
```

- Without a file it opens the same window as on Windows. That needs Tk
  (`sudo apt install python3-tk` on Ubuntu and Debian). Add `--cli` to stay
  in the terminal.
- PawPaint is looked for in Steam's usual folders, including Flatpak and Snap
  installs and extra Steam libraries. If it isn't found, pass
  `--pawpaint-dir FOLDER` or set `PAWPAINT_DIR`.
- When PawPaint runs through Proton it sees your files through drive letters
  (`Z:` is `/`). The converter writes the folder that way for the plugin and
  prints it, e.g. `Z:\home\you\My project (PawPaint import)`. Use that form if
  you paste a folder into the plugin by hand. A Flatpak Steam can only open
  folders it has been given access to.
- Engine *Auto* uses the graphics card through OpenGL 4.3. Where only software
  OpenGL exists (llvmpipe), it uses the CPU engine, which is faster.

**How far this is tested:** the converter gives the same results on Ubuntu
24.04 as on Windows (CPU engine, and the OpenGL engine on software OpenGL),
and installing the plugin and the handoff were tried against a made-up Steam
library. **Not tested yet:** PawPaint itself on Linux picking up a conversion,
and the OpenGL engine on a real Linux graphics driver. If you try it, please
open an issue and say how it went.

## What converts

| Substance | PawPaint | Notes |
|---|---|---|
| Texture sets | Materials | Matched by name, or by hand. Projects from older Substance versions and from 9.1 both work |
| Folders, nesting, visibility, opacity, blend modes | Same | Unsupported blend modes fall back to Normal, with a note |
| Fill layers with colour / value channels | Fill layers with the same values | Base colour and emission converted from linear to sRGB |
| Image fills (bitmaps) | Fill layers with that image | UV tiling, offset and rotation applied; transparency is respected. In masks, planar and (undeformed) warp projections are rebuilt; other projections are applied in UV space, with a note |
| Gradient fills in masks (Substance's starter linear gradient) | Mask image | Position, contrast, tiling, rotation and invert, measured against Substance |
| Polygon-fill masks (triangle, polygon, mesh, UV chunk; click or rectangle; 3D or UV view; symmetry) | Mask image | Replayed against the mesh, so exact |
| Geometry masks (per mesh part) | Folded into the mask image | Exact |
| Brush, path and eraser strokes (3D view or 2D UV view) | Mask / paint image | Substance's brush engine as measured: size spaces, spacing, hardness falloff, flow, pen pressure, symmetry and culling (round brush; other brush shapes are approximated as round) |
| Blur and Levels filters | Reproduced | Blur strength measured against Substance |
| Generators (edge wear, dirt, curvature, AO, thickness, position, light) | Approximated from the baked maps | Rough match only; noted in the report |
| Bevel, noise/grunge textures, other procedurals | Skipped, with a note | The Substance Result layer still shows them. Substance owners can use the exporter for exact layers |
| Baked mesh maps | Exported; baked normal imported as a layer | |
| Custom channels (opacity, specular…) | Custom PawPaint channels | Glossiness also converted to roughness |
| No saved texture cache | Approximate Substance Result built from the layers | Noted |
| UDIM projects | Tile 1001 only | Noted |

**How close is it?** On three hand-painted character projects, measured in
PawPaint itself (its own exported render against Substance's textures), the
editable layers alone match on 99–100% of texels, and 99.97–100% with the
Detail fix layer. Each layer was also checked on its own against Substance's
per-layer output, so that hiding a layer shows what Substance would show.
Projects that lean on Substance-only procedural effects
(fur filters, generators, anchor points) keep the right picture, but those
effects end up baked into layers or in the Detail fix, not as separate
editable layers.

## For developers

```
converter/
  spp2pawpaint/
    sppfile.py    .spp = HDF5. Opens it, decodes texture-cache entries
    docbin.py     paint/document.bin reader (self-describing binary, see below)
    assbin.py     paint/scene.assbin reader (Substance's variant of Assimp binary)
    qtvariant.py  QDataStream QVariantMap headers
    geometry.py   triangle tables, picking, islands, UV rasterising
    images.py     embedded images (tiled JPEG 2000, raw LZ4) -> arrays
    masks.py      replays mask actions (polygon fill, strokes, fills, filters, generators, geometry masks)
    brushmodel.py Substance's brush stamp model (falloff, flow, spacing), as measured
    strokes2d.py  brush strokes on the CPU
    gpu.py        brush strokes on the graphics card (OpenGL 4.3, moderngl)
    procedural.py blur, gradients, projections, UV image sampling, approximate generators
    layers.py     one texture set -> PawPaint layer bundle (manifest.json + PNGs)
    refine.py     fits the rebuilt masks to Substance's final textures; detail fix
    parallel.py   worker processes for heavy masks
    settings.py   saved settings
    exact.py      exact layers from the Substance companion exporter
    glb.py        mesh -> .glb
    install.py    find PawPaint via Steam, install plugin, handoff note
    cli.py        command line / drag-and-drop
    gui.py        the small window
  SppToPawPaint.py   PyInstaller entry point
plugin/scripts/
  spp_importer.lua   the PawPaint plugin
  autoload.lua       loads it at startup
substance_plugin/
  pawpaint_exporter.py  optional Substance plugin (official API) for exact layers
build.ps1           builds the exe and the release zip
release_readme.txt  the 'READ ME FIRST' text that goes into the zip
```

Run from source: `pip install -r converter/requirements.txt`, then
`python -m spp2pawpaint file.spp` from `converter/`.

Build the exe and release zip: `pip install pyinstaller`, then
`powershell -ExecutionPolicy Bypass -File build.ps1`.

### .spp format notes (reverse engineered)
- **Container:** HDF5. Every entry is a 1-D uint8 dataset with an `m3_x64_128`
  (MurmurHash3) attribute.
- **`paint/document.bin`:** magic `0x1b7c2fdd`, then a tree of objects. Each
  object is `u32 endOffset, i32 classIndex` (`-1` = class defined inline as
  name + `(fieldName, typeCode)` list), followed by its fields in order. The
  type codes are listed in `docbin.py`. The root `DataDocument` holds
  `materials[]` (texture sets) → `stacks[]` → `DataStackLayers` →
  `DataLayerColor` / `DataLayerGroup`. Masks are `maskActions` (a stack of
  `DataActionGroup`, `DataActionDecal` (polygon fill), `DataActionPaint`,
  `DataActionFill`, `DataActionFilterProcedural`, …). Lists are bottom first.
- **Polygon fill hits:** `projViewInvMatrix` is column-major. `position` is in
  screen 0..1 for `viewType 1` (3D view) and is a UV coordinate for
  `viewType 0` (UV view). Granularity: 0 triangle, 1 polygon, 2 mesh, 3 UV
  chunk. Rectangle picks select triangles that overlap the rectangle.
  `duplicateMatrix` is the symmetry mirror.
- **Channel flags:** basecolor 1, height 2, specular 4, opacity 8, emissive 16,
  roughness 128, metallic 8192, normal 4194304.
- **Blending modes:** the stored id is the index in Substance's own list
  (2 = Normal, 4 = Multiply, 8 = Screen, 11 = Overlay, 25 = Passthrough, …);
  see `SUBSTANCE_BLEND_LABELS` in `layers.py`.
- **`texture-cache/<sceneMaterialUID>/<stack>/<channelFlag>`:** a QVariantMap
  header (`width`, `height`, `internalFormat` GL enum, `bufferSize`), then
  `u32 BE size + LZ4 block` chunks. The pixels are a GL readback (bottom row
  first). They hold the final composited channels.
- **`paint/scene.assbin`:** a 512 byte header, then chunks prefixed by a
  `u64 size` that includes itself. Every vertex array has a `u32` count. Faces
  are `(u32 polygonStart, u32 n=3, 3×u32)`.

### How this was made
The `.spp` reader was written only by examining `.spp` files and the data
inside them, for interoperability, so people can open their own projects in
another program. No Adobe software was decompiled, disassembled or modified,
and this project contains no Adobe code, libraries, shaders or assets. The
PawPaint side uses only PawPaint's public, documented plugin API.

Where the files alone don't say how Substance behaves (brush falloff, blur
strength, projections, blend order), it was measured from the outside: small
test projects were made in a licensed copy of Substance 3D Painter through its
official Python scripting interface, exported, and the numbers compared.

### PawPaint side
The plugin uses only documented API: `pp.project.openMesh`,
`pp.import.layers` (the layer bundle format), `pp.fill.*`, `pp.layers.*`,
`pp.channels.add` and `pp.ui.panel`. Each imported layer gets a temporary
`#sppN` name suffix, so the plugin can find it after import, set its fill
values and rename it back.

## Disclaimer

- This is an unofficial community tool. It is not affiliated with, made,
  endorsed or supported by PawWorks or Adobe. Please don't send PawWorks bug
  reports about this tool; open an issue on this project instead.
- Only convert projects you own or have the right to use. You are responsible
  for following any licence or terms that apply to your files, models and
  assets, and to the software you made them with.
- Conversion is best effort. Always keep your original `.spp`. The tool only
  reads it and never changes it.
- Provided "as is", without warranty of any kind. The authors are not liable
  for any damage or data loss from using it.
- Substance and Substance 3D Painter are trademarks of Adobe. PawPaint is a
  product of PawWorks. Names are used only to say what the tool works with.
