# PawPaint Exporter (optional Substance 3D Painter plugin)

For people who **own** Substance 3D Painter. It exports every layer of your
project exactly, including procedural filters, generators and smart masks,
which the converter alone can only approximate. People without Substance
don't need this; SppToPawPaint works from the .spp alone.

It uses only Adobe's official Substance 3D Painter Python API, run inside
your own copy of Substance.

## Install
1. Copy `pawpaint_exporter.py` to
   `Documents\Adobe\Adobe Substance 3D Painter\python\plugins\`
2. In Substance: **Python > Reload Plugins Folder**, then tick
   **pawpaint_exporter** in the Python menu.

## Use
1. Open your project and choose **File > Export for PawPaint...**. Pick a resolution (2048 is a good balance; every layer is exported twice, so large projects take a while).
2. It writes `<project> (Substance export)` next to the `.spp`, and restores your layer stack afterwards. It never saves your project.
3. Run **SppToPawPaint** on the `.spp` as usual. It finds that folder and uses the exact layers instead of rebuilt ones. report.txt says "exact layers from Substance".

## How it works
For each layer and texture set, the plugin hides every other layer (keeping
the layer's folders), sets the blend modes of the layer and its folders to
Normal and their opacities to 100 %, and exports twice with a temporary fill
layer underneath: once black, once white. The converter recovers colour and
coverage from the pair (`alpha = 1 - (white - black)`, `colour = black /
alpha`, in linear space). Every channel's blend mode and opacity is written
to `export.json` and put back on in PawPaint. Because folder masks are
applied during export, they end up inside each layer's coverage, so imported
folders carry no masks of their own.

## Status and requirements
- Needs a Substance version with the layer stack editing API (8.3 or newer).
- **Tested end to end on Substance 3D Painter 12.2** (Steam, "2026"): the export, the layer stack left exactly as it was afterwards, and the import in PawPaint 1.28. On two test projects every channel came into PawPaint as in Substance: height 100 %, roughness and metallic 99 to 100 %, normals 96 to 100 %, base colour 99.8 to 100 % on most materials.
- **After updating the plugin, export again.** Exports made with versions before 0.3.0 lack roughness, metallic and emission on Substance 12.2, and bring wrong height and normals.
- Limits of exact layers:
  - Soft edges can look slightly darker in PawPaint than in Substance. Substance mixes layers in linear light, PawPaint doesn't. The converter's own rebuilt layers are corrected against Substance's picture for this; exact layers can't be.
  - Blend modes PawPaint doesn't have (Value, Color, Difference, ...) become Normal, and the report says which layers.
  - A pass-through folder that holds a blend-mode layer is left out (its layers move up a level), because PawPaint folders are always composited on their own.
  - Normal maps come in at 8 bits per channel.
- If anything fails mid-way, the plugin restores visibility, blend modes and opacities and deletes its temporary layer in `finally` blocks.
