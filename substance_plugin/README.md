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
the layer's folders), sets the layer's blend modes to Normal, and exports
twice with a temporary fill layer underneath: once black, once white. The
converter recovers colour and coverage from the pair
(`alpha = 1 - (white - black)`, `colour = black / alpha`, in linear space).
Because folder masks are applied during export, they end up inside each
layer's coverage, so imported folders carry no masks of their own. The
original blend modes are put back on in PawPaint.

## Status and requirements
- Needs a Substance version with the layer stack editing API (8.3 or newer).
- **Not yet run end to end on a real Substance install.** The export method itself (each layer alone over black and over white) was checked in Substance 3D Painter 12.2 during development, but this plugin has not been. It was written against the documented API: `substance_painter.layerstack` (`get_root_layer_nodes`, `insert_fill`, `InsertPosition`, `set_visible`, `set_blending_mode`, `set_source`, `delete_node`), `substance_painter.export.export_project_textures`, `substance_painter.textureset` and `substance_painter.ui`. If a function name differs in your version, the Python log says which. Please report it along with your Substance version.
- If anything fails mid-way, the plugin restores visibility and blend modes and deletes its temporary layer in `finally` blocks.
