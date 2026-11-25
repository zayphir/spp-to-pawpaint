# Changelog

## 0.6.1 – 2026-11-24
Linux, from source (see 'Linux' in the README). Conversions come out the same as with 0.6.0.

- The converter runs on Linux with Python. On Ubuntu 24.04 it gives the same results as on Windows. PawPaint itself on Linux picking up a conversion has not been tested yet.
- PawPaint is also found in Flatpak and Snap installs of Steam, and whatever letter case its program file has.
- When PawPaint runs through Proton, the folder handed to the plugin is written the way PawPaint sees it (`Z:\home\...`).
- 'The folder path is on your clipboard' is only said on Linux when a clipboard tool (wl-copy, xclip, xsel) really took it.
- Engine 'Auto' no longer picks software OpenGL (llvmpipe); the CPU engine is faster there.
- Settings follow `XDG_CONFIG_HOME` on Linux.

## 0.6.0 – 2026-11-23
First public release. The project is still being worked on: the picture matches Substance very closely, but not every editable layer comes out 100% perfect yet (see 'Status' in the README).

- Converts a Substance 3D Painter project (`.spp`) into a folder PawPaint can import: the mesh, every texture set, and the layer stack with folders, fill layers and one mask per layer.
- Works from the `.spp` alone. No copy of Substance is needed.
- Layer masks are rebuilt from the project's own recipe (fills, images, gradients, projections, polygon fills, brush strokes) and then corrected against Substance's final textures.
- Each texture set also gets an exact 'Substance Result' layer, a small 'Detail fix' layer, and the baked normal map.
- Runs on the graphics card when one is available (OpenGL 4.3), otherwise on the CPU.
- Installs the PawPaint plugin for you and hands the conversion over to it.
- Optional Substance plugin for people who own Substance: exports every layer exactly.
