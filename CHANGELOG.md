# Changelog

## 0.6.0 – 2026-11-23
First public release. The project is still being worked on: the picture matches Substance very closely, but not every editable layer comes out 100% perfect yet (see 'Status' in the README).

- Converts a Substance 3D Painter project (`.spp`) into a folder PawPaint can import: the mesh, every texture set, and the layer stack with folders, fill layers and one mask per layer.
- Works from the `.spp` alone. No copy of Substance is needed.
- Layer masks are rebuilt from the project's own recipe (fills, images, gradients, projections, polygon fills, brush strokes) and then corrected against Substance's final textures.
- Each texture set also gets an exact 'Substance Result' layer, a small 'Detail fix' layer, and the baked normal map.
- Runs on the graphics card when one is available (OpenGL 4.3), otherwise on the CPU.
- Installs the PawPaint plugin for you and hands the conversion over to it.
- Optional Substance plugin for people who own Substance: exports every layer exactly.
