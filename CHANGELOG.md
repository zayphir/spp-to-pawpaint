# Changelog

## 0.6.3 – 2026-11-26
The size now follows your graphics card, so big projects come in with all their layers instead of only the flattened result. Nothing about how layers are converted has changed: at the same size, every file the converter writes for the test projects is identical to 0.6.2's.

- **Why some materials came in with only 'Substance Result (flattened)'.** PawPaint keeps every layer on the graphics card and refuses a material that needs more video memory than is free. A layer with a mask takes about 117 MB at 4096, so a material with 120 layers needs about 15 GB there, and a quarter of that at 2048.
- **Plugin: new 'Size' option, on 'Auto (fit graphics card)'.** When PawPaint refuses a material, the plugin opens the project one size smaller and imports everything again with every layer, instead of leaving layers out. It uses the files you already converted (PawPaint scales them) and says so when it is done. 'As converted' keeps the size; 2048, 1024 and 512 set it by hand.
- **Converter: Resolution 'Auto' looks at your graphics card first.** It reads how much video memory the card has and has free, works out roughly what PawPaint will need, and converts at the largest size that can be imported. A smaller size also converts several times faster. It says what it chose and why. Pick a resolution yourself to decide otherwise.
- **The numbers are shown.** The window and `log.txt` say what the project will need, and `report.txt` lists it per material at the converted size and the next one down.
- **When the size can't go down** (importing into the open project, or a size set by hand), a material that doesn't fit is imported without its hidden layers before the plugin falls back to the flattened result alone, and the message gives PawPaint's numbers.
- The window's GPU line shows the card's video memory.
- The plugin is 0.6.3 too, so both parts carry the same number again.

## 0.6.2 – 2026-11-26
Big projects convert much faster, and a long run now says what it is doing. The layers come out exactly as with 0.6.1: every file the converter writes for the test projects is identical, byte for byte. The one exception is the fix at the end of this list.

- **Texture sets with many layers no longer take hours.** The step that corrects the masks against Substance's result ('fitting masks') repeated a lot of work when a texture set had many shading layers (Multiply, Add, Overlay), and it ran on one CPU core. It now does that work once and uses every core. For a project whose biggest texture set has 134 layers at 4096, this step was still running after two hours for the person who reported it. It now takes about two and a half minutes on the test PC.
- **Painting the masks is about three times faster on big projects.** Rectangle selections no longer test every ray against the whole model, selections and brush strokes are only processed over the part of the texture they touch, and brush actions whose result can't be used are skipped. The whole 6-texture-set project above converts in a little over 7 minutes on the test PC.
- **Progress while fitting.** The window shows how far the step is and roughly how long is left.
- **`log.txt`** in the converted folder holds the whole run with times, written line by line. Attach it when you report a slow or failed conversion.
- **Stopping early keeps what is done.** Each texture set is saved when it finishes. A folder from a run that was closed early can be imported, and the plugin says how many texture sets are missing.
- **Less memory while fitting.** The processes that paint masks are stopped once a texture set's masks are in, which frees several gigabytes at 4096. They start again for the next texture set.
- 'CPU cores' also sets how many cores the fitting uses.
- **Fixed:** with 'Substance result' unticked (or `--layers-only`), the masks were not corrected against Substance's result at all and no 'Detail fix' layer was written. Now only the result layer itself is left out.

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
