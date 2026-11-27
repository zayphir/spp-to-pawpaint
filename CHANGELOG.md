# Changelog

## 0.6.6 – 2026-11-27
Fixes projects saved with the current Substance 3D Painter (12.2, the '2026' version). If materials came into PawPaint blank, or without their 'Substance Result' layer, convert the project again with this version. The plugin is unchanged (0.6.5).

- **Projects saved by Substance 12.2 were read wrongly.** That version numbers a material's channels differently in the file (0, 1, 7, 13, 22 where older files have 1, 2, 128, 8192, 4194304). The converter took the height texture for the base colour and didn't recognise the others.
  - A material with a height channel failed completely (`operands could not be broadcast together ...` or `cannot reshape array ...` in the window) and came into PawPaint with nothing on it.
  - A material without one converted, but its 'Substance Result' layer was empty and its layers were not corrected against Substance's picture.
  - Polygon fills are stored differently too (the camera of each pick, and whether it adds or removes), and so are the keys of path strokes. Unread, every polygon fill was skipped and the correction step had to rebuild those masks on its own.
  - All of it is read correctly now. Checked by saving two older test projects with Substance 12.2: the old and the new file of the same project convert to exactly the same files. Files from older Substance versions convert exactly as before.
- **A failing correction step no longer costs you the material.** If fitting the layers to Substance's picture fails for any reason, the layers are imported as rebuilt and the 'Substance Result' layer stays. Before, the whole material was dropped.
- **Base colour kept in another format than the default sRGB8** (RGB8, 16-bit, or greyscale L8 / L16F) came out too dark or failed. Only sRGB8 holds display values; the others hold linear ones, and they are converted now. Measured in Substance.
- **Linux: the graphics card is found in more situations.** The converter asked only the desktop for OpenGL. With no desktop (a terminal over SSH) that failed, and on a remote or virtual desktop without a graphics driver of its own it got software OpenGL (llvmpipe); either way the conversion ran on the processor. It now also asks the graphics devices directly (EGL) and uses the first real card it finds. Tested on an NVIDIA card: with a desktop, with none, and on a virtual desktop that only offered llvmpipe, the converted files are identical. When it had to look further than the desktop, the window and `log.txt` say what each way gave (`OpenGL asked for: ...`).
- `log.txt` now also says how the project numbers its channels and what result textures each material has, which makes reports like the one behind this fix quicker to solve.

## 0.6.5 – 2026-11-26
Fixes the height bumps along painted lines, and the individual layers are truer to Substance, most of all in projects with soft, hand-painted layers. Convert your project again to get it, and let the converter update the plugin (now 0.6.5 too).

- **Bumps along painted lines, with the 'Substance Result' layer hidden.** PawPaint reads a painted layer's height from the image as it is and ignores the image's transparency. Painted layers were written with 'nothing' outside the strokes, which PawPaint took as a deep pit, with a cliff along every stroke edge. A project that has no height at all in Substance came out covered in bumps. A painted layer's height is now flat wherever nothing is painted, and a layer whose strokes carry no height gets no height image at all.
- **Soft edges of painted colour layers were too dark.** The same images stored the colour already multiplied by the stroke's coverage. They now hold the plain colour.
- **Plugin: new 'Height (bumps)' option.** Untick it to import without any height.
- **Soft brush strokes now reach around the model.** A brush with Substance's default alignment paints every surface inside its stamp. The converter skipped the surfaces facing away from the camera, which left stripes on round parts (a tail painted from several sides) and missed the far side of thin ones (fur tufts).
- **Overlapping soft layers keep their own shapes.** Where the rebuilt picture was slightly off, the correction step used to re-explain the spot as 'one layer' or 'one layer over another', which could throw away a third layer or push one to full strength. It now first tries to nudge all the layers there a little, and only rewrites when that can't reproduce Substance's picture. On the project this was tuned on, hiding or recolouring a single layer gives a result much closer to Substance's, and the layers on their own match Substance's picture on more of the texture as well (97.4 % to 98.2 % for its biggest material, measured in PawPaint).
- Conversions are a little faster.

## 0.6.4 – 2026-11-26
Two fixes for layers that came out wrong in PawPaint. Convert your project again to get them. Projects that use neither thing convert exactly as with 0.6.3. The plugin is unchanged (0.6.3).

- **Layers that only showed through the 'Detail fix' layer.** Substance stores blending per channel. A folder or layer whose settings named only some channels (say, height and normal turned down to 0 %) had those settings applied to the whole layer, so a folder could come into PawPaint at 0 % opacity with everything in it invisible. The picture still looked right, because the 'Detail fix' layer painted over it, but the layers were not doing the work: hiding or recolouring them changed nothing. Each channel now gets its own blend mode and opacity, and channels without a setting keep 100 %. On the project this was found on, the layers alone went from 44 % of the texture matching Substance to 97 % in PawPaint, and the 'Detail fix' layer shrank from half the texture to about 1 %.
- **'Random height bumps' on projects with height** (embossed lines, stitches, scales). PawPaint adds a layer's height to the layers below unless that layer's height is set to Replace. The 'Substance Result' layer sat on layers that already had the height, so bumps came out twice as high in patches, with a ridge around each patch. Its height is now set to Replace, so the height is exactly Substance's.
- **Smoother height with the result layer hidden.** The 'Detail fix' layer corrected height only where the layers were clearly off, which left a step around every corrected patch. It now corrects much finer and stores height in 16 bits instead of 8.

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
