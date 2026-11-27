Substance Painter (.spp) -> PawPaint importer
==============================================

EARLY RELEASE - STILL BEING WORKED ON
  The picture in PawPaint matches Substance very closely, because every material gets
  an exact copy of Substance's result. The editable layers underneath are rebuilt, and
  not every layer comes out 100% perfect: hand-painted masks can differ slightly at
  their edges, and Substance-only effects (fur filters, generators, smart masks, some
  blend modes) can't be rebuilt as separate layers. report.txt lists what couldn't be
  converted exactly. Keep your original .spp.

USE (no Substance needed)
  1. Drag your .spp file onto SppToPawPaint.exe (or double-click it and pick the file).
     The first time, it offers to install the PawPaint plugin for you. Say yes.
  2. Open PawPaint (restart it if it was already open, the first time only).
     Plugins > Substance Importer: your conversion is already filled in. Click Import.
  3. When it says Done: File > Save As.

WHAT YOU GET, PER MATERIAL
  - "Substance Result" on top: an exact copy of what Substance showed. Hide it to edit
    the rebuilt layers underneath.
  - The rebuilt Substance layers: folders, fill layers with their colours, and one
    mask per layer, corrected against Substance's own result. Recolour or hide a
    layer and the rest stays as it was.
  - "Detail fix (from Substance)": a small layer with the few spots no layer explains.
  - "Black base (as in Substance)": only where nothing fully covers an area (Substance
    shows those areas over black).
  - "Baked Normal (from Substance)": the baked normal map. Other baked maps (ID,
    curvature, position, thickness...) are in the converted folder under materials\<name>\baked.
  - report.txt in the converted folder lists anything that couldn't be converted exactly.
  - log.txt in the converted folder is the whole run with times. If a conversion is slow
    or fails, attach it to your GitHub issue.

TIPS
  - Want shape keys? In the plugin choose Mesh: "My own model file" and give your FBX.
    Or open your FBX in PawPaint first and choose "The project open now".
  - Texture sets that don't match a material by name can be matched by hand in the panel.
  - Don't want the height (bumps) from Substance? Untick "Height (bumps)" in the plugin
    before you click Import.
  - Windows may warn about an unknown app the first time (it isn't code-signed):
    More info > Run anyway.
  - Big projects take a while: a material with over a hundred layers at 4096 can need
    several minutes for "fitting masks". The window shows how far it is. If you close the
    converter early, the materials that were finished can still be imported.
  - PawPaint keeps every layer on the graphics card. A material with over a hundred
    layers needs about 15 GB of video memory at 4096 and under 4 GB at 2048. Leave
    Resolution (converter) and Size (plugin) on Auto: every material keeps its own
    size, one that doesn't fit your graphics card is made a size smaller with every
    layer kept, and you are told when that happened. (PawPaint 1.28 or newer sizes
    each material on its own; older versions do it for the whole project.) Closing other programs that use the graphics card (Substance, games,
    browsers) frees video memory.

OWN SUBSTANCE? (optional, for exact layers)
  See "Substance plugin (optional)": it exports every layer exactly, including
  procedural effects. Then run SppToPawPaint as usual; it picks the export up.

MANUAL PLUGIN INSTALL (only if the automatic one can't find PawPaint)
  Steam > right-click PawPaint > Manage > Browse local files, then copy both files from
  "PawPaint scripts" into PawPaint's "scripts" folder. If scripts\autoload.lua already
  exists, add our dofile line to it instead of replacing it.

DISCLAIMER
  Unofficial community tool. Not made, endorsed or supported by PawWorks (PawPaint) or
  Adobe. Only convert projects you own or have the right to use. Keep your original .spp,
  which this tool only reads and never changes. Provided "as is", without warranty.
  Substance and Substance 3D Painter are trademarks of Adobe.
