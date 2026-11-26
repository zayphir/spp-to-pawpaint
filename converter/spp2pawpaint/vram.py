"""Video memory: what PawPaint needs for an import, and what this PC has.

PawPaint (1.27 and later) keeps every imported layer on the graphics card and
refuses a texture set whose layers need more video memory than is free. The
plugin then has to leave layers out, so it pays to know the number before a
long conversion at a size that cannot be imported anyway.

How PawPaint counts (measured on PawPaint 1.27 by asking it to import far too
many layers of one kind and reading the number in its refusal; the result is
exact for every combination tried, and the texture sets of a test project add
up to the megabyte):

    at 4096 x 4096                                      MB
    a mask on any layer                                 117.33  (32 for the mask + 85.33 for its image file)
    an image on a fill layer, any channel                85.33
    a painted colour, normal or emission channel         80
    a painted roughness or metallic channel              32
    a painted height channel                             48
    folders, layers without masks or images               0

Everything scales with the texture area (a quarter at 2048). The part for an
image file goes by the file's own size when that is smaller than the project.
"""

import os

MASK_MB = 32.0                    # a layer mask, at the project's size
IMAGE_MB = 256.0 / 3.0            # an image file kept on the card (8-bit RGBA and its smaller copies)
PAINT_MB = {"color": 80.0, "vector": 80.0, "scalar": 32.0, "signed": 48.0}
CHANNEL_KINDS = {"albedo": "color", "emission": "color", "normal": "vector", "roughness": "scalar",
                 "metallic": "scalar", "height": "signed", "opacity": "scalar", "specular": "color",
                 "glossiness": "scalar", "displacement": "signed"}
SIZES = (4096, 2048, 1024, 512)
MIN_AUTO_SIZE = 1024              # 'auto' never goes below this by itself

# What a texture set gets besides its own layers, for the guess made before
# converting: the Substance result (five painted channels), the detail fix
# (colour and roughness) and the baked normal (an image on a fill layer).
EXTRA_LAYERS_MB = 272.0 + 112.0 + IMAGE_MB
# Layers that also carry an image (baked textures, painted layers) add about
# this much to 'every layer has a mask' on the projects looked at.
GUESS_FACTOR = 1.1


def _area(res):
    return (float(res) / 4096.0) ** 2


def need_mb(manifest, res, file_res=None):
    """Video memory PawPaint counts for one texture set's layers at `res`.
    `file_res`: the size of the image files, when it differs from `res`."""
    s = _area(res)
    f = min(s, _area(file_res or res))
    kinds = dict(CHANNEL_KINDS)
    for c in manifest.get("channels") or []:
        if isinstance(c, dict) and c.get("id"):
            kinds[c["id"]] = c.get("kind") or "scalar"
    total = 0.0
    for layer in manifest.get("layers") or []:
        total += len(layer.get("masks") or []) * (MASK_MB * s + IMAGE_MB * f)
        channels = layer.get("channels") or {}
        if layer.get("type") == "Paint":
            total += sum(PAINT_MB.get(kinds.get(cid, "scalar"), 32.0) for cid in channels) * s
        elif layer.get("type") == "Fill":
            total += len(channels) * IMAGE_MB * f
    return total


def count_layers(material):
    """(layers, folders) in a texture set of the document."""
    layers = folders = 0
    todo = [((material.get("stacks") or [{}])[0].get("stack") or {}).get("items") or []]
    while todo:
        for node in todo.pop():
            if not node:
                continue
            sub = node.get("subStack")
            if sub:
                folders += 1
                todo.append(sub.get("items") or [])
            else:
                layers += 1
    return layers, folders


def guess_mb(material, res):
    """need_mb before the texture set is converted: every layer with a mask,
    plus the layers the converter adds."""
    layers, _ = count_layers(material)
    return (layers * (MASK_MB + IMAGE_MB) * GUESS_FACTOR + EXTRA_LAYERS_MB) * _area(res)


# -- this PC ----------------------------------------------------------------

def _registry_cards():
    """[(name, MB)] of the graphics cards Windows knows, from the registry."""
    if os.name != "nt":
        return []
    import winreg
    cards = []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return []
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                    try:
                        size = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")[0]
                    except OSError:
                        size = winreg.QueryValueEx(key, "HardwareInformation.MemorySize")[0]
                    if isinstance(size, bytes):
                        size = int.from_bytes(size, "little")
                    size = int(size)
            except (OSError, ValueError, TypeError):
                continue
            if size > 0:
                cards.append((name, size / 2.0 ** 20))
    return cards


def _linux_cards():
    """[(name, MB)] from the kernel's graphics drivers (AMD and Intel report it)."""
    cards = []
    try:
        import glob
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            with open(path) as f:
                size = int(f.read().strip())
            if size > 0:
                cards.append((os.path.basename(os.path.dirname(os.path.dirname(path))), size / 2.0 ** 20))
    except (OSError, ValueError):
        pass
    return cards


VENDOR_WORDS = ("nvidia", "geforce", "quadro", "radeon", "amd", "intel", "arc")


def _same_card(a, b):
    """True when two card names are from the same maker (all that can be told from names)."""
    a, b = str(a).lower(), str(b).lower()
    return any(w in a and w in b for w in VENDOR_WORDS)


def video_memory(use_opengl=True):
    """{'total': MB or None, 'free': MB or None, 'name': str} for the graphics
    card PawPaint will most likely use, or None when nothing can be found out.

    'free' comes from the graphics driver through the converter's own OpenGL
    context (NVIDIA and AMD report it); 'total' from there or from the
    operating system. With several cards, the one with the most memory."""
    total = free = None
    name = ""
    if use_opengl:
        try:
            from . import gpu
            ok, renderer = gpu.available()
            if ok and not gpu.is_software(renderer):
                total, free = gpu.memory_mb()
                name = str(renderer)
        except Exception:
            total = free = None
    cards = _registry_cards() or _linux_cards()
    if cards:
        best = max(cards, key=lambda c: c[1])
        same = _same_card(name, best[0]) or len(cards) == 1
        if total is None:
            # the driver gave no total (AMD); what it says is free only counts for its own card
            if not same:
                free = None
            total, name = best[1], best[0]
        elif best[1] > total * 1.5 and not same:
            # OpenGL runs on a smaller card than the best one here (a laptop's built-in graphics)
            total, free, name = best[1], None, best[0]
    if total is None and free is None:
        return None
    if total is not None and free is not None:
        free = min(free, total)
    return {"total": total, "free": free, "name": name}


def describe(memory):
    """'NVIDIA ..., 12.0 GB of video memory, 8.3 GB free now' for the log."""
    if not memory:
        return "unknown"
    parts = [memory.get("name") or "unknown"]
    if memory.get("total"):
        parts.append("%s of video memory" % gb(memory["total"]))
    if memory.get("free"):
        parts.append("%s free now" % gb(memory["free"]))
    return ", ".join(parts)


SHARE_OF_TOTAL = 0.75     # of a card's memory: what PawPaint can count on with little else open


def budget_mb(memory):
    """What PawPaint can be expected to have free on this card when it
    imports, in MB. What is free right now is the best number there is, but a
    program that is open now (PawPaint with another project, a game) may be
    closed by then, so it is never taken as less than SHARE_OF_TOTAL of the
    card. Erring on the high side only costs time: the plugin imports at a
    smaller size when PawPaint refuses."""
    if not memory:
        return None
    total, free = memory.get("total"), memory.get("free")
    if total and free:
        return max(float(free), float(total) * SHARE_OF_TOTAL)
    if free:
        return float(free)
    if total:
        return float(total) * SHARE_OF_TOTAL
    return None


USED_SHARE = 0.6          # of what PawPaint counted for a texture set, once it is imported


def fits(set_needs, budget):
    """True when texture sets needing `set_needs` MB each, in the order they
    are imported, can all be imported with `budget` MB free. PawPaint checks
    each texture set against what is free at that moment. An imported set
    then takes about half of what was counted for it (measured: 5.7 GB
    counted for four texture sets, 3.0 GB less free afterwards); USED_SHARE
    of the count is taken off for the sets that follow."""
    used = 0.0
    for need in set_needs:
        if need + used > budget:
            return False
        used += USED_SHARE * need
    return True


def auto_size(native, needs_at, budget):
    """The largest size from `native` down to MIN_AUTO_SIZE whose texture sets
    all fit `budget`; `needs_at(size)` gives their needs in MB. Falls back to
    MIN_AUTO_SIZE (or `native` when that is smaller)."""
    size = native
    while size > MIN_AUTO_SIZE and not fits(needs_at(size), budget):
        size //= 2
    return size


def gb(mb):
    """A number of megabytes for a message: '15.1 GB', or '165 MB' under one gigabyte."""
    return "%d MB" % round(mb) if mb < 1000 else "%.1f GB" % (mb / 1024.0)
