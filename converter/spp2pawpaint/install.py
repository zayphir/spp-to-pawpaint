"""Find PawPaint, install/update the importer plugin, and leave a handoff note.

PawPaint (Steam app 4776250) keeps Lua plugins in <install>/scripts/. The
plugin can read JSON files but can't write any, so after converting we drop
scripts/spp_importer_pending.json; the plugin reads it and pre-fills the
folder, so the user only has to click Import.

On Linux the Windows build of PawPaint runs through Steam's Proton (Wine). It
then sees files through drive letters (Z: is / by default), so the folder in
the handoff note is written the way PawPaint will see it (`pawpaint_path`).
"""

import json
import os
import re
import shutil
import sys
import time

APP_ID = "4776250"
PENDING_NAME = "spp_importer_pending.json"
AUTOLOAD_MARK = "spp_importer.lua"
AUTOLOAD_BLOCK = """
-- Substance (.spp) importer, added by SppToPawPaint.
do
    local ok, err = pcall(dofile, "scripts/spp_importer.lua")
    if not ok then pp.log("Could not load spp_importer.lua: " .. tostring(err)) end
end
"""


def bundled_scripts_dir():
    """Folder holding spp_importer.lua / autoload.lua (inside the exe or the repo)."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, "plugin_scripts")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "plugin", "scripts"))


# Steam folders under the home folder: the usual Linux places, Flatpak, Snap, macOS
HOME_STEAM_DIRS = (
    ".steam/steam", ".steam/root", ".local/share/Steam", ".steam/debian-installation",
    ".var/app/com.valvesoftware.Steam/.local/share/Steam",
    ".var/app/com.valvesoftware.Steam/data/Steam",
    "snap/steam/common/.local/share/Steam", "snap/steam/common/.steam/steam",
    "Library/Application Support/Steam",
)
PROGRAM_EXTS = ("", ".exe", ".x86_64", ".sh", ".appimage")


def _steam_roots():
    roots = []
    try:
        import winreg
        for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam")):
            for value in ("SteamPath", "InstallPath"):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        roots.append(winreg.QueryValueEx(k, value)[0])
                except OSError:
                    pass
    except ImportError:  # not Windows
        pass
    if os.name == "nt":
        roots += [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam"]
    home = os.path.expanduser("~")
    roots += [os.path.join(home, *d.split("/")) for d in HOME_STEAM_DIRS]
    out, seen = [], set()
    for r in roots:
        r = os.path.normpath(r.replace("/", os.sep))
        key = os.path.normcase(os.path.realpath(r))  # ~/.steam/steam is usually a link
        if os.path.isdir(r) and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _steamapps(lib):
    """The library's steamapps folder (old Linux installs spell it SteamApps)."""
    for name in ("steamapps", "SteamApps"):
        p = os.path.join(lib, name)
        if os.path.isdir(p):
            return p
    return os.path.join(lib, "steamapps")


def program_file(folder):
    """Name of the PawPaint program in `folder` (any letter case), or None.
    A native program is preferred over a Windows .exe."""
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    found = None
    for n in sorted(names):
        stem, ext = os.path.splitext(n.lower())
        if stem == "pawpaint" and ext in PROGRAM_EXTS and os.path.isfile(os.path.join(folder, n)):
            if ext != ".exe":
                return n
            found = found or n
    return found


def _libraries(root):
    libs = [root]
    vdf = os.path.join(_steamapps(root), "libraryfolders.vdf")
    try:
        with open(vdf, encoding="utf-8", errors="replace") as f:
            for m in re.finditer(r'"path"\s+"([^"]+)"', f.read()):
                libs.append(m.group(1).replace("\\\\", "\\"))
    except OSError:
        pass
    return libs


def find_pawpaint(explicit=None):
    """Return the PawPaint install folder, or None."""
    for cand in (explicit, os.environ.get("PAWPAINT_DIR")):
        if cand and os.path.isdir(cand):
            return cand
    for root in _steam_roots():
        for lib in _libraries(root):
            manifest = os.path.join(_steamapps(lib), "appmanifest_%s.acf" % APP_ID)
            installdir = "PawPaint"
            if os.path.exists(manifest):
                try:
                    with open(manifest, encoding="utf-8", errors="replace") as f:
                        m = re.search(r'"installdir"\s+"([^"]+)"', f.read())
                        if m:
                            installdir = m.group(1)
                except OSError:
                    pass
            folder = os.path.join(_steamapps(lib), "common", installdir)
            if os.path.isdir(folder) and program_file(folder):
                return folder
    return None


# -- paths as PawPaint sees them (Proton / Wine on Linux) -----------------------------------
def runs_through_wine(pawpaint_dir):
    """True when this PawPaint is the Windows build on a system that isn't Windows."""
    if os.name == "nt" or not pawpaint_dir:
        return False
    return (program_file(pawpaint_dir) or "").lower().endswith(".exe")


def _wine_drives(pawpaint_dir):
    """{drive letter: Linux folder} of the Wine prefix PawPaint runs in."""
    prefixes = []
    for var, sub in (("STEAM_COMPAT_DATA_PATH", "pfx"), ("WINEPREFIX", "")):
        if os.environ.get(var):
            prefixes.append(os.path.join(os.environ[var], sub))
    # <library>/steamapps/common/PawPaint -> <library>/steamapps/compatdata/<app id>/pfx
    steamapps = os.path.dirname(os.path.dirname(os.path.abspath(pawpaint_dir)))
    prefixes.append(os.path.join(steamapps, "compatdata", APP_ID, "pfx"))
    for prefix in prefixes:
        dos = os.path.join(prefix, "dosdevices")
        drives = {}
        try:
            for n in os.listdir(dos):
                if len(n) == 2 and n[1] == ":" and n[0].isalpha():
                    drives[n[0].lower()] = os.path.realpath(os.path.join(dos, n))
        except OSError:
            continue
        if drives:
            return drives
    return {"z": "/"}


def wine_path(path, drives):
    """A Linux path written with the drive letter that reaches it (longest match)."""
    best = None
    for letter, target in sorted(drives.items()):
        t = target.rstrip("/")
        if path == t or path.startswith(t + "/"):
            if best is None or len(t) > len(best[1]):
                best = (letter, t)
    if best is None:
        best = ("z", "")
    letter, t = best
    return letter.upper() + ":" + (path[len(t):].replace("/", "\\") or "\\")


def pawpaint_path(pawpaint_dir, path):
    """`path` the way PawPaint sees it: unchanged, except through Proton/Wine."""
    path = os.path.abspath(path)
    if not runs_through_wine(pawpaint_dir):
        return path
    return wine_path(path, _wine_drives(pawpaint_dir))


def _version(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = re.search(r'IMPORTER_VERSION\s*=\s*"([^"]+)"', f.read())
            return tuple(int(x) for x in m.group(1).split(".")) if m else None
    except (OSError, ValueError):
        return None


def plugin_status(pawpaint_dir):
    """'missing', 'outdated' or 'current'."""
    src = os.path.join(bundled_scripts_dir(), "spp_importer.lua")
    dst = os.path.join(pawpaint_dir, "scripts", "spp_importer.lua")
    if not os.path.exists(dst):
        return "missing"
    sv, dv = _version(src), _version(dst)
    if sv and (not dv or dv < sv):
        return "outdated"
    autoload = os.path.join(pawpaint_dir, "scripts", "autoload.lua")
    try:
        with open(autoload, encoding="utf-8", errors="replace") as f:
            if AUTOLOAD_MARK not in f.read():
                return "outdated"
    except OSError:
        return "outdated"
    return "current"


def install_plugin(pawpaint_dir, log=print):
    """Copy the plugin and make sure autoload.lua loads it. Never overwrites a
    user's own autoload.lua: our block is appended once."""
    src_dir = bundled_scripts_dir()
    scripts = os.path.join(pawpaint_dir, "scripts")
    os.makedirs(scripts, exist_ok=True)
    shutil.copyfile(os.path.join(src_dir, "spp_importer.lua"), os.path.join(scripts, "spp_importer.lua"))
    autoload = os.path.join(scripts, "autoload.lua")
    if not os.path.exists(autoload):
        shutil.copyfile(os.path.join(src_dir, "autoload.lua"), autoload)
        log("  installed spp_importer.lua and autoload.lua into %s" % scripts)
    else:
        with open(autoload, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if AUTOLOAD_MARK not in text:
            with open(autoload, "a", encoding="utf-8") as f:
                f.write(AUTOLOAD_BLOCK)
            log("  installed spp_importer.lua and added it to your existing autoload.lua")
        else:
            log("  updated spp_importer.lua in %s" % scripts)
    return True


def write_pending(pawpaint_dir, out_dir, source, materials):
    """Tell the plugin which conversion to offer next."""
    path = os.path.join(pawpaint_dir, "scripts", PENDING_NAME)
    data = {"format": "spp2pawpaint-pending", "version": 1, "folder": pawpaint_path(pawpaint_dir, out_dir),
            "source": os.path.basename(source), "created": int(time.time()),
            "createdText": time.strftime("%Y-%m-%d %H:%M"), "materials": list(materials)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return path


def copy_to_clipboard(text):
    import subprocess
    if os.name == "nt":
        try:
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        except Exception:
            pass
    else:
        # A Tk clipboard is gone as soon as its window closes on Linux, so only
        # say the path was copied when a real clipboard tool took it.
        tools = [["pbcopy"]] if sys.platform == "darwin" else []
        if os.environ.get("WAYLAND_DISPLAY"):
            tools.append(["wl-copy"])
        if os.environ.get("DISPLAY"):
            tools += [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
        for cmd in tools:
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=5,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except Exception:
                    pass
        return False
    try:
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True
    except Exception:
        return False
