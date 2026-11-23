"""Find PawPaint, install/update the importer plugin, and leave a handoff note.

PawPaint (Steam app 4776250) keeps Lua plugins in <install>/scripts/. The
plugin can read JSON files but can't write any, so after converting we drop
scripts/spp_importer_pending.json; the plugin reads it and pre-fills the
folder, so the user only has to click Import.
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
    roots += [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam",
              os.path.expanduser("~/.steam/steam"), os.path.expanduser("~/.local/share/Steam")]
    out = []
    for r in roots:
        r = os.path.normpath(r.replace("/", os.sep))
        if os.path.isdir(r) and r.lower() not in (x.lower() for x in out):
            out.append(r)
    return out


def _libraries(root):
    libs = [root]
    vdf = os.path.join(root, "steamapps", "libraryfolders.vdf")
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
            manifest = os.path.join(lib, "steamapps", "appmanifest_%s.acf" % APP_ID)
            installdir = "PawPaint"
            if os.path.exists(manifest):
                try:
                    with open(manifest, encoding="utf-8", errors="replace") as f:
                        m = re.search(r'"installdir"\s+"([^"]+)"', f.read())
                        if m:
                            installdir = m.group(1)
                except OSError:
                    pass
            folder = os.path.join(lib, "steamapps", "common", installdir)
            if os.path.isdir(folder) and any(os.path.exists(os.path.join(folder, e))
                                             for e in ("PawPaint.exe", "PawPaint", "pawpaint")):
                return folder
    return None


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
    data = {"format": "spp2pawpaint-pending", "version": 1, "folder": os.path.abspath(out_dir),
            "source": os.path.basename(source), "created": int(time.time()),
            "createdText": time.strftime("%Y-%m-%d %H:%M"), "materials": list(materials)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return path


def copy_to_clipboard(text):
    if os.name == "nt":
        try:
            import subprocess
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        except Exception:
            pass
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
