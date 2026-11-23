"""User settings: processing engine, detail, CPU cores. Saved between runs.

    engine   "auto" (GPU if one works, else CPU) | "gpu" | "cpu"
    detail   CPU only: "fast" (heavy brush masks at up to 2K, brush-sized
             working resolution) | "lossless" (everything at full resolution;
             slower). The GPU always works at full resolution.
    workers  CPU cores for hand-painted masks: 0 = all available
    check    compare rebuilt layers with Substance's result (diagnostic, slower)
"""

import ctypes
import json
import os
from dataclasses import asdict, dataclass

ENGINES = ("auto", "gpu", "cpu")
DETAILS = ("fast", "lossless")


@dataclass
class Settings:
    engine: str = "auto"
    detail: str = "fast"
    workers: int = 0
    check: bool = False
    resolution: int = 0          # 0 = automatic
    mesh_units: str = "cm"
    layers: bool = True
    flat: bool = True
    baked: bool = True

    def normalised(self):
        s = Settings(**asdict(self))
        if s.engine not in ENGINES:
            s.engine = "auto"
        if s.detail not in DETAILS:
            s.detail = "fast"
        s.workers = max(0, int(s.workers or 0))
        return s


def _path():
    base = os.environ.get("APPDATA") or os.path.expanduser("~/.config")
    return os.path.join(base, "SppToPawPaint", "settings.json")


def load():
    try:
        with open(_path(), encoding="utf-8") as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in Settings.__dataclass_fields__}
        return Settings(**known).normalised()
    except (OSError, ValueError, TypeError):
        return Settings()


def save(s):
    try:
        os.makedirs(os.path.dirname(_path()), exist_ok=True)
        with open(_path(), "w", encoding="utf-8") as f:
            json.dump(asdict(s.normalised()), f, indent=1)
    except OSError:
        pass


def available_memory_gb():
    """Free physical memory in GB (best effort)."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / 2 ** 30
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 8.0


def cpu_workers(requested, res, lossless):
    """All cores by default, but never more workers than free memory allows."""
    cores = os.cpu_count() or 2
    want = cores if not requested else min(requested, cores)
    # rough per-worker footprint: mesh + per-part surface maps + buffers
    per = 0.8 + (2.5 if lossless and res >= 4096 else 0.8 if res >= 2048 else 0.3)
    by_mem = int(available_memory_gb() * 0.8 / per)
    return max(1, min(want, by_mem))


def gpu_workers(requested):
    """Worker processes sharing the GPU: each needs a few hundred MB of video
    memory and mostly does CPU-side work around the GPU, so a handful is best."""
    cores = os.cpu_count() or 2
    want = min(4, max(1, cores - 1)) if not requested else min(requested, cores)
    by_mem = int(available_memory_gb() * 0.8 / 1.5)
    return max(1, min(want, by_mem))
