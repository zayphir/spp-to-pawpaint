"""A small window for people who double-click the exe or drop a file on it."""

import os
import queue
import subprocess
import sys
import threading
import traceback

from . import __version__


def _hide_console():
    """The exe is a console app (so the CLI works); hide that window in GUI mode."""
    if os.name == "nt" and getattr(sys, "frozen", False):
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass


def run_gui(initial=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from . import cli

    _hide_console()
    root = tk.Tk()
    root.title("SppToPawPaint %s" % __version__)
    root.minsize(700, 560)
    msgs = queue.Queue()
    state = {"busy": False, "out": None}

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    ttk.Label(frm, text="Substance project (.spp)").grid(row=0, column=0, sticky="w")
    path_var = tk.StringVar(value=initial or "")
    ttk.Entry(frm, textvariable=path_var).grid(row=0, column=1, sticky="ew", padx=6)

    def browse():
        p = filedialog.askopenfilename(title="Choose a Substance 3D Painter project",
                                       filetypes=[("Substance Painter project", "*.spp"), ("All files", "*.*")])
        if p:
            path_var.set(p)
    ttk.Button(frm, text="Browse...", command=browse).grid(row=0, column=2)

    from . import settings as SET
    cfg = SET.load()

    opts = ttk.Frame(frm)
    opts.grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 4))
    ttk.Label(opts, text="Resolution").pack(side="left")
    res_var = tk.StringVar(value=str(cfg.resolution) if cfg.resolution else "Auto")
    ttk.Combobox(opts, textvariable=res_var, width=7, state="readonly",
                 values=["Auto", "4096", "2048", "1024", "512"]).pack(side="left", padx=(4, 14))
    ttk.Label(opts, text="Model units").pack(side="left")
    unit_var = tk.StringVar(value=cfg.mesh_units)
    ttk.Combobox(opts, textvariable=unit_var, width=5, state="readonly",
                 values=["cm", "m", "mm", "in", "raw"]).pack(side="left", padx=(4, 14))
    layers_var, flat_var, baked_var = tk.BooleanVar(value=cfg.layers), tk.BooleanVar(value=cfg.flat), tk.BooleanVar(value=cfg.baked)
    ttk.Checkbutton(opts, text="Rebuilt layers", variable=layers_var).pack(side="left")
    ttk.Checkbutton(opts, text="Substance result", variable=flat_var).pack(side="left", padx=6)
    ttk.Checkbutton(opts, text="Baked maps", variable=baked_var).pack(side="left")

    # processing settings (saved between runs)
    proc = ttk.LabelFrame(frm, text="Processing", padding=(8, 4))
    proc.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
    ENG = {"auto": "Auto (GPU if available)", "gpu": "GPU", "cpu": "CPU"}
    DET = {"fast": "Fast", "lossless": "Full detail (lossless, slower)"}
    ttk.Label(proc, text="Engine").grid(row=0, column=0, sticky="w")
    eng_var = tk.StringVar(value=ENG[cfg.engine])
    ttk.Combobox(proc, textvariable=eng_var, width=24, state="readonly",
                 values=list(ENG.values())).grid(row=0, column=1, sticky="w", padx=(4, 14))
    ttk.Label(proc, text="CPU detail").grid(row=0, column=2, sticky="w")
    det_var = tk.StringVar(value=DET[cfg.detail])
    det_box = ttk.Combobox(proc, textvariable=det_var, width=28, state="readonly", values=list(DET.values()))
    det_box.grid(row=0, column=3, sticky="w", padx=(4, 14))
    ttk.Label(proc, text="CPU cores (0 = all)").grid(row=1, column=0, sticky="w", pady=(4, 0))
    cores_var = tk.IntVar(value=cfg.workers)
    ttk.Spinbox(proc, from_=0, to=max(os.cpu_count() or 1, 1), textvariable=cores_var, width=6).grid(
        row=1, column=1, sticky="w", padx=(4, 14), pady=(4, 0))
    check_var = tk.BooleanVar(value=cfg.check)
    ttk.Checkbutton(proc, text="Compare with Substance's result (slower)", variable=check_var).grid(
        row=1, column=2, columnspan=2, sticky="w", pady=(4, 0))
    gpu_var = tk.StringVar(value="GPU: checking...")
    ttk.Label(proc, textvariable=gpu_var).grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
    ttk.Label(proc, text="The GPU always works at full detail. CPU detail only applies when the CPU is used.",
              foreground="#666").grid(row=3, column=0, columnspan=4, sticky="w")

    def detect_gpu():
        try:
            from . import gpu
            ok, name = gpu.available()
            if ok and gpu.is_software(name):
                msgs.put(("gpu", "GPU: only software OpenGL (%s); Auto uses the CPU, which is faster" % name))
            else:
                msgs.put(("gpu", ("GPU: %s" % name) if ok else "GPU: not available (%s), the CPU will be used" % name))
        except Exception as e:
            msgs.put(("gpu", "GPU: not available (%s), the CPU will be used" % e))
    threading.Thread(target=detect_gpu, daemon=True).start()

    plug = ttk.Frame(frm)
    plug.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
    plug_var = tk.StringVar(value=cli.maybe_install_plugin())
    ttk.Label(plug, textvariable=plug_var).pack(side="left")

    def install():
        plug_var.set(cli.maybe_install_plugin(assume_yes=True, log=lambda m: msgs.put(("log", m))))
    ttk.Button(plug, text="Install / update PawPaint plugin", command=install).pack(side="right")

    bar = ttk.Progressbar(frm, mode="determinate", maximum=100)
    bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    logbox = tk.Text(frm, height=14, wrap="word", state="disabled")
    logbox.grid(row=5, column=0, columnspan=3, sticky="nsew")
    frm.rowconfigure(5, weight=1)

    btns = ttk.Frame(frm)
    btns.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    convert_btn = ttk.Button(btns, text="Convert")
    convert_btn.pack(side="left")
    open_btn = ttk.Button(btns, text="Open folder", state="disabled")
    open_btn.pack(side="left", padx=6)
    copy_btn = ttk.Button(btns, text="Copy path", state="disabled")
    copy_btn.pack(side="left")
    ttk.Label(btns, text="Then in PawPaint: Plugins > Substance Importer > Import").pack(side="right")

    def log(m):
        logbox.configure(state="normal")
        logbox.insert("end", m + "\n")
        logbox.see("end")
        logbox.configure(state="disabled")

    def worker(path, kw):
        try:
            out = cli.convert(path, log=lambda m: msgs.put(("log", m)),
                              progress=lambda d, t, label: msgs.put(("progress", (d, t, label))), **kw)
            msgs.put(("done", out))
        except Exception as e:
            msgs.put(("log", traceback.format_exc()))
            msgs.put(("error", str(e)))

    def start():
        path = path_var.get().strip().strip('"')
        if not path or not os.path.exists(path):
            messagebox.showwarning("SppToPawPaint", "Choose a .spp file first.")
            return
        if state["busy"]:
            return
        state["busy"] = True
        convert_btn.configure(state="disabled")
        r = res_var.get()
        engine = next(k for k, v in ENG.items() if v == eng_var.get())
        detail = next(k for k, v in DET.items() if v == det_var.get())
        try:
            cores = max(0, int(cores_var.get()))
        except (tk.TclError, ValueError):
            cores = 0
        kw = dict(resolution=None if r == "Auto" else int(r), layers=layers_var.get(), flat=flat_var.get(),
                  baked=baked_var.get(), mesh_units=unit_var.get(), engine=engine, detail=detail,
                  workers=cores, check=check_var.get())
        SET.save(SET.Settings(engine=engine, detail=detail, workers=cores, check=check_var.get(),
                              resolution=0 if r == "Auto" else int(r), mesh_units=unit_var.get(),
                              layers=layers_var.get(), flat=flat_var.get(), baked=baked_var.get()))
        bar.configure(value=0)
        threading.Thread(target=worker, args=(path, kw), daemon=True).start()
    convert_btn.configure(command=start)

    def open_folder():
        if state["out"]:
            if os.name == "nt":
                os.startfile(state["out"])
            else:
                subprocess.Popen(["xdg-open", state["out"]])
    open_btn.configure(command=open_folder)

    def copy_path():
        if state["out"]:
            from . import install
            root.clipboard_clear()
            root.clipboard_append(install.pawpaint_path(install.find_pawpaint(), state["out"]))
    copy_btn.configure(command=copy_path)

    def pump():
        try:
            while True:
                kind, val = msgs.get_nowait()
                if kind == "log":
                    log(val)
                elif kind == "gpu":
                    gpu_var.set(val)
                elif kind == "progress":
                    d, t, label = val
                    bar.configure(maximum=max(t, 1), value=d)
                elif kind == "done":
                    state.update(busy=False, out=val)
                    convert_btn.configure(state="normal")
                    open_btn.configure(state="normal")
                    copy_btn.configure(state="normal")
                    log("\nNext: open PawPaint > Plugins > Substance Importer and click Import.")
                elif kind == "error":
                    state["busy"] = False
                    convert_btn.configure(state="normal")
                    messagebox.showerror("SppToPawPaint", "Conversion failed:\n%s" % val)
        except queue.Empty:
            pass
        root.after(100, pump)
    root.after(100, pump)
    if initial:
        root.after(300, start)
    root.mainloop()
    return 0
