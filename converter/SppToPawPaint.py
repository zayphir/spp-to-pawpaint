"""Launcher used for the standalone SppToPawPaint.exe build."""
import multiprocessing
import sys

from spp2pawpaint.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # worker processes inside the frozen exe
    sys.exit(main())
