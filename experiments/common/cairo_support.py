import ctypes.util, os
from pathlib import Path
def configure_cairo():
    original=ctypes.util.find_library
    candidate=os.environ.get("CAIRO_LIBRARY") or original("cairo") or original("cairo-2")
    if candidate:
        ctypes.util.find_library=lambda name: candidate if name in ("cairo","cairo-2") else original(name)
    return candidate
