import ctypes
import importlib.util
import os
import sys
from typing import Optional


def setup_cublas_path():
    spec = importlib.util.find_spec("nvidia.cublas")
    if spec is None:
        return

    pkg_dir: Optional[str] = None
    if spec.submodule_search_locations:
        pkg_dir = list(spec.submodule_search_locations)[0]
    elif spec.origin and spec.origin != "frozen":
        pkg_dir = os.path.dirname(spec.origin)

    if not pkg_dir:
        return

    bin_dir = os.path.join(pkg_dir, "bin" if sys.platform == "win32" else "lib")
    if not os.path.isdir(bin_dir):
        bin_dir = pkg_dir  # Fallback to base package dir if structure varies

    if sys.platform == "win32":
        os.add_dll_directory(bin_dir)
        os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ.get("PATH", "")
    else:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if bin_dir not in ld_path.split(os.path.pathsep):
            os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld_path}" if ld_path else bin_dir

        # Pre-load shared libraries directly into memory on Linux since modifying
        # LD_LIBRARY_PATH after Python process start doesn't always affect C-bindings
        for file in os.listdir(bin_dir):
            if file.startswith("libcublas") and file.endswith(".so"):
                so_path = os.path.join(bin_dir, file)
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
