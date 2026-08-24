"""Compat-Shim: laedt lightx2v_platform.base.global_var als Einzelpfad.

vae.py macht `from lightx2v_platform.base.global_var import AI_DEVICE`.
Das echte lightx2v_platform-__init__ importiert alle Plattform-Treiber
(nvidia, amd, npu, ...) — irrelevant fuer uns und teils nicht installierbar.
Dieser PTH-Hook registriert einen Import-Finder, der NUR global_var als
Einzeldatei laedt und dafuer sorgt, dass sys.modules["lightx2v_platform"]
ein leeres Namespace-Paket ist.
"""
import importlib.util
import sys
from pathlib import Path

_GV_PATH = Path("/usr/local/lib/python3.10/site-packages/lightx2v_platform/base/global_var.py")


def _load_global_var():
    spec = importlib.util.spec_from_file_location(
        "lightx2v_platform.base.global_var", _GV_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lightx2v_platform.base.global_var" and _GV_PATH.exists():
            # Elternpakete als leere Module vorregistrieren
            for parent in ("lightx2v_platform", "lightx2v_platform.base"):
                if parent not in sys.modules:
                    m = importlib.util.module_from_spec(
                        importlib.util.spec_from_loader(parent, loader=None))
                    m.__path__ = []
                    sys.modules[parent] = m
            return importlib.util.spec_from_file_location(fullname, _GV_PATH)
        return None


sys.meta_path.insert(0, _Finder())
