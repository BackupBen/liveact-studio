"""Compat-Shim: laedt lightx2v_platform.base.global_var als Einzelpfad.

vae.py macht `from lightx2v_platform.base.global_var import AI_DEVICE`.
Das echte lightx2v_platform-__init__ wuerde alle Plattform-Treiber importieren
und init_ai_device() laufen lassen. Wir umgehen das: global_var wird direkt
aus der Datei geladen, AI_DEVICE auf "cuda" gesetzt (Worker läuft nur auf GPU)
und EAGER in sys.modules registriert — damit liefert der Import in vae.py
sofort den gepatchten Wert. utils.py's getattr(torch, AI_DEVICE) funktioniert.
"""
import importlib.util
import sys
from pathlib import Path

_SP = Path("/usr/local/lib/python3.10/site-packages")
_GV_PATH = _SP / "lightx2v_platform" / "base" / "global_var.py"


def _register(parent_name: str) -> None:
    if parent_name in sys.modules:
        return
    m = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(parent_name, loader=None))
    m.__path__ = []
    sys.modules[parent_name] = m


def _setup():
    if not _GV_PATH.exists():
        return
    _register("lightx2v_platform")
    _register("lightx2v_platform.base")
    modname = "lightx2v_platform.base.global_var"
    if modname in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(modname, _GV_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Normalerweise setzt lightx2v_platform.init_ai_device() diesen Wert.
    # Wir sind ein CUDA-Worker -> direkt setzen (string, wie torch.cuda erwartet).
    if not getattr(mod, "AI_DEVICE", None):
        mod.AI_DEVICE = "cuda"
    sys.modules[modname] = mod


try:
    _setup()
except Exception:
    pass  # nie den Interpreter-Start blockieren


class _Finder:
    """Fallback fuer den Fall, dass jemand die Module frisch importiert,
    obwohl das eager Setup nicht griff."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lightx2v_platform.base.global_var" and _GV_PATH.exists():
            _register("lightx2v_platform")
            _register("lightx2v_platform.base")
            return importlib.util.spec_from_file_location(fullname, _GV_PATH)
        return None


sys.meta_path.insert(0, _Finder())
