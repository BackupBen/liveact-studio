"""Settings-Store: persistierte RunPod-Konfiguration (data/settings.json).

Reihenfolge: Settings-Store (von der Webapp gesetzt) > ENV.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from . import config

_LOCK = threading.Lock()


def _path() -> Path:
    return config.DATA_DIR / "settings.json"


def load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(settings: dict) -> None:
    with _LOCK:
        p = _path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)


def get_runpod_credentials() -> tuple[Optional[str], Optional[str]]:
    """(api_key, endpoint_id) — Store zuerst, dann ENV-Fallback."""
    s = load()
    return (s.get("runpod_api_key") or None,
            s.get("runpod_endpoint_id") or None)


def get_runpod_setup() -> Optional[dict]:
    """Gespeicherte Provisionierungs-Ergebnisse (volume_id, template_id, ...)."""
    return load().get("runpod_setup") or None
