"""LiveAct Studio — Webapp-Konfiguration aus Umgebungsvariablen.

Alle Einstellungen lassen sich per ENV überschreiben (Coolify macht das trivial).
"""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LIVEACT_DATA_DIR", BASE_DIR / "data"))

# --- RunPod ---
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_PROJECT_ID = os.environ.get("RUNPOD_PROJECT_ID", "")  # optional, für Graphen-Screenshots

# --- Storage ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")          # z. B. https://s3.example.com
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

# --- Webapp ---
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "")  # z. B. https://liveact.example.com
# Zufälliges Secret, falls nicht gesetzt (nicht für Produktion geeignet — ENV setzen!)
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_urlsafe(32))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")    # leer = keine Auth aktiv

# --- Worker-Defaults ---
DEFAULT_GPU = os.environ.get("DEFAULT_GPU", "NVIDIA H100")
DEFAULT_SIZE = os.environ.get("DEFAULT_SIZE", "768*432")
DEFAULT_FPS = int(os.environ.get("DEFAULT_FPS", "20"))
MAX_AUDIO_MINUTES = int(os.environ.get("MAX_AUDIO_MINUTES", "75"))
# Chunklänge in Sekunden für den segmentierten Video-Export (RAM-Bombe vermeiden)
CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "120"))
# RunPod-Ausführungs-Timeout in ms (2 h Standard, max 7 Tage)
EXECUTION_TIMEOUT_MS = int(os.environ.get("EXECUTION_TIMEOUT_MS", "7200000"))
EXECUTION_POLICY = {
    "executionTimeout": EXECUTION_TIMEOUT_MS,
    "ttl": EXECUTION_TIMEOUT_MS * 2,
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
for _sub in ("avatars", "audio", "videos", "jobs"):
    (DATA_DIR / _sub).mkdir(parents=True, exist_ok=True)
