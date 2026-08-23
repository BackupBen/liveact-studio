"""RunPod Serverless Handler für LiveAct Studio.

Flow pro Job:
  1. S3: audio + avatar herunterladen
  2. ffmpeg: audio -> WAV
  3. LiveActRenderer (lazy geladen, Modell bleibt im VRAM des warmen Workers)
  global
  4. Chunked-Export -> MP4
  5. S3-Upload + Callback an die Webapp
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig

# --- RunPod-Umgebung -------------------------------------------------------
RUNPOD_WEBHOOK_POST = os.getenv("RUNPOD_WEBHOOK_POST", "")  # Progress an RunPod
rp = None
try:
    import runpod  # noqa
    rp = runpod
except ImportError:
    rp = None

WORK_DIR = Path("/runpod-volume/inputs") if Path("/runpod-volume").exists() else Path("/tmp/liveact")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("/runpod-volume/outputs") if Path("/runpod-volume").exists() else WORK_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/runpod-volume/models/LiveAct"))
WAV2VEC_DIR = Path(os.getenv("WAV2VEC_DIR", "/runpod-volume/models/chinese-wav2vec2-base"))

# --- S3 (boto3, S3v4, path-style für MinIO/Kompatibles) --------------------
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")


def s3_client():
    kwargs = {
        "aws_access_key_id": S3_ACCESS_KEY,
        "aws_secret_access_key": S3_SECRET_KEY,
        "region_name": S3_REGION,
    }
    cfg = BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"})
    if S3_ENDPOINT:
        return boto3.client("s3", endpoint_url=S3_ENDPOINT, config=cfg, **kwargs)
    return boto3.client("s3", config=cfg, **kwargs)


def download_from_s3(key: str, dest: Path) -> Path:
    s3_client().download_file(S3_BUCKET, key, str(dest))
    return dest


def upload_to_s3(path: Path, key: str) -> str:
    s3_client().upload_file(str(path), S3_BUCKET, key)
    return key


# --- Progress-Reporting: Webapp-Callback + RunPod-Webhook ------------------

CALLBACK_URL = os.getenv("CALLBACK_URL", "")  # überschrieben per job input
CALLBACK_TOKEN = os.getenv("CALLBACK_TOKEN", "")


def send_progress(job_input: dict, pct: int, note: str):
    url = job_input.get("callback_url") or CALLBACK_URL
    token = job_input.get("callback_token") or CALLBACK_TOKEN
    payload = {
        "job_id": job_input.get("job_id"),
        "event": "progress",
        "pct": pct,
        "note": note,
        "worker_id": os.getenv("RUNPOD_POD_ID", ""),
    }
    # 1) Webapp
    if url:
        try:
            import requests
            requests.post(url, json={**payload, "token": token}, timeout=5)
        except Exception:
            pass
    # 2) RunPod-eigene Progress-Webhooks
    if rp and RUNPOD_WEBHOOK_POST:
        try:
            rp.serverless.progress_update({"percentage": pct, "note": note})
        except Exception:
            pass


# --- Renderer global (warm hält das Modell im Speicher) --------------------

_RENDERER = None
_RENDERER_CFG_KEY = None


def get_renderer(cfg_overrides: dict):
    global _RENDERER, _RENDERER_CFG_KEY
    gpu_flags = (
        os.getenv("LIVEACT_FP8_KV", "0") == "1",
        os.getenv("LIVEACT_BLOCK_OFFLOAD", "0") == "1",
        os.getenv("LIVEACT_OFFLOAD_CACHE", "0") == "1",
    )
    size = (cfg_overrides or {}).get("size", os.getenv("LIVEACT_SIZE", "768*432"))
    fps = (cfg_overrides or {}).get("fps", int(os.getenv("LIVEACT_FPS", "25")))
    key = (gpu_flags, size, fps)
    if _RENDERER is not None and _RENDERER_CFG_KEY == key:
        return _RENDERER
    if _RENDERER is not None:
        # Flags geändert -> neu laden ist teuer; wir verwerfen den alten Renderer
        import gc
        import torch
        del _RENDERER
        gc.collect()
        torch.cuda.empty_cache()
    from liveact_runner import LiveActRenderer
    cfg = {
        "ckpt_dir": str(MODEL_DIR),
        "wav2vec_dir": str(WAV2VEC_DIR),
        "size": size,
        "fps": fps,
        "fp8_kv_cache": gpu_flags[0],
        "block_offload": gpu_flags[1],
        "offload_cache": gpu_flags[2],
        "t5_cpu": os.getenv("LIVEACT_T5_CPU", "1") == "1",
    }
    cfg.update(cfg_overrides or {})
    _RENDERER = LiveActRenderer(cfg, progress_cb=None)
    _RENDERER_CFG_KEY = key
    return _RENDERER


# --- Handler ---------------------------------------------------------------

def handler(job):
    """RunPod-Serverless-Entry-Point."""
    job_input = job["input"]

    # --- Sondermodus: Modelle aufs Volume laden (Setup-Helfer) -------------
    if job_input.get("download_models"):
        from huggingface_hub import snapshot_download
        vol = os.getenv("MODEL_DIR", "/runpod-volume/models/LiveAct")
        models_root = str(Path(vol).parent)
        t0 = time.time()
        def dl_progress(pct, note):
            send_progress(job_input, pct, note)
        dl_progress(1, "Lade Soul-AILab/LiveAct (~55 GB) aufs Volume …")
        snapshot_download("Soul-AILab/LiveAct",
                          local_dir=f"{models_root}/LiveAct", max_workers=8)
        dl_progress(80, "Lade chinese-wav2vec2-base (~400 MB) …")
        snapshot_download("TencentGameMate/chinese-wav2vec2-base",
                          local_dir=f"{models_root}/chinese-wav2vec2-base", max_workers=4)
        return {"downloaded": True, "seconds": round(time.time() - t0, 1),
                "models_root": models_root}

    t0 = time.time()
    job_id = job_input.get("job_id", "unknown")

    try:
        s3_audio_key = job_input["s3_audio_key"]
        s3_avatar_key = job_input["s3_avatar_key"]
        prompt = job_input.get("prompt", "a person is talking")
        size = job_input.get("size", "768*432")
        fps = int(job_input.get("fps", 25))
        chunk_seconds = int(job_input.get("chunk_seconds", 120))
    except KeyError as e:
        return {"error": f"input fehlt: {e}"}

    send_progress(job_input, 2, "Lade Dateien …")
    audio_raw = WORK_DIR / f"{job_id}_audio_{Path(s3_audio_key).name}"
    avatar_img = WORK_DIR / f"{job_id}_avatar_{Path(s3_avatar_key).name}"
    download_from_s3(s3_audio_key, audio_raw)
    download_from_s3(s3_avatar_key, avatar_img)

    # Audio -> WAV
    from liveact_runner import convert_to_wav
    audio_wav = convert_to_wav(audio_raw, WORK_DIR / f"{job_id}.wav")

    out_dir = OUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{job_id}.mp4"

    def cb(pct, note):
        send_progress(job_input, pct, note)

    renderer = get_renderer({"size": size, "fps": fps})
    result = renderer.render(
        image_path=str(avatar_img),
        audio_wav=str(audio_wav),
        out_mp4=str(out_mp4),
        prompt=prompt,
        seed=int(job_input.get("seed", 42)),
        chunk_seconds=chunk_seconds,
        workdir=out_dir / "work",
    )

    send_progress(job_input, 95, "Lade Video hoch …")
    s3_key = f"videos/{job_id}.mp4"
    upload_to_s3(out_mp4, s3_key)

    render_seconds = round(time.time() - t0, 1)
    cost_usd = None
    try:
        # grobe Kostenschätzung aus exec-time; echte Abrechnung macht RunPod
        rate = float(os.getenv("LIVEACT_USD_PER_HOUR", "0"))
        if rate:
            cost_usd = round(render_seconds / 3600 * rate, 4)
    except Exception:
        pass

    # Aufräumen (außer auf persistenter Volume, damit Debugging möglich bleibt)
    for f in (audio_raw, avatar_img, audio_wav):
        try: f.unlink()
        except Exception: pass

    # Webapp informieren
    url = job_input.get("callback_url")
    if url:
        try:
            import requests
            requests.post(url, json={
                "job_id": job_id, "event": "done", "token": job_input.get("callback_token", ""),
                "s3_key": s3_key, "render_seconds": render_seconds, "cost_usd": cost_usd,
            }, timeout=10)
        except Exception:
            pass

    return {
        "s3_key": s3_key,
        "render_seconds": render_seconds,
        "cost_usd": cost_usd,
        **{k: v for k, v in result.items() if k != "output"},
    }


if rp:
    rp.serverless.start({"handler": handler})
