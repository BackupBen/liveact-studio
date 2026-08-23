"""Persistenter Job-Store: JSON pro Job in data/jobs/, Thread-Safe.

Der Store läuft ohne DB — Coolify-Volumes gemountet, Restart-safe.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from . import config

_LOCK = threading.Lock()


def _job_path(job_id: str) -> Path:
    return config.DATA_DIR / "jobs" / f"{job_id}.json"


def create_job(payload: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",          # queued -> submitting -> running -> postprocessing -> done | error | cancelled
        "created_at": time.time(),
        "updated_at": time.time(),
        "avatar_id": payload.get("avatar_id"),
        "avatar_name": payload.get("avatar_name", ""),
        "audio_filename": payload.get("audio_filename", ""),
        "audio_duration_s": payload.get("audio_duration_s"),
        "size": payload.get("size", config.DEFAULT_SIZE),
        "fps": payload.get("fps", config.DEFAULT_FPS),
        "prompt": payload.get("prompt", ""),
        "policy": config.EXECUTION_POLICY,
        "runpod_job_id": None,
        "runpod_worker_id": None,
        "progress_pct": 0,
        "progress_note": "",
        "error": None,
        "video_path": None,
        "video_url": None,
        "s3_key": None,
        "render_seconds": None,
        "cost_usd": None,
    }
    write(job)
    return job


def write(job: dict) -> None:
    job["updated_at"] = time.time()
    with _LOCK:
        p = _job_path(job["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)


def get(job_id: str) -> Optional[dict]:
    p = _job_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_jobs(limit: int = 100) -> list[dict]:
    jobs = []
    for p in (config.DATA_DIR / "jobs").glob("*.json"):
        try:
            jobs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return jobs[:limit]


def update(job_id: str, **fields) -> Optional[dict]:
    job = get(job_id)
    if job is None:
        return None
    job.update(fields)
    write(job)
    return job
