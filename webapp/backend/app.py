"""LiveAct Studio — FastAPI-Backend.

Routen:
  /api/avatars            GET/POST/DELETE  Avatar-Verwaltung
  /api/jobs               POST             Audio+Avatar -> RunPod-Job
  /api/jobs               GET              Job-Liste
  /api/jobs/{id}          GET              Status (poll)
  /api/jobs/{id}/cancel   POST             Abbrechen
  /api/video/{id}         GET              Fertiges Video streamen (lokal oder S3-Redirect)
  /api/runpod/callback    POST             Progress vom Worker
  /healthz                GET              Healthcheck
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import runpod_client, s3
from . import settings_store
from .jobstore import create_job, get as get_job, list_jobs, update
from .setup_routes import router as setup_router
from . import config

log = logging.getLogger("liveact")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LiveAct Studio", version="1.1.0")
app.include_router(setup_router)

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac", ".webm"}
ALLOWED_IMAGE = {".png", ".jpg", ".jpeg", ".webp"}
SAFE_NAME = re.compile(r"^[a-zA-Z0-9_\-]+$")

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

async def _save_upload(upload: UploadFile, dest_dir: Path, allowed: set[str], max_bytes: int) -> tuple[Path, str]:
    ext = Path(upload.filename or "file.bin").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Dateityp {ext} nicht erlaubt ({', '.join(sorted(allowed))})")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{int(time.time() * 1000)}_{re.sub(r'[^a-zA-Z0-9._-]', '_', upload.filename or 'file')}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "Datei zu groß")
            f.write(chunk)
    return dest, ext


def _probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _validate_image(path: Path) -> None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        raise HTTPException(400, f"Ungültiges Bild: {e}")


# --------------------------------------------------------------------------
# Avatare
# --------------------------------------------------------------------------

@app.get("/api/avatars")
def list_avatars():
    items = []
    for p in sorted((config.DATA_DIR / "avatars").glob("*")):
        if p.suffix.lower() in ALLOWED_IMAGE and p.is_file():
            items.append({
                "id": p.stem,
                "name": p.stem.replace("_", " "),
                "image": f"/api/avatars/{p.stem}/image",
            })
    return {"avatars": items}


@app.get("/api/avatars/{avatar_id}/image")
def avatar_image(avatar_id: str):
    if not SAFE_NAME.match(avatar_id):
        raise HTTPException(400, "Ungültige Avatar-ID")
    for ext in ALLOWED_IMAGE:
        p = config.DATA_DIR / "avatars" / f"{avatar_id}{ext}"
        if p.exists():
            return FileResponse(p)
    raise HTTPException(404, "Avatar nicht gefunden")


@app.post("/api/avatars")
async def add_avatar(name: str = Form(...), image: UploadFile = File(...)):
    """Avatar anlegen ODER ersetzen (Upsert): gleicher Name = Bild wird
    ueberschrieben. VidForge kann also einfach immer POSTen."""
    if not SAFE_NAME.match(name):
        raise HTTPException(400, "Name darf nur Buchstaben, Zahlen, _ und - enthalten")
    path, _ = await _save_upload(image, config.DATA_DIR / "avatars", ALLOWED_IMAGE, 20 * 1024 * 1024)
    _validate_image(path)
    final = config.DATA_DIR / "avatars" / f"{name}{path.suffix.lower()}"
    if final.exists():
        final.unlink()
    shutil.move(str(path), final)
    return {"id": name, "name": name.replace("_", " "), "image": f"/api/avatars/{name}/image",
            "upsert": True}


@app.delete("/api/avatars/{avatar_id}")
def delete_avatar(avatar_id: str):
    if not SAFE_NAME.match(avatar_id):
        raise HTTPException(400, "Ungültige Avatar-ID")
    removed = False
    for ext in ALLOWED_IMAGE:
        p = config.DATA_DIR / "avatars" / f"{avatar_id}{ext}"
        if p.exists():
            p.unlink()
            removed = True
    if not removed:
        raise HTTPException(404, "Avatar nicht gefunden")
    return {"deleted": avatar_id}


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_render_job(
    audio: UploadFile = File(...),
    avatar_id: str = Form(...),
    prompt: str = Form("a person is talking, natural movements"),
    size: str = Form(config.DEFAULT_SIZE),
    fps: int = Form(config.DEFAULT_FPS),
):
    # Avatar prüfen
    if not SAFE_NAME.match(avatar_id):
        raise HTTPException(400, "Ungültige Avatar-ID")
    avatar_file = None
    for ext in ALLOWED_IMAGE:
        p = config.DATA_DIR / "avatars" / f"{avatar_id}{ext}"
        if p.exists():
            avatar_file = p
            break
    if avatar_file is None:
        raise HTTPException(404, f"Avatar '{avatar_id}' nicht gefunden")

    # Audio speichern & prüfen
    audio_path, _ = await _save_upload(audio, config.DATA_DIR / "audio", ALLOWED_AUDIO, 500 * 1024 * 1024)
    duration = _probe_duration(audio_path)
    if duration <= 0:
        raise HTTPException(400, "Audiodauer konnte nicht ermittelt werden (ffprobe fehlt oder Datei defekt)")
    if duration > config.MAX_AUDIO_MINUTES * 60:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Audio zu lang: {duration/60:.1f} min (max {config.MAX_AUDIO_MINUTES} min)")

    # S3-Upload (Worker zieht sich Dateien selbst)
    try:
        audio_key = s3.upload_file(audio_path, f"audio/{audio_path.name}")
        avatar_key = s3.upload_file(avatar_file, f"avatars/{avatar_file.name}", public=False)
    except Exception as e:
        log.exception("S3-Upload fehlgeschlagen")
        raise HTTPException(502, f"S3-Upload fehlgeschlagen: {e}")

    job = create_job({
        "avatar_id": avatar_id,
        "avatar_name": avatar_id.replace("_", " "),
        "audio_filename": audio_path.name,
        "audio_duration_s": round(duration, 1),
        "size": size,
        "fps": fps,
        "prompt": prompt,
    })

    # RunPod-Job asynchron einreichen (API-Key ggf. aus Settings-Store)
    key, _ = settings_store.get_runpod_credentials()
    endpoint_id = config.RUNPOD_ENDPOINT_ID
    if not endpoint_id:
        _, endpoint_id = settings_store.get_runpod_credentials()
    worker_payload = {
        "job_id": job["id"],
        "s3_audio_key": audio_key,
        "s3_avatar_key": avatar_key,
        "audio_filename": audio_path.name,
        "avatar_name": avatar_id,
        "prompt": prompt,
        "size": size,
        "fps": fps,
        "chunk_seconds": config.CHUNK_SECONDS,
        "callback_url": f"{config.PUBLIC_BASE_URL}/api/runpod/callback",
        "callback_token": config.SESSION_SECRET,
    }

    try:
        resp = runpod_client.submit(endpoint_id, worker_payload,
                                    policy=config.EXECUTION_POLICY, key=key)
        update(job["id"], status="submitting", runpod_job_id=resp.get("id"))
    except Exception as e:
        log.exception("RunPod-Submit fehlgeschlagen")
        update(job["id"], status="error", error=f"RunPod-Submit: {e}")
        raise HTTPException(502, f"RunPod-Submit fehlgeschlagen: {e}")

    return job


@app.get("/api/jobs")
def jobs_list():
    jobs = list_jobs()
    # Auto-Reconcile: Callbacks koennen verloren gehen (PUBLIC_BASE_URL nicht
    # erreichbar etc.) — hengende Jobs direkt bei RunPod nachziehen.
    key, endpoint_id = settings_store.get_runpod_credentials()
    if key and (endpoint_id or config.RUNPOD_ENDPOINT_ID):
        now = time.time()
        stale = [j for j in jobs
                 if j.get("status") in ("queued", "submitting", "running")
                 and now - j.get("updated_at", 0) > 45]
        for j in stale[:3]:  # max 3 pro Poll, um Latenz zu begrenzen
            try:
                reconcile_job(j["id"])
            except Exception as e:
                log.warning(f"Auto-Reconcile {j['id']}: {e}")
    return {"jobs": list_jobs()}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job nicht gefunden")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job nicht gefunden")
    if job.get("runpod_job_id"):
        try:
            key, endpoint_id = _runpod_creds()
            runpod_client.cancel(endpoint_id, job["runpod_job_id"], key=key)
        except Exception as e:
            log.warning(f"Cancel bei RunPod fehlgeschlagen: {e}")
    update(job_id, status="cancelled", progress_note="Abgebrochen")
    return {"cancelled": job_id}


def _runpod_creds() -> tuple[str | None, str | None]:
    key, endpoint_id = settings_store.get_runpod_credentials()
    return key, endpoint_id or config.RUNPOD_ENDPOINT_ID


# --------------------------------------------------------------------------
# Video-Auslieferung
# --------------------------------------------------------------------------

@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job nicht gefunden")
    if job.get("status") != "done":
        raise HTTPException(409, "Video noch nicht fertig")
    if job.get("s3_key"):
        url = s3.presign_get(job["s3_key"])
        if url:
            return RedirectResponse(url)
    if job.get("video_path") and Path(job["video_path"]).exists():
        return FileResponse(job["video_path"], media_type="video/mp4")
    raise HTTPException(410, "Video nicht mehr verfügbar")


# --------------------------------------------------------------------------
# RunPod-Callbacks vom Worker
# --------------------------------------------------------------------------

@app.post("/api/runpod/callback")
async def runpod_callback(request: Request):
    body = await request.json()
    if config.SESSION_SECRET and body.get("token") != config.SESSION_SECRET:
        raise HTTPException(401, "Ungültiger Token")
    job_id = body.get("job_id")
    event = body.get("event")            # progress | done | error
    job = get_job(job_id) if job_id else None
    if job is None:
        return {"ok": False, "error": "unbekannter job"}
    if event == "progress":
        update(job_id,
               status="running",
               progress_pct=int(body.get("pct", 0)),
               progress_note=body.get("note", ""),
               runpod_worker_id=body.get("worker_id"))
        return {"ok": True}
    if event == "done":
        update(job_id,
               status="postprocessing",
               progress_pct=100,
               progress_note="Upload",
               s3_key=body.get("s3_key"),
               render_seconds=body.get("render_seconds"),
               cost_usd=body.get("cost_usd"))
        # Verifizieren: Objekt existiert im Bucket?
        try:
            url = s3.presign_get(body.get("s3_key"))
            update(job_id, status="done", video_url=url, progress_note="")
        except Exception:
            update(job_id, status="done", video_url=None)
        return {"ok": True}
    if event == "error":
        update(job_id, status="error", error=body.get("error", "Worker-Fehler"))
        return {"ok": True}
    return {"ok": False, "error": "unbekanntes event"}


# --------------------------------------------------------------------------
# Reconciliation: RunPod-Status nachziehen, falls Callbacks verloren gingen
# --------------------------------------------------------------------------

@app.post("/api/jobs/{job_id}/reconcile")
def reconcile_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job nicht gefunden")
    rp_id = job.get("runpod_job_id")
    if not rp_id:
        return job
    try:
        key, endpoint_id = _runpod_creds()
        st = runpod_client.status(endpoint_id, rp_id, key=key)
    except Exception as e:
        raise HTTPException(502, f"RunPod-Status: {e}")
    rp_status = st.get("status")
    out = st.get("output") or {}
    if rp_status == "COMPLETED":
        if out.get("s3_key") and job["status"] not in ("done",):
            url = s3.presign_get(out["s3_key"])
            update(job_id, status="done", s3_key=out["s3_key"], video_url=url,
                   render_seconds=out.get("render_seconds"), cost_usd=out.get("cost_usd"))
    elif rp_status in ("FAILED", "CANCELLED"):
        update(job_id, status="error" if rp_status == "FAILED" else "cancelled",
               error=(st.get("error") or out.get("error") or rp_status))
    else:
        update(job_id, status="running")
    return get_job(job_id)


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    key, endpoint_id = settings_store.get_runpod_credentials()
    return {
        "ok": True,
        "runpod_configured": bool((key or config.RUNPOD_API_KEY) and (endpoint_id or config.RUNPOD_ENDPOINT_ID)),
        "s3_configured": bool(config.S3_BUCKET),
        "jobs": len(list_jobs(limit=1000)),
    }


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static"), html=True), name="static")
