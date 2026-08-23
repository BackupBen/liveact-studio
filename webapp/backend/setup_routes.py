"""Setup-Routen: RunPod-Provisionierung per Button aus der Webapp.

POST /api/settings          RunPod-API-Key (+ optional Endpoint-ID) speichern
GET  /api/settings          Status (ohne Secrets)
POST /api/setup             Komplett-Provisionierung:
                             Volume -> Template -> Endpoint (+ optional Download-Pod)
GET  /api/setup/status      Fortschritt der Provisionierung
POST /api/setup/download-pod  Model-Download-Pod starten/prüfen/terminieren
"""
from __future__ import annotations

import threading
import time
import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import runpod_client, settings_store
from . import config

router = APIRouter(prefix="/api", tags=["setup"])

# Provisionierungs-State (in-memory; Ergebnis persistiert in settings.json)
_setup_state: dict = {"running": False, "steps": [], "error": None, "done_at": None}
_setup_lock = threading.Lock()

DOWNLOAD_POD_POLL_KEY = "download_pod"


def _step(name: str, detail: str = "", status: str = "ok") -> None:
    _setup_state["steps"].append({"name": name, "detail": detail, "status": status,
                                  "ts": time.time()})


class SettingsIn(BaseModel):
    runpod_api_key: str | None = None
    runpod_endpoint_id: str | None = None
    worker_image: str | None = None
    volume_name: str | None = None
    datacenter: str | None = None
    gpu: str | None = None


class SetupIn(BaseModel):
    worker_image: str = "docker.io/backupben/liveact-worker:latest"
    volume_name: str = "liveact-models"
    volume_size_gb: int = 200
    datacenter: str = "US-KS-2"
    gpu: str = "NVIDIA H100 PCIe"
    workers_max: int = 1
    start_download_pod: bool = True


# ------------------------------------------------------------- Settings ----

@router.get("/settings")
def get_settings():
    s = settings_store.load()
    setup = s.get("runpod_setup") or {}
    return {
        "has_api_key": bool(s.get("runpod_api_key")),
        "endpoint_id": s.get("runpod_endpoint_id") or "",
        "worker_image": setup.get("worker_image") or "",
        "volume_name": setup.get("volume_name") or "",
        "datacenter": setup.get("datacenter") or "",
        "gpu": setup.get("gpu") or "",
        "setup_done": bool(setup.get("endpoint_id")),
        "download_pod": setup.get("download_pod") or None,
    }


@router.post("/settings")
def save_settings(body: SettingsIn):
    s = settings_store.load()
    if body.runpod_api_key:
        s["runpod_api_key"] = body.runpod_api_key.strip()
    if body.runpod_endpoint_id is not None:
        s["runpod_endpoint_id"] = body.runpod_endpoint_id.strip()
    setup = s.get("runpod_setup") or {}
    if body.worker_image:
        setup["worker_image"] = body.worker_image.strip()
    if body.volume_name:
        setup["volume_name"] = body.volume_name.strip()
    if body.datacenter:
        setup["datacenter"] = datacenter_id(body.datacenter)
    if body.gpu:
        setup["gpu"] = body.gpu.strip()
    s["runpod_setup"] = setup
    settings_store.save(s)
    return {"ok": True}


def datacenter_id(dc: str) -> str:
    """Kurzform -> ID ('US' -> 'US-KS-2' bleibt 'US-KS-2')."""
    mapping = {"US": "US-KS-2", "EU": "EU-SE-1", "NO": "NO-LB-1", "FR": "FR-EU1"}
    return mapping.get(dc.upper(), dc)


# ---------------------------------------------------------------- Setup ----

@router.post("/setup")
def run_setup(body: SetupIn, background_tasks=None):
    if _setup_state["running"]:
        raise HTTPException(409, "Setup läuft bereits")
    key, _ = settings_store.get_runpod_credentials()
    if not key:
        raise HTTPException(400, "Zuerst RunPod-API-Key in den Settings speichern")

    with _setup_lock:
        _setup_state.update(running=True, steps=[], error=None, done_at=None)

    def work():
        try:
            dc = datacenter_id(body.datacenter)
            # 1. Volume
            _step("volume", f"Prüfe/lege Volume '{body.volume_name}' an ({body.volume_size_gb} GB, {dc})")
            vol_id = runpod_client.ensure_volume(key, body.volume_name, body.volume_size_gb, dc)
            _step("volume", f"Volume bereit: {vol_id}")

            # 2. Template
            _step("template", f"Prüfe/lege Serverless-Template an (Image: {body.worker_image})")
            template_id = runpod_client.ensure_template(key, "liveact-worker", body.worker_image)
            _step("template", f"Template bereit: {template_id}")

            # 3. Endpoint
            _step("endpoint", f"Prüfe/lege Endpoint '{body.gpu}' mit Volume an")
            endpoint_id = runpod_client.ensure_endpoint(
                key, "liveact", template_id, body.gpu, vol_id,
                workers_max=body.workers_max,
                execution_timeout_ms=config.EXECUTION_TIMEOUT_MS)
            _step("endpoint", f"Endpoint bereit: {endpoint_id}")

            persisted = {
                "volume_id": vol_id, "template_id": template_id,
                "endpoint_id": endpoint_id, "worker_image": body.worker_image,
                "volume_name": body.volume_name, "datacenter": dc, "gpu": body.gpu,
            }

            # 4. Optional: Download-Pod für die ~55 GB Modelle
            if body.start_download_pod:
                _step("download-pod", "Starte Model-Download-Pod (läuft im Hintergrund weiter)")
                pod_id = runpod_client.launch_download_pod(
                    key, vol_id, dc, repo="https://github.com/BackupBen/liveact-studio")
                persisted["download_pod"] = {"pod_id": pod_id, "started_at": time.time()}
                _step("download-pod", f"Download-Pod läuft: {pod_id} — Modelle landen auf dem Volume")

            s = settings_store.load()
            s["runpod_endpoint_id"] = endpoint_id
            s["runpod_setup"] = persisted
            settings_store.save(s)
            _step("done", "Setup abgeschlossen — Endpoint-ID gespeichert")
        except Exception as e:
            _setup_state["error"] = f"{e}"
            _step("error", str(e), status="error")
            traceback.print_exc()
        finally:
            _setup_state["running"] = False
            _setup_state["done_at"] = time.time()

    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@router.get("/setup/status")
def setup_status():
    s = settings_store.load()
    return {
        "running": _setup_state["running"],
        "steps": _setup_state["steps"],
        "error": _setup_state["error"],
        "setup": s.get("runpod_setup") or None,
    }


class DownloadPodAction(BaseModel):
    action: str  # "status" | "kill" | "restart"


@router.post("/setup/download-pod")
def download_pod(body: DownloadPodAction):
    key, _ = settings_store.get_runpod_credentials()
    if not key:
        raise HTTPException(400, "RunPod-API-Key fehlt")
    s = settings_store.load()
    setup = s.get("runpod_setup") or {}
    pod = setup.get("download_pod") or {}
    pod_id = pod.get("pod_id")

    if body.action == "status":
        if not pod_id:
            return {"pod": None}
        try:
            st = runpod_client.pod_status(key, pod_id)
            return {"pod": {"id": pod_id, **st}}
        except Exception as e:
            return {"pod": {"id": pod_id, "error": str(e)}}

    if body.action == "kill":
        if not pod_id:
            raise HTTPException(404, "Kein Download-Pod bekannt")
        runpod_client.kill_pod(key, pod_id)
        setup["download_pod"] = {"pod_id": pod_id, "killed_at": time.time()}
        s["runpod_setup"] = setup
        settings_store.save(s)
        return {"killed": pod_id}

    if body.action == "restart":
        vol_id = setup.get("volume_id")
        if not vol_id:
            raise HTTPException(400, "Kein Volume vorhanden — zuerst /api/setup")
        pod_id = runpod_client.launch_download_pod(
            key, vol_id, setup.get("datacenter", "US-KS-2"),
            repo="https://github.com/BackupBen/liveact-studio")
        setup["download_pod"] = {"pod_id": pod_id, "started_at": time.time()}
        s["runpod_setup"] = setup
        settings_store.save(s)
        return {"pod_id": pod_id}

    raise HTTPException(400, "action: status | kill | restart")
