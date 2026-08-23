"""Minimaler RunPod-Client (nur was wir brauchen: run / status / cancel / health)."""
from __future__ import annotations

import time
from typing import Optional

import requests

from . import config

API = "https://api.runpod.ai"


class RunPodError(RuntimeError):
    pass


def _headers() -> dict:
    if not config.RUNPOD_API_KEY:
        raise RunPodError("RUNPOD_API_KEY ist nicht gesetzt")
    return {"Authorization": f"Bearer {config.RUNPOD_API_KEY}",
            "Content-Type": "application/json"}


def submit(endpoint_id: str, payload: dict, policy: Optional[dict] = None) -> dict:
    """Job asynchron einreichen (/run). Liefert {id, status}."""
    body = {"input": payload}
    if policy:
        body["policy"] = policy
    r = requests.post(f"{API}/v2/{endpoint_id}/run", headers=_headers(), json=body, timeout=30)
    data = r.json()
    if r.status_code != 200 or data.get("status") in ("FAILED",):
        raise RunPodError(f"RunPod /run Fehler {r.status_code}: {data}")
    return data


def status(endpoint_id: str, runpod_job_id: str) -> dict:
    r = requests.get(f"{API}/v2/{endpoint_id}/status/{runpod_job_id}",
                     headers=_headers(), timeout=30)
    if r.status_code != 200:
        raise RunPodError(f"RunPod /status Fehler {r.status_code}: {r.text}")
    return r.json()


def cancel(endpoint_id: str, runpod_job_id: str) -> dict:
    r = requests.post(f"{API}/v2/{endpoint_id}/cancel/{runpod_job_id}",
                      headers=_headers(), timeout=30)
    if r.status_code != 200:
        raise RunPodError(f"RunPod /cancel Fehler {r.status_code}: {r.text}")
    return r.json()


def endpoint_health(endpoint_id: str) -> dict:
    r = requests.get(f"{API}/v1/oai?endpoint_id={endpoint_id}", headers=_headers(), timeout=30)
    return {"http_status": r.status_code}
