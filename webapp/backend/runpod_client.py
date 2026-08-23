"""Erweiterter RunPod-Client — REST + GraphQL für Volume/Template/Endpoint/Pod.

Der API-Key kann dynamisch (aus dem Settings-Store der Webapp) übergeben werden,
nicht nur aus der statischen ENV.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import requests

REST = "https://rest.runpod.io"
GQL = "https://api.runpod.io/graphql"


class RunPodError(RuntimeError):
    pass


def _resolve_key(override: Optional[str] = None) -> str:
    key = override or os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise RunPodError("RUNPOD_API_KEY fehlt — in den Webapp-Settings eintragen")
    return key


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _gql(key: str, query: str, variables: dict | None = None) -> dict:
    r = requests.post(GQL, params={"api_key": key},
                      json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RunPodError(json.dumps(data["errors"], indent=2, ensure_ascii=False))
    return data["data"]


# --------------------------------------------------------------- Volumes ----

def list_volumes(key: str) -> list[dict]:
    r = requests.get(f"{REST}/v1/networkvolumes", headers=_headers(key), timeout=30)
    r.raise_for_status()
    return r.json() or []


def create_volume(key: str, name: str, size_gb: int, dc: str) -> dict:
    r = requests.post(f"{REST}/v1/networkvolumes", headers=_headers(key), timeout=30,
                      json={"name": name, "size": size_gb, "dataCenterId": dc})
    if r.status_code not in (200, 201):
        raise RunPodError(f"Volume-Create {r.status_code}: {r.text}")
    return r.json()


def ensure_volume(key: str, name: str, size_gb: int, dc: str) -> str:
    """Idempotent: vorhandenes Volume per Name wiederverwenden."""
    for v in list_volumes(key):
        if v.get("name") == name:
            return v["id"]
    return create_volume(key, name, size_gb, dc)["id"]


# -------------------------------------------------------------- Templates ---

def list_templates(key: str) -> list[dict]:
    data = _gql(key, "query { myself { templates { id name imageName isServerless } } }")
    return data["myself"]["templates"] or []


def ensure_template(key: str, name: str, image_name: str,
                    env: list[dict] | None = None,
                    container_disk_gb: int = 5) -> str:
    """Serverless-Template anlegen (idempotent per Name)."""
    for t in list_templates(key):
        if t.get("name") == name:
            return t["id"]
    mutation = """
    mutation($input: SaveTemplateInput!) {
      saveTemplate(input: $input) { id name imageName isServerless }
    }"""
    variables = {"input": {
        "containerDiskInGb": container_disk_gb,
        "dockerArgs": "python -u handler.py",
        "env": env or [],
        "imageName": image_name,
        "isServerless": True,
        "name": name,
        "volumeInGb": 0,
    }}
    try:
        return _gql(key, mutation, variables)["saveTemplate"]["id"]
    except RunPodError:
        # Fallback: Input-Typname variiert je API-Stand
        mutation2 = mutation.replace("SaveTemplateInput!", "PodTemplateInput!")
        return _gql(key, mutation2, variables)["saveTemplate"]["id"]


# ------------------------------------------------------------- Endpoints ----

def list_endpoints(key: str) -> list[dict]:
    r = requests.get(f"{REST}/v1/endpoints", headers=_headers(key), timeout=30)
    r.raise_for_status()
    return r.json() or []


def ensure_endpoint(key: str, name: str, template_id: str, gpu: str,
                    volume_id: str, workers_max: int = 1,
                    execution_timeout_ms: int = 7200000) -> str:
    for e in list_endpoints(key):
        if e.get("name") == name:
            return e["id"]
    body = {
        "name": name,
        "templateId": template_id,
        "gpuTypeIds": [gpu],
        "networkVolumeId": volume_id,
        "workersMin": 0,
        "workersMax": workers_max,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "idleTimeout": 5,
        "executionTimeoutMs": execution_timeout_ms,
        "flashBoot": False,
    }
    r = requests.post(f"{REST}/v1/endpoints", headers=_headers(key), timeout=30, json=body)
    if r.status_code not in (200, 201):
        raise RunPodError(f"Endpoint-Create {r.status_code}: {r.text}")
    return r.json()["id"]


# ---------------------------------------------- Model-Download-Pod ---------

def launch_download_pod(key: str, volume_id: str, dc: str, repo: str,
                         gpu: str = "NVIDIA RTX A4000") -> str:
    cmd = (
        "pip install -q huggingface_hub && "
        f"curl -sL {repo}/raw/main/worker/scripts/download_models.py -o /tmp/dl.py && "
        "python /tmp/dl.py --volume /runpod-volume && "
        "echo MODEL_DOWNLOAD_DONE"
    )
    mutation = """
    mutation($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) { id desiredStatus }
    }"""
    variables = {"input": {
        "cloudType": "ALL",
        "gpuCount": 1,
        "containerDiskInGb": 20,
        "volumeInGb": 0,
        "gpuTypeId": gpu,
        "name": "liveact-model-download",
        "imageName": "runpod/base:0.4.0",
        "dockerArgs": f"bash -c '{cmd}'",
        "networkVolumeId": volume_id,
        "dataCenterId": dc,
        "env": [],
    }}
    try:
        return _gql(key, mutation, variables)["podFindAndDeployOnDemand"]["id"]
    except RunPodError:
        mutation2 = mutation.replace("PodFindAndDeployOnDemandInput!", "PodDeployInput!")
        return _gql(key, mutation2, variables)["podFindAndDeployOnDemand"]["id"]


def pod_status(key: str, pod_id: str) -> dict:
    q = 'query { pod(input: { podId: "%s" }) { id desiredStatus runtimeStatus } }' % pod_id
    return _gql(key, q)["pod"]


def kill_pod(key: str, pod_id: str) -> None:
    _gql(key, 'mutation { podTerminate(input: { podId: "%s" }) { id } }' % pod_id)


# --------------------------------------------------- Job-Submit (wie gehabt)

def submit(endpoint_id: str, payload: dict, policy: dict | None = None,
           key: Optional[str] = None) -> dict:
    k = _resolve_key(key)
    body = {"input": payload}
    if policy:
        body["policy"] = policy
    r = requests.post(f"{REST}/v2/{endpoint_id}/run", headers=_headers(k), json=body, timeout=30)
    data = r.json()
    if r.status_code != 200 or data.get("status") in ("FAILED",):
        raise RunPodError(f"RunPod /run {r.status_code}: {data}")
    return data


def status(endpoint_id: str, runpod_job_id: str, key: Optional[str] = None) -> dict:
    k = _resolve_key(key)
    r = requests.get(f"{REST}/v2/{endpoint_id}/status/{runpod_job_id}",
                     headers=_headers(k), timeout=30)
    if r.status_code != 200:
        raise RunPodError(f"RunPod /status {r.status_code}: {r.text}")
    return r.json()


def cancel(endpoint_id: str, runpod_job_id: str, key: Optional[str] = None) -> dict:
    k = _resolve_key(key)
    r = requests.post(f"{REST}/v2/{endpoint_id}/cancel/{runpod_job_id}",
                      headers=_headers(k), timeout=30)
    if r.status_code != 200:
        raise RunPodError(f"RunPod /cancel {r.status_code}: {r.text}")
    return r.json()
