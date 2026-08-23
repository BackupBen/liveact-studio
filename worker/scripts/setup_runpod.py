#!/usr/bin/env python3
"""RunPod-Setup komplett per API (kein Console-Klicken nötig).

Schritte:
  1. Network Volume anlegen (oder vorhandenes wiederverwenden)  — REST /v1/networkvolumes
  2. (optional) Pod starten, der die ~55 GB Modelle aufs Volume lädt  — GraphQL podFindAndDeployOnDemand
  3. Serverless Endpoint mit gemountetem Volume anlegen            — REST /v1/endpoints

Nur pip install requests nötig.

Beispiele:
  export RUNPOD_API_KEY=rpk_xxx

  # Alles in einem (Volume + Modelle + Endpoint):
  python setup_runpod.py \
      --template-id xkhgg72fuo \
      --image docker.io/DEINDOCKERHUB/liveact-worker:latest \
      --volume-name liveact-models --size 200 --dc US-KS-2 \
      --download-models \
      --repo https://github.com/BackupBen/liveact-studio

  # Nur Volume + Endpoint (Modelle schon drauf):
  python setup_runpod.py --template-id xkhgg72fuo --gpu "NVIDIA H100 PCIe" \
      --volume-name liveact-models --size 200 --dc US-KS-2

Hinweise:
  - Template einmalig in der Console anlegen (Serverless > New Template):
    Image = dein Worker-Image, Registry-Credentials bei privatem Docker Hub.
    Die Template-ID dann hier per --template-id mitgeben.
  - Das Volume muss in einem Datacenter liegen, das auch deine GPU-Class hat
    (Standard US-KS-2 hat H100/A100/4090 in der US-Region).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

REST = "https://rest.runpod.io"
GQL = "https://api.runpod.io/graphql"


def _headers():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY ist nicht gesetzt (export RUNPOD_API_KEY=rpk_...)")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(GQL, params={"api_key": os.environ["RUNPOD_API_KEY"]},
                      json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        sys.exit(f"GraphQL-Fehler: {json.dumps(data['errors'], indent=2)}")
    return data["data"]


# ---------------------------------------------------------------- Volume ----

def list_volumes() -> list[dict]:
    r = requests.get(f"{REST}/v1/networkvolumes", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json() or []


def ensure_volume(name: str, size_gb: int, dc: str) -> str:
    for v in list_volumes():
        if v.get("name") == name:
            print(f"[volume] '{name}' existiert bereits: {v['id']} ({v.get('size')} GB in {v.get('dataCenterId', v.get('dataCenter', '?'))})")
            if int(v.get("size", 0)) < size_gb:
                print(f"[volume] WARNUNG: vorhandenes Volume ist kleiner als {size_gb} GB — Größe kann nur via Support/Console vergrößert werden.")
            return v["id"]
    r = requests.post(f"{REST}/v1/networkvolumes", headers=_headers(), timeout=30,
                      json={"name": name, "size": size_gb, "dataCenterId": dc})
    if r.status_code not in (200, 201):
        sys.exit(f"[volume] Anlegen fehlgeschlagen: {r.status_code} {r.text}")
    vol = r.json()
    print(f"[volume] angelegt: {vol['id']} ({size_gb} GB in {dc})")
    return vol["id"]


# ---------------------------------------------------- Model-Download-Pod ----

def launch_download_pod(volume_id: str, dc: str, repo: str, gpu: str) -> str:
    cmd = (
        "pip install -q huggingface_hub requests && "
        f"curl -sL {repo}/raw/main/worker/scripts/download_models.py -o /tmp/dl.py && "
        "python /tmp/dl.py --volume /runpod-volume && "
        "echo MODEL_DOWNLOAD_DONE && sleep 300"
    )
    q = """
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
        "dockerArgs": f"bash -c \"{cmd}\"",
        "networkVolumeId": volume_id,
        "dataCenterId": dc,
        "env": [],
    }}
    data = _gql(q, variables)
    pod_id = data["podFindAndDeployOnDemand"]["id"]
    print(f"[download-pod] gestartet: {pod_id}")
    print("[download-pod] Logs in der Console beobachten; am Ende erscheint MODEL_DOWNLOAD_DONE.")
    print("[download-pod] Danach Pod stoppen/löschen (Console oder: python setup_runpod.py --kill-pod " + pod_id + ")")
    return pod_id


def kill_pod(pod_id: str) -> None:
    q = 'mutation { podTerminate(input: { podId: "%s" }) { id } }' % pod_id
    _gql(q)
    print(f"[pod] {pod_id} terminiert.")


# --------------------------------------------------------------- Endpoint ----

def list_endpoints() -> list[dict]:
    r = requests.get(f"{REST}/v1/endpoints", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json() or []


def ensure_endpoint(name: str, template_id: str, gpu: str, volume_id: str,
                    workers_max: int, execution_timeout_ms: int) -> str:
    for e in list_endpoints():
        if e.get("name") == name:
            print(f"[endpoint] '{name}' existiert bereits: {e['id']}")
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
    r = requests.post(f"{REST}/v1/endpoints", headers=_headers(), timeout=30, json=body)
    if r.status_code not in (200, 201):
        sys.exit(f"[endpoint] Anlegen fehlgeschlagen: {r.status_code} {r.text}")
    ep = r.json()
    print(f"[endpoint] angelegt: {ep['id']}")
    return ep["id"]


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description="RunPod-Setup per API")
    ap.add_argument("--volume-name", default="liveact-models")
    ap.add_argument("--size", type=int, default=200, help="Volume-Größe in GB (Modelle ~60 GB)")
    ap.add_argument("--dc", default="US-KS-2", help="Datacenter-ID (muss GPU-Verfügbarkeit haben)")
    ap.add_argument("--template-id", help="Serverless-Template-ID (aus der Console)")
    ap.add_argument("--endpoint-name", default="liveact")
    ap.add_argument("--gpu", default="NVIDIA H100 PCIe", help="GPU-Type-ID, z. B. 'NVIDIA H100 PCIe', 'NVIDIA A100 80GB PCIe', 'NVIDIA GeForce RTX 4090'")
    ap.add_argument("--workers-max", type=int, default=1)
    ap.add_argument("--execution-timeout-ms", type=int, default=7200000)
    ap.add_argument("--download-models", action="store_true", help="Download-Pod starten (~55 GB aufs Volume)")
    ap.add_argument("--download-gpu", default="NVIDIA RTX A4000", help="günstige GPU für den Download-Pod")
    ap.add_argument("--repo", default="https://github.com/BackupBen/liveact-studio")
    ap.add_argument("--kill-pod", help="Pod-ID terminieren (nach Model-Download)")
    args = ap.parse_args()

    if args.kill_pod:
        kill_pod(args.kill_pod)
        return

    _headers()  # KEY-Check

    vol_id = ensure_volume(args.volume_name, args.size, args.dc)

    if args.download_models:
        launch_download_pod(vol_id, args.dc, args.repo, args.download_gpu)

    if not args.template_id:
        print("\n[endpoint] Übersprungen: --template-id fehlt.")
        print("  Template einmalig in der Console anlegen (Serverless > New Template, dein Worker-Image),")
        print("  dann erneut laufen lassen. Volume-ID für den Endpoint: " + vol_id)
        return

    ep_id = ensure_endpoint(args.endpoint_name, args.template_id, args.gpu, vol_id,
                            args.workers_max, args.execution_timeout_ms)

    print("\n================ FERTIG ================")
    print(f"Network Volume : {vol_id}")
    print(f"Endpoint-ID    : {ep_id}   <- RUNPOD_ENDPOINT_ID in der Webapp")
    print("Webapp-ENV: RUNPOD_ENDPOINT_ID=" + ep_id)


if __name__ == "__main__":
    main()
