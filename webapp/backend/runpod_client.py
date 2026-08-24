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

REST = "https://rest.runpod.io"      # v1-CRUD-API (volumes, endpoints)
GQL = "https://api.runpod.io/graphql"  # GraphQL (templates, pods)
JOBS = "https://api.runpod.ai"        # v2-Job-API (run/status/cancel) — NICHT rest.runpod.io!


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
    body = None
    try:
        body = r.json()
    except Exception:
        pass
    if r.status_code != 200 or (body and body.get("errors")):
        msgs = []
        if body and body.get("errors"):
            for e in body["errors"]:
                msgs.append(e.get("message", str(e)))
        raise RunPodError(f"GraphQL {r.status_code}: {'; '.join(msgs) or (r.text or '')[:300]}")
    return body["data"]


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
    """Nicht unterstützt: RunPod-GraphQL hat keine Template-List-Query.
    (Dummy für etwaige Zukunft — immer leer.)"""
    return []


def ensure_template(key: str, name: str, image_name: str,
                    env: list[dict] | None = None,
                    container_disk_gb: int = 40,
                    gpu_count: int = 1) -> str:
    """Serverless-Template anlegen.

    Es gibt keine List-Query — daher: kanonischer Name versuchen;
    falls der Name schon existiert (RunPod erzwingt eindeutige Namen),
    wird ein Name mit Zeitstempel-Suffix angelegt. Vorteil: Ein neues
    Worker-Image landet garantiert in einem neuen Template statt in einem
    eventuell veralteten.

    gpu_count > 1 => Multi-GPU-Worker (Sequence-Parallelism via torchrun);
    der Name haengt dann am gpu_count, damit der Wechsel ein frisches
    Template ergibt. dockerArgs zeigt auf runpod_launcher.py, der bei
    >=2 GPUs automatisch torchrun startet.
    """
    full_name = name if gpu_count == 1 else f"{name}-{gpu_count}gpu"

    def build_mutation(tname: str) -> str:
        env_str = ", ".join(
            f'{{ key: {json.dumps(e["key"])}, value: {json.dumps(e["value"])} }}'
            for e in (env or [])
        )
        return (
            "mutation { saveTemplate(input: { "
            f"containerDiskInGb: {container_disk_gb}, "
            f"dockerArgs: {json.dumps('python -u runpod_launcher.py')}, "
            f"env: [{env_str}], "
            f"gpuCount: {gpu_count}, "
            f"imageName: {json.dumps(image_name)}, "
            "isServerless: true, "
            f"name: {json.dumps(tname)}, "
            "volumeInGb: 0, "
            f"volumeMountPath: {json.dumps('/runpod-volume')} "
            "}) { id name } }"
        )

    try:
        return _gql(key, build_mutation(full_name))["saveTemplate"]["id"]
    except RunPodError as e:
        msg = str(e).lower()
        if "gpucount" in msg:
            raise RunPodError(
                "RunPod-API lehnt gpuCount ab — Multi-GPU-Serverless in diesem "
                "API-Stand nicht verfuegbar. Setup mit gpu_count=1 wiederholen.")
        if "unique" not in msg and "exist" not in msg and "duplicate" not in msg:
            raise
    # Name belegt -> frisches Template mit Suffix
    import time as _time
    return _gql(key, build_mutation(f"{full_name}-{int(_time.time())}"))["saveTemplate"]["id"]


# ------------------------------------------------------------- Endpoints ----

def list_endpoints(key: str) -> list[dict]:
    r = requests.get(f"{REST}/v1/endpoints", headers=_headers(key), timeout=30)
    r.raise_for_status()
    return r.json() or []


def delete_endpoint(key: str, endpoint_id: str) -> None:
    _gql(key, 'mutation { deleteEndpoint(id: "%s") }' % endpoint_id)


def update_endpoint(key: str, endpoint_id: str, **fields) -> dict:
    """Endpoint-Settings per PATCH aendern (z. B. workersMax)."""
    if not fields:
        raise RunPodError("update_endpoint ohne Felder")
    r = requests.patch(f"{REST}/v1/endpoints/{endpoint_id}",
                       headers=_headers(key), json=fields, timeout=30)
    if r.status_code not in (200, 201):
        raise RunPodError(f"Endpoint-Patch {r.status_code}: {(r.text or '')[:300]}")
    try:
        return r.json()
    except Exception:
        return {}


def restart_workers(key: str, endpoint_id: str) -> dict:
    """Worker deterministisch recyclen: workersMax->0, warten, zurueck.

    Killt warme Worker (die ihr Image behalten) und zaechtigt so ein
    frisches Image-Pull beim naechsten Job.
    """
    eps = list_endpoints(key)
    current = next((e for e in eps if e["id"] == endpoint_id), None)
    workers_max = (current or {}).get("workersMax") or 1
    update_endpoint(key, endpoint_id, workersMax=0)
    import time as _t
    _t.sleep(25)  # RunPod die Worker terminieren lassen
    update_endpoint(key, endpoint_id, workersMax=workers_max)
    return {"workersMax_restored": workers_max}


def ensure_endpoint(key: str, name: str, template_id: str, gpus: list[str],
                    volume_id: str, workers_max: int = 1,
                    execution_timeout_ms: int = 7200000) -> str:
    """Endpoint mit GPU-Prioritaetsliste (idempotent, DC-tolerant).

    Nicht jeder Datacenter bietet jeden GPU-Typ (z. B. US-KS-2: nur H100 80GB).
    Daher: volle Liste versuchen, dann schrittweise vom Ende (niedrigste
    Prioritaet) kuerzen, bis der Create klappt.
    Existierenden Endpoint nur neu anlegen, wenn die primaere GPU abweicht
    oder GPUs enthalten sind, die nicht mehr gewuenscht sind (kein Endlos-
    Recycle durch gekuerzte Listen).
    """
    want = list(dict.fromkeys(gpus))  # dedupliziert, Reihenfolge bleibt
    for e in list_endpoints(key):
        if e.get("name") == name:
            have = e.get("gpuTypeIds") or []
            same_primary = bool(have) and bool(want) and have[0] == want[0]
            obsolete = [g for g in have if g not in want]
            same_template = e.get("templateId") == template_id
            if same_primary and not obsolete and same_template:
                return e["id"]
            delete_endpoint(key, e["id"])
            break

    base_body = {
        "name": name,
        "templateId": template_id,
        "networkVolumeId": volume_id,
        "workersMin": 0,
        "workersMax": workers_max,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "idleTimeout": 5,
        "executionTimeoutMs": execution_timeout_ms,
        "flashboot": False,
    }
    last_err = ""
    for n in range(len(want), 0, -1):  # volle Liste, dann vom Ende kuerzen
        body = {**base_body, "gpuTypeIds": want[:n]}
        r = requests.post(f"{REST}/v1/endpoints", headers=_headers(key), timeout=30, json=body)
        if r.status_code in (200, 201):
            if n < len(want):
                print(f"[endpoint] nur {n} GPU-Typ(en) verfuegbar: {want[:n]}")
            return r.json()["id"]
        last_err = f"{r.status_code}: {r.text[:200]}"
    raise RunPodError(f"Endpoint-Create fuer keine GPU-Liste erfolgreich ({want}): {last_err}")


# ---------------------------------------------- Model-Download-Pod ---------

def launch_download_pod(key: str, volume_id: str, dc: str, repo: str,
                         gpu: str = "NVIDIA RTX A4000") -> str:
    cmd = (
        "pip3 install -q huggingface_hub requests && "
        f"curl -sL {repo}/raw/main/worker/scripts/download_models.py -o /tmp/dl.py && "
        "python3 /tmp/dl.py --volume /runpod-volume && "
        "echo MODEL_DOWNLOAD_DONE"
    )
    # GraphQL-Literal ohne Typ-Annotationen im Input-Objekt (robust gegen Schema-Schwankungen)
    def lit(v: str) -> str:
        return json.dumps(v)

    def mutation_for(gpu_type: str) -> str:
        return (
            "mutation { podFindAndDeployOnDemand(input: { "
            "cloudType: ALL, "
            "gpuCount: 1, "
            "containerDiskInGb: 20, "
            "volumeInGb: 0, "
            f"gpuTypeId: {lit(gpu_type)}, "
            f"name: {lit('liveact-model-download')}, "
            f"imageName: {lit('runpod/base:1.1.0-ubuntu2204')}, "
            f"dockerArgs: {lit('bash -c ' + json.dumps(cmd))}, "
            f"networkVolumeId: {lit(volume_id)}, "
            "env: [] "
            "}) { id desiredStatus } }"
        )

    # GPU-Fallback: Der Download-Pod braucht keine Rechenleistung — es zaehlt
    # nur Verfuegbarkeit im Datacenter des Volumes. Bei "no instances available"
    # den naechsten Typ versuchen.
    gpu_chain = [gpu] + [g for g in [
        "NVIDIA RTX A5000",
        "NVIDIA A40",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 4090",
    ] if g != gpu]
    last_err: RunPodError | None = None
    tried: list[str] = []
    for gpu_type in gpu_chain:
        tried.append(gpu_type)
        try:
            return _gql(key, mutation_for(gpu_type))["podFindAndDeployOnDemand"]["id"]
        except RunPodError as e:
            last_err = e
            if "no longer any instances" not in str(e) and "availability" not in str(e).lower():
                raise  # anderer Fehler -> nicht weiterversuchen
    raise RunPodError(
        f"Alle GPU-Typen im Datacenter des Volumes ausgelastet "
        f"(versucht: {', '.join(tried)}). Spaeter erneut versuchen oder Volume in "
        f"ein anderes Datacenter legen. Letzter RunPod-Fehler: {last_err}"
    )


def pod_status(key: str, pod_id: str) -> dict:
    q = 'query { pod(input: { podId: "%s" }) { id desiredStatus runtimeStatus } }' % pod_id
    return _gql(key, q)["pod"]


def kill_pod(key: str, pod_id: str) -> None:
    _gql(key, 'mutation { podTerminate(input: { podId: "%s" }) { id } }' % pod_id)


# --------------------------------------------------- Job-Submit (wie gehabt)

def submit(endpoint_id: str, payload: dict, policy: dict | None = None,
           key: Optional[str] = None) -> dict:
    k = _resolve_key(key)
    if not endpoint_id:
        raise RunPodError("Keine Endpoint-ID konfiguriert (Setup erneut ausführen)")
    body = {"input": payload}
    if policy:
        body["policy"] = policy
    r = requests.post(f"{JOBS}/v2/{endpoint_id}/run", headers=_headers(k), json=body, timeout=30)
    try:
        data = r.json()
    except Exception:
        raise RunPodError(f"RunPod /run {r.status_code}: {(r.text or '(leerer Body)')[:300]}")
    if r.status_code != 200 or data.get("status") in ("FAILED",):
        raise RunPodError(f"RunPod /run {r.status_code}: {data}")
    return data


def status(endpoint_id: str, runpod_job_id: str, key: Optional[str] = None) -> dict:
    k = _resolve_key(key)
    r = requests.get(f"{JOBS}/v2/{endpoint_id}/status/{runpod_job_id}",
                     headers=_headers(k), timeout=30)
    if r.status_code != 200:
        raise RunPodError(f"RunPod /status {r.status_code}: {(r.text or '(leerer Body)')[:300]}")
    return r.json()


def cancel(endpoint_id: str, runpod_job_id: str, key: Optional[str] = None) -> dict:
    k = _resolve_key(key)
    r = requests.post(f"{JOBS}/v2/{endpoint_id}/cancel/{runpod_job_id}",
                      headers=_headers(k), timeout=30)
    if r.status_code != 200:
        raise RunPodError(f"RunPod /cancel {r.status_code}: {(r.text or '(leerer Body)')[:300]}")
    return r.json()
