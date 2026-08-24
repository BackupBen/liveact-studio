#!/usr/bin/env python3
"""Handler-Wrapper: startet den eigentlichen Handler, bei >=2 GPUs pro
Worker mit torchrun (RANK/WORLD_SIZE/LOCAL_RANK gesetzt -> Sequence-Parallel).

RunPod Serverless ruft `handler.handler(job)` im Worker-Prozess auf. Wir
ersetzen das Modul: beim Import wird geprueft, wie viele GPUs sichtbar sind.
- 1 GPU: transparenter Durchlauf (wie bisher)
- >=2 GPUs: einmalig einen torchrun-Subprozess (nprocpernode=GPUs) starten,
  der einen Render-Server (SimpleQueueQueue auf 127.0.0.1) bedient; dieser
  Prozess bleibt der RunPod-Handler und leitet jeden Job an den Server weiter.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request

PORT = int(os.getenv("RENDER_SERVER_PORT", "8791"))


def _gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _server_up(retries: int = 60, wait: float = 2.0) -> bool:
    for _ in range(retries):
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                return True
        except OSError:
            time.sleep(wait)
    return False


def _start_render_server() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "RENDER_SERVER_PORT": str(PORT)}
    subprocess.Popen(
        [sys.executable, "-m", "torch.distributed.run",
         f"--nproc_per_node={_gpu_count()}",
         os.path.join(here, "render_server.py")],
        env=env, cwd=here,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    if not _server_up():
        raise RuntimeError("Render-Server kam nicht hoch (siehe Worker-Log)")


def _post_job(job: dict, timeout_s: int = 0) -> dict:
    data = json.dumps(job).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/render", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s or None) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------- Single-GPU:
if _gpu_count() <= 1:
    from handler import handler  # noqa: F401,E402

# ---------------------------------------------------------------- Multi-GPU:
else:
    _start_render_server()

    def handler(job):  # noqa: F811
        # Model-Download: nur eine GPU/CPU-Sache — direkt im Launcher-Prozess
        # ausfuehren (handler.handler behandelt download_models selbst).
        if (job.get("input") or {}).get("download_models"):
            from handler import handler as _single_handler
            return _single_handler(job)
        return _post_job(job)
