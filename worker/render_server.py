"""Multi-GPU-Render-Server: laeuft unter torchrun (1 Prozess pro GPU).

Rank 0 stellt einen winzigen HTTP-Server auf 127.0.0.1:PORT bereit, nimmt
Job-Payloads an und rendert sie mit dem sequence-parallelen Renderer
(liveact_runner aktiviert dist automatisch per RANK/WORLD_SIZE-ENV von
torchrun). Alle Ranks takten gemeinsam ueber torch.distributed barrier.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.distributed as dist

PORT = int(os.getenv("RENDER_SERVER_PORT", "8791"))
RANK = int(os.getenv("RANK", 0))
WORLD = int(os.getenv("WORLD_SIZE", 1))

_renderer = None
_renderer_key = None


def _get_renderer():
    global _renderer, _renderer_key
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    key = (gpu, os.getenv("LIVEACT_SIZE", "760*432"), os.getenv("LIVEACT_FPS", "20"))
    if _renderer is None or _renderer_key != key:
        from liveact_runner import LiveActRenderer
        _renderer = LiveActRenderer({
            "ckpt_dir": os.getenv("MODEL_DIR", "/runpod-volume/models/LiveAct"),
            "wav2vec_dir": os.getenv("WAV2VEC_DIR", "/runpod-volume/models/chinese-wav2vec2-base"),
            "size": key[1], "fps": int(key[2]),
            "fp8_kv_cache": os.getenv("LIVEACT_FP8_KV", "1") == "1",
            "block_offload": os.getenv("LIVEACT_BLOCK_OFFLOAD", "0") == "1",
            "t5_cpu": os.getenv("LIVEACT_T5_CPU", "1") == "1",
        })
        _renderer_key = key
    return _renderer


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # RunPod-Log ruhig halten
        pass

    def do_POST(self):
        global _renderer
        if self.path != "/render":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        job = json.loads(self.rfile.read(length).decode())

        # Alle Ranks muessen rendern — Rank 0 broadcastet das Startsignal
        if RANK == 0:
            obj_list = [job]
        else:
            obj_list = [None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        job = obj_list[0]

        try:
            if job.get("input", {}).get("download_models"):
                if RANK == 0:
                    from handler import model_download
                    out = model_download(job)
                else:
                    out = {}
            else:
                renderer = _get_renderer()
                out = renderer.render_job(job) if hasattr(renderer, "render_job") else \
                      _legacy_render(renderer, job)
            ok, payload = True, out
        except Exception as e:  # noqa: BLE001
            import traceback
            payload = {"error": f"{type(e).__name__}: {e}",
                       "tb": traceback.format_exc()[-2000:]}
            ok = False

        if RANK == 0:
            body = json.dumps({"ok": ok, "output": payload}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        _barrier()


def _legacy_render(renderer, job):
    """Adapter: ruft handler.render pipeline aehnlich handler.py, aber im
    Multi-GPU-Kontext (Rank 0 liefert das Ergebnis zurueck)."""
    ji = job.get("input", job)
    import tempfile, time as _t
    from handler import download_inputs, upload_to_s3  # reuse helpers
    t0 = _t.time()
    local_audio, local_avatar = download_inputs(ji)
    out_dir = tempfile.mkdtemp(prefix="render_")
    out_mp4 = os.path.join(out_dir, f"{ji.get('job_id', 'job')}.mp4")
    renderer.render(
        image_path=local_avatar,
        audio_wav=local_audio,
        out_mp4=out_mp4,
        prompt=ji.get("prompt", "a person is talking, natural movements"),
    )
    s3_key = f"videos/{ji.get('job_id', 'job')}.mp4"
    upload_to_s3(out_mp4, s3_key)
    dur = _t.time() - t0
    return {"s3_key": s3_key, "render_seconds": round(dur, 1),
            "cost_usd": round(dur / 3600 * float(os.getenv("H100_PRICE", "4.55")) * WORLD, 2)}


if RANK == 0:
    print(f"[render_server] rank {RANK}/{WORLD} — HTTP auf 127.0.0.1:{PORT}", flush=True)
HTTPServer(("127.0.0.1", PORT), _Handler).serve_forever()
