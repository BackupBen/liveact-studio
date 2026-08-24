#!/usr/bin/env python3
"""LiveAct Runner — angepasster generate.py-Ablauf für RunPod Serverless.

Unterschiede zu generate.py (Upstream ac8579b, 2026-06-15):
  1. Bibliothek statt Skript: render_video(cfg) statt CLI + input_json
  2. MP3/M4A/OGG -> WAV via ffmpeg (Upstream lädt nur WAV sauber)
  3. CHUNKED EXPORT: decoded frames werden alle N Sekunden als
     Segment-MP4 auf Platte geflusht (libx264) und am Ende per
     ffmpeg concat (stream copy) zusammengefügt + Audio gemuxt.
     => System-RAM bleibt konstant, 30-60-min-Videos möglich.
  4. Progress-Callback an die Webapp (HTTP POST).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from util_liveact import *  # noqa: F401,F403 — Upstream-Helpers

from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
from fp4_gemm import FP4GemmOptions, enable_fp4_gemm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

FFMPEG = "ffmpeg"


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def convert_to_wav(src: Path, dst: Path) -> Path:
    """MP3/M4A/OGG/... -> 16-bit PCM WAV (48 kHz, mono downmix)."""
    if src.suffix.lower() == ".wav":
        return src
    subprocess.run(
        [FFMPEG, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "48000",
         "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True,
    )
    return dst


# ---------------------------------------------------------------------------
# Segmentierter Video-Writer
# ---------------------------------------------------------------------------

class ChunkedVideoWriter:
    """Sammelt Frames im RAM bis chunk_frames erreicht, dann libx264-Segment.

    Frames kommen als uint8-fähige float-Tensoren [T, H, W, C] (0..1).
    Am Ende: ffmpeg concat demuxer (stream copy) => ein MP4 ohne Re-Encode.
    """

    def __init__(self, workdir: Path, fps: int, width: int, height: int,
                 chunk_frames: int = 4 * 300, crf: int = 17):
        self.dir = workdir / "segments"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.w, self.h = width, height
        self.chunk_frames = chunk_frames
        self.crf = crf
        self.segments: list[Path] = []
        self._buf: list[np.ndarray] = []
        self._frames_written = 0

    def write(self, frames: np.ndarray) -> None:
        """frames: [T, H, W, C] float 0..1 oder uint8."""
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
        self._buf.append(frames)
        self._frames_written += frames.shape[0]
        if sum(b.shape[0] for b in self._buf) >= self.chunk_frames:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        arr = np.concatenate(self._buf, axis=0)
        self._buf.clear()
        idx = len(self.segments)
        seg = self.dir / f"seg_{idx:05d}.mp4"
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.w}x{self.h}", "-r", str(self.fps),
             "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
             "-pix_fmt", "yuv420p", str(seg)],
            input=arr.tobytes(), check=True,
        )
        self.segments.append(seg)

    def finalize(self) -> Path:
        self._flush()
        if not self.segments:
            raise RuntimeError("Keine Segmente geschrieben")
        if len(self.segments) == 1:
            return self.segments[0]
        lst = self.dir / "concat.txt"
        lst.write_text("\n".join(f"file '{s.as_posix()}'" for s in self.segments), encoding="utf-8")
        out = self.dir.parent / "video_silent.mp4"
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(out)],
            check=True, capture_output=True,
        )
        return out

    def mux_audio(self, silent: Path, audio: Path, out: Path) -> Path:
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error",
             "-i", str(silent), "-i", str(audio),
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", "-movflags", "+faststart", str(out)],
            check=True, capture_output=True,
        )
        return out


# ---------------------------------------------------------------------------
# Haupt-Render-Funktion (Struktur 1:1 aus generate.py, Zeilennummern @ ac8579b)
# ---------------------------------------------------------------------------

class LiveActRenderer:
    def __init__(self, cfg: dict, progress_cb: Optional[Callable[[int, str], None]] = None):
        self.cfg = cfg
        self.cb = progress_cb or (lambda pct, note: None)
        self.report(1, "Lade Modelle …")

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
            from xfuser.core.distributed import (init_distributed_environment,
                                                 initialize_model_parallel)
            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(),
                                      ring_degree=1, ulysses_degree=world_size)

        if world_size > 1:
            from model_liveact.model_memory_sp import WanModel
        else:
            from model_liveact.model_memory import WanModel

        size = cfg["size"]  # "W*H" wie im Upstream (width*height)
        width, height = [int(_) for _ in size.split("*")]
        self.width, self.height = width, height
        fps = cfg.get("fps", 25)
        self.fps = fps
        vae_stride = (4, 8, 8)
        patch_size = (1, 2, 2)
        timesteps = [torch.tensor([_]).to(device, dtype=torch.float32)
                     for _ in [1000.0, 937.5, 833.33333333, 0.0]]
        blksz_lst = [6, 8]
        frame_len = (height // (patch_size[1] * vae_stride[1])) * (width // (patch_size[2] * vae_stride[2]))
        kv_cache_tokens = frame_len * sum(blksz_lst) // world_size
        kv_cache_device = "cpu" if cfg.get("offload_cache") else device
        kv_cache_dtype = torch.float8_e4m3fn if cfg.get("fp8_kv_cache") else torch.bfloat16
        kv_scale_shape = (1, kv_cache_tokens, 40, 1)
        kv_cache = {i: {layer_id: {
            "k": torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
            "v": torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
            "k_scale": torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if cfg.get("fp8_kv_cache") else None,
            "v_scale": torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if cfg.get("fp8_kv_cache") else None,
            "mean_memory": cfg.get("mean_memory", False),
            "offload_cache": cfg.get("offload_cache", False),
            "fp8_kv_cache": cfg.get("fp8_kv_cache", False),
        } for layer_id in range(40)} for i in range(len(timesteps) - 1)}
        if cfg.get("audio_cfg", 1.0) > 1.0:
            kv_cache_null_audio = {i: {layer_id: {
                "k": torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                "v": torch.zeros([1, kv_cache_tokens, 40, 128], dtype=kv_cache_dtype, device=kv_cache_device),
                "k_scale": torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if cfg.get("fp8_kv_cache") else None,
                "v_scale": torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device) if cfg.get("fp8_kv_cache") else None,
                "mean_memory": cfg.get("mean_memory", False),
                "offload_cache": cfg.get("offload_cache", False),
                "fp8_kv_cache": cfg.get("fp8_kv_cache", False),
            } for layer_id in range(40)} for i in range(len(timesteps) - 1)}

        ckpt_dir = cfg["ckpt_dir"]
        wan_i2v_model = WanModel.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        wan_i2v_model = wan_i2v_model.to(dtype=torch.bfloat16)
        for n in range(40):
            wan_i2v_model.blocks[n].self_attn.init_kvidx(frame_len, world_size)

        vae = LightVAE(vae_path=os.path.join(ckpt_dir, "Wan2.1_VAE.pth"), dtype=torch.bfloat16, device=device,
                       use_lightvae=False, parallel=(world_size > 1))

        clip = CLIPModel(
            checkpoint_path=os.path.join(ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            tokenizer_path=os.path.join(ckpt_dir, "xlm-roberta-large"), dtype=torch.bfloat16, device=device)
        clip.model = clip.model.to(device, dtype=torch.bfloat16)

        text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16,
                                      device="cpu" if cfg.get("t5_cpu") else device,
                                      checkpoint_path=os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                                      tokenizer_path=os.path.join(ckpt_dir, "google/umt5-xxl"))

        audio_encoder = Wav2Vec2Model.from_pretrained(
            cfg["wav2vec_dir"], local_files_only=True, torch_dtype=torch.bfloat16
        ).to(device, dtype=torch.bfloat16).eval()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg["wav2vec_dir"], local_files_only=True)

        audio_encoder.feature_extractor._freeze_parameters()
        wan_i2v_model.freqs = wan_i2v_model.freqs.to(device)
        for _model in [wan_i2v_model, clip.model, audio_encoder, vae.model]:
            for name, param in _model.named_parameters():
                param.requires_grad = False

        if cfg.get("fp8_gemm"):
            enable_fp8_gemm(wan_i2v_model, options=FP8GemmOptions())
        if cfg.get("fp4_gemm"):
            def filter_fn(name, module):
                return "blocks." in name
            enable_fp4_gemm(wan_i2v_model, options=FP4GemmOptions(), module_filter=filter_fn)
        if cfg.get("block_offload"):
            for name, child in wan_i2v_model.named_children():
                if name != "blocks":
                    child.to(device)
            wan_i2v_model.enable_block_offload(onload_device=torch.device(f"cuda:{device}"))
        else:
            wan_i2v_model = wan_i2v_model.to(device)
        wan_i2v_model.eval()
        # torch.compile nur wenn Dynamo aktiv (TORCHDYNAMO_DISABLE=1 => eager).
        # Eager + SageAttention ist deterministisch und hier schneller als
        # compile + flash_attn (Custom-Ops untracebar, Recompiles bei dynamischen
        # Sequenzlaengen). Siehe Dockerfile-Kommentar.
        if os.getenv("TORCHDYNAMO_DISABLE", "0") != "1":
            wan_i2v_model = torch.compile(wan_i2v_model)

        vae.model.eval()
        if os.getenv("TORCHDYNAMO_DISABLE", "0") != "1":
            vae.encode = torch.compile(vae.encode)
        torch_gc()

        self.rank, self.world_size, self.device = rank, world_size, device
        self.model, self.vae, self.clip, self.text_encoder = wan_i2v_model, vae, clip, text_encoder
        self.audio_encoder, self.wav2vec_fx = audio_encoder, wav2vec_feature_extractor
        self.timesteps, self.blksz_lst = timesteps, blksz_lst
        self.vae_stride = vae_stride
        self.frame_len = frame_len
        self.kv_cache = kv_cache
        self.kv_cache_null_audio = kv_cache_null_audio if cfg.get("audio_cfg", 1.0) > 1.0 else None

    def report(self, pct: int, note: str):
        print(f"[liveact {pct:3d}%] {note}", flush=True)
        try:
            self.cb(pct, note)
        except Exception:
            pass

    def render(self, image_path: str, audio_wav: str, out_mp4: str,
               prompt: str = "a person is talking", edit_prompts: Optional[dict] = None,
               seed: int = 42, chunk_seconds: int = 120, workdir: Optional[Path] = None) -> dict:
        t_start = time.time()
        cfg = self.cfg
        device, fps = self.device, self.fps
        height, width = self.height, self.width
        vae_stride, blksz_lst, timesteps = self.vae_stride, self.blksz_lst, self.timesteps

        workdir = workdir or Path(out_mp4).parent / "work"
        workdir.mkdir(parents=True, exist_ok=True)

        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (height, width))),
            transforms.ToTensor(),
            transforms.Resize((height, width)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        edit_prompts = edit_prompts or {}
        self.report(3, "Kodiere Prompt …")
        context = [self.text_encoder(texts=prompt, device="cpu" if cfg.get("t5_cpu") else device)[0]
                   .to(device, dtype=torch.bfloat16)]
        if edit_prompts:
            edit_prompts = {k: self.text_encoder(texts=v, device="cpu" if cfg.get("t5_cpu") else device)[0]
                            .to(device, dtype=torch.bfloat16) for k, v in edit_prompts.items()}

        image = Image.open(image_path).convert("RGB")
        cond_image = transform(image).unsqueeze(1).unsqueeze(0).to(device, torch.bfloat16)
        self.clip.model.to(device)
        clip_context = self.clip.visual(cond_image)
        self.clip.model.cpu()
        torch_gc()

        audio_ori, sr_ori = torchaudio.load(audio_wav)

        def resample_audio(audio, sr, fps):
            rate = 25 / fps
            effects = [["tempo", f"{rate}"]]
            y, sr = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
            resampler = T.Resample(sr, 16000)
            return resampler(y) * 3.0, 16000

        self.report(4, "Audio-Embeddings …")
        audio, sr = resample_audio(audio_ori, sr_ori, fps)
        audio_embedding = get_embedding(audio[0], self.wav2vec_fx, self.audio_encoder, device=device)
        audio_len = audio_ori.size(1) / sr_ori

        ref_target_masks = torch.ones(3, height // vae_stride[1], width // vae_stride[2]).to(device, torch.bfloat16)
        frame_num = (sum(blksz_lst) - 1) * 4 + 1
        msk = get_msk(frame_num, cond_image, vae_stride, device)

        def get_y(frame_num):
            video_frames = torch.zeros(
                1, cond_image.shape[1], frame_num - cond_image.shape[2], height, width
            ).to(cond_image.device, cond_image.dtype)
            padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
            y = self.vae.encode(padding_frames_pixels_values.to(self.vae.device)).to(self.model.device).unsqueeze(0)
            y = torch.concat([msk, y], dim=1)
            return y

        y = get_y(frame_num)

        iter_total_num = int(audio_len / (vae_stride[0] * blksz_lst[-1] / fps)) + 1
        total_frames_expected = iter_total_num * blksz_lst[-1] * vae_stride[0]
        writer = ChunkedVideoWriter(workdir, fps, width, height,
                                    chunk_frames=max(1, chunk_seconds * fps))
        self.report(5, f"Generiere {iter_total_num} Blöcke (~{total_frames_expected} Frames) …")

        torch.manual_seed(seed)
        gen_start = time.time()
        last_report = time.time()
        pre_latent = None
        frames_out = 0

        for _ in range(iter_total_num):
            t1 = time.time()
            audio_start_idx, audio_end_idx = 0, frame_num
            if (_ - 1) * blksz_lst[-1] * vae_stride[0] > 0:
                audio_start_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]
                audio_end_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]

            if not cfg.get("steam_audio"):
                audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, device)
            else:
                audio, sr = resample_audio(
                    audio_ori[:1, int(sr_ori * (audio_start_idx / fps)):int(sr_ori * ((audio_end_idx + 2) / fps))],
                    sr_ori, fps)
                audio_embedding = get_embedding(audio[0], self.wav2vec_fx, self.audio_encoder, device=device)
                audio_embs = get_audio_emb(audio_embedding, 0, frame_num, device)

            y_cut = y[:, :, : frame_num // 4 + 1, ...]

            _context = context
            if edit_prompts:
                for k, v in edit_prompts.items():
                    if ast.literal_eval(k)[0] <= _ <= ast.literal_eval(k)[1]:
                        _context = [v]
                        break

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                f = _ if _ <= 1 else 1
                latent = torch.randn(16, blksz_lst[f], height // vae_stride[1], width // vae_stride[2],
                                     dtype=torch.bfloat16, device=device)
                for i in range(len(timesteps) - 1):
                    timestep = timesteps[i]
                    arg_c = {"context": _context, "clip_fea": clip_context,
                             "ref_target_masks": ref_target_masks,
                             "audio": audio_embs,
                             "y": y_cut[:, :, sum(blksz_lst[:f]):sum(blksz_lst[:f + 1])],
                             "start_idx": sum(blksz_lst[:f]) * self.frame_len,
                             "end_idx": sum(blksz_lst[:f + 1]) * self.frame_len,
                             "update_cache": _ > 1}
                    noise_pred = self.model([latent.to(device)], t=timestep, kv_cache=self.kv_cache[i],
                                            skip_audio=False if i in [1, 2] else True, **arg_c)[0]

                    if cfg.get("audio_cfg", 1.0) > 1.0 and i in [1, 2]:
                        arg_null_audio = dict(arg_c)
                        arg_null_audio["audio"] = torch.zeros_like(audio_embs)
                        noise_pred_drop_audio = self.model(
                            [latent.to(device)], t=timestep, kv_cache=self.kv_cache_null_audio[i],
                            **arg_null_audio)[0]
                        noise_pred = noise_pred_drop_audio + cfg["audio_cfg"] * (noise_pred - noise_pred_drop_audio)

                    x0_pred = latent + (-noise_pred) * (timesteps[i][0] / 1000 - 0.0)
                    latent = (1 - timesteps[i + 1][0] / 1000) * x0_pred + torch.randn_like(x0_pred) * (timesteps[i + 1][0] / 1000)

                if f == 0:
                    _latent = latent
                    _videos = self.vae.decode(_latent.squeeze(0))
                else:
                    _latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
                    _videos = self.vae.decode(_latent.squeeze(0))[:, :, 9:]
                pre_latent = latent
                frames = ((_videos.permute(0, 2, 3, 4, 1)[0] + 1.0) / 2).float().cpu().numpy()
                writer.write(frames)
                frames_out += frames.shape[0]

                now = time.time()
                if now - last_report > 15 or _ == iter_total_num - 1:
                    last_report = now
                    pct = 5 + int(90 * frames_out / max(1, total_frames_expected))
                    fps_now = frames_out / max(1e-9, now - gen_start)
                    eta_min = (total_frames_expected - frames_out) / max(0.1, fps_now) / 60
                    self.report(min(pct, 95), f"Frame {frames_out}/{total_frames_expected} · {fps_now:.1f} FPS · ETA {eta_min:.0f} min")

        self.report(96, "Video zusammensetzen …")
        silent = writer.finalize()
        self.report(97, "Audio muxen …")
        final = writer.mux_audio(silent, audio_wav, Path(out_mp4))
        render_seconds = round(time.time() - t_start, 1)
        self.report(100, f"Fertig in {render_seconds}s")
        return {
            "output": str(final),
            "render_seconds": render_seconds,
            "frames": frames_out,
            "fps": fps,
            "width": width,
            "height": height,
        }


# Bewusst simpel: frame_len_local wird in __init__ berechnet — Alias für Lesbarkeit
