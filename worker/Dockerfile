# LiveAct Studio — Worker-Image für RunPod Serverless
#
# Build-Strategie (GH-Actions-Runner-freundlich):
#   - KEIN SageAttention-Compile: LiveAct nutzt es OPTIONAL (try/except-Fallback
#     auf FlashAttention2 / PyTorch-SDPA im Upstream-Code). nvcc-Kompilierung
#     war der Runner-Killer (RAM/Disk).
#   - KEIN Conda: Ubuntu 22.04 bringt Python 3.10 nativ.
#   - devel-Base nur wg. eventuellen Header-Bedarfs; final schlank.
# Modelle liegen NICHT im Image, sondern auf dem RunPod Network Volume.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/runpod-volume/models/LiveAct \
    WAV2VEC_DIR=/runpod-volume/models/chinese-wav2vec2-base \
    LIVEACT_SIZE=768*432 \
    LIVEACT_FPS=25 \
    LIVEACT_T5_CPU=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg sox libsox-dev libsox-fmt-all build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upstream-Repo gepinnt auf getesteten Commit
ARG LIVEACT_COMMIT=ac8579b907756f228e975922a5159765c93c6a34
RUN git clone https://github.com/Soul-AILab/SoulX-LiveAct.git /LiveAct \
    && cd /LiveAct && git checkout ${LIVEACT_COMMIT} && rm -rf /LiveAct/.git

WORKDIR /LiveAct

# PyTorch zuerst (cu128-Wheels bringen eigene CUDA-Runtimes mit).
# typing-extensions VORHER von PyPI: das 4.16-Wheel wird von neuem pip wegen
# Name-Metadata verworfen und der sdist-Build scheitert am Torch-Index (kein flit_core).
RUN pip install --no-cache-dir "typing-extensions==4.12.2"
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Rest der Upstream-Requirements (ohne torch-Pins)
RUN grep -v -E "^(torch|torchaudio|torchvision)" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && rm -rf /root/.cache /tmp/req.txt

# flash-attn v2 (Prebuilt-Wheel, kein Compile) — schnelle Attention statt SDPA-Fallback
# sm89=RTX 4090, sm90=H100. Wheel-Namen: cp310 / torch2.8 / cu12
RUN pip install --no-cache-dir flash-attn --no-build-isolation \
        --find-links https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3 \
    || echo "WARN: flash-attn Wheel nicht gefunden — SDPA-Fallback wird genutzt"

# LightVAE aus LightX2V — av+cython vorher (av hat fertige Wheels; sonst
# wuerde setup_vae.py av per easy_install aus dem Quellcode bauen).
# Fallback: direkter Copy des Pure-Python-Pakets in site-packages.
RUN pip install --no-cache-dir av cython \
    && git clone --depth 1 https://github.com/ModelTC/LightX2V.git /tmp/LightX2V \
    && cd /tmp/LightX2V \
    && (python setup_vae.py install || cp -r lightx2v /usr/local/lib/python3.10/site-packages/) \
    && python -c "from lightx2v.models.video_encoders.hf.wan.vae import WanVAE; print('LightVAE import OK')" \
    && rm -rf /tmp/LightX2V /root/.cache

# Unsere Dateien + Serverless-Extras
COPY liveact_runner.py handler.py /LiveAct/
RUN pip install --no-cache-dir runpod boto3 requests

# Smoke-Test: Struktur-Check ohne GPU/Gewichte (einzeilig — Dockerfile-Parsing)
RUN python -c "src = open('/LiveAct/liveact_runner.py').read(); assert 'class ChunkedVideoWriter' in src and 'def convert_to_wav' in src; print('liveact_runner Struktur OK')"

ENV HF_HOME=/runpod-volume/hf

CMD [ "python", "-u", "handler.py" ]
