# LiveAct Studio — Worker-Image für RunPod Serverless
# Build: siehe README (GitHub Container Registry oder RunPod-eigene Registry)
#
# Basemodelle liegen NICHT im Image, sondern auf dem RunPod Network Volume:
#   MODEL_DIR  = /runpod-volume/models/LiveAct
#   WAV2VEC_DIR = /runpod-volume/models/chinese-wav2vec2-base
# (siehe scripts/download_models.py — einmalig in einem Pod ausführen)

FROM continuumio/miniconda3:24.7.1-0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/runpod-volume/models/LiveAct \
    WAV2VEC_DIR=/runpod-volume/models/chinese-wav2vec2-base \
    LIVEACT_SIZE=768*432 \
    LIVEACT_FPS=25 \
    LIVEACT_T5_CPU=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg sox libsox-dev libsox-fmt-all build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10 env (wie Upstream)
RUN conda create -n liveact python=3.10 -y
ENV PATH=/opt/conda/envs/liveact/bin:$PATH

# Upstream-Repo gepinnt auf getesteten Commit
ARG LIVEACT_COMMIT=ac8579b907756f228e975922a5159765c93c6a34
RUN git clone https://github.com/Soul-AILab/SoulX-LiveAct.git /LiveAct \
    && cd /LiveAct \
    && git checkout ${LIVEACT_COMMIT}

WORKDIR /LiveAct

# PyTorch zuerst (cu128-Wheels), dann Rest der Upstream-Requirements
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128
RUN grep -v -E "^(torch|torchaudio|torchvision)" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

# SageAttention v2.2.0 (wie Upstream-README)
RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && cd /tmp/SageAttention && git checkout v2.2.0 \
    && python setup.py install

# LightVAE aus LightX2V (wie Upstream-README)
RUN git clone https://github.com/ModelTC/LightX2V.git /tmp/LightX2V \
    && cd /tmp/LightX2V && python setup_vae.py install

# Unsere Dateien
COPY liveact_runner.py /LiveAct/liveact_runner.py
COPY handler.py /LiveAct/handler.py

# Serverless-Extras
RUN pip install --no-cache-dir runpod boto3 requests

# Kleiner Smoke-Test, dass die Imports existieren (ohne GPU, ohne Gewichte)
RUN python -c "import liveact_runner" || echo "WARN: Import-Test fehlgeschlagen (nicht fatal)"

ENV HF_HOME=/runpod-volume/hf

CMD [ "python", "-u", "handler.py" ]
