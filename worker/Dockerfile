# LiveAct Studio — Worker-Image für RunPod Serverless
# Wichtig: devel-Base (mit nvcc) — SageAttention und LightVAE kompilieren CUDA-Kernel.
# Modelle liegen NICHT im Image, sondern auf dem RunPod Network Volume.
#
# Build via GitHub Actions: .github/workflows/docker-worker.yml

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/runpod-volume/models/LiveAct \
    WAV2VEC_DIR=/runpod-volume/models/chinese-wav2vec2-base \
    LIVEACT_SIZE=768*432 \
    LIVEACT_FPS=25 \
    LIVEACT_T5_CPU=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget bzip2 ca-certificates \
        ffmpeg sox libsox-dev libsox-fmt-all \
        build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

# Python 3.10 env (wie Upstream)
RUN conda create -n liveact python=3.10 -y
ENV PATH=/opt/conda/envs/liveact/bin:$PATH

# Upstream-Repo gepinnt auf getesteten Commit
ARG LIVEACT_COMMIT=ac8579b907756f228e975922a5159765c93c6a34
RUN git clone https://github.com/Soul-AILab/SoulX-LiveAct.git /LiveAct \
    && cd /LiveAct \
    && git checkout ${LIVEACT_COMMIT}

WORKDIR /LiveAct

# PyTorch zuerst (cu128-Wheels), dann Rest der Upstream-Requirements ohne torch-Pins
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128
RUN grep -v -E "^(torch|torchaudio|torchvision)" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

# SageAttention v2.2.0 (wie Upstream-README; kompiliert CUDA-Kernel via nvcc)
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

# Smoke-Test: Imports existieren (ohne GPU, ohne Gewichte — tolerant)
RUN python -c "import liveact_runner" || echo "WARN: Import-Test fehlgeschlagen (nicht fatal)"

ENV HF_HOME=/runpod-volume/hf

CMD [ "python", "-u", "handler.py" ]
