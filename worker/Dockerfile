# LiveAct Studio — Worker-Image für RunPod Serverless
# Schlanker Build (GH-Actions-tauglich):
#   - KEIN Conda: Ubuntu 22.04 bringt Python 3.10 nativ mit
#   - devel-Base nur fuer die Build-Phase (nvcc fuer SageAttention/LightVAE)
#   - Cleanup nach jedem grossen Schritt (Runner-Disk ist begrenzt)
# Modelle liegen NICHT im Image, sondern auf dem RunPod Network Volume.

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
    TORCH_CUDA_ARCH_LIST="8.9;9.0" \
    MAX_JOBS=4

# System-Pakete + Python 3.10 (Ubuntu 22.04 Default) statt Conda
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates \
        ffmpeg sox libsox-dev libsox-fmt-all \
        build-essential ninja-build \
        python3.10 python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Upstream-Repo gepinnt auf getesteten Commit
ARG LIVEACT_COMMIT=ac8579b907756f228e975922a5159765c93c6a34
RUN git clone https://github.com/Soul-AILab/SoulX-LiveAct.git /LiveAct \
    && cd /LiveAct && git checkout ${LIVEACT_COMMIT} && rm -rf /LiveAct/.git

WORKDIR /LiveAct

# PyTorch zuerst (cu128-Wheels bringen eigene CUDA-Runtimes mit)
RUN pip3 install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Rest der Upstream-Requirements (ohne torch-Pins)
RUN grep -v -E "^(torch|torchaudio|torchvision)" requirements.txt > /tmp/req.txt \
    && pip3 install --no-cache-dir -r /tmp/req.txt \
    && rm -rf /root/.cache /tmp/req.txt

# SageAttention v2.2.0 — kompiliert CUDA-Kernel (nur SM 8.9 + 9.0, seq.)
RUN git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && cd /tmp/SageAttention && python3 setup.py install \
    && rm -rf /tmp/SageAttention /root/.cache

# LightVAE aus LightX2V
RUN git clone --depth 1 https://github.com/ModelTC/LightX2V.git /tmp/LightX2V \
    && cd /tmp/LightX2V && python3 setup_vae.py install \
    && rm -rf /tmp/LightX2V /root/.cache

# Unsere Dateien + Serverless-Extras
COPY liveact_runner.py handler.py /LiveAct/
RUN pip3 install --no-cache-dir runpod boto3 requests

# Smoke-Test: Module-Struktur ohne GPU/Gewichte (tolerant)
RUN python3 -c "import liveact_runner" || echo "WARN: Import-Test fehlgeschlagen (nicht fatal)"

# Letzter Disk-Cleanup
RUN rm -rf /tmp/* /root/.cache /usr/local/cuda/doc /usr/local/cuda/nsight-computation* /usr/local/cuda/samples 2>/dev/null; true

ENV HF_HOME=/runpod-volume/hf

CMD [ "python3", "-u", "handler.py" ]
