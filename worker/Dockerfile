# LiveAct Studio — Worker-Image für RunPod Serverless
#
# Build-Strategie (GH-Actions-Runner-freundlich):
#   - python:3.10-slim + cu128-Torch-Wheels
#   - KEIN SageAttention-Compile (Upstream hat try/except-Fallback auf SDPA)
#   - KEIN LightX2V-pip-Install: dessen __init__ zieht die Pipeline-Bibliothek
#     (zmq etc.) rein. Wir vendoren NUR die 3 Dateien, die generate.py braucht.
# Modelle liegen NICHT im Image, sondern auf dem RunPod Network Volume.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/runpod-volume/models/LiveAct \
    WAV2VEC_DIR=/runpod-volume/models/chinese-wav2vec2-base \
    LIVEACT_SIZE=768*432 \
    LIVEACT_FPS=20 \
    LIVEACT_T5_CPU=1 \
    LIVEACT_FP8_KV=1 \
    TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg sox libsox-dev libsox-fmt-all build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upstream-Repo gepinnt auf getesteten Commit
ARG LIVEACT_COMMIT=ac8579b907756f228e975922a5159765c93c6a34
RUN git clone https://github.com/Soul-AILab/SoulX-LiveAct.git /LiveAct \
    && cd /LiveAct && git checkout ${LIVEACT_COMMIT} && rm -rf /LiveAct/.git \
    # clip.py ruft flash_attention hart mit version=2 (assert-crash ohne flash-attn).
    # version=None -> attention.py nutzt fa3/fa2 falls da, sonst SDPA-Fallback.
    && sed -i 's/version=2/version=None/g' wan/modules/clip.py \
    && grep -c "version=None" wan/modules/clip.py

WORKDIR /LiveAct

# typing-extensions von PyPI vorab (pip/torch-index Konflikt vermeiden)
RUN pip install --no-cache-dir "typing-extensions==4.12.2"

# PyTorch (cu128-Wheels bringen eigene CUDA-Runtimes mit)
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Rest der Upstream-Requirements (ohne torch-Pins)
RUN grep -v -E "^(torch|torchaudio|torchvision)" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && rm -rf /root/.cache /tmp/req.txt

# flash-attn v2.8.3 als Prebuilt-Wheel (cu12/torch2.8/cp310) — DIREKTER Download.
# (pip --find-links auf GitHub-Releases findet das Wheel nicht zuverlaessig und
# wuerde aus dem Quellcode bauen -> CUDA_HOME-Crash. clip.py braucht flash_attn
# zwingend: flash_attention() hat keinen SDPA-Fallback.)
# ABI-Variante (cxx11abi TRUE/FALSE) dynamisch von torch ablesen.
RUN python -c "import torch, urllib.request, os; \
abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'; \
whl = f'flash_attn-2.8.3+cu12torch2.8cxx11abi{abi}-cp310-cp310-linux_x86_64.whl'; \
print('downloading', whl); \
urllib.request.urlretrieve('https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/' + whl, '/tmp/' + whl)" \
    && WHL=$(python -c "import torch; abi='TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'; print(f'flash_attn-2.8.3+cu12torch2.8cxx11abi{abi}-cp310-cp310-linux_x86_64.whl')") \
    && pip install --no-cache-dir "/tmp/${WHL}" \
    && rm -f "/tmp/${WHL}" \
    && python -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'import OK')"

# --- SageAttention v2 (optional, Attention-Speedup im 14B-Backbone) ---------
# KEINE Prebuilt-Wheels; die pip nvidia-cuda-nvcc-Wheels enthalten KEIN nvcc
# (nur ptxas). Daher nvcc aus NVIDIAs debian12-apt-Repo (Base = python:3.10-slim
# auf Debian 12). Nur sm_90 (H100) + MAX_JOBS=2 gegen Runner-RAM-Tod.
# Fehlschlag = WARN + Image ohne Sage (Upstream faellt auf flash_attn zurueck).
# KEIN cuda-keyring-.deb: es legt eine eigene (SHA1-signierte) Sources-List an,
# die mit unserer trusted-Liste kollidiert. Wir registrieren das Repo direkt.
RUN echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64 /" > /etc/apt/sources.list.d/cuda-x86_64.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH \
    && nvcc --version \
    && git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/Sage \
    && (cd /tmp/Sage && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="9.0" pip install --no-cache-dir --no-build-isolation .) \
    && rm -rf /tmp/Sage /root/.cache \
    && python -c "from sageattention import sageattn; print('sageattn import OK')" \
    || echo "WARN: SageAttention-Compile fehlgeschlagen — Image ohne Sage (flash_attn aktiv)"

# --- LightVAE: minimal vendoren statt pip-Install --------------------------
# lightx2v/__init__.py importiert die gesamte Pipeline (zmq etc.) — die braucht
# generate.py NICHT. Wir legen nur die benoetigten Module an; alle __init__.py
# bleiben LEER, damit kein Treiber-/Pipeline-Ballast laedt. Der Import von
# global_var in vae.py passiert als Einzelpfad via importlib (siehe
# liveact_runner-Compat-shim in site-packages).
RUN git clone --depth 1 https://github.com/ModelTC/LightX2V.git /tmp/LightX2V \
    && LX2V=/usr/local/lib/python3.10/site-packages/lightx2v \
    && PLAT=/usr/local/lib/python3.10/site-packages/lightx2v_platform \
    && mkdir -p $LX2V/models/video_encoders/hf/wan $LX2V/utils \
                $PLAT/base \
    && cp /tmp/LightX2V/lightx2v/models/video_encoders/hf/wan/vae.py \
          $LX2V/models/video_encoders/hf/wan/vae.py \
    && cp /tmp/LightX2V/lightx2v/utils/envs.py $LX2V/utils/envs.py \
    && cp /tmp/LightX2V/lightx2v/utils/utils.py $LX2V/utils/utils.py 2>/dev/null || true \
    && cp /tmp/LightX2V/lightx2v_platform/base/global_var.py $PLAT/base/global_var.py \
    && find $LX2V $PLAT -type d -exec touch {}/__init__.py \; \
    && python -c "import ast; ast.parse(open('$LX2V/models/video_encoders/hf/wan/vae.py').read()); print('VAE-Datei syntaktisch OK')" \
    && rm -rf /tmp/LightX2V /root/.cache

# Unsere Dateien + Serverless-Extras
COPY liveact_runner.py handler.py lightx2v_compat.py runpod_launcher.py render_server.py /LiveAct/
RUN pip install --no-cache-dir runpod boto3 requests pyzmq \
    && SP=/usr/local/lib/python3.10/site-packages \
    && cp /LiveAct/lightx2v_compat.py $SP/lightx2v_compat.py \
    && echo "import lightx2v_compat" > $SP/lightx2v_compat_pth.py \
    && printf 'import lightx2v_compat\n' > $SP/zz_lightx2v_compat.pth \
    && rm -f $SP/lightx2v_compat_pth.py

# Smoke-Test: Struktur-Check ohne GPU/Gewichte (einzeilig — Dockerfile-Parsing)
RUN python -c "src = open('/LiveAct/liveact_runner.py').read(); assert 'class ChunkedVideoWriter' in src and 'def convert_to_wav' in src; print('liveact_runner Struktur OK')"

ENV HF_HOME=/runpod-volume/hf

CMD [ "python", "-u", "runpod_launcher.py" ]
