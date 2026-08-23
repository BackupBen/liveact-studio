# 🎬 LiveAct Studio

Self-Hosted **HeyGen-Alternative** auf Basis von [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct):
Voiceover-Audio (MP3/WAV) + Avatar-Foto rein → sprechender Avatar raus. 16:9, beliebig lang
(dank ConvKV bleibt der VRAM-Bedarf konstant — Chunked-Export macht 30–60-min-Videos RAM-technisch möglich).

```
┌─────────────┐   POST /api/jobs    ┌──────────────────────┐    S3-Pull/Push   ┌──────────────────┐
│  Webapp     │ ──────────────────► │ RunPod Serverless    │ ────────────────► │ S3/MinIO Bucket  │
│  (Coolify)  │ ◄────────────────── │ Worker (dieses Repo) │ ◄──────────────── │ (audio/avatar/   │
│             │  Callbacks + Poll   │ 1× H100 (o.ä.)       │                   │  videos/)        │
└─────────────┘                     └──────────────────────┘                   └──────────────────┘
```

## Komponenten

| Pfad | Was |
|---|---|
| `webapp/` | FastAPI + Vanilla-JS-Frontend. Avatar-Verwaltung, Audio-Upload, Job-Status, Video-Player. Deploy auf **Coolify**. |
| `worker/` | RunPod-Serverless-Worker: `handler.py` (S3-Download → Render → S3-Upload → Callback), `liveact_runner.py` (Upstream `generate.py` als Bibliothek **mit Chunked-Export** statt RAM-Sammeln), Dockerfile (angepinnt auf Upstream-Commit `ac8579b`). |
| `worker/scripts/download_models.py` | Einmaliger Download der ~55 GB Modellgewichte auf das RunPod Network Volume. |

## Setup-Übersicht (Details unten)

1. **GitHub Repo pushen** (dieses Verzeichnis)
2. **RunPod**: Network Volume anlegen → Pod starten → Modelle downloaden → Worker-Image bauen → Serverless Endpoint erstellen
3. **S3-Bucket** anlegen (MinIO auf dem Coolify-Host o.ä.)
4. **Coolify**: Webapp aus dem GitHub-Repo deployen, ENV-Variablen setzen
5. Loslegen

---

## 1️⃣ GitHub

```bash
git init && git add . && git commit -m "LiveAct Studio"
git remote add origin git@github.com:DEINUSER/liveact-studio.git
git push -u origin main
```

## 2️⃣ RunPod

### Network Volume + Modelle

1. RunPod → **Storage → Network Volume** → anlegen (200 GB, Region frei — muss zur Endpoint-Region passen)
2. Beliebigen **GPU-Pod** mit dem Volume gemountet starten (z. B. RTX 4090, PyTorch-Template)
3. Im Pod-Jupyter/Terminal:

```bash
pip install huggingface_hub
curl -O https://raw.githubusercontent.com/DEINUSER/liveact-studio/main/worker/scripts/download_models.py
python download_models.py --volume /runpod-volume
```

Danach liegt unter `/runpod-volume/models/` → `LiveAct/` + `chinese-wav2vec2-base/`. Pod stoppen.

### Worker-Image bauen

Entweder auf demselben Pod (Docker ist dort verfügbar):

```bash
git clone https://github.com/DEINUSER/liveact-studio.git
cd liveact-studio/worker
docker build -t docker.io/DEINDOCKERHUB/liveact-worker:latest .
docker push docker.io/DEINDOCKERHUB/liveact-worker:latest
```

… oder über GitHub Actions (Registry deiner Wahl). Das Image enthält **keine** Modellgewichte.

### Serverless Endpoint

RunPod → **Serverless → New Endpoint**:

| Einstellung | Wert |
|---|---|
| Container image | `docker.io/DEINDOCKERHUB/liveact-worker:latest` |
| GPU | **H100** (empfohlen) / A100 80GB / RTX 4090 (mit `LIVEACT_FP8_KV=1 LIVEACT_BLOCK_OFFLOAD=1`) |
| Volume Mount | Network Volume → `/runpod-volume` |
| Env | `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` (denselben Bucket wie die Webapp!) |
| Active Workers | 0 (Scale-to-Zero) — Cold Start lädt ~55 GB vom Volume (2–5 min) |
| Execution Timeout | 7200000 (2 h; für 60-min-Videos höher) |

Die **Endpoint-ID** notieren.

## 3️⃣ S3 / MinIO

Beliebig S3-kompatibel. Wichtig: Der RunPod-Worker muss den Bucket erreichen —
bei MinIO auf dem Coolify-Host also die **öffentliche** URL (nicht `localhost`) in `S3_ENDPOINT` tragen.

```
liveact-studio-bucket/
  audio/xxxx_meinvoiceover.mp3
  avatars/theo.png
  videos/<jobid>.mp4
```

## 4️⃣ Coolify (Webapp)

1. Coolify → **New Resource → Dockerfile** → Repo auswählen
2. **Dockerfile-Speicherort**: `/webapp/Dockerfile`
3. Port `8000` freigeben, Domain setzen
4. ENV aus `.env.example` übernehmen:

```
RUNPOD_API_KEY=rpk_xxx
RUNPOD_ENDPOINT_ID=<aus Schritt 2>
S3_ENDPOINT=https://s3.deinhost.de        # öffentlich erreichbar!
S3_BUCKET=liveact-studio
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_REGION=us-east-1
PUBLIC_BASE_URL=https://liveact.deinhost.de
SESSION_SECRET=<zufällig lang>
```

5. Volume auf `/app/data` mounten (Avatare + Job-JSONs überleben Restarts)
6. Deploy → `https://liveact.deinhost.de` öffnen

## 5️⃣ Benutzung

1. **+ Avatar hinzufügen** → frontales Porträt hochladen (Mund gut sichtbar, Beleuchtung egal)
2. Voiceover-MP3 reinziehen → **🚀 Video rendern**
3. Job läuft: `queued → running (mit % und ETA) → done`
4. ▶ Abspielen / Download. Das Video liegt im S3-Bucket unter `videos/<jobid>.mp4`.

## Kosten-Richtwerte (RunPod Serverless, active-compute)

| Video | H100 | A100 80GB |
|---|---|---|
| 1 min | ~$0.15 | ~$0.25 |
| 9 min | ~$1.40 | ~$2.30 |
| 30 min | ~$4.50 | ~$7 |
| 60 min | ~$9 | ~$14 |

(Dazu Network Volume ~$0.07/GB·Monat für ~57 GB ≈ $4/Monat.)

## Troubleshooting

- **Job hängt in `submitting`**: Endpoint-ID/API-Key prüfen; `/api/jobs/{id}/reconcile` ziehen
- **Worker-CRASH / OOM bei >20 min**: sollte durch Chunked-Export nicht passieren — `CHUNK_SECONDS` notfalls senken
- **Callback kommt nie an**: `PUBLIC_BASE_URL` muss von RunPod aus erreichbar sein (https, korrektes Zertifikat); Fallback ist das 5-s-Polling + `reconcile`
- **Lip-Sync wirkt zeitversetzt**: Audio unbedingt als WAV/MP3 **ohne langes Lead-In** liefern; erste ~2 s sind Referenz-Frame

## Hinweise

- Upstream-Quellen in `worker/liveact/` sind der Referenz-Commit `ac8579b` (2026-06-15); `liveact_runner.py` folgt der Struktur von `generate.py` mit denselben Zeilen-Kommentaren.
- Lizenzen: SoulX-LiveAct (Repo) + Wan2.1-Gewichte vor kommerzieller Nutzung prüfen.
