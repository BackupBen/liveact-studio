# LiveAct Studio — HTTP-API für VidForge

Alle Routen relativ zur Webapp-Basis-URL (z. B. `https://liveact.example.com`).
Für VidForge sind nur **3 Calls** relevant: Avatar anlegen (einmalig) → Job einreichen → Status pollen → Video holen.

---

## 1️⃣ Avatar anlegen (einmalig pro Avatar)

```http
POST /api/avatars
Content-Type: multipart/form-data

name=Theo
image=<avatar.png>        # PNG/JPG/WEBP, frontales Porträt, max 20 MB
```

**Response `200`:**
```json
{ "id": "Theo", "name": "Theo", "image": "/api/avatars/Theo/image", "upsert": true }
```

**Upsert:** Postet VidForge einen bereits existierenden Namen, wird das Bild
stillschweigend ersetzt (kein DELETE nötig). VidForge kann also vor jedem
Produktionslauf einfach den aktuellen Avatar POSTen — idempotent.

Avatare auflisten: `GET /api/avatars` → `{"avatars":[{"id":"Theo", ...}]}`
Avatar-Bild abrufen: `GET /api/avatars/{id}/image` (PNG/JPG/WEBP)
Löschen: `DELETE /api/avatars/{id}`

---

## 2️⃣ Render-Job einreichen

```http
POST /api/jobs
Content-Type: multipart/form-data

audio=<voiceover.mp3>     # MP3/WAV/M4A/OGG/FLAC/OPUS/AAC/WEBM, max 500 MB, max 75 min
avatar_id=Theo            # Avatar-ID aus Schritt 1
prompt=a person is talking  # optional
size=768*432              # optional — PRO JOB wählbar:
                          #   768*432  Standard (100 % Rechenzeit)
                          #   512*288  Overlay-Ecke (~2× schneller)
                          #   384*224  Overlay-Ecke (~4× schneller)
                          #   720*416 / 512*512 ebenfalls möglich
fps=20                    # optional (20 Standard | 25 | 24 | 16 Overlay-Sparmodus)
```

**Renderzeit-Regel (Faustformel, 1×H100, warm):**
`Minuten ≈ Audio_min × 2520 / (Breite×Höhe-Pixels relativ zu 768×432) × (fps/20)`
- 768*432 @ 20: ~66 min pro 10 min Audio
- 512*288 @ 20: ~30 min · 384*224 @ 16: ~14 min

**VidForge-Overlay-Muster:** Avatar-Segmente (klein in der Ecke) mit
`size=384*224&fps=16` rendern, Fullscreen-Avatare mit `768*432&fps=20`.
Mische resolution **nicht innerhalb eines Videos** ohne Neuskalierung im
Schnitt — die Segmente unterschiedlicher Größe müssen von VidForge auf
einheitliche Zielgröße gescalet werden (aufrundig empfohlen).

**Response `200`** — Job-Objekt:
```json
{
  "id": "a1b2c3d4e5f6",
  "status": "submitting",
  "avatar_name": "Theo",
  "audio_duration_s": 540.2,
  "size": "768*432",
  "fps": 25,
  "runpod_job_id": "…",
  ...
}
```

**Fehler:** `400` (Audio zu lang/defekt, Avatar fehlt), `502` (S3/RunPod nicht erreichbar)

---

## 3️⃣ Job-Status pollen

```http
GET /api/jobs/{job_id}
```

**Status-Werte:** `queued` → `submitting` → `running` → `postprocessing` → `done` | `error` | `cancelled`

Während `running`:
```json
{
  "status": "running",
  "progress_pct": 47,
  "progress_note": "Frame 5000/10800 · 4.2 FPS · ETA 23 min",
  ...
}
```

Fertig:
```json
{ "status": "done", "video_url": "https://s3.../videos/a1b2c3d4e5f6.mp4?...", "render_seconds": 812.4, "cost_usd": 0.94 }
```

Video direkt streamen (redirect auf S3-Presigned-URL): `GET /api/video/{job_id}`

---

## Beispiel-Integration in VidForge (Node)

```js
// 1) Avatar (einmalig)
const fd = new FormData();
fd.append('name', 'Theo');
fd.append('image', fs.createReadStream('theo.png'));
const av = await fetch(`${BASE}/api/avatars`, { method: 'POST', body: fd }).then(r => r.json());

// 2) Job
const fd2 = new FormData();
fd2.append('audio', fs.createReadStream('9-transkript-reinschrift.mp3'));
fd2.append('avatar_id', av.id);
fd2.append('size', '768*432');
const job = await fetch(`${BASE}/api/jobs`, { method: 'POST', body: fd2 }).then(r => r.json());

// 3) Pollen bis fertig
let done = false;
while (!done) {
  const st = await fetch(`${BASE}/api/jobs/${job.id}`).then(r => r.json());
  console.log(st.status, st.progress_pct + '%', st.progress_note);
  if (st.status === 'done') { console.log('Video:', st.video_url); done = true; }
  if (st.status === 'error') throw new Error(st.error);
  await new Promise(r => setTimeout(r, 15000));
}
```

---

## Alle Routen (Referenz)

| Methode | Route | Zweck |
|---|---|---|
| `GET` | `/healthz` | Healthcheck + Konfigurationsstatus |
| `GET/POST` | `/api/avatars` | Avatare listen/anlegen |
| `GET` | `/api/avatars/{id}/image` | Avatar-Bild |
| `DELETE` | `/api/avatars/{id}` | Avatar löschen |
| `POST` | `/api/jobs` | Render-Job einreichen |
| `GET` | `/api/jobs` | Alle Jobs |
| `GET` | `/api/jobs/{id}` | Job-Status (für VidForge-Polling) |
| `POST` | `/api/jobs/{id}/cancel` | Job abbrechen |
| `POST` | `/api/jobs/{id}/reconcile` | Status von RunPod nachziehen |
| `GET` | `/api/video/{id}` | Video (307 → S3-Presigned) |
| `POST` | `/api/runpod/callback` | Worker-Callbacks (token-geschützt) |
| `GET/POST` | `/api/settings` | RunPod-API-Key & Setup-Parameter |
| `POST` | `/api/setup` | 🔘 Ein-Klick-RunPod-Provisionierung (Volume+Template+Endpoint) |
| `GET` | `/api/setup/status` | Provisionierungs-Fortschritt |
| `POST` | `/api/setup/download-pod` | Model-Download-Pod starten/prüfen/terminieren |

**Hinweis S3:** Die Webapp lädt Audio+Avatar in einen S3-Bucket hoch (der Worker zieht sie sich von dort). Der Bucket muss per ENV (`S3_*`) konfiguriert sein — das ist der eine manuelle Coolify-Schritt, der bleibt.

Interaktive Doku (aus den FastAPI-Schemas generiert): `GET /docs` (Swagger UI) bzw. `GET /openapi.json`.
