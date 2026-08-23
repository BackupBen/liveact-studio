# LiveAct Studio — Webapp (Coolify)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LIVEACT_DATA_DIR=/app/data

# ffprobe für Audiolängen-Prüfung
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY static ./static

RUN mkdir -p /app/data
VOLUME /app/data

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
