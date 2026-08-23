"""Einmaliger Modell-Download auf das RunPod Network Volume.

Ausführen in einem beliebigen Pod mit gemountetem Volume:
    python scripts/download_models.py --volume /runpod-volume

Lädt ~55 GB (LiveAct) + ~400 MB (wav2vec2) von HuggingFace.
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="/runpod-volume", help="Mount-Punkt des Network Volumes")
    args = ap.parse_args()

    vol = Path(args.volume)
    models = vol / "models"
    models.mkdir(parents=True, exist_ok=True)

    print(">> Lade Soul-AILab/LiveAct (~55 GB) …")
    snapshot_download(
        "Soul-AILab/LiveAct",
        local_dir=str(models / "LiveAct"),
        max_workers=8,
    )
    print(">> Lade TencentGameMate/chinese-wav2vec2-base (~400 MB) …")
    snapshot_download(
        "TencentGameMate/chinese-wav2vec2-base",
        local_dir=str(models / "chinese-wav2vec2-base"),
        max_workers=4,
    )
    print("Fertig. Verzeichnis:", models)


if __name__ == "__main__":
    main()
