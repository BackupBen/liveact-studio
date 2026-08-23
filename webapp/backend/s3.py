"""S3-Upload (boto3) mit Public-/Presigned-URL-Erzeugung.

Benötigt einen S3-kompatiblen Speicher (MinIO, Hetzner, AWS, Cloudflare R2, ...).
Der Worker lädt bei jedem Job Audio+Avatar direkt aus dem Bucket,
und lädt das fertige Video in denselben Bucket hoch.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig

from . import config


def _client():
    if not (config.S3_ENDPOINT or config.S3_BUCKET):
        return None
    kwargs = {
        "aws_access_key_id": config.S3_ACCESS_KEY,
        "aws_secret_access_key": config.S3_SECRET_KEY,
        "region_name": config.S3_REGION,
        "config": BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    }
    if config.S3_ENDPOINT:
        kwargs["endpoint_url"] = config.S3_ENDPOINT
    return boto3.client("s3", **kwargs)


def upload_file(local_path: Path, key: str, public: bool = False) -> str:
    s3 = _client()
    if s3 is None:
        raise RuntimeError("S3 ist nicht konfiguriert (S3_ENDPOINT/S3_BUCKET fehlen)")
    extra = {"ContentType": mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"}
    if public:
        extra["ACL"] = "public-read"
    s3.upload_file(str(local_path), config.S3_BUCKET, key, ExtraArgs=extra)
    return key


def presign_get(key: str, expires_s: int = 7 * 24 * 3600) -> str | None:
    s3 = _client()
    if s3 is None:
        return None
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.S3_BUCKET, "Key": key},
        ExpiresIn=expires_s,
    )
