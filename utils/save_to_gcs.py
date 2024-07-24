from fastapi import UploadFile
from google.cloud import storage

from config import settings


async def save_to_gcs(file: UploadFile, name: str) -> str:
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(settings.bucket)
    blob = bucket.blob(name)
    content = await file.read()
    blob.upload_from_string(content, content_type=file.content_type)

    return f"{settings.bucket}/{name}"
