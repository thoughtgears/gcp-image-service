import os
from pydantic import BaseModel

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""


class Settings(BaseModel):
    project_id: str = os.getenv('GCP_PROJECT_ID')
    region: str = os.getenv('GCP_REGION')
    firestore_collection: str = os.getenv('FIRESTORE_COLLECTION', 'image-data')
    bucket: str = os.getenv('BUCKET_NAME', 'image-data')


settings = Settings()
