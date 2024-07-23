from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ColorWeight(BaseModel):
    name: str
    shade: str
    weight: float


class Metadata(BaseModel):
    height: int
    width: int
    labels: Optional[list[str]] = None
    color_weights: Optional[list[ColorWeight]] = None


class ImageDocument(BaseModel):
    bucket: str
    imageId: str
    imageUrl: str
    imageName: str
    imagePath: str
    imageDescription: Optional[str] = None
    metadata: Metadata
    published: Optional[bool] = False
    valid: Optional[bool] = False
    timeCreated: datetime
    timeUpdated: datetime
    text_embedding_field: Optional[Vector] = None
    image_embedding_field: Optional[Vector] = None

    class Config:
        arbitrary_types_allowed = True


class DBService:
    def __init__(self, project_id, collection):
        self._client = firestore.Client(project=project_id)
        self._collection = collection

        if not self._collection:
            raise ValueError("Collection name must be set")

    def get_documents(self, limit: int = 1000) -> list[ImageDocument]:
        docs = []
        query = self._client.collection(self._collection).limit(limit)

        for doc in query.stream():
            docs.append(ImageDocument(**doc.to_dict()))

        return docs

    def get_document_by_id(self, document_id: str) -> ImageDocument:
        doc_ref = self._client.collection(self._collection).document(document_id)
        doc = doc_ref.get()
        return ImageDocument(**doc.to_dict())

    def add_document(self, data: ImageDocument):
        doc_ref = self._client.collection(self._collection).document()
        doc_ref.set(data, merge=True)
        return

    def update_document(self, data: ImageDocument):
        doc_ref = self._client.collection(self._collection).document(data.imageId)
        doc_ref.set(data, merge=True)
        return

    def delete_document(self, document_id: str):
        doc_ref = self._client.collection(self._collection).document(document_id)
        doc_ref.delete()
        return
