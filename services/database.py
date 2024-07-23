from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import hashlib


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
    companyId: Optional[str] = None
    albumId: Optional[str] = None
    text_embedding_field: Optional[Vector] = None
    image_embedding_field: Optional[Vector] = None
    text_embedding_field_1480: Optional[Vector] = None
    image_embedding_field_1480: Optional[Vector] = None

    class Config:
        arbitrary_types_allowed = True


class DBService:
    def __init__(self, project_id, collection):
        self._client = firestore.Client(project=project_id)
        self._collection = collection

        if not self._collection:
            raise ValueError("Collection name must be set")

    @staticmethod
    def encode_image_id(image_path: str) -> str:
        # Generate MD5 hash
        hash_object = hashlib.md5(image_path.encode())
        # Convert hash to hexadecimal string
        hex_dig = hash_object.hexdigest()
        return hex_dig

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
        image_id = self.encode_image_id(data.imagePath)
        doc_ref = self._client.collection(self._collection).document(image_id)
        try:
            doc_ref.set(
                document_data=data.model_dump(),
                merge=True,
            )
        except Exception as e:
            print(f"Error inserting document {data.imageId}: {e}")
        return

    def update_document(self, data: ImageDocument):
        doc_ref = self._client.collection(self._collection).document(data.imageId)
        try:
            doc_ref.set(
                document_data=data.model_dump(),
                merge=True,
            )
        except Exception as e:
            print(f"Error inserting document {data.imageId}: {e}")
        return

    def delete_document(self, document_id: str):
        doc_ref = self._client.collection(self._collection).document(document_id)
        doc_ref.delete()
        return
