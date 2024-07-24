from services.database import DBService, ImageDocument
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector


class VectorSearchService(DBService):
    def __init__(self, project_id, collection):
        super().__init__(project_id, collection)

    def find_nearest(self, vector: list[float], limit: int = 5) -> list[ImageDocument]:
        collection_ref = self._client.collection(self._collection)

        vector_query = collection_ref.find_nearest(
            vector_field="image_embedding_field",
            query_vector=Vector(vector),
            distance_measure=DistanceMeasure.DOT_PRODUCT,
            limit=limit
        )

        docs = []
        for doc in vector_query.stream():
            docs.append(doc.to_dict())

        return docs
