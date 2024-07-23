from services.ai import AIService
from services.database import DBService
import os
from dotenv import load_dotenv

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
region = os.getenv("GCP_REGION")
firestore_collection = os.getenv("FIRESTORE_COLLECTION")

ai = AIService(project_id, region)
db = DBService(project_id, firestore_collection)

if __name__ == "__main__":
    docs = db.get_documents(limit=2000)
    for doc in docs:
        print(f"Processing document {doc.imageId}")
        image_path = f"gs://{doc.bucket}/{doc.imagePath}"
        props = ai.image_properties(image_path)

        doc.imageDescription = props.description
        doc.metadata.labels = props.labels
        doc.metadata.color_weights = props.colors
        doc.text_embedding_field = props.text_embedding
        doc.image_embedding_field = props.image_embedding

        db._client.collection("vector-image-data").document(doc.imageId).set(doc.model_dump(), merge=True)
        print(f"Updated document {doc.imageId}")
