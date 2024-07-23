from services.ai import AIService
from services.database import DBService
import os
from dotenv import load_dotenv
from google.cloud.firestore_v1.vector import Vector

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
region = os.getenv("GCP_REGION")
firestore_collection = os.getenv("FIRESTORE_COLLECTION")

ai = AIService(project_id, region)
db = DBService(project_id, firestore_collection)
db2 = DBService(project_id, "vector-image-data")

if __name__ == "__main__":
    docs = db.get_documents(limit=2000)
    docs2 = db2.get_documents(limit=2000)
    docs2_ids = {doc.imageId for doc in docs2}
    for doc in docs:
        if doc.imageId in docs2_ids:
            print(f"Skipping document {doc.imageId} as it is already in docs2")
            continue  # Skip the processing for this document

        print(f"Processing document {doc.imageId}")
        image_path = f"gs://{doc.bucket}/{doc.imagePath}"

        try:
            props = ai.image_properties(image_path)

            doc.imageDescription = props.description
            doc.metadata.labels = props.labels
            doc.metadata.color_weights = props.colors

            # Ensure embeddings are not empty
            if not props.text_embedding or not props.image_embedding:
                print(f"Skipping document {doc.imageId} due to empty embeddings")
                continue

            doc.text_embedding_field = Vector(props.text_embedding)
            doc.image_embedding_field = Vector(props.image_embedding)

            # Update the document in Firestore
            db._client.collection("vector-image-data").document(doc.imageId).set(doc.dict(exclude={'id'}), merge=True)
            print(f"Updated document {doc.imageId}")
        except Exception as e:
            print(f"An error occurred while processing document {doc.imageId}: {e}")
            continue  # Skip to the next document if an error occurs

        props = ai.image_properties(image_path)

        doc.imageDescription = props.description
        doc.metadata.labels = props.labels
        doc.metadata.color_weights = props.colors
        doc.text_embedding_field = Vector(props.text_embedding)
        doc.image_embedding_field = Vector(props.image_embedding)

        db._client.collection("vector-image-data").document(doc.imageId).set(doc.model_dump(), merge=True)
        print(f"Updated document {doc.imageId}")
