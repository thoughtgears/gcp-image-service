from services.ai import AIService
from services.database import DBService
import os
from dotenv import load_dotenv
from google.cloud.firestore_v1.vector import Vector
import json

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
region = os.getenv("GCP_REGION")
firestore_collection = os.getenv("FIRESTORE_COLLECTION")

ai = AIService(project_id, region)
db = DBService(project_id, firestore_collection)
db2 = DBService(project_id, "vector-image-data")


def save_last_document_id(last_document_id: str, file_path: str = "docs/last_document"):
    with open(file_path, 'w') as f:
        f.write(last_document_id)


def load_last_document_id(file_path: str = "docs/last_document") -> str:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return None


def load_data_from_file(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data.update(entry)
    return data


companies_with_albums = load_data_from_file("data/companies_with_albums.json")


def find_image(image_id_query):
    return companies_with_albums.get(image_id_query, None)


if __name__ == "__main__":
    limit = 500
    docs = db.get_documents(limit=limit, start_at=load_last_document_id())
    for doc in docs:
        image_id = doc.imageId

        result = find_image(image_id)
        if result:
            doc.metadata.companyId = result["company_id"]
            doc.metadata.albumId = result["album_id"]

        image_path = f"gs://{doc.bucket}/{doc.imagePath}"

        try:
            props = ai.image_properties(image_path=image_path)
            if not doc.imageDescription:
                doc.imageDescription = props.description
            if not doc.metadata.labels:
                doc.metadata.labels = props.labels
            if not doc.metadata.color_weights:
                doc.metadata.color_weights = props.colors

            text = doc.imageDescription + " " + ", ".join(doc.metadata.labels)
            if not doc.text_embedding_field or not doc.image_embedding_field:
                text_embedding_512, image_embedding_512 = ai.get_embeddings(image_path, text, 512)

                # Ensure embeddings are not empty
                if not text_embedding_512 or not text_embedding_512:
                    print(f"Skipping document {doc.imageId} due to empty embeddings")
                    continue

                doc.text_embedding_field = Vector(text_embedding_512)
                doc.image_embedding_field = Vector(image_embedding_512)

            if not doc.text_embedding_field_1408 or not doc.image_embedding_field_1408:
                text_embedding_1408, image_embedding_1408 = ai.get_embeddings(image_path, text, 1408)

                if not text_embedding_1408 or not image_embedding_1408:
                    print(f"Skipping document {doc.imageId} due to empty embeddings")
                    continue

                doc.text_embedding_field_1408 = Vector(text_embedding_1408)
                doc.image_embedding_field_1408 = Vector(image_embedding_1408)

            very_likely = "VERY_LIKELY"
            safe_search_flags = [
                props.safe_search.adult,
                props.safe_search.spoof,
                props.safe_search.medical,
                props.safe_search.violence,
                props.safe_search.racy
            ]
            doc.valid = all(flag != very_likely for flag in safe_search_flags)
            db2.update_document(doc)
            print(f"Updated document {doc.imageId}")
        except Exception as e:
            print(f"An error occurred while processing document {doc.imageId}: {e}")
            continue  # Skip to the next document if an error occurs

    if docs:
        save_last_document_id(last_document_id=docs[-1].imageId)
