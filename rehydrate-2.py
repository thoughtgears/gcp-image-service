from services.ai import AIService
from services.database import DBService
import os
from dotenv import load_dotenv
from google.cloud.firestore_v1.vector import Vector
from google.cloud import vision
import json

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

load_dotenv()

project_id = os.getenv("GCP_PROJECT_ID")
region = os.getenv("GCP_REGION")

ai = AIService(project_id, region)
db = DBService(project_id, "vector-image-data")  # This is the collection to get the data from
client = vision.ImageAnnotatorClient()


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
    docs = db.get_documents(limit=1)
    for doc in docs:
        image_id = doc.imageId

        result = find_image(image_id)
        if result:
            print(f"Found image {image_id} in company {result['company_id']}, album {result['album_id']}")
            doc.companyId = result['company_id']
            doc.albumId = result['album_id']

        print(f"Processing document {doc.imageId}")
        image_path = f"gs://{doc.bucket}/{doc.imagePath}"

        try:
            image = vision.Image()
            image.source.image_uri = image_path
            labels_response = client.label_detection(image=image)
            labels_annotations = labels_response.label_annotations
            labels = [label.description for label in labels_annotations]

            properties_response = client.image_properties(image=image)
            props = properties_response.image_properties_annotation

            colors = ""

            for color in props.dominant_colors.colors:
                colors += f"score: {color.score}\n"
                colors += f"\tr: {color.color.red}\n"
                colors += f"\tg: {color.color.green}\n"
                colors += f"\tb: {color.color.blue}\n"

            colors_list = ai._get_colors(colors)

            description = ai._get_image_description(image_path, labels)

            text = description + " " + ", ".join(labels)
            text_embeddings_1408, image_embeddings_1408 = ai._get_embeddings(image_path, text, 1408)
            text_embeddings_512, image_embeddings_512 = ai._get_embeddings(image_path, text, 512)

            doc.imageDescription = description
            doc.metadata.labels = labels
            doc.metadata.color_weights = colors_list

            # Ensure embeddings are not empty
            if not text_embeddings_1408 or not image_embeddings_1408:
                if not text_embeddings_512 or not image_embeddings_512:
                    print(f"Skipping document {doc.imageId} due to empty embeddings")
                    continue

            doc.text_embedding_field = Vector(text_embeddings_512)
            doc.image_embedding_field = Vector(image_embeddings_512)
            doc.text_embedding_field_1480 = Vector(text_embeddings_1408)
            doc.image_embedding_field_1480 = Vector(image_embeddings_1408)

            print(doc.companyId)
            print(doc.albumId)

            # Update the document in Firestore
            # db._client.collection("vector-image-data").document(doc.imageId).set(doc.model_dump(), merge=True)
            # print(f"Updated document {doc.imageId}")
        except Exception as e:
            print(f"An error occurred while processing document {doc.imageId}: {e}")
            continue  # Skip to the next document if an error occurs
