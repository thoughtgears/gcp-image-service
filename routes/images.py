from fastapi import APIRouter, Request, Response, UploadFile, File
from services.database import DBService, ImageDocument, Metadata
from services.ai import AIService
from services.images import ImageService
from config import settings
import utils
from google.cloud.firestore_v1.vector import Vector
import datetime

router = APIRouter()
db = DBService(settings.project_id, settings.firestore_collection)
ai = AIService(settings.project_id, settings.region)
image = ImageService(settings.bucket)


@router.get("")
async def get_images():
    return db.get_documents()


@router.get("/{image_id}")
async def get_image(image_id: str):
    return db.get_document_by_id(image_id)


@router.post("")
async def process_images(file: UploadFile = File(...), image_name: str = None) -> Response:
    if image_name is None:
        image_name = file.filename

    image_path = await utils.save_to_gcs(file, image_name)
    prefix_image_path = f"gs://{image_path}"
    props = ai.image_properties(prefix_image_path)
    height, width = utils.get_image_dimensions(file)
    serving_url = image.get_serving_url(image_path)
    doc = ImageDocument(
        imageId=db.encode_image_id(image_name),
        imagePath=image_path,
        bucket=settings.bucket,
        imageName=image_path.split("/")[-1],
        imageDescription=props.description,
        imageUrl=serving_url,
        published=False,
        valid=props.valid,
        timeCreated=datetime.datetime.now(datetime.UTC),
        timeUpdated=datetime.datetime.now(datetime.UTC),
        text_embedding_field=Vector(props.text_embedding),
        image_embedding_field=Vector(props.image_embedding),
        text_embedding_field_1480=Vector(props.text_embedding_1480),
        image_embedding_field_1480=Vector(props.image_embedding_1480),
        metadata=Metadata(
            height=height,
            width=width,
            labels=props.labels,
            color_weights=props.colors,
        )
    )

    try:
        db.add_document(doc)
    except Exception as e:
        return Response(status_code=500, content=f"An error occurred: {e}")

    return Response(status_code=201)


@router.delete("/{image_id}")
async def delete_image(image_id: str):
    doc = db.get_document_by_id(image_id)
    image.delete_serving_url(doc.imagePath)
    return db.delete_document(image_id)
