from fastapi import APIRouter, Request, Response
from services.database import DBService, ImageDocument, Metadata
from services.ai import AIService
from config import settings
import utils
from google.cloud.firestore_v1.vector import Vector
import datetime

router = APIRouter()
db = DBService(settings.project_id, settings.firestore_collection)
ai = AIService(settings.project_id, settings.region)


@router.get("")
async def get_images():
    return db.get_documents()


@router.get("/{image_id}")
async def get_image(image_id: str):
    return db.get_document_by_id(image_id)


@router.post("")
async def process_images(req: Request) -> Response:
    body = await req.json()
    image_path = f"gs://{body['bucket']}/{body['image_path']}"
    props = ai.image_properties(image_path)
    height, width = utils.get_image_dimensions(body['bucket'], body['image_path'])
    doc = ImageDocument(
        imageId=db.encode_image_id(body['image_path']),
        imagePath=body['image_path'],
        bucket=body['bucket'],
        imageName=body['image_path'].split("/")[-1],
        imageDescription=props.description,
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
            albumId=body['album_id'],
            companyId=body['company_id']
        )
    )

    return {
        "labels": props.labels,
        "colors": props.colors,
        "safe": props.safe_search,
        "description": props.description
    }


@router.delete("/{image_id}")
async def delete_image(image_id: str):
    return db.delete_document(image_id)
