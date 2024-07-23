from fastapi import APIRouter, Request, Response
from services.database import DBService
from services.ai import AIService
from config import settings

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
    return {
        "labels": props.labels,
        "colors": props.colors,
        "safe": props.safe_search,
        "description": props.description
    }


@router.delete("/{image_id}")
async def delete_image(image_id: str):
    return db.delete_document(image_id)
