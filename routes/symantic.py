from fastapi import APIRouter, Request, Response
from services.vector import VectorSearchService
from config import settings

router = APIRouter()
db = VectorSearchService(settings.project_id, settings.firestore_collection)


@router.get("")
async def get_images(req: Request) -> Response:
    docs = db.get_documents(limit=1)
    print(docs)
    return db.find_nearest(limit=5)
