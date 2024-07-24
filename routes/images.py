from fastapi import APIRouter, Response, UploadFile, File, HTTPException
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
image = ImageService()


@router.get("", response_model=list[ImageDocument], summary="Retrieve all images", description="Get a list of all images stored in the database.")
async def get_images():
    """
    Retrieve all images.
    """
    return db.get_documents()


@router.get("/{image_id}", response_model=ImageDocument, summary="Retrieve a single image by ID", description="Get details of an image using its ID.")
async def get_image(image_id: str):
    """
    Retrieve a single image by ID.
    - **image_id**: The ID of the image to retrieve.
    """
    doc = db.get_document_by_id(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Image not found")
    return doc


@router.post("", status_code=201, summary="Process and store an image", description="Upload, process, and store an image.")
async def process_images(file: UploadFile = File(...), image_name: str = None) -> Response:
    """
    Process and store an image.
    - **file**: The image file to upload.
    - **image_name**: The name of the image. If not provided, the filename of the uploaded file will be used.
    """
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


@router.delete("/{image_id}", status_code=204, summary="Delete an image by ID", description="Delete an image and its associated data using its ID.")
async def delete_image(image_id: str):
    """
    Delete an image by ID.
    - **image_id**: The ID of the image to delete.
    """
    doc = db.get_document_by_id(image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Image not found")
    image.delete_serving_url(doc.imagePath)
    db.delete_document(image_id)
    return Response(status_code=204)
