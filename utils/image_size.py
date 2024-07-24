from PIL import Image
from google.cloud import storage
import io


def get_image_dimensions(bucket_name, image_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(image_path)
    img_bytes = blob.download_as_bytes()
    img = Image.open(io.BytesIO(img_bytes))
    width, height = img.size
    return width, height
