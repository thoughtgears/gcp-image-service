from PIL import Image
from fastapi import UploadFile
import io


def get_image_dimensions(file: UploadFile):
    file.file.seek(0)  # Reset file pointer to the beginning
    img = Image.open(io.BytesIO(file.file.read()))
    return img.size
