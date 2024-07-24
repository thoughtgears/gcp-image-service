from google.appengine.api import images


class ImageService:
    def __init__(self):
        self._gs_prefix = "/gs/"
        pass

    def get_serving_url(self, image_path: str):
        try:
            serving_image = images.get_serving_url(None, filename=self._gs_prefix + image_path, secure_url=True)
            return serving_image
        except images.AccessDeniedError as e:
            raise Exception(f"Access denied to image, Ensure the GAE service account has access to the object in Google Cloud Storage. {e}")
        except images.TransformationError as e:
            raise Exception(f"Error transforming image. {e}")
        except images.ObjectNotFoundError as e:
            raise Exception(f"Image not found. {e}")
        except images.LargeImageError as e:
            raise Exception(f"Image too large. {e}")

    def delete_serving_url(self, image_path: str):
        try:
            images.delete_serving_url(self._gs_prefix + image_path)
        except images.AccessDeniedError as e:
            raise Exception(f"Access denied to image, Ensure the GAE service account has access to the object in Google Cloud Storage. {e}")
        except images.ObjectNotFoundError as e:
            raise Exception(f"Image not found. {e}")
