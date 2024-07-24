# Image Service

A simple image service that allows anyone to post an image and then get image data back. The service uses AI to generate extra data about the image together
with vector lists of the image. The service is built using FastAPI and uses a firestore database to store the image data.
It will take a image in the body, together with a optional name and then run it through a AI model to generate extra data about the image. The image is then
stored in a gcs bucket and image data is uploaded to a firestore database and returned to the user.
It will also create a Google CDN URL to the image and return that to the user. It will run in app engine to use the Google APIs to generate the public URL to
the image.