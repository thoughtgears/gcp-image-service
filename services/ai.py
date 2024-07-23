from google.cloud import vision
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from services.database import ColorWeight
import json
from pydantic import BaseModel


class SafeSearch(BaseModel):
    adult: str
    spoof: str
    medical: str
    violence: str
    racy: str


class ImageProperties(BaseModel):
    labels: list[str]
    colors: list[ColorWeight]
    safe_search: SafeSearch
    description: str = None
    text_embedding: list[float] = None
    image_embedding: list[float] = None


class AIService:
    def __init__(self, project_id: str, location: str):
        vertexai.init(project=project_id, location=location)
        self._model_name = "gemini-1.5-flash-001"

    @staticmethod
    def _get_embeddings(image_uri: str, description: str) -> tuple[list[float], list[float]]:
        try:
            model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
            image = Image.load_from_file(image_uri)

            embeddings = model.get_embeddings(
                contextual_text=description,
                image=image,
                dimension=512,
            )

            return embeddings.text_embedding, embeddings.image_embedding
        except Exception as e:
            print(f"An error occurred while getting embeddings: {e}")
            return [], []

    def _get_image_description(self, image_path: str, labels: list[str], emphasis: str = None) -> str:
        model = GenerativeModel(
            self._model_name,
        )

        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 0.95,
        }

        image = Part.from_uri(
            mime_type="image/jpeg",
            uri=image_path,
        )

        prompt = f"""Describe the image with only 150 tokens.
        You must assess the images properly to ensure the description matches the image presented.
        The description MUST be something that could be shown in an ALT IMG tag.
        You can use the following labels to help you describe the image: {labels}
        You can also use the following emphasis to help you describe the image: {emphasis}
        """

        responses = model.generate_content(
            [prompt, image],
            generation_config=generation_config,
        )

        return responses.candidates[0].content.parts[0].text

    def _get_colors(self, colors: str) -> list[ColorWeight]:
        model = GenerativeModel(
            self._model_name,
        )

        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 0.95,
            "response_mime_type": "application/json",
        }

        responses = model.generate_content(
            [f"""Based on the input colors, give me the colors that the RGB values make.
            A color MUST be a single word, i.e. 'red', 'blue', 'green', etc
            The response must be a list of objects with the following keys:
            name: name of color, i.e. 'red', 'blue', 'green', etc
            shade: shade of color, i.e. 'light', 'medium', 'dark', etc
            weight: weight of color in float numbers
            
            {colors}
            """],
            generation_config=generation_config,
        )

        data = json.loads(responses.candidates[0].content.parts[0].text)

        return [ColorWeight(**item) for item in data]

    def image_properties(self, image_path: str) -> ImageProperties:
        client = vision.ImageAnnotatorClient()
        image = vision.Image()
        image.source.image_uri = image_path

        labels_response = client.label_detection(image=image)
        labels = labels_response.label_annotations

        properties_response = client.image_properties(image=image)
        props = properties_response.image_properties_annotation

        colors = ""

        for color in props.dominant_colors.colors:
            colors += f"score: {color.score}\n"
            colors += f"\tr: {color.color.red}\n"
            colors += f"\tg: {color.color.green}\n"
            colors += f"\tb: {color.color.blue}\n"

        colors_list = self._get_colors(colors)

        response = client.safe_search_detection(image=image)
        safe = response.safe_search_annotation
        likelihood_name = (
            "UNKNOWN",
            "VERY_UNLIKELY",
            "UNLIKELY",
            "POSSIBLE",
            "LIKELY",
            "VERY_LIKELY",
        )

        description = self._get_image_description(image_path, labels)

        text_embedding, image_embedding = self._get_embeddings(image_path, description)

        image_props = ImageProperties(
            labels=[label.description for label in labels],
            colors=colors_list,
            description=description,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            safe_search=SafeSearch(
                adult=likelihood_name[safe.adult],
                spoof=likelihood_name[safe.spoof],
                medical=likelihood_name[safe.medical],
                violence=likelihood_name[safe.violence],
                racy=likelihood_name[safe.racy],
            ),
        )

        return image_props
