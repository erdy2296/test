from typing import Any

import replicate
from cog import BasePredictor, Input, Path

from private_config import REPLICATE_API_TOKEN


TARGET_MODEL = "kwaivgi/kling-v3-motion-control"


def output_to_text(output: Any) -> str:
    if isinstance(output, list):
        return "\n".join(str(item) for item in output)
    return str(output)


class Predictor(BasePredictor):
    def setup(self):
        self.client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    def predict(
        self,
        image: Path = Input(
            description="Upload gambar karakter / start image"
        ),
        video: Path = Input(
            description="Upload video referensi gerakan / motion reference"
        ),
        prompt: str = Input(
            default="",
            description="Prompt tambahan"
        ),
        character_orientation: str = Input(
            default="video",
            choices=["image", "video"],
            description="image = ikut orientasi gambar, video = ikut orientasi video"
        ),
        mode: str = Input(
            default="std",
            choices=["std", "pro"],
            description="std = 720p, pro = 1080p"
        ),
    ) -> str:
        input_data = {
            "image": open(str(image), "rb"),
            "video": open(str(video), "rb"),
            "character_orientation": character_orientation,
            "mode": mode,
        }

        if prompt.strip():
            input_data["prompt"] = prompt.strip()

        output = self.client.run(
            TARGET_MODEL,
            input=input_data
        )

        return output_to_text(output)
