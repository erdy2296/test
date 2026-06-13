import os
from typing import Any

import replicate
from cog import BasePredictor, Input, Secret


TARGET_MODEL = "kwaivgi/kling-v3-motion-control"


def output_to_text(output: Any) -> str:
    if isinstance(output, list):
        if len(output) == 1:
            return str(output[0])
        return "\n".join(str(item) for item in output)

    return str(output)


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        image: str = Input(
            description="URL gambar karakter / model / start image"
        ),
        video: str = Input(
            description="URL video referensi gerakan / motion reference"
        ),
        prompt: str = Input(
            default="",
            description="Prompt tambahan untuk menjaga konsistensi karakter, outfit, background, dan motion"
        ),
        character_orientation: str = Input(
            default="video",
            choices=["image", "video"],
            description="image = mengikuti arah gambar, video = mengikuti orientasi video referensi"
        ),
        mode: str = Input(
            default="std",
            choices=["std", "pro"],
            description="std = 720p, pro = 1080p"
        ),
    ) -> str:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_token.get_secret_value()

        input_data = {
            "image": image,
            "video": video,
            "character_orientation": character_orientation,
            "mode": mode,
        }

        if prompt.strip():
            input_data["prompt"] = prompt

        output = replicate.run(
            TARGET_MODEL,
            input=input_data
        )

        return output_to_text(output)
