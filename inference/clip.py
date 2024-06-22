from dataclasses import dataclass
from typing import Callable, Literal

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from inference.frame_video import VideoFrame, create_key_frames_for_video


@dataclass
class CLIP:
    processor: CLIPProcessor
    model: CLIPModel

    _create_key_frames_for_video: Callable[[str], list[VideoFrame]] = create_key_frames_for_video

    def __call__(self, encode_source: str, encode_type: Literal["text"] | Literal["video"]) -> list[float]:
        if encode_type == "text":
            return self._encode_text(encode_source)

        if encode_type == "video":
            return self._encode_video(encode_source)

    def _encode_text(self, description: str) -> list[float]:
        text_inputs = self.processor(text=[description], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.tolist()[0]

    def _encode_video(self, link: str) -> list[float]:
        images = self._create_key_frames_for_video(link)
        image_inputs = []
        for image in images:
            image = Image.open(image.file)
            image_input = self.processor(images=image, return_tensors="pt")
            image_inputs.append(image_input)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs[0])
            for image_input in image_inputs[1:]:
                image_feature = self.model.get_image_features(**image_input)
                image_features = torch.cat((image_features, image_feature), dim=0)

            features = torch.mean(image_features, dim=0)
            features /= features.norm(dim=-1, keepdim=True)

        return features.tolist()[0]
