from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from processors.frame_extracting import create_thumbnails_for_video_message
from models import Feature


def process_text(text: str, processor: CLIPProcessor, clip_model: CLIPModel) -> list[float]:
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)

    features /= features.norm(dim=-1, keepdim=True)
    return features.tolist()


def process_video(video_url: str, processor: CLIPProcessor, clip_model: CLIPModel) -> Feature:
    video_inputs = []
    keyframes = create_thumbnails_for_video_message(video_url)
    for frame in keyframes:
        image = Image.open(frame.file)
        image_input = processor(images=image, return_tensors="pt")
        video_inputs.append(image_input)
    with torch.no_grad():
        video_features = clip_model.get_image_features(**video_inputs[0])
        for video_input in video_inputs[1:]:
            video_feature = clip_model.get_image_features(**video_input)
            video_features = torch.cat((video_features, video_feature), dim=0)
        
        video_features = torch.mean(video_features, dim=0)
        video_features /= video_features.norm(dim=-1, keepdim=True)
    return Feature(
        url=video_url,
        features=video_features.tolist()
    )
