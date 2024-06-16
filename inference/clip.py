import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from deps import Model, Processor, lifespan
from frame_video import create_thumbnails_for_video

app = FastAPI(lifespan=lifespan)

class EncodeRequest(BaseModel):
    text: Optional[str] = None
    video_url: Optional[str] = None

@app.get("/")
async def root():
    return JSONResponse(content={"ok": True})

@app.post("/encode")
async def encode(request: EncodeRequest, processor: Processor, model: Model):
    text = request.text
    video_url = request.video_url
    if not any((text, video_url)):
        raise HTTPException(status_code=400, detail="Please provide either 'text' as string or 'video_url' as video URL, or both.")
    text_features, image_features = None, None
    if text:
        text_inputs = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    if video_url:
        images = create_thumbnails_for_video(video_url)
        image_inputs = []
        for image in images:
            image = Image.open(image.file)
            image_input = processor(images=image, return_tensors="pt")
            image_inputs.append(image_input)
        with torch.no_grad():
            image_features_list = [model.get_image_features(**image_input) for image_input in image_inputs]
            image_features = torch.mean(torch.stack(image_features_list), dim=0)
            image_features /= image_features.norm(dim=-1, keepdim=True)
    if text and video_url:
        text_weight = 1.0
        video_weight = 2.0  # Giving more importance to video
        unified_features = (text_features * text_weight + image_features * video_weight) / (text_weight + video_weight)
        return {"features": unified_features.tolist()[0]}
    elif text:
        return {"features": text_features.tolist()[0]}
    elif video_url:
        return {"features": image_features.tolist()[0]}
