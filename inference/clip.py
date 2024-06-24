import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from deps import Model, Processor, KeyFrameExtractor, lifespan
from frame_video import FrameExtractor

app = FastAPI(lifespan=lifespan)

class EncodeRequest(BaseModel):
    link: Optional[str] = None
    description: Optional[str] = None

@app.get("/")
async def root():
    return JSONResponse(content={"ok": True})

@app.post("/encode")
async def encode(request: EncodeRequest, processor: Processor, model: Model, keyframe_extractor: KeyFrameExtractor):
    if not any((request.description, request.link)):
        raise HTTPException(
            status_code=400, detail="Please provide either 'description' as string or 'link' as video URL, or both."
        )
    
    text_features, image_features = None, None
    
    if request.description:
        request.description = request.description[:65]  # Meet CLIP character limit
        text_inputs = processor(text=[request.description], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
    if request.link:
        images = keyframe_extractor.extract_key_frames(request.link)
        image_inputs = []
        for image in images:
            image = Image.open(image.file)
            image_input = processor(images=image, return_tensors="pt")
            image_inputs.append(image_input)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs[0])
            for image_input in image_inputs[1:]:
                image_feature = model.get_image_features(**image_input)
                image_features = torch.cat((image_features, image_feature), dim=0)

            features = torch.mean(image_features, dim=0)
            features /= features.norm(dim=-1, keepdim=True)

    if request.description and request.link:
        text_weight = 1.0
        video_weight = 2.0  # Giving more importance to video
        # Merged weighted vectors of text and video didn't work so well, leave off for now
        unified_features = (text_features * text_weight + image_features * video_weight) / (text_weight + video_weight)
        return {"features": image_features.tolist()[0]}

    elif request.description:
        return {"features": text_features.tolist()[0]}

    elif request.link:
        return {"features": image_features.tolist()[0]}
