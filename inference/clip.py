import torch
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from deps import Model, Processor, lifespan
from keymap import create_thumbnails_for_video_message

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

    features = None
    features = None

    if all((text, video_url)):
        raise HTTPException(status_code=400, detail="Please provide either 'text' as string or 'video_url' as video URL, not both.") 
    if not any((text, video_url)):
        raise HTTPException(status_code=400, detail="Please provide either 'text' as string or 'video_url' as video URL, or both.")

    if text:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features /= features.norm(dim=-1, keepdim=True)

    if video_url:
        images = create_thumbnails_for_video_message(video_url)

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

    return {"features": features.tolist()}
