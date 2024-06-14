import av
import torch
import requests
from io import BytesIO
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from deps import Model, Processor, lifespan
from keymap import read_video_pyav, sample_frame_indices

app = FastAPI(lifespan=lifespan)

class EncodeRequest(BaseModel):
    text: str = ''
    video_url: Optional[str] = None

@app.get("/")
async def root():
    return JSONResponse(content={"ok": True})


@app.post("/encode")
async def encode(request: EncodeRequest, processor: Processor, model: Model):
    video_data = BytesIO(requests.get(request.video_url).content)
    container = av.open(video_data)
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)
    inputs = processor(
        text=[request.text],
        videos=list(video),
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.video_embeds
    features /= features.norm(dim=-1, keepdim=True)

    return {"features": features.tolist()}
