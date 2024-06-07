from fastapi import FastAPI, HTTPException

import torch
import uvicorn
from PIL import Image
from pydantic import BaseModel

from deps import Faiss, Model, Processor, Storage, lifespan
from keymap import create_thumbnails_for_video_message
from storage import Feature


app = FastAPI(lifespan=lifespan)

class EncodeRequest(BaseModel):
    texts: list[str] | None = None
    video_links: list[str] | None = None

@app.post("/encode")
async def encode(request: EncodeRequest, processor: Processor, clip_model: Model, storage: Storage):
    texts = request.texts
    videos = request.video_links

    if not any((texts, videos)):
        raise HTTPException(status_code=400, detail="Please provide either 'texts' as list of strings or 'video_links' as list of Image URLs.")

    if all((texts, videos)):
        raise HTTPException(status_code=400, detail="Please provide either texts or video URLs, not both.")

    if texts:
        #TODO: подумать как сохранять текст
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)

    if videos:
        video_data = []
        for video_url in videos:
            video_data.append(
                process_video(video_url=video_url, processor=processor, clip_model=clip_model)
            )

        storage.add_features(video_data)
    
    return {"status": "ok"}


class SearchRequest(BaseModel):
    search: str
    return_amount: int = 5


@app.post("/find")
async def find_similar(request: SearchRequest, faiss: Faiss):
    return faiss(search_request=request.search, k=request.return_amount)


def process_video(video_url: str, processor: Processor, clip_model: Model) -> Feature:
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
    video_features /= video_features.norm(dim=-1, keepdim=True)
    return Feature(
        url=video_url,
        features=video_features.tolist()
    )


if __name__ == "__main__":
    uvicorn.run(app)
