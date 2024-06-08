from fastapi import FastAPI

import uvicorn

from deps import Faiss, Model, Processor, Storage, lifespan
from models import EncodeRequest, SearchRequest
from processors.encoding import process_video


app = FastAPI(lifespan=lifespan)

@app.post("/encode")
async def encode(
        request: EncodeRequest,
        processor: Processor,
        clip_model: Model,
        storage: Storage
) -> dict[str, str]:
    video_data = []
    for encodable in request.videos:
        video_data.append(
            process_video(
                video_url=encodable.url,
                processor=processor,
                clip_model=clip_model
            )
        )
    storage.add_features(video_data)
    
    return {"status": "ok"}


@app.post("/find")
async def find_similar(request: SearchRequest, faiss: Faiss):
    return faiss(search_request=request.search, k=request.return_amount)


if __name__ == "__main__":
    uvicorn.run(app)
