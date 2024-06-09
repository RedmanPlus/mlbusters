from fastapi import FastAPI

import uvicorn

from deps import Model, Processor, Storage, lifespan
from models import EncodeRequest, SearchRequest
from processors.encoding import process_text, process_video


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
async def find_similar(
        request: SearchRequest,
        processor: Processor,
        clip_model: Model,
        store: Storage
) -> dict[str, list[str]]:
    search_vector = process_text(
        request.search,
        processor=processor,
        clip_model=clip_model
    )
    results = store.get_relevant_features(
        feature_vector=search_vector[0],
        top_k=request.return_amount
    )
    return {"results": [result.url for result in results]}


if __name__ == "__main__":
    uvicorn.run(app)
