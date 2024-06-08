from fastapi import FastAPI, Depends
import uvicorn

from deps import Faiss, Clip, Storage, lifespan
from models import EncodeRequest, SearchRequest

app = FastAPI(lifespan=lifespan)

@app.post("/encode")
async def encode(request: EncodeRequest, clip: Clip, storage: Storage):
    feature = await clip.get_video_embedding(request)
    storage.add_features(features=[feature])
    return {"status": "ok", "features": feature.features}

@app.get("/find")
async def find_similar(faiss: Faiss, params: SearchRequest = Depends()):
    return await faiss.get_similar_video_urls(search_request=params.search, k=params.return_amount)

if __name__ == "__main__":
    uvicorn.run(app)
