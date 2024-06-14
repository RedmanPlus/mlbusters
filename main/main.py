from fastapi import FastAPI, Depends
from fastapi_cache.decorator import cache

from deps import Opus, Clip, Chroma, lifespan
from settings import Settings
from models import EncodeRequest, SearchRequest

app = FastAPI(lifespan=lifespan)

@app.post("/encode")
async def encode(request: EncodeRequest, clip: Clip, chroma: Chroma):
    feature = await clip.get_video_embedding(request)
    chroma.add_feature(feature=feature)
    return {"status": "ok", "features": feature.features}

@app.get("/find")
@cache(expire=Settings.cache_lifetime)
async def find_similar(
        clip: Clip,
        chroma: Chroma,
        translator: Opus,
        params: SearchRequest = Depends()
) -> dict[str, list[str]]:
    search_vector = await clip.get_text_embedding(
        EncodeRequest(
            text=translator(params.search)
        )
    )
    return {"results": chroma.find_relevant_videos(search_feature=search_vector, top_k=params.return_amount)}
