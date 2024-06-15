from fastapi import FastAPI, Depends
from fastapi_cache.decorator import cache

from deps import Opus, Clip, Chroma, Speller, lifespan
from main.settings import Settings
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
        speller: Speller,
        params: SearchRequest = Depends()
) -> dict[str, list[str]]:
    spelled_search = speller(params.search)
    translated_search = translator(spelled_search)
    search_vector = await clip.get_text_embedding(
        EncodeRequest(
            text=translated_search
        )
    )
    return {"results": chroma.find_relevant_videos(search_feature=search_vector, top_k=params.return_amount)}
