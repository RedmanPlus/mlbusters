from fastapi import FastAPI, Depends
from fastapi_cache.decorator import cache

from deps import Opus, Clip, Chroma, lifespan
from settings import Settings
from models import Video, Text

app = FastAPI(lifespan=lifespan)

@app.post("/index")
async def add_video_to_index(request: Video, clip: Clip, chroma: Chroma):
    """Добавляет новое видео в хранилище - индекс"""
    feature = await clip.get_video_embedding(request)
    chroma.add_feature(feature=feature)
    return request.model_dump(mode="dict")

@app.get("/search")
@cache(expire=Settings.cache_lifetime)
async def search_for_related_videos(
        clip: Clip,
        chroma: Chroma,
        translator: Opus,
        params: Text = Depends()
) -> dict[str, list[str]]:
    """Ищет наиболее релевантные видео под запрос"""
    search_vector = await clip.get_text_embedding(
        Video(
            description=translator(params.text)
        )
    )
    return {"results": chroma.search_relevant_videos(search_feature=search_vector, top_k=params.return_amount)}
