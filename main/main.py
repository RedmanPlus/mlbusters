from fastapi import FastAPI, Depends
from fastapi_cache.decorator import cache

from deps import Opus, Clip, Chroma, Speller, lifespan
from settings import Settings
from models import SearchFeature, Video, Text, SuggestRequest

app = FastAPI(lifespan=lifespan)


@app.post("/index")
async def add_video_to_index(request: Video, clip: Clip, chroma: Chroma) -> Video:
    """Добавляет новое видео в хранилище - индекс"""
    features = await clip.get_video_embeddings(request)
    if request.description is not None:
        chroma.add_text_search_suggestion(suggestion_query=request.description)

    for feature in features:
        chroma.add_feature(feature=feature)
    return request


@app.get("/search")
@cache(expire=Settings.cache_lifetime)
async def search_for_related_videos(
        clip: Clip,
        chroma: Chroma,
        translator: Opus,
        speller: Speller,
        params: Text = Depends()
) -> dict[str, list[str]]:
    """Ищет наиболее релевантные видео под запрос"""
    spelled_search = speller(params.text)
    translated_search = translator(spelled_search)
    search_vector = await clip.get_text_embedding(
        SearchFeature(
            query=translated_search
        )
    )
    return {"results": chroma.search_relevant_videos(search_feature=search_vector, top_k=params.return_amount)}


@app.get("/suggest")
@cache(expire=Settings.cache_lifetime)
async def suggest_search_prompt(
        chroma: Chroma,
        params: SuggestRequest = Depends(),
) -> dict[str, list[str]]:
    """Предлагает подсказки по текстовому запросу"""
    return {"results": chroma.get_text_search_suggestions(search_query=params.text)}
