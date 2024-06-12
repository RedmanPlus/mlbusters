from fastapi import FastAPI, Depends
import uvicorn

from deps import T5, Clip, Chroma, lifespan
from models import EncodeRequest, SearchRequest

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(lifespan=lifespan)

@app.post("/encode")
async def encode(request: EncodeRequest, clip: Clip, chroma: Chroma):
    feature = await clip.get_video_embedding(request)
    chroma.add_feature(feature=feature)
    return {"status": "ok", "features": feature.features}

@app.get("/find")
async def find_similar(
        clip: Clip,
        chroma: Chroma,
        translator: T5,
        params: SearchRequest = Depends()
) -> dict[str, list[str]]:
    search_vector = await clip.get_text_embedding(
        EncodeRequest(
            text=translator(params.search)
        )
    )
    return {"results": chroma.find_relevant_videos(search_feature=search_vector, top_k=params.return_amount)}

if __name__ == "__main__":
    uvicorn.run(app)
