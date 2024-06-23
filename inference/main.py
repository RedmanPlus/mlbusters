import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from deps import Model, Processor, lifespan
from clip import CLIP
from models import EncodeRequest, EncodeSearchRequest

app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__name__)

@app.get("/")
async def root():
    return JSONResponse(content={"ok": True})

@app.post("/encode")
async def encode(
        request: EncodeRequest,
        processor: Processor,
        model: Model,
):
    logger.info("Initializing CLIP module...")
    clip = CLIP(processor=processor, model=model, logger=logger)
    logger.info("CLIP module successfully initialized")

    video_features = clip(request.link, encode_type="video")
    return {
        "video": video_features,
    }

@app.post("/encode-search")
async def encode_search(
        request: EncodeSearchRequest, processor: Processor, model: Model
):
    logger.info("Initializing CLIP module...")
    clip = CLIP(processor=processor, model=model, logger=logger)
    logger.info("CLIP module successfully initialized")

    features = clip(request.query, encode_type="text")

    return {"features": features}
