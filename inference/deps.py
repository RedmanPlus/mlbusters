from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from transformers import CLIPModel, CLIPProcessor
from frame_video import FrameExtractor
from settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clip_model = CLIPModel.from_pretrained(
        Settings.clip_model,
        cache_dir="./model_cache"
    )
    app.state.processor = CLIPProcessor.from_pretrained(
        Settings.clip_model,
        cache_dir="./model_cache"
    )
    app.state.frame_extractor = FrameExtractor()
    yield


def _get_clip_model(request: Request) -> CLIPModel:
    return request.app.state.clip_model


def _get_clip_processor(request: Request) -> CLIPProcessor:
    return request.app.state.processor

def _get_frame_extractor(request: Request) -> FrameExtractor:
    return request.app.state.frame_extractor


Processor = Annotated[CLIPProcessor, Depends(_get_clip_processor)]
Model = Annotated[CLIPModel, Depends(_get_clip_model)]
KeyFrameExtractor = Annotated[FrameExtractor, Depends(_get_frame_extractor)]
