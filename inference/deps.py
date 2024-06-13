from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from transformers import CLIPModel, CLIPProcessor

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
    yield


def _get_clip_model(request: Request) -> CLIPModel:
    return request.app.state.clip_model


def _get_clip_processor(request: Request) -> CLIPProcessor:
    return request.app.state.processor


Processor = Annotated[CLIPProcessor, Depends(_get_clip_processor)]
Model = Annotated[CLIPModel, Depends(_get_clip_model)]
