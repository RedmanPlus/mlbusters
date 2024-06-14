from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from transformers import AutoProcessor, AutoModel

from settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clip_model = AutoModel.from_pretrained(
        Settings.clip_model,
        cache_dir="./model_cache"
    )
    app.state.processor = AutoProcessor.from_pretrained(
        Settings.clip_model,
        cache_dir="./model_cache"
    )
    yield


def _get_clip_model(request: Request) -> AutoModel:
    return request.app.state.clip_model


def _get_clip_processor(request: Request) -> AutoProcessor:
    return request.app.state.processor


Processor = Annotated[AutoProcessor, Depends(_get_clip_processor)]
Model = Annotated[AutoModel, Depends(_get_clip_model)]
