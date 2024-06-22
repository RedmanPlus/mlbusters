from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from transformers import CLIPModel, CLIPProcessor

from inference.whisper import WhisperService
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
    app.state.whisper_model = WhisperService()
    yield


def _get_clip_model(request: Request) -> CLIPModel:
    return request.app.state.clip_model


def _get_clip_processor(request: Request) -> CLIPProcessor:
    return request.app.state.processor


def _get_whisper(request: Request) -> WhisperService:
    return request.app.state.whisper_model


Processor = Annotated[CLIPProcessor, Depends(_get_clip_processor)]
Model = Annotated[CLIPModel, Depends(_get_clip_model)]
Whisper = Annotated[WhisperService, Depends(_get_whisper)]
