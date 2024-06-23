from contextlib import asynccontextmanager
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from transformers import CLIPModel, CLIPProcessor

from whisper import WhisperService
from settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger(__name__)
    logger.info("Setting up CLIP model...")
    app.state.clip_model = CLIPModel.from_pretrained(
        Settings.clip_model,
        cache_dir="./model_cache"
    )
    logger.info("Setting up CLIP processor...")
    app.state.processor = CLIPProcessor.from_pretrained(
        Settings.clip_model,
        cache_dir="./model_cache"
    )
    logger.info("Setting up Whisper service...")
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
