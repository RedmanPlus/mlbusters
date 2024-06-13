from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, Request
from chroma import ChromaStorage
from clip import CLIPService
from search_translate import T5TranslatorModel
from settings import Settings


def get_clip_service() -> CLIPService:
    return CLIPService(
        url=Settings.clip_url
    )

def get_chroma_storage() -> ChromaStorage:
    return ChromaStorage()


def get_t5_translator() -> T5TranslatorModel:
    return T5TranslatorModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clip = get_clip_service()
    app.state.chroma = get_chroma_storage()
    app.state.t5 = get_t5_translator()
    yield

def _get_clip(request: Request) -> CLIPService:
    return request.app.state.clip

def _get_chroma(request: Request) -> ChromaStorage:
    return request.app.state.chroma

def _get_t5(request: Request) -> T5TranslatorModel:
    return request.app.state.t5

Clip = Annotated[CLIPService, Depends(_get_clip)]
Chroma = Annotated[ChromaStorage, Depends(_get_chroma)]
T5 = Annotated[T5TranslatorModel, Depends(_get_t5)]
