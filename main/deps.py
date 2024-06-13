from contextlib import asynccontextmanager
from typing import Annotated

from aiomcache import Client
from fastapi import Depends, FastAPI, Request
from fastapi_cache import FastAPICache
from fastapi_cache.backends.memcached import MemcachedBackend
from chroma import ChromaStorage
from clip import CLIPService

from search_translate import OpusTranslatorModel
from settings import Settings


def get_clip_service() -> CLIPService:
    return CLIPService(
        url=Settings.clip_url
    )

def get_chroma_storage() -> ChromaStorage:
    return ChromaStorage()


def get_opus_translator() -> OpusTranslatorModel:
    return OpusTranslatorModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    memcached = Client(host=Settings.memcached_host)
    FastAPICache.init(backend=MemcachedBackend(memcached), prefix="search-cache")
    app.state.clip = get_clip_service()
    app.state.chroma = get_chroma_storage()
    app.state.opus = get_opus_translator()
    yield

def _get_clip(request: Request) -> CLIPService:
    return request.app.state.clip

def _get_chroma(request: Request) -> ChromaStorage:
    return request.app.state.chroma

def _get_opus(request: Request) -> OpusTranslatorModel:
    return request.app.state.opus

Clip = Annotated[CLIPService, Depends(_get_clip)]
Chroma = Annotated[ChromaStorage, Depends(_get_chroma)]
Opus = Annotated[OpusTranslatorModel, Depends(_get_opus)]
