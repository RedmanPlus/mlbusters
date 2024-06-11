import os
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, Request
from chroma import ChromaStorage
from clip import CLIPService


def get_clip_service() -> CLIPService:
    return CLIPService(
        url=os.getenv("CLIP_URL", "http://localhost:8000/encode")
    )

def get_chroma_storage() -> ChromaStorage:
    return ChromaStorage()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clip = get_clip_service()
    app.state.chroma = get_chroma_storage()
    yield

def _get_clip(request: Request) -> CLIPService:
    return request.app.state.clip

def _get_chroma(request: Request) -> ChromaStorage:
    return request.app.state.chroma

Clip = Annotated[CLIPService, Depends(_get_clip)]
Chroma = Annotated[ChromaStorage, Depends(_get_chroma)]
