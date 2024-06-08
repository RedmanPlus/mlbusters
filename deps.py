import os
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, Request

from similarity import FaissService
from storage import FeatureStorage
from clip import CLIPService
from dotenv import load_dotenv
load_dotenv()


def get_clip_service() -> CLIPService:
    return CLIPService(
        url=os.getenv("CLIP_URL", "http://localhost:8000/encode")
    )

def get_feature_storage() -> FeatureStorage:
    return FeatureStorage(
        conn_addr=os.getenv(
            "MONGO_URL", "mongodb://localhost:27017"
        ), 
        db_name=os.getenv(
            "MONGO_DB", "features"
        ),
        collection_name=os.getenv(
            "MONGO_COLLECTION", "features"
        )
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = get_feature_storage()
    app.state.clip = get_clip_service()
    app.state.faiss = FaissService(
        clip=app.state.clip,
        storage=app.state.db
    )
    yield
    app.state.db.deinit()


def _get_storage(request: Request) -> FeatureStorage:
    return request.app.state.db


def _get_faiss(request: Request) -> FaissService:
    return request.app.state.faiss

def _get_clip(request: Request) -> CLIPService:
    return request.app.state.clip

Clip = Annotated[CLIPService, Depends(_get_clip)]
Storage = Annotated[FeatureStorage, Depends(_get_storage)]
Faiss = Annotated[FaissService, Depends(_get_faiss)]
