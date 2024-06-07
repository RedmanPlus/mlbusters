import os
from contextlib import contextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, Request
from transformers import CLIPModel, CLIPProcessor

from similarity import FaissService
from storage import FeatureStorage
from dotenv import load_dotenv
load_dotenv()

def get_clip_processor() -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(
        os.getenv("CLIP_MODEL", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    )

def get_clip_model() -> CLIPModel:
    return CLIPModel.from_pretrained(
        os.getenv("CLIP_MODEL", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
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

@contextmanager
def lifespan(app: FastAPI):
    app.state.db = get_feature_storage()
    app.state.processor = get_clip_processor()
    app.state.model = get_clip_model()
    app.state.faiss = FaissService(
        processor=app.state.processor,
        clip_model=app.state.model,
        storage=app.state.db
    )
    yield
    app.state.db.deinit()


def _get_processor(request: Request) -> CLIPProcessor:
    return request.app.state.processor


def _get_model(request: Request) -> CLIPModel:
    return request.app.state.model


def _get_storage(request: Request) -> FeatureStorage:
    return request.app.state.db


def _get_faiss(request: Request) -> FaissService:
    return request.app.state.faiss


Processor = Annotated[CLIPProcessor, Depends(_get_processor)]
Model = Annotated[CLIPModel, Depends(_get_model)]
Storage = Annotated[FeatureStorage, Depends(_get_storage)]
Faiss = Annotated[FaissService, Depends(_get_faiss)]
