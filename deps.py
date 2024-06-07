from contextlib import contextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, Request
from transformers import CLIPModel, CLIPProcessor

from similarity import FaissService
from storage import FeatureStorage


def get_clip_processor() -> CLIPProcessor:
    ...


def get_clip_model() -> CLIPModel:
    ...


def get_feature_storage() -> FeatureStorage:
    return FeatureStorage(conn_addr="", db_name="features", collection_name="features")

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
