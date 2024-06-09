from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, Request
from transformers import CLIPModel, CLIPProcessor

from db import FeatureStorage
from dotenv import load_dotenv

from settings import settings


load_dotenv()


def get_clip_processor() -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(settings.clip_id)


def get_clip_model() -> CLIPModel:
    return CLIPModel.from_pretrained(settings.clip_id)


def get_feature_storage() -> FeatureStorage:
    return FeatureStorage(
        conn_addr=settings.mongodb_conn,
        db_name="features",
        collection_name="features"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = get_feature_storage()
    app.state.processor = get_clip_processor()
    app.state.model = get_clip_model()
    yield
    app.state.db.deinit()


def _get_processor(request: Request) -> CLIPProcessor:
    return request.app.state.processor


def _get_model(request: Request) -> CLIPModel:
    return request.app.state.model


def _get_storage(request: Request) -> FeatureStorage:
    return request.app.state.db


Processor = Annotated[CLIPProcessor, Depends(_get_processor)]
Model = Annotated[CLIPModel, Depends(_get_model)]
Storage = Annotated[FeatureStorage, Depends(_get_storage)]
