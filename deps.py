from typing import Annotated
from fastapi import Depends
from transformers import CLIPModel, CLIPProcessor

from storage import FeatureStorage


def get_clip_processor() -> CLIPProcessor:
    ...


def get_clip_model() -> CLIPModel:
    ...


def get_feature_storage():
    storage = FeatureStorage(conn_addr="", db_name="features", collection_name="features")
    yield storage
    storage.deinit()


Processor = Annotated[CLIPProcessor, Depends(get_clip_processor)]
Model = Annotated[CLIPModel, Depends(get_clip_model)]
Storage = Annotated[FeatureStorage, Depends(get_feature_storage)]
