from dataclasses import dataclass

from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


class Feature(BaseModel):
    url: str
    features: list[float]


@dataclass
class FeatureStorage:

    _client: MongoClient
    _db: Database
    _collection: Collection

    def __init__(self, conn_addr: str, db_name: str, collection_name: str) -> None:
        self._client = MongoClient(conn_addr)
        self._db = getattr(self._client, db_name)
        self._collection = getattr(self._db, collection_name)

    def deinit(self) -> None:
        ...

    def add_features(self, features: list[Feature]) -> None:
        self._collection.insert_many(
            [feature.model_dump(mode="json") for feature in features]
        )
