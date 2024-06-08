from dataclasses import dataclass
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from models import Feature

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
        self._client.close()

    def add_features(self, features: list[Feature]) -> None:
        self._collection.insert_many(
            [feature.model_dump(mode="json") for feature in features]
        )

    def get_all_features(self) -> list[Feature]:
        return [
            Feature(
                url=data["url"],
                features=data["features"],
            )
            for data in self._collection.find({})
        ]

    def get_relevant_features(self, feature_vector: list[float], top_k: int = 5) -> list[Feature]:
        pipeline = [
            {
                "$search": {
                    "knnBeta": {
                        "vector": feature_vector,
                        "path": "features",
                        "k": top_k
                    }
                }
            },
            {
                "$limit": top_k
            }
        ]

        results = self._collection.aggregate(pipeline)
        return [Feature(**result) for result in results]
