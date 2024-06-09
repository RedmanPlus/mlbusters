from dataclasses import dataclass
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

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
        self._add_feature_search_index()

    def deinit(self) -> None:
        self._client.close()

    def add_features(self, features: list[Feature]) -> None:
        self._collection.insert_many(
            [feature.__dict__ for feature in features]
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
        print(feature_vector)
        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'vector_index', 
                    'path': 'features', 
                    'queryVector': feature_vector,
                    'numCandidates': 150, 
                    'limit': top_k,
                }
            }, {
                '$project': {
                    '_id': 0,
                    'url': 1,
                    'features': 1,
                }
            }
        ]

        results = self._collection.aggregate(pipeline)
        return [Feature(**result) for result in results]

    def _add_feature_search_index(self):
        definition = {
            "fields": [{
                "type": "vector",
                "path": "features",
                "numDimensions": 1024,
                "similarity": "cosine",
            }]
        }
        index = SearchIndexModel(definition, name="vector_index", type="vectorSearch")
        self._collection.create_search_index(model=index)
