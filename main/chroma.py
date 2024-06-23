from uuid import uuid4
import chromadb
from chromadb.server import Settings as ChromaSettings
from models import Feature
from settings import Settings

class ChromaStorage:
    def __init__(
            self,
            collection_name: str = 'features',
            desc_collection_name: str = "descriptions",
    ) -> None:
        self.client = chromadb.HttpClient(
            host=Settings.db_host,
            port=Settings.db_port,
            settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.desc_collection = self.client.get_or_create_collection(
            name=desc_collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_feature(self, feature: Feature) -> None:
        self.collection.add(
            ids=[str(uuid4())],
            embeddings=[feature.features],
            uris=[feature.link],
            metadatas=[{"feature_type": feature.feature_type}]
        )
    
    def search_relevant_videos(self, search_feature: Feature, top_k: int = 100) -> list[str]:
        results = self.collection.query(
            query_embeddings=search_feature.features,
            n_results=top_k
        )
        return results['uris'][0]

    def add_text_search_suggestion(self, suggestion_query: str) -> None:
        subsearches = suggestion_query.split()
        self.desc_collection.add(
            documents=[suggestion_query] + subsearches,
            ids=[str(uuid4()) for _ in [suggestion_query] + subsearches]
        )

    def get_text_search_suggestions(self, search_query: str, top_k: int = 20) -> list[str]:
        results = self.desc_collection.query(
            query_texts=[search_query],
            n_results=top_k,
        )
        return results["documents"][0]
