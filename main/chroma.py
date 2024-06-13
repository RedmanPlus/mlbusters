import chromadb
from models import Feature
from settings import Settings

class ChromaStorage:
    def __init__(self, collection_name: str = 'features') -> None:
        self.client = chromadb.HttpClient(port=Settings.db_port)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_feature(self, feature: Feature) -> None:
        self.collection.add(
            ids=[feature.url],
            embeddings=[feature.features],
        )
    
    def find_relevant_videos(self, search_feature: Feature, top_k: int = 100) -> list[str]:
        results = self.collection.query(
            query_embeddings=search_feature.features,
            n_results=top_k
        )
        return results['ids'][0]
