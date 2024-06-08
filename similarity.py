import numpy as np
import faiss

from storage import Feature
from models import EncodeRequest


class FaissService:

    def __init__(self, clip: 'Clip', storage: 'Storage') -> list[str]:
        # Message IDs are UUIDs > incompatible with FAISS > make mapping for {FAISS index ID: Message}
        self.index_ids_to_messages = {}
        self.clip = clip

        all_features = storage.get_all_features()
        self.index = self.setup_medias_index(all_features)

    async def get_similar_video_urls(self, search_request: str, k=5):
        """ Find k most similar video urls for a given search request using FAISS. """
        try:
            features = await self.clip.get_text_embedding(EncodeRequest(text=search_request))
            input_vector = features.features.numpy().astype('float32').reshape(1, -1)
            _, indices = self.index.search(input_vector, k)
            vector_ids = indices.flatten().tolist()
            urls = [
                self.index_ids_to_messages[vector_id] 
                for vector_id in vector_ids
                if vector_id in self.index_ids_to_messages
            ]
            return urls
        except Exception as e:
            print(f"Error resolving media: {e}")
            return []

    def setup_medias_index(self, video_features: list[Feature]) -> faiss.IndexIVFFlat:
        """ Build a FAISS index from media messages' vectors. """
        d = 1024
        nlist = len(video_features)  # Number of clusters
        quantizer = faiss.IndexFlatL2(d)  # The quantizer for the IVF index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        vectors = []
        ids = []
        for i, feature in enumerate(video_features):
            vector = np.array(feature.features, dtype='float32')
            if vector.size != d:
                vector = np.resize(vector, (d,))
            vectors.append(vector)
            self.index_ids_to_messages[i] = feature.url
            ids.append(i)
        vectors = np.array(vectors).astype('float32')
        ids = np.array(ids).astype('int64')
        if not index.is_trained:
            index.train(vectors)
        index.add_with_ids(vectors, ids)
        return index

    def update_medias_index(self, video_features: list[Feature]) -> None:
        self.index_ids_to_messages = {}
        self.index = self.setup_medias_index(video_features)
