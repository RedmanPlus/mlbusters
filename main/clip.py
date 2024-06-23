import aiohttp
from settings import Settings
from models import SearchFeature, Video, Feature

class CLIPService:
    def __init__(self) -> None:
        self.encode_clip_url = Settings.encode_clip_url
        self.search_clip_url = Settings.search_clip_url
        self.session_timeout = aiohttp.ClientTimeout(60 * 5)
    
    async def get_video_embeddings(self, request: Video) -> list[Feature]:
        async with aiohttp.ClientSession(timeout=self.session_timeout).post(
            url=f"{self.encode_clip_url}encode", 
            json=request.model_dump(mode="json")
        ) as resp:
            features = await resp.json()

        return [
            Feature(
                features=v,
                link=request.link,
                description=request.description,
                feature_type=k
            )
            for k, v in features.items()
        ]
     
    async def get_text_embedding(
            self, 
            request: SearchFeature, 
    ) -> Feature:
        async with aiohttp.ClientSession().post(
            f"{self.search_clip_url}encode-search", 
            json=request.model_dump(mode="json")
        ) as resp:
            features = await resp.json()
        return Feature(features=features['features'], feature_type="description")
