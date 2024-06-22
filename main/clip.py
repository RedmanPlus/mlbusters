import aiohttp
from models import SearchFeature, Video, Feature

class CLIPService:
    def __init__(self, url: str) -> None:
        self.clip_url = url
    
    async def get_video_embeddings(self, request: Video) -> list[Feature]:
        async with aiohttp.ClientSession().post(
            url=f"{self.clip_url}/encode", 
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
            f"{self.clip_url}/encode-search", 
            json=request.model_dump(mode="json")
        ) as resp:
            features = await resp.json()
        return Feature(features=features['features'], feature_type="description")
