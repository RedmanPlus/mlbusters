import aiohttp
from models import EncodeRequest, Feature
from googletrans import Translator

class CLIPService:
    def __init__(self, url: str) -> None:
        self.clip_url = url
    
    async def get_video_embedding(self, request: EncodeRequest) -> Feature:
        async with aiohttp.ClientSession().post(
            url=self.clip_url, 
            json=request.model_dump(mode="json")
        ) as resp:
            features = await resp.json()

        return Feature(features=features['features'], url=request.video_url)
     
    async def get_text_embedding(self, request: EncodeRequest) -> Feature:
        request.text = Translator().translate(text=request.text).text
        async with aiohttp.ClientSession().post(
            self.clip_url, 
            json=request.model_dump(mode="json")
        ) as resp:
            features = await resp.json()
            print(f"{features=}")
        return Feature(features=features['features'][0])
