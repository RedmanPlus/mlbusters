from typing import Optional
from pydantic import BaseModel

class EncodeRequest(BaseModel):
    text: Optional[str] = None
    video_url: Optional[str] = None

class SearchRequest(BaseModel):
    search: str
    return_amount: int = 20

class SuggestRequest(BaseModel):
    search_prompt: str

class Feature(BaseModel):
    url: Optional[str] = None
    features: list[float]
