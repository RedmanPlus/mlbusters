from io import BytesIO
from pydantic import BaseModel


class Encodable(BaseModel):
    url: str
    description: str | None = None


class EncodeRequest(BaseModel):
    videos: list[Encodable]


class SearchRequest(BaseModel):
    search: str
    return_amount: int = 5


class Feature(BaseModel):
    url: str
    features: list[float]


class VideoFrame(BaseModel):
    video_url: str
    file: BytesIO
