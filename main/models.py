from typing import Literal, Optional
from pydantic import BaseModel

class Video(BaseModel):
    """Represents a Link to Video with text description 
    to be vectorized and added to index"""
    link: str
    description: str | None = None

class Text(BaseModel):
    """Represents a text query to search related videos"""
    text: str
    return_amount: int = 50

class SearchFeature(BaseModel):
    query: str

class SuggestRequest(BaseModel):
    """Represents a text query to suggest related completions"""
    text: str

class Feature(BaseModel):
    """Represents an Embedding of a video"""
    link: Optional[str] = None
    description: Optional[str] = None
    features: list[float]
    feature_type: Literal["description"] | Literal["video"] | Literal["audio"]
