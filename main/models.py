from typing import Optional
from pydantic import BaseModel

class Video(BaseModel):
    """Represents a Link to Video with text description 
    to be vectorized and added to index"""
    description: Optional[str] = None
    link: Optional[str] = None

class Text(BaseModel):
    """Represents a text query to search related videos"""
    text: str
    return_amount: int = 50

class Feature(BaseModel):
    """Represents an Embedding of a video"""
    link: Optional[str] = None
    description: Optional[str] = None
    features: list[float]
