from pydantic import BaseModel


class EncodeRequest(BaseModel):
    link: str
    description: str | None = None


class EncodeSearchRequest(BaseModel):
    query: str
