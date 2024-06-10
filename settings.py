import os
from pydantic import BaseModel, MongoDsn


class Settings(BaseModel):
    mongodb_conn: MongoDsn = os.getenv("MONGO_URL", "mongodb://user:pass@localhost:27019/?directConnection=true")
    clip_id: str = os.getenv('CLIP_ID', 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K')

settings = Settings()
