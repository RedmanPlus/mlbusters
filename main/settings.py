from environs import Env


env = Env()


class Settings:
    db_host: str = env.str("DB_HOST", default="chroma_db")
    db_port: int = env.int("DB_PORT", default=8080)
    encode_clip_url: str = env.str("ENCODE_CLIP_URL", default="http://encode:8040/")
    search_clip_url: str = env.str("SEARCH_CLIP_URL", default="http://search:8050/")
    memcached_host: str = env.str("MEMCACHED_HOST", default="request_cache")
    cache_lifetime: int = env.int("CACHE_LIFETIME", default=3600)
