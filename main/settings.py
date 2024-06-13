from environs import Env


env = Env()


class Settings:
    db_host: str = env.str("DB_HOST", default="chroma_db")
    db_port: int = env.int("DB_PORT", default=8080)
    clip_url: str = env.str("CLIP_URL", default="http://inference:8040/encode")
    memcached_host: str = env.str("MEMCACHED_HOST", default="request_cache")
    cache_lifetime: int = env.str("CACHE_LIFETIME", default=3600)
