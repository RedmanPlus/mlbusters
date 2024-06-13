from environs import Env


env = Env()


class Settings:

    db_port: int = env.int("DB_PORT", default=8080)
    clip_url: str = env.str("CLIP_URL", default="http://localhost:8787/encode")
