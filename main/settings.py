from environs import Env
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


env = Env()


class Settings:

    db_port: int = env.int("DB_PORT", default=8080)
    clip_model: str = env.str("CLIP_MODEL")
    clip_url: str = env.str("CLIP_URL", default="http://localhost:8787/encode")
