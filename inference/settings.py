from environs import Env


env = Env()


class Settings:
    clip_model: str = env.str("CLIP_MODEL")
