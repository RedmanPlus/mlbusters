from environs import Env


env = Env()


class Settings:
    clip_model: str = env.str("CLIP_MODEL")
    whisper_path: str = env.str("WHISPER_PATH")
