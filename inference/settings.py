from environs import Env


env = Env()


class Settings:
    clip_model: str = env.str("CLIP_MODEL")
    summarization_model: str = env.str("SUMMARIZATION_MODEL")
    whisper_model: str = env.str("WHISPER_MODEL")
    translation_model: str = env.str("TRANSLATION_MODEL")
