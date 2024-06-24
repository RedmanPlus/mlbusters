from environs import Env


env = Env()


class Settings:
    clip_model: str = env.str("CLIP_MODEL")
    summarization_model: str = env.str("SUMMARIZATION_MODEL")
    translation_model: str = env.str("TRANSLATION_MODEL")
