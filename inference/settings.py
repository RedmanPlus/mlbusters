from environs import Env


env = Env()


class Settings:
    clip_model: str = env.str("CLIP_MODEL", default="laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    translation_model: str = env.str("TRANSLATION_MODEL", default="Helsinki-NLP/opus-mt-ru-en")
