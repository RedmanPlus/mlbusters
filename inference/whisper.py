from dataclasses import dataclass, field
from io import BytesIO

import requests
from whisper_cpp_python import Whisper

from settings import Settings


@dataclass
class WhisperService:
    _service: Whisper = field(default_factory=lambda: Whisper(model_path=Settings.whisper_path))

    def __call__(self, link: str) -> str:
        
        video_data = BytesIO(requests.get(link).content)
        data = self._service.transcribe(video_data)
        return data["text"]
