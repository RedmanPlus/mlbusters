from dataclasses import dataclass, field
from io import BytesIO
import tempfile

import requests
from whisper_cpp_python import Whisper

from settings import Settings


@dataclass
class WhisperService:
    _service: Whisper = field(default_factory=lambda: Whisper(model_path=Settings.whisper_path))

    def __call__(self, link: str) -> str:
        
        video_data = BytesIO(requests.get(link).content)
        with tempfile.NamedTemporaryFile(delete_on_close=False) as tp:
            tp.write(video_data.read())
            tp.close()
            data = self._service.transcribe(open(tp.name))
        return data["text"]
