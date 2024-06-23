from dataclasses import dataclass, field
from io import BytesIO
import tempfile
import os
from typing import Callable

import requests
from whisper_cpp_python import Whisper

from inference.frame_video import get_audio_in_ram
from settings import Settings


@dataclass
class WhisperService:
    _service: Whisper = field(
        default_factory=lambda: Whisper(
            model_path=Settings.whisper_path,
            n_threads=4
        )
    )
    _get_audio_in_ram: Callable[[str], BytesIO] = get_audio_in_ram

    def __call__(self, link: str) -> str:
        
        video_data = BytesIO(requests.get(link).content)
        with tempfile.NamedTemporaryFile() as video:
            video.write(video_data.read())
            audio_data = self._get_audio_in_ram(video.name)

        with tempfile.NamedTemporaryFile(delete=False) as audio:
            audio.write(audio_data.read())
            audio.close()
            data = self._service.translate(
                audio.name, prompt=""
            )
        os.unlink(audio.name)
        return data["text"]
