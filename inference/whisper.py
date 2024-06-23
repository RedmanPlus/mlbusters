from dataclasses import dataclass, field
from io import BytesIO
import logging
import tempfile
import os
from typing import Callable

import requests
from transformers import Pipeline, pipeline
from whisper_cpp_python import Whisper

from frame_video import get_audio_in_ram
from settings import Settings


@dataclass
class WhisperService:
    _service: Whisper = field(
        default_factory=lambda: Whisper(
            model_path=Settings.whisper_path,
            n_threads=4
        )
    )
    _summary_pipeline: Pipeline = field(
        default_factory=lambda: pipeline(
            "summarization",
            model=Settings.summarization_model
        )
    )
    _logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__)
    )
    _get_audio_in_ram: Callable[[str], BytesIO] = get_audio_in_ram

    def __call__(self, link: str) -> str:
        self._logger.info("Converting video file to WAV")
        video_data = BytesIO(requests.get(link).content)
        with tempfile.NamedTemporaryFile(delete=False) as video:
            video.write(video_data.read())
            video.close()
            audio_data = self._get_audio_in_ram(video.name)
        os.unlink(video.name)
        self._logger.info("Processing WAV file by whisper")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio:
            audio.write(audio_data.read())
            audio.close()
            data = self._service.translate(
                audio.name, prompt=""
            )
        os.unlink(audio.name)
        self._logger.info("summarizing transcript into 77 CLIP tokens")
        text = data["text"]
        summary = self._summary_pipeline(text, max_length=77)
        result: str = summary[0]["summary_text"]  # type: ignore
        self._logger.info("Processed video file into text description: %s, total length: %s", result, len(result))
        return result
