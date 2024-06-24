from dataclasses import dataclass, field
from io import BytesIO
import logging
from typing import Callable

import requests
from transformers import Pipeline, pipeline
from faster_whisper import WhisperModel

from frame_video import get_audio_in_ram
from translator import OpusTranslatorModel
from settings import Settings


model = WhisperModel


@dataclass
class WhisperService:
    _whisper: WhisperModel = field(
        default_factory=lambda: WhisperModel(
            Settings.whisper_model,
            device="cpu",
            compute_type="float16",
            cpu_threads=8,
            num_workers=4,
        )
    )
    _translator: OpusTranslatorModel = field(
        default_factory=OpusTranslatorModel
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
        self._logger.info("Converting video file to transcript")
        video_data = BytesIO(requests.get(link).content)
        segments, info = self._whisper.transcribe(
            video_data,
            language="ru",
            beam_size=5
        )
        if info.language_probability < 0.5:
            self._logger.info(
                "Cannot properly identify speech, probability=%s, returning empty string",
                info.language_probability
            )
            return ""
        self._logger.info("summarizing transcript into 77 CLIP tokens")
        full_translation = ""
        for segment in segments:
            if segment.no_speech_prob > 0.5:
                continue
            translated_segment = self._translator(segment.text)
            full_translation += " " + translated_segment
        summary = self._summary_pipeline(full_translation, max_length=77)
        result: str = summary[0]["summary_text"]  # type: ignore
        self._logger.info("Processed video file into text description: %s, total length: %s", result, len(result))
        return result
