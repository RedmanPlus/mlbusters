import os
import subprocess
import tempfile
from io import BytesIO

from pydantic import BaseModel

import requests
from scenedetect import detect, ContentDetector


class MessageThumbnail(BaseModel):
    message: str
    file: BytesIO


def create_thumbnails_for_video_message(
        message: str, 
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 10
    ) -> list[MessageThumbnail]:
    thumbnails = []
    video_data = BytesIO(requests.get(message).content)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name

    # Setup Scene Detection
    scenes = detect(video_path, ContentDetector(threshold=frame_change_threshold))

    # Gradully reduce number of key frames with a sliding window
    while len(scenes) > num_of_thumbnails:
        scenes.pop()
        scenes.pop(0)
    for i, scene in enumerate(scenes):
        scene_start, _ = scene
        output_path = f'key_frame_{i}.jpg'
        save_frame(video_path, scene_start.get_timecode(), output_path)
        with open(output_path, 'rb') as frame_data:
            thumbnail: MessageThumbnail = MessageThumbnail(message=message, file=BytesIO(frame_data.read()))
            thumbnails.append(thumbnail)
        os.remove(output_path)
    os.unlink(video_path)
    return thumbnails

def save_frame(video_path: str, timecode, output_path: str):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-ss', str(timecode), '-vframes', '1', output_path])
