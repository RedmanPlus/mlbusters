import os
import subprocess
import tempfile
from io import BytesIO

import requests
from scenedetect import detect, ContentDetector

from models import VideoFrame


def create_thumbnails_for_video_message(
        video_url: str, 
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 10
    ) -> list[VideoFrame]:

    frames: list[VideoFrame] = []
    video_data = BytesIO(requests.get(video_url).content)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name

    # Setup Scene Detection
    scenes = detect(video_path, ContentDetector(threshold=frame_change_threshold))

    # Gradually reduce number of key frames with a sliding window
    while len(scenes) > num_of_thumbnails:
        scenes.pop()
        scenes.pop(0)
    for i, scene in enumerate(scenes):
        scene_start, _ = scene
        output_path = f'key_frame_{i}.jpg'
        save_frame(video_path, scene_start.get_timecode(), output_path)
        with open(output_path, 'rb') as frame_data:
            frame: VideoFrame = VideoFrame(video_url=video_url, file=BytesIO(frame_data.read()))
            frames.append(frame)
        os.remove(output_path)
    os.unlink(video_path)
    
    # Sometimes threshold is too high to find at least 1 key frame.
    if not frames and frame_change_threshold > 2.6:
        return create_thumbnails_for_video_message(
            video_url=video_url,
            frame_change_threshold=frame_change_threshold - 2.5,
            num_of_thumbnails=num_of_thumbnails
        )
    return frames

def save_frame(video_path: str, timecode, output_path: str):
    subprocess.call(['ffmpeg', '-y', '-i', video_path, '-ss', str(timecode), '-vframes', '1', output_path])
