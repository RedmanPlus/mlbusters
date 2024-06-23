import os
import subprocess
import tempfile
from io import BytesIO
from dataclasses import dataclass
import requests
from scenedetect import detect, ContentDetector

@dataclass
class VideoFrame:
    video_link: str
    file: BytesIO

def create_key_frames_for_video(
        video_link: str, 
        frame_change_threshold: float = 7.5,
        num_of_thumbnails: int = 10
    ) -> list[VideoFrame]:
    frames: list[VideoFrame] = []
    video_data = BytesIO(requests.get(video_link).content)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_data.getvalue())
        video_path = tmp_file.name
    scenes = detect(video_path, ContentDetector(threshold=frame_change_threshold))

    # Gradually reduce number of key frames with a sliding window
    while len(scenes) > num_of_thumbnails:
        scenes.pop()
        scenes.pop(0)
    for i, scene in enumerate(scenes):
        scene_start, _ = scene
        frame_data = create_frame_in_ram(video_path, scene_start.get_timecode())
        if frame_data:
            frame: VideoFrame = VideoFrame(video_link=video_link, file=frame_data)
            frames.append(frame)
    os.unlink(video_path)
    # Sometimes threshold is too high to find at least 1 key frame.
    if not frames and frame_change_threshold > 2.6:
        return create_key_frames_for_video(
            video_link=video_link,
            frame_change_threshold=frame_change_threshold - 2.5,
            num_of_thumbnails=num_of_thumbnails
        )
    return frames

def create_frame_in_ram(video_path: str, timecode: str) -> BytesIO:
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-ss', str(timecode),
        '-vframes', '1',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_data, _ = process.communicate()
    return BytesIO(frame_data)

def get_audio_in_ram(video_path: str) -> BytesIO:
    command = [
        "ffmpeg",
        "-i", video_path, 
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000", 
        "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_data, _ = process.communicate()
    return BytesIO(audio_data)
