from typing import Literal
import subprocess
from io import BytesIO
import tempfile
from dataclasses import dataclass
import numpy as np
import cv2
import requests
from scipy.signal import argrelextrema

@dataclass
class FrameRef:
    frame: np.ndarray
    sum_abs_diff: int

@dataclass
class VideoFrame:
    video_link: str
    file: BytesIO

class FrameExtractor:
    """Class for extraction of key frames from video,
    based on sum of absolute differences in LUV colorspace from given video
    """
    def __init__(
            self, 
            use_local_maxima=True,    # Setting local maxima criteria

            len_window=10,            # Length of sliding window taking difference

            max_frames_in_chunk=2500, # Chunk size of Images to be processed at a time in memory

            window_type: Literal[     # Type of smoothening window
                "flat", "hanning", "hamming", 
                "bartlett", "blackman"
            ] = "hanning",                  
        ):
        self.USE_LOCAL_MAXIMA = use_local_maxima
        self.len_window = len_window
        self.max_frames_in_chunk = max_frames_in_chunk
        self.window_type = window_type

    def extract_key_frames(self, video_link: str) -> list[VideoFrame]:
        """Given an input video link, returns a list of all candidate key-frames.
        
        :param video_link: Input video link
        :type video_link: str
        :return: List of VideoFrame objects containing the key-frames
        :rtype: list[VideoFrame]
        """
        video_data = BytesIO(requests.get(video_link).content)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data.getvalue())
            videopath = tmp_file.name
    
        candidate_frames = []
        frame_extractor_from_video_generator = self.__extract_all_frames_from_video(videopath=videopath)
    
        for frames, frame_diffs in frame_extractor_from_video_generator:
            if self.USE_LOCAL_MAXIMA:
                extracted_candidate_key_frames = self.__get_frames_in_local_maxima(frames, frame_diffs)
                for frame in extracted_candidate_key_frames:
                    success, encoded_image = cv2.imencode('.jpeg', frame)
                    if success:
                        image_bytes = BytesIO(encoded_image.tobytes())
                        candidate_frames.append(VideoFrame(video_link=video_link, file=image_bytes)) 
        return candidate_frames

    def __calculate_frame_difference(self, frame, curr_frame, prev_frame):
        """Function to calculate the difference between current frame and previous frame
        :param frame: frame from the video
        :type frame: numpy array
        :param curr_frame: current frame from the video in LUV format
        :type curr_frame: numpy array
        :param prev_frame: previous frame from the video in LUV format
        :type prev_frame: numpy array
        :return: difference count and frame if None is empty or undefined else None
        :rtype: tuple
        """

        if curr_frame is not None and prev_frame is not None:
            # Calculating difference between current and previous frame
            diff = cv2.absdiff(curr_frame, prev_frame)
            count = np.sum(diff)
            frame = FrameRef(frame, count)

            return count, frame
        return None

    def __process_frame(self, frame, prev_frame, frame_diffs, frames):
        """Function to calculate the difference between current frame and previous frame
        :param frame: frame from the video
        :type frame: numpy array
        :param prev_frame: previous frame from the video in LUV format
        :type prev_frame: numpy array
        :param frame_diffs: list of frame differences
        :type frame_diffs: list of int
        :param frames: list of frames
        :type frames: list of numpy array
        :return: previous frame and current frame
        """
        # For LUV images
        # luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        # curr_frame = luv

        # For GrayScale images
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        curr_frame = grey

        # Calculating the frame difference for previous and current frame
        frame_diff = self.__calculate_frame_difference(frame, curr_frame, prev_frame)
        
        if frame_diff is not None:
            count, frame = frame_diff
            frame_diffs.append(count)
            frames.append(frame)
        prev_frame = curr_frame

        return prev_frame, curr_frame

    def __extract_all_frames_from_video(self, videopath: str):
        """Generator function for extracting frames from a input video which are sufficiently different from each other,
        and return result back as list of opencv images in memory
        :param videopath: inputvideo path
        :return: Generator with extracted frames in max_process_frames chunks and difference between frames
        :rtype: generator object with content of type [numpy.ndarray, numpy.ndarray]
        """
        cap = cv2.VideoCapture(str(videopath))
        ret, frame = cap.read()
        i = 1
        chunk_no = 0
        while ret:
            curr_frame = None
            prev_frame = None

            frame_diffs = []
            frames = []
            for _ in range(0, self.max_frames_in_chunk):
                if ret:
                    # Calling process frame function to calculate the frame difference and adding the difference
                    # in **frame_diffs** list and frame to **frames** list
                    prev_frame, curr_frame = self.__process_frame(frame, prev_frame, frame_diffs, frames)
                    i += 1
                    ret, frame = cap.read()
                else:
                    cap.release()
                    break
            chunk_no = chunk_no + 1
            yield frames, frame_diffs
        cap.release()

    def __get_frames_in_local_maxima(self, frames, frame_diffs):
        """ Internal function for getting local maxima of key frames
        This functions Returns one single image with strongest change from its vicinity of frames
        ( vicinity defined using window length )
        :param object: base class inheritance
        :type object: class:`Object`
        :param frames: list of frames to do local maxima on
        :type frames: `list of images`
        :param frame_diffs: list of frame difference values
        :type frame_diffs: `list of images`
        """
        extracted_key_frames = []
        diff_array = np.array(frame_diffs)
        # Normalizing the frame differences based on windows parameters
        sm_diff_array = self.__smooth(diff_array)

        # sm_diff_array = diff_array
        # Get the indexes of those frames which have maximum differences
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]

        for frame_index in frame_indexes:
            extracted_key_frames.append(frames[frame_index - 1].frame)
        return extracted_key_frames

    def __smooth(self, frame_data: np.ndarray):
        """smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        example:
        import numpy as np
        t = np.linspace(-2,2,0.1)
        x = np.sin(t)+np.random.randn(len(t))*0.1
        y = smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        
        :param x: the frame difference list
        :type x: numpy.ndarray
        :param window_len: the dimension of the smoothing window
        :type window_len: slidding window length
        :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' flat window will produce a moving average smoothing.
        :type window: str
        :return: the smoothed signal
        :rtype: ndarray
        """
        # This function takes
        if frame_data.ndim != 1:
            raise (ValueError, "smooth only accepts 1 dimension arrays.")

        if frame_data.size < self.len_window:
            raise (ValueError, "Input vector needs to be bigger than window size.")

        if self.len_window < 3:
            return frame_data

        if not self.window_type in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise (
                ValueError,
                "Smoothing Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'",
            )

        # Doing row-wise merging of frame differences wrt window length. frame difference
        # by factor of two and subtracting the frame differences from index == window length in reverse direction
        s = np.r_[
            2 * frame_data[0] - frame_data[self.len_window:1:-1], 
            frame_data, 
            2 * frame_data[-1] - frame_data[-1:-self.len_window:-1]
        ]

        if self.window_type == "flat":  # moving average
            w = np.ones(self.len_window, "d")
        else:
            w = getattr(np, self.window_type)(self.len_window)
        y = np.convolve(w / w.sum(), s, mode="same")
        return y[self.len_window - 1 : -self.len_window + 1]


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
