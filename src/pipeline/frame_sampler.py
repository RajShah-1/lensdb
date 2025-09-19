from typing import Generator

import cv2


class FrameSampler:
    """Samples frames from a video stream at a fixed interval."""

    def __init__(self, every_n_frames: int = 30):
        self.every_n_frames = every_n_frames

    def sample(self, video_stream) -> Generator:
        for idx, frame in enumerate(video_stream):
            if idx % self.every_n_frames == 0:
                yield frame

class TimeBasedSampler:
    """Samples frames every `interval_sec` seconds."""

    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec

    def sample(self, video_path: str) -> Generator:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Could not determine FPS for {video_path}")

        frame_interval = int(round(fps * self.interval_sec))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                yield frame
            frame_idx += 1

        cap.release()

