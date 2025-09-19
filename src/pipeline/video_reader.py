import cv2
from pathlib import Path
from typing import Generator


class VideoReader:
    """
    Reads a video file and yields frames.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def __iter__(self) -> Generator:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {self.video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
