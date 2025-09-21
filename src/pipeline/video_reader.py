from pathlib import Path
from typing import Generator, Literal

import cv2

class VideoReader:
    """
    Reads and samples frames efficiently from a video file.

    Modes:
      - "frame": sample every N frames
      - "time":  sample every T seconds
    """
    def __init__(
        self,
        video_path: str,
        mode: Literal["frame", "time"],
        every_n_frames: int,
        interval_sec: float,
        batch_size: int,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.mode = mode
        self.every_n_frames = every_n_frames
        self.interval_sec = interval_sec
        self.batch_size = batch_size

    def __iter__(self) -> Generator:
        cap = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print("Warn: FPS Unknown")
            fps = 30

        if self.mode == "frame":
            step = self.every_n_frames
        elif self.mode == "time":
            step = int(round(fps * self.interval_sec))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        frame_idx = 0
        batch = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                batch.append(frame)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            frame_idx += 1

        if batch:
            yield batch

        cap.release()
