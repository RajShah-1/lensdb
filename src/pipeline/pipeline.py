from pathlib import Path
import os
import cv2
import numpy as np

from src.pipeline.video_reader import VideoReader
from src.embeddings.embedder import Embedder
from src.utils import get_best_device

class VideoPipeline:
    """End-to-end pipeline: read video → sample frames → embed."""
    def __init__(self, video_path: str, embedder: Embedder, out_dir: str | None = None):
        self.video_path = Path(video_path)
        self.embedder = embedder
        self.out_dir = Path(out_dir) if out_dir else Path("data") / self.video_path.stem
        self.frames_dir = self.out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def run(self, save: bool):
        reader = VideoReader(str(self.video_path))
        embeddings = []
        count = 0
        frame_counter = 0
        print("Using device:", get_best_device())
        for batch in reader:
            embs = self.embedder.embed(batch)
            embeddings.extend(embs)
            if save: # save frames as images
                for f in batch:
                    frame_path = self.frames_dir / f"frame_{frame_counter:05d}.jpg"
                    cv2.imwrite(str(frame_path), f)
                    frame_counter += 1
            count += len(batch)
            if count % 100 == 0:
                print(f"Processed {count} frames")

        embeddings = np.array(embeddings)
        if save:
            np.save(self.out_dir / "embds.npy", embeddings)
            print(f"Saved {embeddings.shape[0]} embeddings to {self.out_dir/'embds.npy'}")
            print(f"Saved {frame_counter} frames under {self.frames_dir}/")

        return embeddings
